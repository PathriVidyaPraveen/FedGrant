"""Statistical analysis of results/summary/runs.csv (Section 14).

    python -m fedgrant.analysis.stats --runs results/summary/runs.csv \
        --out results/summary/stats --tables results/tables

Seeds are the unit of replication; rounds are never treated as samples.
Every run is used: no outlier exclusion. Runs with status "missing", and runs of
the design (Section 11) that are absent from the file, are listed in
stats_missing.csv and never dropped silently; "rerun" runs count as valid.

Outputs (CSV in --out, LaTeX in --tables):
  stats_groups.csv          per (experiment, config, policy, metric): n, mean, sd, 95% t-CI,
                            every seed value, censored count, missing seeds
  stats_paired.csv          primary comparisons per config: mean paired difference, 95% t-CI,
                            relative effect, two-sided paired t-test, Holm-adjusted p
  stats_ablation.csv        (secondary) each ablation variant vs FG-GRU of the same config
  stats_mininet_vs_sim.csv  M1 minus its M-sim twin (eta, utilization, wasted_grants)
  stats_missing.csv         absent / missing / unexpected runs
  stats_meta.json           counts and settings
  tables/paired_<exp>.tex, tables/groups_<exp>.tex, tables/mininet_vs_sim.tex

SPEC-NOTE: Holm's family is the comparisons defined for the experiment's policy
set (Section 14: five in N-main, four elsewhere; an experiment whose policy set
lacks a policy, e.g. M1 has no B7, gets the comparisons it can form). A
comparison that cannot be computed because runs are missing keeps its place in
the family (m is not reduced), which is conservative. FL has two primary metrics;
Holm is applied per metric.
SPEC-NOTE: a paired t-test on differences with zero variance returns p = 1 if
the mean difference is 0 and p = 0 otherwise (SciPy returns NaN).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import t as student_t

from fedgrant.analysis.runs_schema import (
    FL_EXPERIMENTS,
    FLOAT_METRICS,
    TTA_CENSOR_ROUNDS,
    Run,
    expected_runs,
    read_runs_csv,
)

ALPHA: float = 0.05
PRIMARY_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("FG-GRU", "B3"), ("FG-GRU", "B6"), ("FG-GRU", "B7"), ("B6", "B4"), ("FG-GRU", "FG-MLP"),
)
N_MAIN_ONLY: frozenset[tuple[str, str]] = frozenset({("FG-GRU", "FG-MLP")})
NET_PRIMARY: tuple[str, ...] = ("eta",)
FL_PRIMARY: tuple[str, ...] = ("tta_rounds", "final_acc")
MININET_VS_SIM_METRICS: tuple[str, ...] = ("eta", "utilization", "wasted_grants")
ABLATION_PARENT: dict[str, str] = {"N-abl": "N-main", "F-abl": "F-main"}


def primary_metrics(experiment: str) -> tuple[str, ...]:
    return FL_PRIMARY if experiment in FL_EXPERIMENTS else NET_PRIMARY


# --------------------------------------------------------------------------- statistics


def t_ci(values: Sequence[float], conf: float = 0.95) -> dict[str, float]:
    """n, mean, sample sd, Student-t CI (t_{0.975,4} = 2.776, t_{0.975,2} = 4.303). NaN values dropped."""
    v = np.asarray([x for x in values if not math.isnan(x)], dtype=np.float64)
    n = int(v.size)
    nan = float("nan")
    if n == 0:
        return dict(n=0, mean=nan, sd=nan, ci_lo=nan, ci_hi=nan, t_crit=nan)
    mean = float(v.mean())
    if n == 1:
        return dict(n=1, mean=mean, sd=nan, ci_lo=nan, ci_hi=nan, t_crit=nan)
    sd = float(v.std(ddof=1))
    tc = float(student_t.ppf(0.5 + conf / 2.0, n - 1))
    half = tc * sd / math.sqrt(n)
    return dict(n=n, mean=mean, sd=sd, ci_lo=mean - half, ci_hi=mean + half, t_crit=tc)


def paired_t(diffs: Sequence[float]) -> tuple[float, float]:
    """Two-sided paired t-test on the differences: (t statistic, p)."""
    d = np.asarray(diffs, dtype=np.float64)
    n = d.size
    if n < 2:
        return float("nan"), float("nan")
    if np.all(d == d[0]):                            # zero variance (std() may round to 1e-17, not 0)
        return (0.0, 1.0) if d[0] == 0.0 else (math.copysign(math.inf, float(d[0])), 0.0)
    sd = float(d.std(ddof=1))
    mean = float(d.mean())
    t = mean / (sd / math.sqrt(n))
    return float(t), float(2.0 * student_t.sf(abs(t), n - 1))


def holm(pvals: Sequence[float], m: int | None = None) -> list[float]:
    """Holm step-down adjusted p-values. NaN entries stay NaN; m defaults to the number of
    entries (NaN entries still count toward m, which is conservative)."""
    p = list(pvals)
    m = len(p) if m is None else int(m)
    idx = sorted((i for i, x in enumerate(p) if not math.isnan(x)), key=lambda i: p[i])
    out = [float("nan")] * len(p)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


# --------------------------------------------------------------------------- indexing


class RunIndex:
    """Valid runs (status ok/rerun) indexed by (experiment, config, policy) -> {seed: Run}."""

    def __init__(self, runs: Iterable[Run]) -> None:
        self.valid: dict[tuple[str, str, str], dict[int, Run]] = defaultdict(dict)
        self.missing_status: list[Run] = []
        for r in runs:
            if r.status == "missing":
                self.missing_status.append(r)
            else:
                self.valid[(r.experiment_id, r.config_id, r.policy)][r.seed] = r

    def configs(self, experiment: str) -> list[str]:
        return sorted({c for (e, c, _) in self.valid if e == experiment})

    def policies(self, experiment: str, config: str) -> list[str]:
        return sorted({p for (e, c, p) in self.valid if e == experiment and c == config})

    def seeds(self, experiment: str, config: str, policy: str) -> dict[int, Run]:
        return self.valid.get((experiment, config, policy), {})


def _design() -> tuple[dict[tuple[str, str, str], set[int]], dict[str, set[str]]]:
    seeds: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    policies: dict[str, set[str]] = defaultdict(set)
    for e, c, p, k in expected_runs():
        seeds[(e, c, p)].add(k)
        policies[e].add(p)
    return seeds, policies


DESIGN_SEEDS, DESIGN_POLICIES = _design()


def comparisons_for(experiment: str) -> list[tuple[str, str]]:
    """Primary comparisons whose two policies are both in the experiment's policy set."""
    pol = DESIGN_POLICIES.get(experiment, set())
    return [(a, b) for a, b in PRIMARY_COMPARISONS
            if a in pol and b in pol and ((a, b) not in N_MAIN_ONLY or experiment == "N-main")]


# --------------------------------------------------------------------------- tables


def _fmt_seeds(pairs: Iterable[tuple[int, float]]) -> str:
    return ";".join(f"{k}:{v:.10g}" for k, v in pairs)


def group_rows(idx: RunIndex, metrics: Sequence[str] = FLOAT_METRICS) -> list[dict[str, Any]]:
    rows = []
    for (e, c, p), by_seed in sorted(idx.valid.items()):
        design = DESIGN_SEEDS.get((e, c, p), set())
        miss = sorted(design - set(by_seed))
        for m in metrics:
            vals = [(k, by_seed[k].value(m)) for k in sorted(by_seed)]
            vals = [(k, v) for k, v in vals if not math.isnan(v)]
            if not vals:
                continue
            s = t_ci([v for _, v in vals])
            n_cens = sum(1 for k in by_seed if m == "tta_rounds" and by_seed[k].tta_censored)
            rows.append(dict(experiment_id=e, config_id=c, policy=p, metric=m, n_seeds=s["n"], mean=s["mean"],
                             sd=s["sd"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"], t_crit=s["t_crit"],
                             seed_values=_fmt_seeds(vals), n_censored=n_cens,
                             missing_seeds=";".join(map(str, miss)),
                             primary=m in primary_metrics(e)))
    return rows


def compare(idx: RunIndex, e: str, c: str, a: str, b: str, m: str,
            e_b: str | None = None) -> dict[str, Any]:
    """Paired comparison a - b on metric m (b may come from experiment e_b, e.g. the ablation parent)."""
    ra, rb = idx.seeds(e, c, a), idx.seeds(e_b or e, c, b)
    common = sorted(k for k in set(ra) & set(rb)
                    if not math.isnan(ra[k].value(m)) and not math.isnan(rb[k].value(m)))
    unpaired = sorted((set(ra) | set(rb)) - set(common))
    diffs = [ra[k].value(m) - rb[k].value(m) for k in common]
    ref = [rb[k].value(m) for k in common]
    s = t_ci(diffs)
    t, p = paired_t(diffs)
    ref_mean = float(np.mean(ref)) if ref else float("nan")
    rel = s["mean"] / ref_mean if ref and ref_mean != 0 else float("nan")
    cens = m == "tta_rounds"
    return dict(experiment_id=e, config_id=c, comparison=f"{a} vs {b}", policy_a=a, policy_b=b, metric=m,
                n_pairs=s["n"], mean_diff=s["mean"], sd=s["sd"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"],
                t_crit=s["t_crit"], ref_mean=ref_mean, rel_effect=rel, t_stat=t, p_raw=p,
                n_censored_a=sum(1 for k in common if cens and ra[k].tta_censored),
                n_censored_b=sum(1 for k in common if cens and rb[k].tta_censored),
                seed_diffs=_fmt_seeds(zip(common, diffs)), unpaired_seeds=";".join(map(str, unpaired)))


def paired_rows(idx: RunIndex) -> list[dict[str, Any]]:
    rows = []
    for e in sorted({e for (e, _, _) in idx.valid}):
        comps = comparisons_for(e)
        if not comps:
            continue
        for c in idx.configs(e):
            for m in primary_metrics(e):
                fam = [compare(idx, e, c, a, b, m) for a, b in comps]
                adj = holm([r["p_raw"] for r in fam], m=len(comps))
                for r, pa in zip(fam, adj):
                    r.update(p_holm=pa, holm_m=len(comps),
                             significant=(not math.isnan(pa)) and pa < ALPHA)
                rows.extend(fam)
    return rows


def ablation_rows(idx: RunIndex) -> list[dict[str, Any]]:
    """Secondary: FG-GRU-A* minus FG-GRU of the parent experiment, same config and seeds."""
    rows = []
    for e, parent in ABLATION_PARENT.items():
        for c in idx.configs(e):
            for a in idx.policies(e, c):
                for m in primary_metrics(e):
                    r = compare(idx, e, c, a, "FG-GRU", m, e_b=parent)
                    r["reference_experiment"] = parent
                    rows.append(r)
    return rows


def mininet_vs_sim_rows(idx: RunIndex) -> list[dict[str, Any]]:
    rows = []
    for c in idx.configs("M1"):
        for p in idx.policies("M1", c):
            twin = idx.seeds("M-sim", c, p)
            if not twin:
                rows.append(dict(config_id=c, policy=p, metric="", n_pairs=0, note="no M-sim twin"))
                continue
            mn = idx.seeds("M1", c, p)
            for m in MININET_VS_SIM_METRICS:
                common = sorted(k for k in set(mn) & set(twin)
                                if not math.isnan(mn[k].value(m)) and not math.isnan(twin[k].value(m)))
                diffs = [mn[k].value(m) - twin[k].value(m) for k in common]
                s = t_ci(diffs)
                rows.append(dict(config_id=c, policy=p, metric=m, n_pairs=s["n"], mean_diff=s["mean"], sd=s["sd"],
                                 ci_lo=s["ci_lo"], ci_hi=s["ci_hi"], t_crit=s["t_crit"],
                                 seed_diffs=_fmt_seeds(zip(common, diffs)),
                                 unpaired_seeds=";".join(map(str, sorted((set(mn) | set(twin)) - set(common)))),
                                 note=""))
    return rows


def missing_rows(runs: Sequence[Run]) -> list[dict[str, Any]]:
    present = {r.key for r in runs}
    rows = [dict(experiment_id=e, config_id=c, policy=p, seed=k, kind="absent")
            for e, c, p, k in expected_runs() if (e, c, p, k) not in present]
    rows += [dict(experiment_id=r.experiment_id, config_id=r.config_id, policy=r.policy, seed=r.seed,
                  kind="missing") for r in runs if r.status == "missing"]
    design = set(expected_runs())
    rows += [dict(experiment_id=r.experiment_id, config_id=r.config_id, policy=r.policy, seed=r.seed,
                  kind="unexpected") for r in runs if r.key not in design]
    return sorted(rows, key=lambda r: (r["kind"], r["experiment_id"], r["config_id"], r["policy"], r["seed"]))


# --------------------------------------------------------------------------- output


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(float(v)) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


def _tex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _num(x: float, d: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if isinstance(x, float) and math.isinf(x):
        return r"$\infty$" if x > 0 else r"$-\infty$"
    return f"{x:.{d}g}"


def latex_paired(rows: list[dict[str, Any]], experiment: str) -> str:
    out = [r"\begin{tabular}{llllllll}", r"\toprule",
           r"Config & Comparison & Metric & $n$ & Mean diff. & 95\% CI & Rel. & $p_{\mathrm{Holm}}$ \\",
           r"\midrule"]
    for r in rows:
        if r["experiment_id"] != experiment:
            continue
        cens = ""
        if r["metric"] == "tta_rounds" and (r["n_censored_a"] or r["n_censored_b"]):
            cens = rf"$^{{\dagger}}$"
        rel = "--" if math.isnan(r["rel_effect"]) else f"{100 * r['rel_effect']:+.1f}\\%"
        sig = "*" if r.get("significant") else ""
        out.append(f"{_tex(r['config_id'])} & {_tex(r['comparison'])} & {_tex(r['metric'])}{cens} & {r['n_pairs']} & "
                   f"{_num(r['mean_diff'])} & [{_num(r['ci_lo'])}, {_num(r['ci_hi'])}] & {rel} & "
                   f"{_num(r.get('p_holm', float('nan')), 3)}{sig} \\\\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"% * p_Holm < 0.05; $\dagger$ includes TTA censored at 300 rounds (entered as 300)"]
    return "\n".join(out) + "\n"


def latex_groups(rows: list[dict[str, Any]], experiment: str) -> str:
    out = [r"\begin{tabular}{lllll}", r"\toprule",
           r"Config & Policy & Metric & $n$ & Mean [95\% CI] \\", r"\midrule"]
    for r in rows:
        if r["experiment_id"] != experiment or not r["primary"]:
            continue
        cens = rf" ({r['n_censored']} cens.)" if r["n_censored"] else ""
        miss = rf" (missing {r['missing_seeds']})" if r["missing_seeds"] else ""
        out.append(f"{_tex(r['config_id'])} & {_tex(r['policy'])} & {_tex(r['metric'])} & {r['n_seeds']} & "
                   f"{_num(r['mean'])} [{_num(r['ci_lo'])}, {_num(r['ci_hi'])}]{cens}{_tex(miss)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def latex_mininet(rows: list[dict[str, Any]]) -> str:
    out = [r"\begin{tabular}{llllll}", r"\toprule",
           r"Config & Policy & Metric & $n$ & Mininet $-$ sim & 95\% CI \\", r"\midrule"]
    for r in rows:
        if not r.get("metric"):
            continue
        out.append(f"{_tex(r['config_id'])} & {_tex(r['policy'])} & {_tex(r['metric'])} & {r['n_pairs']} & "
                   f"{_num(r['mean_diff'])} & [{_num(r['ci_lo'])}, {_num(r['ci_hi'])}] \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


GROUP_FIELDS = ["experiment_id", "config_id", "policy", "metric", "primary", "n_seeds", "mean", "sd", "ci_lo",
                "ci_hi", "t_crit", "seed_values", "n_censored", "missing_seeds"]
PAIRED_FIELDS = ["experiment_id", "config_id", "comparison", "policy_a", "policy_b", "metric", "n_pairs",
                 "mean_diff", "sd", "ci_lo", "ci_hi", "t_crit", "ref_mean", "rel_effect", "t_stat", "p_raw",
                 "p_holm", "holm_m", "significant", "n_censored_a", "n_censored_b", "seed_diffs", "unpaired_seeds"]
ABL_FIELDS = ["experiment_id", "reference_experiment", "config_id", "comparison", "policy_a", "policy_b", "metric",
              "n_pairs", "mean_diff", "sd", "ci_lo", "ci_hi", "t_crit", "ref_mean", "rel_effect", "t_stat", "p_raw",
              "n_censored_a", "n_censored_b", "seed_diffs", "unpaired_seeds"]
MVS_FIELDS = ["config_id", "policy", "metric", "n_pairs", "mean_diff", "sd", "ci_lo", "ci_hi", "t_crit",
              "seed_diffs", "unpaired_seeds", "note"]
MISSING_FIELDS = ["kind", "experiment_id", "config_id", "policy", "seed"]


def analyze(runs: Sequence[Run]) -> dict[str, list[dict[str, Any]]]:
    idx = RunIndex(runs)
    return dict(groups=group_rows(idx), paired=paired_rows(idx), ablation=ablation_rows(idx),
                mininet_vs_sim=mininet_vs_sim_rows(idx), missing=missing_rows(runs))


def run(runs_csv: str | Path, out_dir: str | Path = "results/summary/stats",
        tables_dir: str | Path = "results/tables", log=print) -> dict[str, list[dict[str, Any]]]:
    runs = read_runs_csv(runs_csv)
    res = analyze(runs)
    out, tab = Path(out_dir), Path(tables_dir)
    _write_csv(out / "stats_groups.csv", res["groups"], GROUP_FIELDS)
    _write_csv(out / "stats_paired.csv", res["paired"], PAIRED_FIELDS)
    _write_csv(out / "stats_ablation.csv", res["ablation"], ABL_FIELDS)
    _write_csv(out / "stats_mininet_vs_sim.csv", res["mininet_vs_sim"], MVS_FIELDS)
    _write_csv(out / "stats_missing.csv", res["missing"], MISSING_FIELDS)
    tab.mkdir(parents=True, exist_ok=True)
    exps = sorted({r.experiment_id for r in runs})
    for e in exps:
        (tab / f"groups_{e}.tex").write_text(latex_groups(res["groups"], e))
        if any(r["experiment_id"] == e for r in res["paired"]):
            (tab / f"paired_{e}.tex").write_text(latex_paired(res["paired"], e))
    if res["mininet_vs_sim"]:
        (tab / "mininet_vs_sim.tex").write_text(latex_mininet(res["mininet_vs_sim"]))
    kinds = defaultdict(int)
    for r in res["missing"]:
        kinds[r["kind"]] += 1
    meta = dict(runs_csv=str(runs_csv), n_rows=len(runs), n_expected=len(expected_runs()),
                missing_counts=dict(kinds), alpha=ALPHA, tta_censor_rounds=TTA_CENSOR_ROUNDS,
                experiments=exps)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats_meta.json").write_text(json.dumps(meta, indent=1))
    log(f"{len(runs)} runs read; absent {kinds['absent']}, missing {kinds['missing']}, "
        f"unexpected {kinds['unexpected']} (see {out / 'stats_missing.csv'})")
    sig = [r for r in res["paired"] if r.get("significant")]
    log(f"{len(res['paired'])} primary comparisons, {len(sig)} significant after Holm; tables in {tab}")
    return res


def main(argv: list[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    ap = argparse.ArgumentParser(description="Statistical analysis (Section 14).")
    ap.add_argument("--runs", default="results/summary/runs.csv")
    ap.add_argument("--out", default="results/summary/stats")
    ap.add_argument("--tables", default="results/tables")
    a = ap.parse_args(argv)
    return run(a.runs, a.out, a.tables, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()