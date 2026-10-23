"""Tests for fedgrant/analysis/stats.py and runs_schema.py (Sections 11, 12, 14), on a mock
runs.csv generated from the full experimental matrix with known effects."""

from __future__ import annotations

import csv
import math
import random

import numpy as np
import pytest
from scipy.stats import ttest_rel

from fedgrant.analysis.runs_schema import (
    COLUMNS,
    FL_EXPERIMENTS,
    RunsSchemaError,
    expected_runs,
    make_config_id,
    make_row,
    parse_row,
    read_runs_csv,
    write_runs_csv,
)
from fedgrant.analysis.stats import (
    comparisons_for,
    holm,
    main,
    paired_t,
    t_ci,
)

# known effects on eta (network) and on final_acc / tta (FL)
ETA_EFFECT = {"FG-GRU": 0.06, "FG-MLP": 0.04, "B7": 0.03, "B6": 0.02, "B4": 0.0, "B3": 0.0, "B5": 0.0,
              "B2": -0.05, "B1": -0.08, "ORACLE": 0.09, "FG-GRU-A2": 0.04, "FG-GRU-A3": 0.05, "FG-GRU-A4": 0.055}


def _parse_cfg(cid: str) -> tuple[str, float, int, float]:
    s, b, n, c = cid.split("-")
    return s, float(b[1:]), int(n[1:]), float(c[1:])


def mock_rows(noise: float = 0.01, seed: int = 0, drop: set | None = None, censor_policy: str = "B2"):
    """One row per design run. eta = base(config) + seed effect + policy effect + noise."""
    rng = np.random.default_rng(seed)
    seed_eff = rng.normal(0, 0.03, 5)                       # shared by all policies: pairing removes it
    rows = []
    for e, cid, p, k in expected_runs():
        if drop and (e, cid, p, k) in drop:
            continue
        s, b, n, c = _parse_cfg(cid)
        n_rounds = 300 if e in FL_EXPERIMENTS else (100 if e in ("M1", "M-sim") else 200)
        base = 0.5 + 0.5 * b
        eta = base + seed_eff[k] + ETA_EFFECT.get(p, 0.0) + rng.normal(0, noise)
        if e == "M1":
            eta += 0.02                                     # Mininet systematically higher than its twin
        m = dict(eta=eta, utilization=0.8 + rng.normal(0, 0.01), wasted_grants=0.1 + rng.normal(0, 0.01))
        cens = None
        if e in FL_EXPERIMENTS:
            acc = 0.80 + (0.0 if p == "FULL" else ETA_EFFECT.get(p, 0.0) / 2) + seed_eff[k] / 5
            cens = p == censor_policy and e == "F-main"
            m.update(final_acc=acc, tta_seconds=100.0)
            if not cens:
                m["tta_rounds"] = 150 - 300 * ETA_EFFECT.get(p, 0.0) + 10 * k
            if p == "FULL":
                m.pop("eta")
        rows.append(make_row(e, s, b, n, p, k, n_rounds, cross_mbps=c, tta_censored=cens, **m))
    return rows


@pytest.fixture(scope="module")
def analysed(tmp_path_factory):
    d = tmp_path_factory.mktemp("stats")
    rows = mock_rows()
    random.Random(1).shuffle(rows)                          # pairing must not depend on row order
    write_runs_csv(rows, d / "runs.csv")
    res = main(["--runs", str(d / "runs.csv"), "--out", str(d / "out"), "--tables", str(d / "tables")])
    return d, res


# --------------------------------------------------------------------------- schema


def test_design_size_matches_section_11() -> None:
    runs = expected_runs()
    assert len(runs) == len(set(runs)) == 675 + 80 + 80 + 60 + 96 + 3 + 12 + 36 + 2 + 30
    assert sum(1 for r in runs if r[0] == "N-main") == 675


def test_config_id_and_row_validation() -> None:
    assert make_config_id("L2", 0.1, 20, 0) == "L2-b0.10-N20-c0"
    assert make_config_id("L4", 0.10, 20, 3.0) == "L4-b0.10-N20-c3"
    r = make_row("N-main", "L2", 0.1, 20, "FG-GRU", 0, 200, eta=0.7)
    assert list(r) == list(COLUMNS)
    with pytest.raises(RunsSchemaError):
        make_row("N-main", "L2", 0.1, 20, "FG-GRU", 0, 200)                 # no eta
    with pytest.raises(RunsSchemaError):
        make_row("F-main", "L2", 0.1, 20, "FG-GRU", 0, 300, final_acc=0.8)  # tta_censored missing
    with pytest.raises(RunsSchemaError):
        make_row("N-main", "L2", 0.1, 20, "FG-XYZ", 0, 200, eta=0.7)
    with pytest.raises(RunsSchemaError):
        make_row("N-main", "L2", 0.1, 20, "FG-GRU", 0, 200, eta=0.7, bogus=1.0)
    bad = dict(r, config_id="L2-b0.1-N20-c0")
    with pytest.raises(RunsSchemaError):
        parse_row(bad)
    with pytest.raises(RunsSchemaError):
        parse_row(dict(r, platform="mininet"))                                # N-main is sim
    m = make_row("N-main", "L2", 0.1, 20, "FG-GRU", 0, 200, status="missing")  # missing: metrics may be blank
    assert parse_row(m).status == "missing"


def test_duplicates_rejected(tmp_path) -> None:
    r = make_row("N-main", "L2", 0.1, 20, "FG-GRU", 0, 200, eta=0.7)
    with pytest.raises(RunsSchemaError):
        write_runs_csv([r, r], tmp_path / "x.csv")


def test_censored_tta_entered_as_300(tmp_path) -> None:
    r = make_row("F-main", "L2", 0.1, 20, "B2", 1, 300, final_acc=0.6, tta_censored=True)
    write_runs_csv([r], tmp_path / "r.csv")
    run = read_runs_csv(tmp_path / "r.csv")[0]
    assert run.tta_censored is True and math.isnan(run.metrics["tta_rounds"]) and run.value("tta_rounds") == 300.0


# --------------------------------------------------------------------------- statistics


def test_t_ci_and_paired_t_match_scipy() -> None:
    s = t_ci([1.0, 2.0, 4.0, 4.5, 7.0])
    assert s["t_crit"] == pytest.approx(2.776, abs=5e-4)
    assert t_ci([1.0, 2.0, 3.0])["t_crit"] == pytest.approx(4.303, abs=5e-4)
    a, b = np.array([0.7, 0.8, 0.65, 0.9, 0.75]), np.array([0.6, 0.72, 0.66, 0.8, 0.7])
    t, p = paired_t(a - b)
    ref = ttest_rel(a, b)
    assert t == pytest.approx(ref.statistic) and p == pytest.approx(ref.pvalue)
    assert paired_t([0.1, 0.1, 0.1]) == (math.inf, 0.0) and paired_t([0.0, 0.0])[1] == 1.0
    assert math.isnan(paired_t([0.1])[1])


def test_holm_hand_computed() -> None:
    p = [0.01, 0.04, 0.03, 0.005, 0.20]
    # sorted: 0.005(x5)=0.025, 0.01(x4)=0.04, 0.03(x3)=0.09, 0.04(x2)=0.08 -> max 0.09, 0.20(x1)=0.20
    assert holm(p) == pytest.approx([0.04, 0.09, 0.09, 0.025, 0.20])
    assert holm([0.3, 0.5]) == pytest.approx([0.6, 0.6])
    out = holm([0.01, float("nan"), 0.02], m=3)                 # NaN keeps its place in the family
    assert out[0] == pytest.approx(0.03) and math.isnan(out[1]) and out[2] == pytest.approx(0.04)


def test_comparison_families() -> None:
    assert len(comparisons_for("N-main")) == 5
    assert len(comparisons_for("N-X1")) == len(comparisons_for("F-main")) == 4
    assert comparisons_for("M1") == [("FG-GRU", "B3"), ("FG-GRU", "B6"), ("B6", "B4")]
    assert comparisons_for("N-abl") == [] and comparisons_for("F-ref") == []


# --------------------------------------------------------------------------- end to end


def _read(path):
    return list(csv.DictReader(path.open()))


def test_outputs_written(analysed) -> None:
    d, _ = analysed
    for f in ("stats_groups.csv", "stats_paired.csv", "stats_ablation.csv", "stats_mininet_vs_sim.csv",
              "stats_missing.csv", "stats_meta.json"):
        assert (d / "out" / f).is_file(), f
    for f in ("paired_N-main.tex", "groups_N-main.tex", "paired_F-main.tex", "mininet_vs_sim.tex"):
        assert (d / "tables" / f).is_file(), f
    assert r"\begin{tabular}" in (d / "tables" / "paired_N-main.tex").read_text()


def test_ci_coverage_of_known_effect(analysed) -> None:
    """FG-GRU minus B3 has true effect 0.06 in every network config; most CIs must cover it."""
    _, res = analysed
    rows = [r for r in res["paired"] if r["comparison"] == "FG-GRU vs B3" and r["metric"] == "eta"
            and r["experiment_id"] in ("N-main", "N-X1", "N-X2")]
    assert len(rows) == 15 + 2 + 2
    covered = sum(r["ci_lo"] <= 0.06 <= r["ci_hi"] for r in rows)
    assert covered >= 15                                     # 95% nominal; 19 configs
    assert all(r["n_pairs"] == 5 and r["holm_m"] in (4, 5) for r in rows)
    assert all(r["significant"] for r in rows)               # effect 0.06 vs noise 0.014 per diff
    r = rows[0]
    assert r["rel_effect"] == pytest.approx(r["mean_diff"] / r["ref_mean"])


def test_pairing_removes_shared_seed_effect(analysed) -> None:
    """seed effect sd 0.03 is shared: the paired sd must be near the noise level, not 0.03."""
    _, res = analysed
    sds = [r["sd"] for r in res["paired"] if r["comparison"] == "B6 vs B4" and r["experiment_id"] == "N-main"]
    assert np.median(sds) < 0.025


def test_holm_applied_within_config(analysed) -> None:
    _, res = analysed
    fam = [r for r in res["paired"] if r["experiment_id"] == "N-main" and r["config_id"] == "L1-b0.05-N20-c0"]
    assert len(fam) == 5
    assert [r["p_holm"] for r in fam] == pytest.approx(holm([r["p_raw"] for r in fam]))


def test_censored_tta_flagged(analysed) -> None:
    _, res = analysed
    r = next(r for r in res["paired"] if r["experiment_id"] == "F-main" and r["metric"] == "tta_rounds"
             and r["comparison"] == "FG-GRU vs B3")
    assert r["n_censored_a"] == 0 and r["n_censored_b"] == 0
    g = next(g for g in res["groups"] if g["experiment_id"] == "F-main" and g["policy"] == "B2"
             and g["metric"] == "tta_rounds")
    assert g["n_censored"] == 3 and g["mean"] == 300.0
    assert {r["metric"] for r in res["paired"] if r["experiment_id"] == "F-main"} == {"tta_rounds", "final_acc"}


def test_mininet_vs_sim(analysed) -> None:
    _, res = analysed
    rows = res["mininet_vs_sim"]
    b1 = [r for r in rows if r["policy"] == "B1"]
    assert b1 and all(r["note"] == "no M-sim twin" for r in b1)
    eta = [r for r in rows if r["metric"] == "eta"]
    assert len(eta) == 2 * 5 and all(r["n_pairs"] == 3 for r in eta)
    assert all(r["ci_lo"] <= 0.02 <= r["ci_hi"] for r in eta)


def test_ablation_vs_parent(analysed) -> None:
    _, res = analysed
    rows = [r for r in res["ablation"] if r["experiment_id"] == "N-abl"]
    assert len(rows) == 3 * 4                                # 3 variants x 4 configs
    assert all(r["reference_experiment"] == "N-main" and r["n_pairs"] == 5 for r in rows)
    true = {"FG-GRU-A2": -0.02, "FG-GRU-A3": -0.01, "FG-GRU-A4": -0.005}
    covered = sum(r["ci_lo"] <= true[r["policy_a"]] <= r["ci_hi"] for r in rows)
    assert covered >= 9                                      # 95% nominal on 12 intervals
    f = [r for r in res["ablation"] if r["experiment_id"] == "F-abl"]
    assert f and all(r["reference_experiment"] == "F-main" and r["n_pairs"] == 3 for r in f)


def test_missing_runs_reported_not_dropped(tmp_path) -> None:
    gone = ("N-main", "L2-b0.10-N20-c0", "B3", 2)
    rows = mock_rows(drop={gone})
    for r in rows:
        if (r["experiment_id"], r["config_id"], r["policy"], r["seed"]) == ("N-main", "L2-b0.10-N20-c0", "FG-GRU", 4):
            r.update(status="missing", eta="")
    write_runs_csv(rows, tmp_path / "runs.csv")
    res = main(["--runs", str(tmp_path / "runs.csv"), "--out", str(tmp_path / "o"), "--tables", str(tmp_path / "t")])
    miss = {(m["kind"], m["policy"], m["seed"]) for m in res["missing"]}
    assert ("absent", "B3", 2) in miss and ("missing", "FG-GRU", 4) in miss
    r = next(r for r in res["paired"] if r["config_id"] == "L2-b0.10-N20-c0" and r["comparison"] == "FG-GRU vs B3"
             and r["experiment_id"] == "N-main")
    assert r["n_pairs"] == 3 and r["unpaired_seeds"] == "2;4"
    g = next(g for g in res["groups"] if g["config_id"] == "L2-b0.10-N20-c0" and g["policy"] == "B3"
             and g["metric"] == "eta" and g["experiment_id"] == "N-main")
    assert g["n_seeds"] == 4 and g["missing_seeds"] == "2"
    assert "missing" in (tmp_path / "t" / "groups_N-main.tex").read_text()