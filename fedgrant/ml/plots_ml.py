"""ML figures (Section 4.7), drawn ONLY from the CSVs in results/summary/.

    python -m fedgrant.ml.plots_ml --summary results/summary --out results/figures

Inputs (written by fedgrant.ml.evaluate and scripts/bench_predictors.py):
  ml_reliability.csv, ml_metrics.csv, ml_metrics_summary.csv, pred_latency.csv
Outputs (PDF):
  ml_reliability.pdf  reliability diagrams, 15 bins, one panel per predictor,
                      Test-ID and Test-OOD rows. P-MLP / P-GRU bins are pooled over
                      seeds (counts summed, means count-weighted).
  ml_metrics.pdf      Brier / ECE / ROC-AUC / BCE, one panel per metric (no shared
                      axis), bars per predictor for ID and OOD; P-MLP / P-GRU show
                      the mean over seeds with the 95% t-interval.
  ml_latency.pdf      table of pred_latency.csv against the spec reference values.
Any input that is absent is skipped with a message, so the command never fails
because one stage has not run yet.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PREDICTORS: tuple[str, ...] = ("ewma", "hmm", "mlp", "gru", "oracle")
LABEL: dict[str, str] = {"ewma": "P-EWMA", "hmm": "P-HMM", "mlp": "P-MLP", "gru": "P-GRU", "oracle": "P-ORACLE"}
SPLITS: tuple[str, ...] = ("test_id", "test_ood")
SPLIT_LABEL: dict[str, str] = {"test_id": "Test-ID", "test_ood": "Test-OOD"}
METRICS: tuple[str, ...] = ("brier", "ece", "auc", "bce")
METRIC_LABEL: dict[str, str] = {"brier": "Brier (lower is better)", "ece": "ECE (lower is better)",
                                "auc": "ROC-AUC (higher is better)", "bce": "BCE (lower is better)"}
# Validated categorical slots (light mode), assigned in fixed order; the two
# splits use slots 1 and 2, every predictor panel is a single series.
SPLIT_COLOR: dict[str, str] = {"test_id": "#2a78d6", "test_ood": "#eb6834"}
SERIES = "#2a78d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
})


def _read(path: Path) -> list[dict[str, str]] | None:
    if not path.is_file():
        print(f"skip: {path} not found")
        return None
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


# --------------------------------------------------------------------------- reliability


def pooled_bins(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, float]]]:
    """{(split, predictor): 15 bins} with counts summed over seeds and count-weighted means."""
    acc: dict[tuple[str, str, int], dict[str, float]] = defaultdict(lambda: dict(n=0.0, sp=0.0, sy=0.0))
    edges: dict[int, tuple[float, float]] = {}
    for r in rows:
        b, n = int(r["bin"]), _f(r["n"])
        a = acc[(r["split"], r["predictor"], b)]
        edges[b] = (_f(r["lo"]), _f(r["hi"]))
        if n > 0:
            a["n"] += n
            a["sp"] += n * _f(r["mean_p"])
            a["sy"] += n * _f(r["mean_y"])
    out: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for (s, p, b), a in sorted(acc.items()):
        n = a["n"]
        out[(s, p)].append(dict(bin=b, lo=edges[b][0], hi=edges[b][1], n=n,
                                mean_p=a["sp"] / n if n else math.nan, mean_y=a["sy"] / n if n else math.nan))
    return out


def plot_reliability(rows: list[dict[str, str]], out: Path) -> Path:
    bins = pooled_bins(rows)
    preds = [p for p in PREDICTORS if any(k[1] == p for k in bins)]
    splits = [s for s in SPLITS if any(k[0] == s for k in bins)]
    fig, axes = plt.subplots(len(splits), len(preds), figsize=(2.0 * len(preds) + 0.4, 2.1 * len(splits) + 0.5),
                             squeeze=False, sharex=True, sharey=True)
    for i, s in enumerate(splits):
        for j, p in enumerate(preds):
            ax = axes[i][j]
            ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=(0, (3, 3)), label="perfect calibration")
            b = [x for x in bins.get((s, p), []) if x["n"] > 0]
            if b:
                ax.plot([x["mean_p"] for x in b], [x["mean_y"] for x in b], color=SERIES, lw=2,
                        marker="o", ms=4, mec="#fcfcfb", mew=1.0)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            if i == 0:
                ax.set_title(LABEL[p] + (" (pooled seeds)" if p in ("mlp", "gru") else ""), fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{SPLIT_LABEL[s]}\nobserved delivery rate")
            if i == len(splits) - 1:
                ax.set_xlabel("predicted p̂")
    fig.suptitle("Reliability diagrams (15 equal-width bins)", fontsize=10, color=INK)
    fig.tight_layout()
    path = out / "ml_reliability.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- metric bars


def _metric_values(metrics: list[dict[str, str]], summary: list[dict[str, str]] | None):
    """{(split, predictor, metric): (mean, ci_lo, ci_hi, n_seeds)}."""
    vals: dict[tuple[str, str, str], tuple[float, float, float, int]] = {}
    if summary:
        for r in summary:
            vals[(r["split"], r["predictor"], r["metric"])] = (_f(r["mean"]), _f(r["ci_lo"]), _f(r["ci_hi"]),
                                                               int(_f(r["n_seeds"]) or 0))
        return vals
    acc: dict[tuple[str, str, str], list[float]] = defaultdict(list)       # fallback: from ml_metrics.csv
    for r in metrics:
        acc[(r["split"], r["predictor"], r["metric"])].append(_f(r["value"]))
    for k, v in acc.items():
        vals[k] = (sum(v) / len(v), math.nan, math.nan, len(v))
    return vals


def plot_metrics(metrics: list[dict[str, str]], summary: list[dict[str, str]] | None, out: Path) -> Path:
    vals = _metric_values(metrics, summary)
    preds = [p for p in PREDICTORS if any(k[1] == p for k in vals)]
    splits = [s for s in SPLITS if any(k[0] == s for k in vals)]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.0 * len(METRICS), 2.8), squeeze=False)
    width = 0.8 / max(1, len(splits))
    for m_i, m in enumerate(METRICS):
        ax = axes[0][m_i]
        for s_i, s in enumerate(splits):
            xs, hs, lo, hi = [], [], [], []
            for p_i, p in enumerate(preds):
                v = vals.get((s, p, m))
                if v is None or math.isnan(v[0]):
                    continue
                x = p_i + (s_i - (len(splits) - 1) / 2) * width
                xs.append(x)
                hs.append(v[0])
                lo.append(v[0] - v[1] if not math.isnan(v[1]) else 0.0)
                hi.append(v[2] - v[0] if not math.isnan(v[2]) else 0.0)
            ax.bar(xs, hs, width=width * 0.92, color=SPLIT_COLOR[s], label=SPLIT_LABEL[s], zorder=2)
            if any(lo) or any(hi):
                ax.errorbar(xs, hs, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
        ax.set_xticks(range(len(preds)))
        ax.set_xticklabels([LABEL[p] for p in preds], rotation=30, ha="right")
        ax.set_title(METRIC_LABEL[m])
        ax.grid(axis="x", visible=False)
        if m == "auc":
            ax.set_ylim(0.5, 1.0)
    axes[0][0].legend(loc="upper left", fontsize=7)
    fig.text(0.5, 0.005, "P-MLP / P-GRU: mean over training seeds with 95% t-interval; "
             "P-EWMA, P-HMM, P-ORACLE: single value.", ha="center", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = out / "ml_metrics.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- latency table


def plot_latency(rows: list[dict[str, str]], out: Path) -> Path:
    head = ["Case", "Clients", "Median (µs)", "p99 (µs)", "Spec (µs)"]
    cell = []
    for r in rows:
        spec = _f(r.get("spec_median_us", ""))
        cell.append([r["case"], r["n_clients"], f"{_f(r['median_us']):.1f}", f"{_f(r['p99_us']):.1f}",
                     "–" if math.isnan(spec) else f"{spec:.0f}"])
    fig, ax = plt.subplots(figsize=(6.4, 0.4 * (len(cell) + 1) + 0.7))
    ax.axis("off")
    t = ax.table(cellText=cell, colLabels=head, loc="center", cellLoc="right", colLoc="right",
                 colWidths=[0.36, 0.11, 0.17, 0.17, 0.19])
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.4)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor(GRID)
        if i == 0:
            c.set_text_props(weight="bold", color=INK)
        if j == 0:
            c.set_text_props(ha="left")
            c._loc = "left"
    src = re.sub(r"\S*/", "", rows[0].get("model_source", "")) if rows else ""      # file names only
    ax.set_title(f"NumPy predictor latency, single thread ({rows[0]['reps'] if rows else 0} reps)",
                 fontsize=9, color=INK)
    if src:
        fig.text(0.5, 0.02, f"models: {src}", ha="center", fontsize=6, color=INK2)
    path = out / "ml_latency.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- CLI


def run(summary_dir: str | Path = "results/summary", out_dir: str | Path = "results/figures") -> list[Path]:
    sd, od = Path(summary_dir), Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    rel = _read(sd / "ml_reliability.csv")
    if rel:
        made.append(plot_reliability(rel, od))
    met = _read(sd / "ml_metrics.csv")
    if met:
        made.append(plot_metrics(met, _read(sd / "ml_metrics_summary.csv"), od))
    lat = _read(sd / "pred_latency.csv")
    if lat:
        made.append(plot_latency(lat, od))
    for p in made:
        print(f"wrote {p}")
    return made


def main(argv: list[str] | None = None) -> list[Path]:
    ap = argparse.ArgumentParser(description="ML figures from results/summary CSVs (Section 4.7).")
    ap.add_argument("--summary", default="results/summary")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args(argv)
    return run(a.summary, a.out)


if __name__ == "__main__":
    main()