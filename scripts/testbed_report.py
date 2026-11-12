#!/usr/bin/env python3
"""Regenerate the testbed/traces report tables and figures from raw artifacts.

Inputs (all local, none fetched):
  results/mininet/ge_L{1,2,3}.json         100k-packet GE validation trials
  data/traces/traces_summary.json + CSVs   binned cellular traces (make data)
  results/mininet/*/trace_replay_client_*.csv   per-change replay timing

Outputs (byte-stable for the same inputs):
  results/summary/testbed_ge_validation.csv, testbed_trace_summary.csv,
  testbed_replay_lag.csv, testbed_tables.tex
  results/figures/testbed_ge_validation.pdf, testbed_traces.pdf,
  testbed_replay_lag.pdf

Missing inputs are reported and skipped; nothing is invented.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCENARIOS = ("L1", "L2", "L3")
TRACE_PANELS = (  # (file, label): the segments L4 (ID test) and L5 (OOD) replay
    ("Verizon-LTE-driving_test.csv", "Verizon LTE driving (ID, test)"),
    ("Verizon-LTE-short_test.csv", "Verizon LTE short (ID, test)"),
    ("ATT-LTE-driving_test.csv", "AT&T LTE driving (ID, test)"),
    ("TMobile-UMTS-driving_ood.csv", "T-Mobile UMTS driving (OOD)"),
    ("Verizon-EVDO-driving_ood.csv", "Verizon EVDO driving (OOD)"),
)
# Reference palette (dataviz skill): single-series marks use slot 1; targets
# and reference lines use muted ink, so identity never depends on color alone.
INK, MUTED, GRID, SERIES = "#1f1f1e", "#6b6a64", "#d9d8d2", "#2a78d6"
# Categorical slots 1-3 of the reference palette, in fixed order per applier.
APPLIER_COLORS = {"runner-thread": "#2a78d6", "tc": "#eb6834", "netlink": "#1baf7a"}
APPLIER_LABELS = {"runner-thread": "runner threads via host shell", "tc": "per-client agent, tc exec",
                  "netlink": "per-client agent, netlink"}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4, "pdf.fonttype": 42,
        "svg.hashsalt": "fedgrant", "legend.frameon": False, "legend.fontsize": 7,
    })
    return plt


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None, "Producer": None})


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# --- Gilbert-Elliott validation --------------------------------------------------------

def ge_rows(ge_dir: Path) -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        path = ge_dir / f"ge_{scenario}.json"
        if not path.is_file():
            print(f"skip: {path} not found", file=sys.stderr)
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        loss_ok = d["loss_rate_ci_low"] <= d["target_loss_rate"] <= d["loss_rate_ci_high"]
        burst_ok = d["mean_burst_ci_low"] <= d["target_mean_burst"] <= d["mean_burst_ci_high"]
        rows.append({
            "scenario": scenario, "packets": d["n"], "lost": d["lost"], "bursts": d["n_bursts"],
            "target_loss": f"{d['target_loss_rate']:.4f}", "loss": f"{d['loss_rate']:.5f}",
            "loss_ci_low": f"{d['loss_rate_ci_low']:.5f}", "loss_ci_high": f"{d['loss_rate_ci_high']:.5f}",
            "target_burst": f"{d['target_mean_burst']:.4f}", "burst": f"{d['mean_burst']:.4f}",
            "burst_ci_low": f"{d['mean_burst_ci_low']:.4f}", "burst_ci_high": f"{d['mean_burst_ci_high']:.4f}",
            "target_in_both_ci": "yes" if loss_ok and burst_ok else "no",
            "loss_ci_method": d.get("loss_interval_method", ""), "burst_ci_method": d.get("burst_interval_method", ""),
        })
    return rows


def plot_ge(rows: list[dict], path: Path) -> None:
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.2))
    x = np.arange(len(rows))
    for ax, key, label, scale in ((axes[0], "loss", "Loss rate (%)", 100.0),
                                  (axes[1], "burst", "Mean loss burst (packets)", 1.0)):
        value = np.array([float(r[key]) for r in rows]) * scale
        low = np.array([float(r[f"{key}_ci_low"]) for r in rows]) * scale
        high = np.array([float(r[f"{key}_ci_high"]) for r in rows]) * scale
        target = np.array([float(r[f"target_{key}"]) for r in rows]) * scale
        ax.errorbar(x, value, yerr=[value - low, high - value], fmt="o", color=SERIES, ms=5, capsize=3,
                    elinewidth=1.2, label="Measured (95% CI)")
        ax.scatter(x + 0.18, target, marker="D", s=22, facecolors="none", edgecolors=MUTED, linewidths=1.2,
                   label="Target", zorder=3)
        ax.set_xticks(x, [r["scenario"] for r in rows])
        ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.set_ylabel(label)
        if key == "burst":
            ax.set_yscale("log")
            ax.set_yticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
            ax.minorticks_off()
        else:
            ax.set_ylim(0, max(high.max(), target.max()) * 1.25)
    axes[0].legend(loc="lower left")
    fig.suptitle("Gilbert–Elliott uplink loss, 100,000 packets per scenario (netem, Fedora testbed)",
                 fontsize=8, color=INK)
    _save(fig, path)
    plt.close(fig)


# --- Traces ----------------------------------------------------------------------------------

def trace_rows(trace_dir: Path) -> list[dict]:
    summary_path = trace_dir / "traces_summary.json"
    if not summary_path.is_file():
        print(f"skip: {summary_path} not found (run make data)", file=sys.stderr)
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for name in sorted(summary):
        s = summary[name]
        trace, split = name.removesuffix(".csv").rsplit("_", 1)
        rows.append({"trace": trace, "split": split, "bins": s["n_bins"], "duration_s": f"{s['duration_ms'] / 1000:.1f}",
                     "mean_mbps": f"{s['mean_bps'] / 1e6:.3f}", "p05_mbps": f"{s['p05_bps'] / 1e6:.3f}",
                     "p95_mbps": f"{s['p95_bps'] / 1e6:.3f}"})
    return rows


def plot_traces(trace_dir: Path, path: Path) -> bool:
    available = [(f, label) for f, label in TRACE_PANELS if (trace_dir / f).is_file()]
    if not available:
        print(f"skip: no trace CSVs under {trace_dir}", file=sys.stderr)
        return False
    plt = _style()
    fig, axes = plt.subplots(len(available), 1, figsize=(6.0, 1.05 * len(available)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, (fname, label) in zip(axes, available, strict=True):
        data = np.genfromtxt(trace_dir / fname, delimiter=",", names=True)
        t_s, mbps = data["t_ms"] / 1000.0, data["rate_bps"] / 1e6
        ax.plot(t_s, mbps, color=SERIES, linewidth=0.7)
        ax.axhline(float(np.mean(mbps)), color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_title(f"{label} — mean {np.mean(mbps):.2f} Mbit/s (dashed)", loc="left", color=INK)
        ax.set_ylabel("Mbit/s")
        ax.set_xlim(t_s[0], t_s[-1])
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel("Time within segment (s); 100 ms bins, before per-client scaling to 2/4/6 Mbit/s")
    fig.tight_layout(h_pad=0.6)
    _save(fig, path)
    plt.close(fig)
    return True


# --- Trace replay timing ------------------------------------------------------------------------

def replay_lags(run_glob_root: Path) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows, per_run = [], {}
    for run_dir in sorted({p.parent for p in run_glob_root.glob("*/trace_replay_client_*.csv")}):
        lags = []
        for path in sorted(run_dir.glob("trace_replay_client_*.csv")):
            with path.open(newline="", encoding="utf-8") as stream:
                lags += [int(r["lag_ns"]) / 1e6 for r in csv.DictReader(stream)]
        if not lags:
            continue
        arr = np.asarray(lags)
        per_run[run_dir.name] = arr
        meta_path = run_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        appliers = sorted({line.split("applier=")[1].split()[0] for log in run_dir.glob("replay_agent_*.log")
                           for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                           if "applier=" in line}) or (
            ["tc"] if any(run_dir.glob("replay_agent_*.log")) else ["runner-thread"])
        rows.append({"run": run_dir.name, "scenario": meta.get("scenario", ""), "clients": meta.get("N", ""),
                     "applier": "+".join(appliers),
                     "rate_changes": arr.size, "lag_p50_ms": f"{np.percentile(arr, 50):.3f}",
                     "lag_p99_ms": f"{np.percentile(arr, 99):.3f}", "lag_max_ms": f"{arr.max():.3f}",
                     "late_over_10ms": int(np.count_nonzero(arr > 10.0))})
    return rows, per_run


def plot_replay(rows: list[dict], lags: dict[str, np.ndarray], path: Path, *, min_changes: int = 100) -> bool:
    """CDF of replay lag per M1-scale run, one line per applier (latest run of each)."""
    latest: dict[str, dict] = {}
    for row in rows:
        if int(row["rate_changes"]) >= min_changes:
            latest[row["applier"]] = row  # rows are sorted by run name; keep the last per applier
    order = [a for a in APPLIER_COLORS if a in latest]
    if not order:
        return False
    plt = _style()
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    for applier in order:
        row = latest[applier]
        x = np.sort(lags[row["run"]])
        y = np.arange(1, x.size + 1) / x.size
        ax.step(x, y, where="post", color=APPLIER_COLORS[applier], linewidth=1.6,
                label=f"{APPLIER_LABELS.get(applier, applier)} (p50 {float(row['lag_p50_ms']):.1f} ms)")
    ax.axvline(10, color=MUTED, linewidth=0.8, linestyle="--")
    ax.annotate("10 ms (10% of a bin)", (10, 0.04), xytext=(3, 0), textcoords="offset points", color=INK, fontsize=7)
    ax.set_xlabel("tc rate-change lag after its 100 ms bin boundary (ms)")
    ax.set_ylabel("Fraction of changes")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    n = int(latest[order[0]]["clients"] or 0)
    ax.set_title(f"Trace replay timing, N = {n} clients, 25 rounds", loc="left", color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    _save(fig, path)
    plt.close(fig)
    return True


# --- LaTeX tables ----------------------------------------------------------------------------------

def latex_tables(ge: list[dict], traces: list[dict], replay: list[dict]) -> str:
    out = ["% Generated by scripts/testbed_report.py -- do not edit by hand.", ""]
    if ge:
        out += [r"\begin{tabular}{lrrrrc}", r"\toprule",
                r"Scenario & Target loss & Measured loss (95\% CI) & Target burst & Measured burst (95\% CI) & Both covered \\",
                r"\midrule"]
        for r in ge:
            out.append(
                f"{r['scenario']} & {100 * float(r['target_loss']):.1f}\\% & "
                f"{100 * float(r['loss']):.3f}\\% [{100 * float(r['loss_ci_low']):.3f}, {100 * float(r['loss_ci_high']):.3f}] & "
                f"{float(r['target_burst']):.2f} & {float(r['burst']):.3f} [{float(r['burst_ci_low']):.3f}, "
                f"{float(r['burst_ci_high']):.3f}] & {r['target_in_both_ci']} \\\\")
        out += [r"\bottomrule", r"\end{tabular}", ""]
    if traces:
        out += [r"\begin{tabular}{llrrrr}", r"\toprule",
                r"Trace & Split & Duration (s) & Mean (Mbit/s) & p5 & p95 \\", r"\midrule"]
        for r in traces:
            out.append(f"{r['trace']} & {r['split']} & {r['duration_s']} & {r['mean_mbps']} & "
                       f"{r['p05_mbps']} & {r['p95_mbps']} \\\\")
        out += [r"\bottomrule", r"\end{tabular}", ""]
    if replay:
        out += [r"\begin{tabular}{lrrrr}", r"\toprule",
                r"Run & Changes & p50 lag (ms) & p99 lag (ms) & Max (ms) \\", r"\midrule"]
        for r in replay:
            out.append(f"{r['run'].replace('_', r'\_')} & {r['rate_changes']} & {r['lag_p50_ms']} & "
                       f"{r['lag_p99_ms']} & {r['lag_max_ms']} \\\\")
        out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ge-dir", default="results/mininet")
    parser.add_argument("--trace-dir", default="data/traces")
    parser.add_argument("--runs-root", default="results/mininet")
    parser.add_argument("--summary-out", default="results/summary")
    parser.add_argument("--figures-out", default="results/figures")
    args = parser.parse_args(argv)
    summary, figures = Path(args.summary_out), Path(args.figures_out)

    ge = ge_rows(Path(args.ge_dir))
    if ge:
        _write_csv(summary / "testbed_ge_validation.csv", list(ge[0]), ge)
        plot_ge(ge, figures / "testbed_ge_validation.pdf")
    traces = trace_rows(Path(args.trace_dir))
    if traces:
        _write_csv(summary / "testbed_trace_summary.csv", list(traces[0]), traces)
    plot_traces(Path(args.trace_dir), figures / "testbed_traces.pdf")
    replay, lags = replay_lags(Path(args.runs_root))
    if replay:
        _write_csv(summary / "testbed_replay_lag.csv", list(replay[0]), replay)
        plot_replay(replay, lags, figures / "testbed_replay_lag.pdf")
    (summary / "testbed_tables.tex").write_text(latex_tables(ge, traces, replay), encoding="utf-8")
    print(f"GE scenarios: {len(ge)}, trace segments: {len(traces)}, replay runs: {len(replay)}")


if __name__ == "__main__":
    main()
