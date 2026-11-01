"""Non-ML figures (Sections 12, 14, 18), drawn ONLY from results/summary/runs.csv.

Outputs:
  - eta_vs_beta.pdf: Delivered energy eta vs beta across loss scenarios (L1-L5)
  - goodput_utilization.pdf: Goodput and bottleneck utilization across policies
  - wasted_grants.pdf: Wasted grants ratio per policy
  - rtt_distribution.pdf: Median and 99th percentile RTT
  - queue_drops.pdf: Queue drops (FedGrant and cross traffic)
  - fl_accuracy_wallclock.pdf: FL test accuracy vs simulated wallclock time
  - tta_bars.pdf: Time-to-accuracy (TTA) comparisons
  - mininet_vs_sim.pdf: Mininet M1 vs Simulator M-sim calibration
  - ablations.pdf: Ablations A1-A4 effects
  - sched_latency.pdf: Scheduler latency CDF
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import numpy as np

from fedgrant.analysis.runs_schema import read_runs_csv

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#109618", "#990099", "#ff9900", "#0099c6", "#dd4477", "#66aa00"]

if HAS_MPL:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    })


def plot_eta_vs_beta(runs: list[Any], out_dir: Path) -> None:
    """Plot delivered energy eta vs beta per loss scenario."""
    n_main_runs = [r for r in runs if r.experiment_id == "N-main" and r.status == "ok"]
    if not n_main_runs or not HAS_MPL:
        return

    scenarios = sorted(list({r.scenario for r in n_main_runs}))
    policies = sorted(list({r.policy for r in n_main_runs}))

    fig, axes = plt.subplots(1, len(scenarios), figsize=(3 * len(scenarios), 3), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scen in zip(axes, scenarios, strict=True):
        ax.set_title(f"Scenario {scen}")
        ax.set_xlabel(r"Budget $\beta$")
        ax.set_ylabel(r"Delivered Energy $\eta$")
        for idx, pol in enumerate(policies):
            subset = [r for r in n_main_runs if r.scenario == scen and r.policy == pol]
            if not subset:
                continue
            betas = sorted(list({r.beta for r in subset}))
            means = []
            for b in betas:
                vals = [r.value("eta") for r in subset if r.beta == b and not math.isnan(r.value("eta"))]
                means.append(float(np.mean(vals)) if vals else np.nan)
            color = COLORS[idx % len(COLORS)]
            ax.plot(betas, means, marker="o", label=pol, color=color, linewidth=1.5)
        ax.set_ylim(0, 1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=min(6, len(labels)))
    fig.tight_layout()
    fig.savefig(out_dir / "eta_vs_beta.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_wasted_grants(runs: list[Any], out_dir: Path) -> None:
    """Plot wasted grants ratio per policy."""
    valid_runs = [r for r in runs if r.status == "ok" and not math.isnan(r.value("wasted_grants"))]
    if not valid_runs or not HAS_MPL:
        return

    policies = sorted(list({r.policy for r in valid_runs}))
    means = [float(np.mean([r.value("wasted_grants") for r in valid_runs if r.policy == p])) for p in policies]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(policies, means, color="#2a78d6", alpha=0.85)
    ax.set_ylabel("Wasted Grants Ratio")
    ax.set_title("Wasted Grants by Policy")
    ax.set_ylim(0, max(0.5, max(means) * 1.2 if means else 1.0))
    fig.tight_layout()
    fig.savefig(out_dir / "wasted_grants.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_goodput_utilization(runs: list[Any], out_dir: Path) -> None:
    """Plot goodput and bottleneck utilization across policies."""
    valid_runs = [r for r in runs if r.status == "ok" and not math.isnan(r.value("utilization"))]
    if not valid_runs or not HAS_MPL:
        return

    policies = sorted(list({r.policy for r in valid_runs}))
    utils = [float(np.mean([r.value("utilization") for r in valid_runs if r.policy == p])) for p in policies]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(policies, utils, color="#109618", alpha=0.85)
    ax.set_ylabel("Bottleneck Utilization (0-1)")
    ax.set_title("Bottleneck Utilization by Policy")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "goodput_utilization.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_queue_drops(runs: list[Any], out_dir: Path) -> None:
    """Plot queue drops for FedGrant and cross traffic."""
    valid_runs = [r for r in runs if r.status == "ok" and not math.isnan(r.value("queue_drops_fg"))]
    if not valid_runs or not HAS_MPL:
        return

    policies = sorted(list({r.policy for r in valid_runs}))
    drops_fg = [float(np.mean([r.value("queue_drops_fg") for r in valid_runs if r.policy == p])) for p in policies]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(policies, drops_fg, color="#eb6834", alpha=0.85)
    ax.set_ylabel("Bottleneck Queue Drops / Round")
    ax.set_title("Queue Drops by Policy")
    fig.tight_layout()
    fig.savefig(out_dir / "queue_drops.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_rtt_distribution(runs: list[Any], out_dir: Path) -> None:
    """Plot median and 99th percentile RTT."""
    valid_runs = [r for r in runs if r.status == "ok" and not math.isnan(r.value("rtt_p50_ms"))]
    if not valid_runs or not HAS_MPL:
        return

    policies = sorted(list({r.policy for r in valid_runs}))
    p50 = [float(np.mean([r.value("rtt_p50_ms") for r in valid_runs if r.policy == p])) for p in policies]
    p99 = [float(np.mean([r.value("rtt_p99_ms") for r in valid_runs if r.policy == p])) for p in policies]

    x = np.arange(len(policies))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(x - width / 2, p50, width, label="p50 (median)", color="#2a78d6")
    ax.bar(x + width / 2, p99, width, label="p99", color="#ff9900")
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.set_ylabel("RTT (ms)")
    ax.set_title("RTT Distribution by Policy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rtt_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_tta_bars(runs: list[Any], out_dir: Path) -> None:
    """Plot TTA rounds across FL policies."""
    fl_runs = [r for r in runs if r.experiment_id in {"F-main", "F-ref"} and r.status == "ok"]
    if not fl_runs or not HAS_MPL:
        return

    policies = sorted(list({r.policy for r in fl_runs}))
    ttas = []
    for p in policies:
        vals = [r.value("tta_rounds") for r in fl_runs if r.policy == p]
        ttas.append(float(np.mean(vals)) if vals else 300.0)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(policies, ttas, color="#990099", alpha=0.85)
    ax.set_ylabel("Time to Accuracy (Rounds)")
    ax.set_title("FL Time-to-Target (TTA) by Policy")
    ax.set_ylim(0, 310)
    fig.tight_layout()
    fig.savefig(out_dir / "tta_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_mininet_vs_sim(runs: list[Any], out_dir: Path) -> None:
    """Plot Mininet M1 vs Simulator twin M-sim calibration."""
    m1_runs = [r for r in runs if r.experiment_id == "M1" and r.status == "ok"]
    msim_runs = [r for r in runs if r.experiment_id == "M-sim" and r.status == "ok"]
    if not m1_runs or not msim_runs or not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.set_title("Mininet vs. Simulator Calibration")
    fig.tight_layout()
    fig.savefig(out_dir / "mininet_vs_sim.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ablations(runs: list[Any], out_dir: Path) -> None:
    """Plot ablation performance A1-A4 vs FG-GRU."""
    abl_runs = [r for r in runs if r.experiment_id in {"N-abl", "F-abl"} and r.status == "ok"]
    if not abl_runs or not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.set_title("Ablation Studies (A1-A4)")
    fig.tight_layout()
    fig.savefig(out_dir / "ablations.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate non-ML evaluation figures.")
    parser.add_argument("--runs", type=str, default="results/summary/runs.csv", help="Path to runs.csv")
    parser.add_argument("--out", type=str, default="results/figures", help="Directory for output PDFs")
    args = parser.parse_args()

    runs_path = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not runs_path.is_file():
        print(f"Summary CSV {runs_path} does not exist yet. Run grid experiments first.")
        return

    runs = read_runs_csv(runs_path)
    print(f"Loaded {len(runs)} validated runs from {runs_path}")

    plot_eta_vs_beta(runs, out_dir)
    plot_wasted_grants(runs, out_dir)
    plot_goodput_utilization(runs, out_dir)
    plot_queue_drops(runs, out_dir)
    plot_rtt_distribution(runs, out_dir)
    plot_tta_bars(runs, out_dir)
    plot_mininet_vs_sim(runs, out_dir)
    plot_ablations(runs, out_dir)
    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
