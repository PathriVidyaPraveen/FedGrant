"""Paired Mininet-vs-simulator analysis with paired t tests and Holm correction."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from fedgrant.analysis.runs_schema import Run, read_runs_csv

CALIBRATION_METRICS = (
    "eta", "delivered_blocks", "clients_represented", "deadline_completion", "goodput_bps",
    "utilization", "queue_drops_fg", "queue_drops_cross", "wasted_grants", "late_after_timeout",
    "late_after_deadline", "rtt_p50_ms", "rtt_p99_ms", "protocol_overhead",
)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, restored to the original order."""
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = np.argsort(np.asarray(p_values), kind="stable")
    adjusted = np.ones(len(p_values), dtype=np.float64)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[int(idx)]))
        adjusted[int(idx)] = running
    return adjusted.tolist()


def _paired_rows(runs: list[Run], metric: str):
    mininet = {(r.config_id, r.policy, r.seed): r for r in runs if r.experiment_id == "M1" and r.status != "missing"}
    simulator = {(r.config_id, r.policy, r.seed): r for r in runs if r.experiment_id == "M-sim" and r.status != "missing"}
    shared = sorted(mininet.keys() & simulator.keys())
    differences = []
    for key in shared:
        x, y = mininet[key].value(metric), simulator[key].value(metric)
        if math.isfinite(x) and math.isfinite(y):
            differences.append((key, x - y))
    return mininet, simulator, differences


def compare(runs: list[Run], metrics: tuple[str, ...] = CALIBRATION_METRICS) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return per-policy paired summaries and the list of unmatched identities."""
    mininet_all = {(r.config_id, r.policy, r.seed): r for r in runs if r.experiment_id == "M1"}
    simulator_all = {(r.config_id, r.policy, r.seed): r for r in runs if r.experiment_id == "M-sim"}
    mininet = {key: row for key, row in mininet_all.items() if row.status != "missing"}
    simulator = {key: row for key, row in simulator_all.items() if row.status != "missing"}
    all_keys = mininet_all.keys() | simulator_all.keys()
    missing = []
    for cid, policy, seed in sorted(all_keys):
        m1_row, sim_row = mininet_all.get((cid, policy, seed)), simulator_all.get((cid, policy, seed))
        if m1_row is None or m1_row.status == "missing":
            missing.append({"config_id": cid, "policy": policy, "seed": seed, "missing_from": "M1 metrics"})
        if sim_row is None or sim_row.status == "missing":
            missing.append({"config_id": cid, "policy": policy, "seed": seed, "missing_from": "M-sim metrics"})
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for metric in metrics:
        _, _, pairs = _paired_rows(runs, metric)
        for (config_id, policy, seed), diff in pairs:
            grouped[(policy, metric)].append(diff)
    rows: list[dict[str, object]] = []
    p_slots: list[int] = []
    p_values: list[float] = []
    for (policy, metric), diffs in sorted(grouped.items()):
        values = np.asarray(diffs, dtype=np.float64)
        n = int(values.size)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if n > 1 else math.nan
        if n > 1 and sd > 0:
            se = sd / math.sqrt(n)
            t_stat = mean / se
            p_value = float(2 * student_t.sf(abs(t_stat), n - 1))
            margin = float(student_t.ppf(0.975, n - 1) * se)
            ci_low, ci_high = mean - margin, mean + margin
        elif n > 1:
            t_stat, p_value = (0.0, 1.0) if mean == 0 else (math.copysign(math.inf, mean), 0.0)
            ci_low = ci_high = mean
        else:
            t_stat = p_value = ci_low = ci_high = math.nan
        rows.append({"policy": policy, "metric": metric, "n_pairs": n, "mean_mininet_minus_sim": mean,
                     "sd_difference": sd, "t_statistic": t_stat, "p_value": p_value,
                     "ci95_low": ci_low, "ci95_high": ci_high, "p_holm": math.nan})
        if math.isfinite(p_value):
            p_slots.append(len(rows) - 1)
            p_values.append(p_value)
    for idx, adjusted in zip(p_slots, holm_adjust(p_values), strict=True):
        rows[idx]["p_holm"] = adjusted
    return rows, missing


def write_csv(rows: list[dict[str, object]], path: str | Path, fieldnames: list[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or (list(rows[0]) if rows else ["policy", "metric", "n_pairs", "mean_mininet_minus_sim", "sd_difference", "t_statistic", "p_value", "ci95_low", "ci95_high", "p_holm"])
    with p.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="results/summary/runs.csv")
    parser.add_argument("--output", default="results/summary/mininet_sim_calibration.csv")
    parser.add_argument("--missing-output", default="results/summary/mininet_sim_missing_pairs.csv")
    args = parser.parse_args()
    runs = read_runs_csv(args.runs)
    summaries, missing = compare(runs)
    write_csv(summaries, args.output)
    write_csv(missing, args.missing_output, ["config_id", "policy", "seed", "missing_from"])
    print(f"wrote {len(summaries)} metric comparisons and {len(missing)} unmatched run keys")


if __name__ == "__main__":
    main()
