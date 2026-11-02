"""Per-configuration paired Mininet-vs-simulator confidence intervals."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from fedgrant.analysis.runs_schema import Run, expected_runs, read_runs_csv

CALIBRATION_METRICS = (
    # Section 14 explicitly defines these three M1-vs-simulator metrics.
    "eta", "utilization", "wasted_grants",
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


def compare(runs: list[Run], metrics: tuple[str, ...] = CALIBRATION_METRICS,
            *, include_expected: bool = True) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return configuration/policy paired summaries and unmatched identities."""
    mininet_all = {(r.config_id, r.policy, r.seed): r for r in runs if r.experiment_id == "M1"}
    simulator_all = {(r.config_id, r.policy, r.seed): r for r in runs if r.experiment_id == "M-sim"}
    mininet = {key: row for key, row in mininet_all.items() if row.status != "missing"}
    simulator = {key: row for key, row in simulator_all.items() if row.status != "missing"}
    expected_m1 = {(cid, policy, seed) for exp, cid, policy, seed in expected_runs() if exp == "M1"}
    expected_sim = {(cid, policy, seed) for exp, cid, policy, seed in expected_runs() if exp == "M-sim"}
    all_keys = mininet_all.keys() | simulator_all.keys()
    if include_expected:
        all_keys |= expected_m1 | expected_sim
    missing = []
    for cid, policy, seed in sorted(all_keys):
        m1_row, sim_row = mininet_all.get((cid, policy, seed)), simulator_all.get((cid, policy, seed))
        if m1_row is None or m1_row.status == "missing":
            missing.append({"config_id": cid, "policy": policy, "seed": seed, "missing_from": "M1 metrics"})
        # B1 is a TCP Mininet baseline and has no M-sim twin in the frozen matrix.
        if policy != "B1" and (sim_row is None or sim_row.status == "missing"):
            missing.append({"config_id": cid, "policy": policy, "seed": seed, "missing_from": "M-sim metrics"})
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for metric in metrics:
        _, _, pairs = _paired_rows(runs, metric)
        for (config_id, policy, seed), diff in pairs:
            grouped[(config_id, policy, metric)].append(diff)
    rows: list[dict[str, object]] = []
    for (config_id, policy, metric), diffs in sorted(grouped.items()):
        values = np.asarray(diffs, dtype=np.float64)
        n = int(values.size)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if n > 1 else math.nan
        if n > 1:
            se = sd / math.sqrt(n)
            margin = float(student_t.ppf(0.975, n - 1) * se)
            ci_low, ci_high = mean - margin, mean + margin
        elif n == 1:
            ci_low = ci_high = math.nan
        else:
            ci_low = ci_high = math.nan
        rows.append({"config_id": config_id, "policy": policy, "metric": metric, "n_pairs": n, "mean_mininet_minus_sim": mean,
                     "sd_difference": sd, "ci95_low": ci_low, "ci95_high": ci_high})
    return rows, missing


def write_csv(rows: list[dict[str, object]], path: str | Path, fieldnames: list[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or (list(rows[0]) if rows else ["config_id", "policy", "metric", "n_pairs", "mean_mininet_minus_sim", "sd_difference", "ci95_low", "ci95_high"])
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
    summaries, missing = compare(runs, include_expected=True)
    write_csv(summaries, args.output)
    write_csv(missing, args.missing_output, ["config_id", "policy", "seed", "missing_from"])
    print(f"wrote {len(summaries)} metric comparisons and {len(missing)} unmatched run keys")


if __name__ == "__main__":
    main()
