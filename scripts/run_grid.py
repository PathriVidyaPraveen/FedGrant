"""Run experiment grid for FedGrant (Sections 11, 14).

Expands configurations from Section 11, assigns runs across machines,
supports resuming existing runs, and records per-run logs and summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from fedgrant.analysis.runs_schema import (
    EXPERIMENTS,
    POLICIES,
    SCENARIOS,
    Run,
    expected_runs,
    make_config_id,
    make_row,
    read_runs_csv,
    write_runs_csv,
)
from fedgrant.policy.types import PolicyConfig
from fedgrant.sim.run_net import run_network_round


def expand_grid(experiments: list[str] | None = None) -> list[dict[str, Any]]:
    """Return all planned run specifications."""
    all_runs = expected_runs()
    selected = []
    for exp, cid, pol, seed in all_runs:
        if experiments and exp not in experiments:
            continue
        parts = cid.split("-")
        scenario = parts[0]
        beta = float(parts[1][1:])
        N = int(parts[2][1:])
        cross = float(parts[3][1:])
        selected.append({
            "experiment_id": exp,
            "config_id": cid,
            "scenario": scenario,
            "beta": beta,
            "N": N,
            "cross_mbps": cross,
            "policy": pol,
            "seed": seed,
            "n_rounds": 300 if exp in {"F-main", "F-ref", "F-abl", "M2"} else (100 if exp in {"M1", "M-sim"} else 200),
        })
    return selected


def partition_jobs(jobs: list[dict[str, Any]], machine_id: int, num_machines: int) -> list[dict[str, Any]]:
    """Partition jobs evenly across machines."""
    if num_machines <= 1:
        return jobs
    return [job for idx, job in enumerate(jobs) if idx % num_machines == machine_id]


def _policy_to_config(policy: str) -> PolicyConfig:
    """Map Section 11 policy string to PolicyConfig."""
    mapping = {
        "B2": PolicyConfig(selector="value", predictor="ewma", pred_in_pacing=False, pred_in_eligibility=False),
        "B4": PolicyConfig(selector="rr", predictor="ewma"),
        "B5": PolicyConfig(selector="srpt", predictor="ewma"),
        "B6": PolicyConfig(selector="value", predictor="ewma"),
        "B7": PolicyConfig(selector="value", predictor="hmm"),
        "FG-MLP": PolicyConfig(selector="value", predictor="mlp"),
        "FG-GRU": PolicyConfig(selector="value", predictor="gru"),
        "ORACLE": PolicyConfig(selector="value", predictor="oracle"),
        "FG-GRU-A2": PolicyConfig(selector="value", predictor="gru", retransmit=False),
        "FG-GRU-A3": PolicyConfig(selector="value", predictor="gru", pred_in_pacing=False),
        "FG-GRU-A4": PolicyConfig(selector="value", predictor="gru", pred_in_eligibility=False),
    }
    return mapping.get(policy, PolicyConfig(selector="value", predictor="ewma"))


def execute_job(job: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    """Execute one simulation run, save raw record, and return validated CSV row dict."""
    exp = job["experiment_id"]
    scen = job["scenario"]
    beta = job["beta"]
    N = job["N"]
    pol = job["policy"]
    seed = job["seed"]
    n_rounds = job["n_rounds"]
    cross = job["cross_mbps"]

    # Only sim runs can be executed directly in the simulator grid runner
    if exp in {"M1", "M2"}:
        return make_row(exp, scen, beta, N, pol, seed, n_rounds, cross_mbps=cross, status="missing")

    p_cfg = _policy_to_config(pol)
    raw_file = raw_dir / f"{exp}_{job['config_id']}_{pol}_s{seed}.json"

    # Check resume
    if raw_file.is_file():
        try:
            with open(raw_file, "r") as f:
                data = json.load(f)
            return make_row(exp, scen, beta, N, pol, seed, n_rounds, cross_mbps=cross, status="ok", **data["summary_metrics"])
        except Exception:
            pass

    # Run rounds 1..n_rounds
    round_metrics = []
    gateway = None
    status = "ok"

    try:
        for r in range(1, n_rounds + 1):
            res = run_network_round(
                round_id=r,
                num_clients=N,
                beta=beta,
                scenario=scen,
                policy_config=p_cfg,
                seed=seed,
                gateway=gateway,
            )
            gateway = res["gateway"]
            round_metrics.append({
                "round_id": r,
                "eta": res["eta"],
                **res["metrics"],
            })
    except Exception as e:
        status = "missing"
        return make_row(exp, scen, beta, N, pol, seed, n_rounds, cross_mbps=cross, status=status)

    # Average metrics over rounds 21..last (first 20 excluded as warm-up)
    eval_rounds = round_metrics[20:] if len(round_metrics) > 20 else round_metrics
    eta_mean = float(np.mean([m["eta"] for m in eval_rounds]))
    util_mean = float(np.mean([m.get("utilization", 0.0) for m in eval_rounds]))
    goodput_mean = float(np.mean([m.get("goodput_bps", 0.0) for m in eval_rounds]))
    wasted_mean = float(np.mean([m.get("wasted_grants_ratio", 0.0) for m in eval_rounds]))
    rtt_p50 = float(np.mean([m.get("rtt_median_ms", 0.0) for m in eval_rounds]))
    rtt_p99 = float(np.mean([m.get("rtt_p99_ms", 0.0) for m in eval_rounds]))
    drops_fg = float(np.mean([m.get("drops_queue_bn_client", 0.0) for m in eval_rounds]))
    drops_cross = float(np.mean([m.get("drops_queue_bn_cross", 0.0) for m in eval_rounds]))
    blocks_mean = float(np.mean([m.get("grants_delivered", 0.0) for m in eval_rounds]))
    clients_rep = float(np.mean([m.get("clients_represented", 0.0) for m in eval_rounds]))
    deadline_comp = float(np.mean([m.get("deadline_completion", 0.0) for m in eval_rounds]))
    late_to = float(np.mean([m.get("late_after_timeout", 0.0) for m in eval_rounds]))

    summary = {
        "eta": eta_mean,
        "utilization": util_mean,
        "goodput_bps": goodput_mean,
        "wasted_grants": wasted_mean,
        "rtt_p50_ms": rtt_p50,
        "rtt_p99_ms": rtt_p99,
        "queue_drops_fg": drops_fg,
        "queue_drops_cross": drops_cross,
        "delivered_blocks": blocks_mean,
        "clients_represented": clients_rep,
        "deadline_completion": deadline_comp,
        "late_after_timeout": late_to,
    }

    # Save raw record
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_file, "w") as f:
        json.dump({
            "job": job,
            "summary_metrics": summary,
            "rounds": round_metrics,
        }, f, indent=2)

    return make_row(exp, scen, beta, N, pol, seed, n_rounds, cross_mbps=cross, status="ok", **summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="FedGrant Experiment Grid Runner")
    parser.add_argument("--exp", nargs="+", choices=EXPERIMENTS, default=None, help="Filter by experiment_id")
    parser.add_argument("--machine-id", type=int, default=0, help="Machine index (0-based)")
    parser.add_argument("--num-machines", type=int, default=1, help="Total number of machines")
    parser.add_argument("--dry-run", action="store_true", help="Print run plan without executing")
    parser.add_argument("--run", action="store_true", help="Execute planned runs on this machine")
    parser.add_argument("--max-runs", type=int, default=None, help="Maximum number of runs to execute")
    parser.add_argument("--output-csv", type=str, default="results/summary/runs.csv", help="Path to summary runs.csv")
    parser.add_argument("--raw-dir", type=str, default="results/raw", help="Directory for raw run records")

    args = parser.parse_args()

    jobs = expand_grid(args.exp)
    my_jobs = partition_jobs(jobs, args.machine_id, args.num_machines)
    if args.max_runs:
        my_jobs = my_jobs[:args.max_runs]

    print(f"=== FedGrant Grid Plan ===")
    print(f"Total matching runs: {len(jobs)}")
    print(f"Machine {args.machine_id + 1}/{args.num_machines} assigned: {len(my_jobs)} runs")
    print(f"Summary CSV: {args.output_csv}")
    print(f"Raw results dir: {args.raw_dir}")

    if args.dry_run or not args.run:
        print("\n[Dry Run] Sample first 5 jobs:")
        for j in my_jobs[:5]:
            print(f"  {j['experiment_id']} | {j['config_id']} | {j['policy']} | seed {j['seed']} | {j['n_rounds']} rounds")
        if not args.run:
            print("\nPass --run to start execution.")
        return

    raw_dir = Path(args.raw_dir)
    csv_path = Path(args.output_csv)
    rows = []

    # If CSV already exists, load existing rows
    existing_keys = set()
    if csv_path.is_file():
        try:
            for r in read_runs_csv(csv_path):
                existing_keys.add(r.key)
                rows.append(make_row(r.experiment_id, r.scenario, r.beta, r.N, r.policy, r.seed, r.n_rounds,
                                     cross_mbps=r.cross_mbps, status=r.status, tta_censored=r.tta_censored, **r.metrics))
        except Exception:
            pass

    print(f"\nStarting execution of {len(my_jobs)} runs...")
    t0 = time.time()
    for idx, job in enumerate(my_jobs, start=1):
        key = (job["experiment_id"], job["config_id"], job["policy"], job["seed"])
        if key in existing_keys:
            print(f"[{idx}/{len(my_jobs)}] Skipping already completed run {key}")
            continue
        print(f"[{idx}/{len(my_jobs)}] Running {job['experiment_id']} | {job['config_id']} | {job['policy']} | seed {job['seed']}...")
        row = execute_job(job, raw_dir)
        rows.append(row)
        write_runs_csv(rows, csv_path)

    elapsed = time.time() - t0
    print(f"\nExecution finished in {elapsed:.1f}s. Wrote {len(rows)} validated rows to {csv_path}")


if __name__ == "__main__":
    main()
