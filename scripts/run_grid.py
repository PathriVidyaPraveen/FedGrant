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
    FL_EXPERIMENTS,
    POLICIES,
    SCENARIOS,
    Run,
    expected_runs,
    make_config_id,
    make_row,
    read_runs_csv,
    write_runs_csv,
)
from fedgrant.analysis.metrics import compute_final_accuracy, compute_time_to_target
from fedgrant.policy.types import PolicyConfig
from fedgrant.sim.run_net import compute_deadline_ns, run_network_round


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

    if exp in FL_EXPERIMENTS:
        return _execute_fl_job(job, raw_dir)

    # Only sim runs can be executed directly in the simulator grid runner
    if exp in {"M1", "M2"}:
        return make_row(exp, scen, beta, N, pol, seed, n_rounds, cross_mbps=cross, status="missing")

    if pol in {"B2", "B3"}:
        # Their distinct push/plan transports are owned by Prajin. The current
        # network-round driver only implements the grant protocol, so refusing
        # these rows is safer than silently scoring them as FG-GRU variants.
        reason = f"{pol} transport is waiting for Prajin's baseline implementation"
        raw_file = raw_dir / f"{exp}_{job['config_id']}_{pol}_s{seed}.json"
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        raw_file.write_text(json.dumps({"job": job, "status": "missing", "reason": reason}, indent=2) + "\n")
        return make_row(exp, scen, beta, N, pol, seed, n_rounds, cross_mbps=cross, status="missing")

    p_cfg = _policy_to_config(pol)
    raw_file = raw_dir / f"{exp}_{job['config_id']}_{pol}_s{seed}.json"

    # Check resume
    if raw_file.is_file():
        try:
            with open(raw_file, "r") as f:
                data = json.load(f)
            summary = dict(data["summary_metrics"])
            saved_rounds = data.get("rounds", [])
            sample_rows = [r for r in saved_rounds if int(r.get("round_id", 0)) > 20
                           and "rtt_samples_ns" in r]
            if sample_rows:
                samples = [int(x) for r in sample_rows for x in r["rtt_samples_ns"]]
                summary["rtt_p50_ms"] = float(np.percentile(samples, 50) / 1e6) if samples else 0.0
                summary["rtt_p99_ms"] = float(np.percentile(samples, 99) / 1e6) if samples else 0.0
            elif "rtt_p50_ms" in summary or "rtt_p99_ms" in summary:
                # Older grid records held only per-round quantiles, which cannot
                # be pooled back into the required sample-level percentiles.
                raise ValueError("legacy raw run lacks pooled RTT samples; rerunning this seed")
            return make_row(exp, scen, beta, N, pol, seed, n_rounds, cross_mbps=cross, status="ok", **summary)
        except Exception:
            pass

    # Run rounds 1..n_rounds
    round_metrics = []
    pooled_rtt_samples_ns: list[int] = []
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
                "rtt_samples_ns": list(res.get("rtt_samples_ns", ())),
            })
            if r > 20:
                pooled_rtt_samples_ns.extend(int(v) for v in res.get("rtt_samples_ns", ()))
    except Exception as e:
        status = "missing"
        return make_row(exp, scen, beta, N, pol, seed, n_rounds, cross_mbps=cross, status=status)

    # Average metrics over rounds 21..last (first 20 excluded as warm-up)
    eval_rounds = round_metrics[20:] if len(round_metrics) > 20 else round_metrics
    eta_mean = float(np.mean([m["eta"] for m in eval_rounds]))
    util_mean = float(np.mean([m.get("utilization", 0.0) for m in eval_rounds]))
    goodput_mean = float(np.mean([m.get("goodput_bps", 0.0) for m in eval_rounds]))
    wasted_mean = float(np.mean([m.get("wasted_grants_ratio", 0.0) for m in eval_rounds]))
    rtt_p50 = float(np.percentile(pooled_rtt_samples_ns, 50) / 1e6) if pooled_rtt_samples_ns else 0.0
    rtt_p99 = float(np.percentile(pooled_rtt_samples_ns, 99) / 1e6) if pooled_rtt_samples_ns else 0.0
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


def _load_full_reference_final_accuracies() -> list[float]:
    """Load the three frozen FULL reference runs required to define a*."""
    values = []
    for seed in range(3):
        path = Path("data/runs") / f"femnist_full_s{seed}.json"
        if not path.is_file():
            raise FileNotFoundError(f"run F-ref seeds 0..2 before FL comparisons; missing {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        history = data.get("history", [])
        evals = [row for row in history if "test_accuracy" in row]
        values.append(compute_final_accuracy([float(row["test_accuracy"]) for row in evals]))
    return values


def _execute_fl_job(job: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    """Run FL experiments through the FEMNIST/FedAvg path, never network-only."""
    exp, pol, seed = job["experiment_id"], job["policy"], int(job["seed"])
    scen, beta, n_clients = job["scenario"], float(job["beta"]), int(job["N"])
    rounds, cross = int(job["n_rounds"]), float(job["cross_mbps"])
    raw_file = raw_dir / f"{exp}_{job['config_id']}_{pol}_s{seed}.json"
    # B2 push and B3 plan-then-send require Prajin's client/gateway adapters.
    # Ignore any old raw record produced by the former incorrect dispatch.
    if pol in {"B2", "B3"}:
        reason = f"{pol} FL transport is waiting for Prajin's baseline adapter"
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        raw_file.write_text(json.dumps({"job": job, "status": "missing", "reason": reason}, indent=2) + "\n")
        return make_row(exp, scen, beta, n_clients, pol, seed, rounds, cross_mbps=cross, status="missing")
    if raw_file.is_file():
        try:
            data = json.loads(raw_file.read_text(encoding="utf-8"))
            return make_row(exp, scen, beta, n_clients, pol, seed, rounds, cross_mbps=cross,
                            status="ok", tta_censored=data.get("tta_censored"),
                            **data["summary_metrics"])
        except Exception:
            pass

    try:
        full_final = None if exp == "F-ref" else _load_full_reference_final_accuracies()
        # Lazy import keeps the non-ML simulator/test commands independent of torch.
        from fedgrant.fl.run_fl import run_experiment

        bank_path = "data/bank/femnist_full_s0.npz" if exp == "F-ref" and seed == 0 else None
        no_ef = exp == "F-abl" and pol == "FG-GRU-A1"
        history = run_experiment(seed=seed, rounds=rounds, policy="FULL" if exp == "F-ref" else pol,
                                 beta=beta, scenario=scen, no_ef=no_ef, bank_path=bank_path)
        evaluations = [row for row in history if "test_accuracy" in row]
        accuracies = [float(row["test_accuracy"]) for row in evaluations]
        final_acc = compute_final_accuracy(accuracies)
        summary: dict[str, Any] = {"final_acc": final_acc}
        censored = None
        if exp != "F-ref":
            assert full_final is not None
            target = 0.9 * float(np.mean(full_final))
            tta_rounds, tta_seconds = compute_time_to_target(
                evaluations, target, max_round=rounds, t_c_s=1.0,
                d_s=compute_deadline_ns(beta, n_clients) / 1e9,
            )
            censored = tta_rounds is None
            summary.update(tta_rounds=(float(tta_rounds) if tta_rounds is not None else None),
                           tta_seconds=(float(tta_seconds) if tta_seconds is not None else None))
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        raw_file.write_text(json.dumps({"job": job, "status": "ok", "tta_censored": censored,
                                        "summary_metrics": summary, "history": history}, indent=2) + "\n")
        return make_row(exp, scen, beta, n_clients, pol, seed, rounds, cross_mbps=cross,
                        status="ok", tta_censored=censored, **summary)
    except Exception as exc:
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        raw_file.write_text(json.dumps({"job": job, "status": "missing", "error": repr(exc)}, indent=2) + "\n")
        return make_row(exp, scen, beta, n_clients, pol, seed, rounds, cross_mbps=cross, status="missing")


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
                if r.status != "missing":
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
        # A missing result is retryable; replace its previous placeholder row.
        rows = [r for r in rows if not (r.key == key and r.status == "missing")]
        print(f"[{idx}/{len(my_jobs)}] Running {job['experiment_id']} | {job['config_id']} | {job['policy']} | seed {job['seed']}...")
        row = execute_job(job, raw_dir)
        rows.append(row)
        write_runs_csv(rows, csv_path)

    elapsed = time.time() - t0
    print(f"\nExecution finished in {elapsed:.1f}s. Wrote {len(rows)} validated rows to {csv_path}")


if __name__ == "__main__":
    main()
