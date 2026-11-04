"""Run experiment grid for FedGrant (Sections 11, 14).

Expands configurations from Section 11, assigns runs across machines,
supports resuming existing runs, and records per-run logs and summaries.
"""

from __future__ import annotations

import argparse
import json
import hashlib
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
from fedgrant.analysis.metrics import (
    compute_final_accuracy,
    compute_target_accuracy,
    compute_time_to_target,
)
from fedgrant.policy.types import PolicyConfig
from fedgrant.sim.run_net import compute_deadline_ns, run_network_round
from fedgrant.sim.run_baselines import BaselineState, run_baseline_round
from fedgrant.sim.seeding import network_round_seed
from fedgrant.sim.value_bank import load_value_bank
from fedgrant.sim.traces import ID_TRACE_NAMES, OOD_TRACE_NAMES

FULL_REFERENCE_DIR = Path("data/runs")


def _load_full_history(seed: int, reference_dir: Path = FULL_REFERENCE_DIR) -> list[dict[str, Any]]:
    """Load one existing FULL reference history; this helper never starts training."""
    path = reference_dir / f"femnist_full_s{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(f"existing FULL reference record is missing: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if int(record.get("rounds", 0)) != 300:
        raise ValueError(f"FULL reference {path} does not contain a 300-round run")
    if "seed" in record and int(record["seed"]) != seed:
        raise ValueError(f"FULL reference seed mismatch in {path}")
    history = record.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"FULL reference history is empty or invalid: {path}")
    evals = sorted(
        (row for row in history if "test_accuracy" in row and "round" in row),
        key=lambda row: int(row["round"]),
    )
    if len(evals) < 10 or int(evals[-1]["round"]) != 300:
        raise ValueError(f"FULL reference history in {path} must include evaluations through round 300")
    if [int(row["round"]) for row in evals[-10:]] != list(range(255, 301, 5)):
        raise ValueError(f"FULL reference history in {path} must evaluate every 5 rounds through 300")
    return evals


def _full_target_accuracy(reference_dir: Path = FULL_REFERENCE_DIR) -> float:
    finals = [
        compute_final_accuracy([float(row["test_accuracy"]) for row in _load_full_history(seed, reference_dir)])
        for seed in range(3)
    ]
    return compute_target_accuracy(finals)


def _pooled_rtt_percentiles(rounds: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """Compute run RTT percentiles from raw grant samples in rounds 21 onward."""
    samples = [
        float(sample)
        for round_row in rounds
        if int(round_row.get("round_id", round_row.get("round", 0))) >= 21
        for sample in round_row.get("rtt_samples_ns", ())
    ]
    if not samples:
        return None, None
    return float(np.percentile(samples, 50) / 1e6), float(np.percentile(samples, 99) / 1e6)


def _network_summary(round_metrics: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    """Aggregate per-round network metrics and pooled RTT samples."""
    eval_rounds = round_metrics[20:] if len(round_metrics) > 20 else round_metrics
    summary: dict[str, Any] = {
        "eta": float(np.mean([m["eta"] for m in eval_rounds])),
        "utilization": float(np.mean([m.get("utilization", 0.0) for m in eval_rounds])),
        "goodput_bps": float(np.mean([m.get("goodput_bps", 0.0) for m in eval_rounds])),
        "queue_drops_fg": float(np.mean([m.get("drops_queue_bn_client", 0.0) for m in eval_rounds])),
        "queue_drops_cross": float(np.mean([m.get("drops_queue_bn_cross", 0.0) for m in eval_rounds])),
        "delivered_blocks": float(np.mean([m.get("delivered_blocks", m.get("grants_delivered", 0.0)) for m in eval_rounds])),
        "clients_represented": float(np.mean([m.get("clients_represented", 0.0) for m in eval_rounds])),
        "late_after_deadline": float(np.mean([m.get("late_after_deadline", 0.0) for m in eval_rounds])),
    }
    if policy not in {"B2", "B3"}:
        summary["wasted_grants"] = float(np.mean([m.get("wasted_grants_ratio", 0.0) for m in eval_rounds]))
        summary["deadline_completion"] = float(np.mean([m.get("deadline_completion", 0.0) for m in eval_rounds]))
        summary["late_after_timeout"] = float(np.mean([m.get("late_after_timeout", 0.0) for m in eval_rounds]))
        summary["rtt_p50_ms"], summary["rtt_p99_ms"] = _pooled_rtt_percentiles(round_metrics)
    return summary


def _row_from_raw(job: dict[str, Any], raw_file: Path) -> dict[str, Any]:
    data = json.loads(raw_file.read_text(encoding="utf-8"))
    summary = dict(data["summary_metrics"])
    tta_censored = summary.pop("tta_censored", None)
    if job["policy"] not in {"B2", "B3"}:
        # Old raw runs stored only per-round percentiles; those cannot be pooled.
        # Keep RTT blank unless the raw grant samples are present.
        summary["rtt_p50_ms"], summary["rtt_p99_ms"] = _pooled_rtt_percentiles(data.get("rounds", []))
    return make_row(
        job["experiment_id"], job["scenario"], job["beta"], job["N"], job["policy"], job["seed"],
        job["n_rounds"], cross_mbps=job["cross_mbps"], status="ok", tta_censored=tta_censored,
        **summary,
    )


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
    """Map grant-based policies to PolicyConfig.

    Push and plan baselines need their own transport schedulers and must never
    be represented by the grant-based network round.
    """
    mapping = {
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
    if policy in {"B2", "B3"}:
        raise NotImplementedError(f"{policy} requires its dedicated push/plan transport scheduler")
    try:
        return mapping[policy]
    except KeyError as exc:
        raise ValueError(f"no simulator implementation for policy {policy!r}") from exc


def _fl_evaluation_metrics(
    job: dict[str, Any], history: list[dict[str, Any]], *, reference_dir: Path = FULL_REFERENCE_DIR,
) -> dict[str, Any]:
    evals = sorted(
        (row for row in history if "test_accuracy" in row and "round" in row),
        key=lambda row: int(row["round"]),
    )
    if not evals:
        raise ValueError("FL run returned no test-accuracy evaluations")
    if int(evals[-1]["round"]) != int(job["n_rounds"]):
        raise ValueError("FL evaluation history does not reach the configured final round")
    if int(job["n_rounds"]) == 300 and [int(row["round"]) for row in evals[-10:]] != list(range(255, 301, 5)):
        raise ValueError("300-round FL history must evaluate every 5 rounds from 255 through 300")
    target = _full_target_accuracy(reference_dir)
    final_acc = compute_final_accuracy([float(row["test_accuracy"]) for row in evals])
    deadline_s = (
        compute_deadline_ns(job["beta"], job["N"]) / 1e9
        if job["experiment_id"] != "F-ref"
        else 0.0
    )
    tta_round, tta_seconds = compute_time_to_target(
        evals,
        target,
        max_round=int(job["n_rounds"]),
        t_c_s=1.0,
        d_s=deadline_s,
    )
    result: dict[str, Any] = {
        "final_acc": final_acc,
        "tta_censored": tta_round is None,
    }
    if tta_round is not None:
        result["tta_rounds"] = float(tta_round)
        result["tta_seconds"] = float(tta_seconds)
    return result


def _fl_round_metrics(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rounds = []
    for record in history:
        metrics = dict(record.get("metrics", {}))
        late = record.get("late_metrics", {})
        rounds.append({
            "round_id": int(record["round"]),
            "eta": float(record["eta"]),
            **metrics,
            "late_after_deadline": late.get("late_after_deadline", 0.0),
            "rtt_samples_ns": record.get("rtt_samples_ns", []),
        })
    return rounds


def _network_provenance(
    job: dict[str, Any], *, bank_sha256: str | None, weights: np.ndarray | None,
) -> dict[str, Any]:
    """Record the actual simulator inputs needed to interpret a raw run."""
    trace_split = "ood" if job["scenario"] == "L5" else "test"
    trace_names = OOD_TRACE_NAMES if job["scenario"] == "L5" else ID_TRACE_NAMES
    trace_files = []
    if job["scenario"] in {"L4", "L5"}:
        for name in trace_names:
            path = Path("data/traces") / f"{name}_{trace_split}.csv"
            trace_files.append({
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    model_hashes = {}
    policy = job["policy"]
    predictor = (
        "gru" if policy.startswith("FG-GRU") else
        "mlp" if policy == "FG-MLP" else
        None if policy in {"B2", "B3"} else
        _policy_to_config(policy).predictor
    )
    if predictor in {"mlp", "gru"}:
        model_path = Path("models") / f"{predictor}_prod.npz"
        if model_path.is_file():
            model_hashes[str(model_path)] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    source_paths = (
        Path("scripts/run_grid.py"), Path("fedgrant/sim/run_net.py"),
        Path("fedgrant/sim/run_baselines.py"), Path("fedgrant/sim/links.py"),
        Path("fedgrant/sim/ge.py"), Path("fedgrant/baselines/b2_udp.py"),
        Path("fedgrant/baselines/b3_plan.py"), Path("fedgrant/baselines/b3_pacing.py"),
        Path("fedgrant/core/gateway.py"),
    )
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths if path.is_file()
    }
    return {
        "runner": "scripts.run_grid.execute_job",
        "command_argv": list(sys.argv),
        "cross_mbps": float(job["cross_mbps"]),
        "seed_scheme": "numpy.SeedSequence([master_seed, round_id - 1]).generate_state(1)[0]",
        "bank_path": "data/bank/femnist_full_s0.npz" if bank_sha256 else None,
        "bank_sha256": bank_sha256,
        "energy_profile_scheme": "uniform FULL round index independently for each (run, round, client)" if bank_sha256 else None,
        "client_weights": None if weights is None else np.asarray(weights, dtype=float).tolist(),
        "trace_split": trace_split if trace_files else None,
        "trace_files": trace_files,
        "model_checkpoint_sha256": model_hashes,
        "source_sha256": source_hashes,
    }


def _transport_summary(round_metrics: list[dict[str, Any]], policy: str) -> dict[str, float]:
    """Transport counters are raw-run diagnostics, separate from the shared CSV schema."""
    eval_rounds = round_metrics[20:] if len(round_metrics) > 20 else round_metrics
    keys = (
        ("data_packets_sent", "data_packets_delivered") if policy == "B2" else
        ("plans_issued", "blocks_planned", "data_packets_sent", "data_bytes_sent", "data_packets_delivered")
    )
    return {key: float(np.mean([float(row.get(key, 0) or 0) for row in eval_rounds])) for key in keys}


def _full_reference_row(job: dict[str, Any], reference_dir: Path) -> dict[str, Any]:
    history = _load_full_history(job["seed"], reference_dir)
    metrics = _fl_evaluation_metrics(job, history, reference_dir=reference_dir)
    tta_censored = bool(metrics.pop("tta_censored"))
    return make_row(
        job["experiment_id"], job["scenario"], job["beta"], job["N"], job["policy"], job["seed"],
        job["n_rounds"], cross_mbps=job["cross_mbps"], status="ok", tta_censored=tta_censored,
        **metrics,
    )


def execute_job(job: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    """Execute one grid job, save raw record where applicable, and validate its row."""
    exp, scen, beta, N = job["experiment_id"], job["scenario"], job["beta"], job["N"]
    pol, seed, n_rounds, cross = job["policy"], job["seed"], job["n_rounds"], job["cross_mbps"]

    # M1 and M2 are Mininet-only by Section 7; do not synthesize their results.
    if exp in {"M1", "M2"}:
        print(f"{exp} requires the Mininet runner; no simulator result was fabricated")
        return make_row(exp, scen, beta, N, pol, seed, n_rounds, cross_mbps=cross, status="missing")

    # F-ref is sourced only from existing FULL histories; this path never trains.
    if exp == "F-ref":
        try:
            return _full_reference_row(job, FULL_REFERENCE_DIR)
        except Exception as exc:
            print(f"F-ref missing: seed {seed}: {type(exc).__name__}: {exc}")
            return make_row(exp, scen, beta, N, pol, seed, n_rounds, cross_mbps=cross, status="missing")

    is_fl = exp in {"F-main", "F-abl"}
    is_baseline = pol in {"B2", "B3"}
    p_cfg = None if (is_baseline or is_fl) else _policy_to_config(pol)
    raw_suffix = "_desim_v1" if is_baseline else ""
    raw_file = raw_dir / f"{exp}_{job['config_id']}_{pol}_s{seed}{raw_suffix}.json"

    if raw_file.is_file():
        try:
            existing = json.loads(raw_file.read_text(encoding="utf-8"))
            if existing.get("job") != job or not isinstance(existing.get("simulation"), dict):
                raise ValueError("raw record lacks matching job and simulator provenance")
            if is_fl:
                expected_simulation = _network_provenance(job, bank_sha256=None, weights=None)
            else:
                bank = load_value_bank("data/bank/femnist_full_s0.npz", N)
                expected_simulation = _network_provenance(
                    job, bank_sha256=bank.sha256, weights=bank.client_weights
                )
            if existing["simulation"] != expected_simulation:
                raise ValueError("raw simulator inputs differ from the current validated configuration")
            return _row_from_raw(job, raw_file)
        except Exception as exc:
            print(
                f"Existing raw artifact is not reusable and will not be overwritten: {raw_file}: "
                f"{type(exc).__name__}: {exc}"
            )
            return make_row(
                exp, scen, beta, N, pol, seed, n_rounds,
                cross_mbps=cross, status="missing",
            )

    try:
        if is_fl:
            from fedgrant.fl.run_fl import run_experiment

            ablation_a1 = exp == "F-abl"
            history = run_experiment(
                seed=seed,
                rounds=n_rounds,
                policy="FG-GRU" if ablation_a1 else pol,
                beta=beta,
                scenario=scen,
                no_ef=ablation_a1,
            )
            round_metrics = _fl_round_metrics(history)
            summary = _network_summary(round_metrics, pol)
            summary.update(_fl_evaluation_metrics(job, history))
            raw_record = {
                "job": job,
                "summary_metrics": summary,
                "rounds": round_metrics,
                "simulation": _network_provenance(job, bank_sha256=None, weights=None),
                "eval_history": [
                    row for row in history if "test_accuracy" in row
                ],
            }
        else:
            value_bank = load_value_bank("data/bank/femnist_full_s0.npz", N)
            round_metrics = []
            gateway = None
            baseline_state: BaselineState | None = None
            for r in range(1, n_rounds + 1):
                if is_baseline:
                    res = run_baseline_round(
                        pol,
                        round_id=r,
                        num_clients=N,
                        client_weights=value_bank.client_weights,
                        client_energies=value_bank.energies_for_run_round(seed, r),
                        beta=beta,
                        scenario=scen,
                        seed=network_round_seed(seed, r),
                        state=baseline_state,
                        cross_mbps=cross,
                    )
                    baseline_state = res["baseline_state"]
                else:
                    res = run_network_round(
                        round_id=r,
                        num_clients=N,
                        client_weights=value_bank.client_weights,
                        client_energies=value_bank.energies_for_run_round(seed, r),
                        beta=beta,
                        scenario=scen,
                        policy_config=p_cfg,
                        seed=network_round_seed(seed, r),
                        gateway=gateway,
                        cross_mbps=cross,
                    )
                    gateway = res["gateway"]
                round_metrics.append({
                    "round_id": r,
                    "eta": res["eta"],
                    **res["metrics"],
                    "late_after_deadline": res.get("late_metrics", {}).get("late_after_deadline", 0.0),
                    "rtt_samples_ns": [] if is_baseline else res.get("rtt_samples_ns", []),
                })
            summary = _network_summary(round_metrics, pol)
            raw_record = {
                "job": job,
                "summary_metrics": summary,
                "rounds": round_metrics,
                "simulation": _network_provenance(
                    job, bank_sha256=value_bank.sha256, weights=value_bank.client_weights
                ),
                **({"transport_metrics": _transport_summary(round_metrics, pol)} if is_baseline else {}),
            }
    except Exception as exc:
        print(
            f"Run failed; recording status=missing: {exp} {job['config_id']} {pol} seed {seed}: "
            f"{type(exc).__name__}: {exc}"
        )
        return make_row(exp, scen, beta, N, pol, seed, n_rounds, cross_mbps=cross, status="missing")

    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_text(json.dumps(raw_record, indent=2), encoding="utf-8")
    tta_censored = summary.pop("tta_censored", None)
    return make_row(
        exp, scen, beta, N, pol, seed, n_rounds, cross_mbps=cross, status="ok",
        tta_censored=tta_censored, **summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="FedGrant Experiment Grid Runner")
    parser.add_argument("--exp", nargs="+", choices=EXPERIMENTS, default=None, help="Filter by experiment_id")
    parser.add_argument("--scenario", nargs="+", choices=SCENARIOS, default=None, help="Filter by scenario")
    parser.add_argument("--policy", nargs="+", choices=POLICIES, default=None, help="Filter by policy")
    parser.add_argument("--seed", nargs="+", type=int, default=None, help="Filter by run seed")
    parser.add_argument("--beta", nargs="+", type=float, default=None, help="Filter by beta")
    parser.add_argument("--clients", nargs="+", type=int, default=None, help="Filter by client count N")
    parser.add_argument("--machine-id", type=int, default=0, help="Machine index (0-based)")
    parser.add_argument("--num-machines", type=int, default=1, help="Total number of machines")
    parser.add_argument("--dry-run", action="store_true", help="Print run plan without executing")
    parser.add_argument("--run", action="store_true", help="Execute planned runs on this machine")
    parser.add_argument("--max-runs", type=int, default=None, help="Maximum number of runs to execute")
    parser.add_argument("--output-csv", type=str, default="results/summary/runs.csv", help="Path to summary runs.csv")
    parser.add_argument("--raw-dir", type=str, default="results/raw", help="Directory for raw run records")

    args = parser.parse_args()

    jobs = expand_grid(args.exp)
    if args.scenario:
        jobs = [job for job in jobs if job["scenario"] in args.scenario]
    if args.policy:
        jobs = [job for job in jobs if job["policy"] in args.policy]
    if args.seed is not None:
        jobs = [job for job in jobs if job["seed"] in args.seed]
    if args.beta is not None:
        jobs = [job for job in jobs if any(math.isclose(job["beta"], value, abs_tol=1e-12) for value in args.beta)]
    if args.clients is not None:
        jobs = [job for job in jobs if job["N"] in args.clients]
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
    existing_status: dict[tuple[str, str, str, int], str] = {}
    if csv_path.is_file():
        try:
            for r in read_runs_csv(csv_path):
                existing_status[r.key] = r.status
                rows.append(make_row(r.experiment_id, r.scenario, r.beta, r.N, r.policy, r.seed, r.n_rounds,
                                     cross_mbps=r.cross_mbps, status=r.status, tta_censored=r.tta_censored, **r.metrics))
        except Exception:
            pass

    print(f"\nStarting execution of {len(my_jobs)} runs...")
    t0 = time.time()
    for idx, job in enumerate(my_jobs, start=1):
        key = (job["experiment_id"], job["config_id"], job["policy"], job["seed"])
        if key in existing_status and existing_status[key] != "missing":
            print(f"[{idx}/{len(my_jobs)}] Skipping already completed run {key}")
            continue
        print(f"[{idx}/{len(my_jobs)}] Running {job['experiment_id']} | {job['config_id']} | {job['policy']} | seed {job['seed']}...")
        try:
            row = execute_job(job, raw_dir)
        except Exception as exc:
            print(f"Run failed; recording status=missing: {type(exc).__name__}: {exc}")
            row = make_row(
                job["experiment_id"], job["scenario"], job["beta"], job["N"],
                job["policy"], job["seed"], job["n_rounds"],
                cross_mbps=job["cross_mbps"], status="missing",
            )
        rows = [existing for existing in rows if existing.get("config_id") != job["config_id"]
                or existing.get("experiment_id") != job["experiment_id"]
                or existing.get("policy") != job["policy"]
                or str(existing.get("seed")) != str(job["seed"])]
        rows.append(row)
        existing_status[key] = row["status"]
        write_runs_csv(rows, csv_path)

    elapsed = time.time() - t0
    print(f"\nExecution finished in {elapsed:.1f}s. Wrote {len(rows)} validated rows to {csv_path}")


if __name__ == "__main__":
    main()
