"""Deterministic experiment-grid expansion, execution, resume, and summary.
The schema's ``expected_runs`` remains the canonical experiment matrix. This
runner executes grant-based and B2/B3 simulator configurations through the
Task 16 run-directory pipeline, and FEMNIST/FedAvg FL jobs through the existing
FL entry point. FL runs receive per-run metadata, run.csv, fl_history.csv, and
fl_result.json; network-only packet/queue logs are deliberately not fabricated.
F-ref publishes the frozen reference files consumed by F-main/F-abl. Production
Mininet client/baseline adapters remain unavailable for M1/M2.
"""

from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
import numpy as np
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from fedgrant.analysis import raw_metrics
from fedgrant.analysis.metrics import compute_final_accuracy, compute_time_to_target
from fedgrant.analysis.runs_schema import (
    EXPERIMENTS,
    FL_EXPERIMENTS,
    POLICIES,
    SCENARIOS,
    RunsSchemaError,
    expected_runs,
    make_config_id,
    make_row,
    parse_row,
    read_runs_csv,
    write_runs_csv,
)
from fedgrant.common.logschema import SCHEMAS, csv_writer, write_rows
from fedgrant.baselines.sim_adapter import run_baseline_network_round
from fedgrant.policy.types import PolicyConfig
from fedgrant.sim.run_net import compute_deadline_ns, run_network_round
from fedgrant.traces.client_trace import POOL_ID, POOL_OOD
GRID_DIR = Path(__file__).resolve().parents[1] / "configs" / "grids"
SCENARIO_DIR = Path(__file__).resolve().parents[1] / "configs" / "scenarios"
_CONFIG_ID_RE = re.compile(
    r"^(?P<scenario>.+)-b(?P<beta>[0-9]+(?:\.[0-9]+)?)-N(?P<N>[0-9]+)-c(?P<cross>[0-9]+(?:\.[0-9]+)?)$"
)
_ROUND_COUNTS = {
    "N-main": 200,
    "N-X1": 200,
    "N-X2": 200,
    "N-abl": 200,
    "F-main": 300,
    "F-ref": 300,
    "F-abl": 300,
    "M1": 100,
    "M2": 300,
    "M-sim": 100,
}
_SIMULATOR_POLICY_CONFIGS: dict[str, PolicyConfig] = {
    "B4": PolicyConfig("rr", "ewma"),
    "B5": PolicyConfig("srpt", "ewma"),
    "B6": PolicyConfig("value", "ewma"),
    "B7": PolicyConfig("value", "hmm"),
    "FG-GRU": PolicyConfig("value", "gru"),
    "FG-MLP": PolicyConfig("value", "mlp"),
    "ORACLE": PolicyConfig("value", "oracle"),
    "FG-GRU-A2": PolicyConfig("value", "gru", retransmit=False),
    "FG-GRU-A3": PolicyConfig("value", "gru", pred_in_pacing=False),
    "FG-GRU-A4": PolicyConfig("value", "gru", pred_in_eligibility=False),
}
_LOG_FILES = {
    "rounds.csv": "rounds",
    "packets.csv": "packets",
    "queues.csv": "queues",
    "round_metrics.csv": "round_metrics",
    "rtt_samples.csv": "rtt_samples",
    "scheduler_samples.csv": "scheduler_samples",
    "processes.csv": "processes",
    "run.csv": "run",
}
class GridError(ValueError):
    """Invalid matrix configuration or unsafe resume state."""
def _parse_config_id(config_id: str) -> tuple[str, float, int, float]:
    match = _CONFIG_ID_RE.fullmatch(config_id)
    if match is None:
        raise GridError(f"invalid config_id {config_id!r}")
    scenario = match.group("scenario")
    beta, n_clients, cross = float(match.group("beta")), int(match.group("N")), float(match.group("cross"))
    if make_config_id(scenario, beta, n_clients, cross) != config_id:
        raise GridError(f"config_id {config_id!r} is not canonical")
    return scenario, beta, n_clients, cross
def _matrix_keys() -> set[tuple[str, str, str, int]]:
    return set(expected_runs())
def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise GridError("PyYAML is required to validate repository grid/scenario YAML; install the testbed extra") from exc
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise GridError(f"cannot load configuration {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise GridError(f"configuration {path} must contain a YAML mapping")
    return value
def validate_repository_configs(
    grid_dir: str | Path = GRID_DIR,
    scenario_dir: str | Path = SCENARIO_DIR,
) -> None:
    """Validate existing YAML declarations against the canonical run schema."""
    grids = Path(grid_dir)
    expected = _matrix_keys()
    for experiment in ("M1", "M-sim"):
        path = grids / ("mn_m1.yaml" if experiment == "M1" else "mn_sim.yaml")
        if not path.is_file():
            raise GridError(f"missing declared grid configuration: {path}")
        config = _read_yaml(path)
        if config.get("experiment_id") != experiment:
            raise GridError(f"{path}: experiment_id must be {experiment!r}")
        expected_platform = "mininet" if experiment in {"M1", "M2"} else "sim"
        if config.get("platform") != expected_platform:
            raise GridError(f"{path}: platform must be {expected_platform!r}")
        scenarios = config.get("scenario")
        betas = config.get("beta")
        policies = config.get("policies")
        seeds = config.get("seeds")
        if not all(isinstance(value, list) and value for value in (scenarios, betas, policies, seeds)):
            raise GridError(f"{path}: scenario, beta, policies, and seeds must be non-empty lists")
        n_clients = config.get("clients", config.get("N"))
        rounds = config.get("rounds")
        cross = config.get("cross_traffic_mbps", 0.0)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (*seeds, rounds, n_clients)):
            raise GridError(f"{path}: clients, rounds, and seeds must be non-negative integers")
        if rounds != _ROUND_COUNTS[experiment]:
            raise GridError(f"{path}: rounds={rounds} disagrees with runs_schema ({_ROUND_COUNTS[experiment]})")
        declared: set[tuple[str, str, str, int]] = set()
        for scenario in scenarios:
            for beta in betas:
                for policy in policies:
                    for seed in seeds:
                        if policy not in POLICIES:
                            raise GridError(f"{path}: unknown policy {policy!r}")
                        config_id = make_config_id(str(scenario), float(beta), int(n_clients), float(cross))
                        declared.add((experiment, config_id, str(policy), int(seed)))
        canonical = {key for key in expected if key[0] == experiment}
        if declared != canonical:
            missing, extra = sorted(canonical - declared), sorted(declared - canonical)
            raise GridError(f"{path}: declared run matrix differs from runs_schema; missing={missing[:3]}, extra={extra[:3]}")
    scenarios_root = Path(scenario_dir)
    for scenario in { _parse_config_id(key[1])[0] for key in expected } - {"none"}:
        path = scenarios_root / f"{scenario}.yaml"
        if not path.is_file():
            raise GridError(f"missing scenario configuration: {path}")
        config = _read_yaml(path)
        if config.get("scenario") != scenario:
            raise GridError(f"{path}: scenario field must be {scenario!r}")
def _run_id(experiment_id: str, config_id: str, policy: str, seed: int) -> str:
    key = (experiment_id, config_id, policy, int(seed))
    digest = hashlib.sha256(json.dumps(key, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
    return f"{experiment_id}__{config_id}__{policy}__s{seed}__{digest}"
def expand_grid(
    experiments: list[str] | None = None,
    *,
    policies: list[str] | None = None,
    scenarios: list[str] | None = None,
    validate_configs: bool = True,
) -> list[dict[str, Any]]:
    """Expand the canonical Section 11 matrix in stable schema order."""
    if experiments is not None:
        unknown = set(experiments) - set(EXPERIMENTS)
        if unknown:
            raise GridError(f"unknown experiment(s): {sorted(unknown)}")
    if policies is not None:
        unknown = set(policies) - set(POLICIES)
        if unknown:
            raise GridError(f"unknown policy(s): {sorted(unknown)}")
    if scenarios is not None:
        unknown = set(scenarios) - set(SCENARIOS)
        if unknown:
            raise GridError(f"unknown scenario(s): {sorted(unknown)}")
    if validate_configs:
        validate_repository_configs()
    selected = []
    for experiment_id, config_id, policy, seed in expected_runs():
        if experiments is not None and experiment_id not in experiments:
            continue
        scenario, beta, n_clients, cross = _parse_config_id(config_id)
        if policies is not None and policy not in policies:
            continue
        if scenarios is not None and scenario not in scenarios:
            continue
        job = {
            "experiment_id": experiment_id,
            "platform": "mininet" if experiment_id in {"M1", "M2"} else "sim",
            "config_id": config_id,
            "scenario": scenario,
            "beta": beta,
            "N": n_clients,
            "n_clients": n_clients,
            "cross_mbps": cross,
            "policy": policy,
            "seed": int(seed),
            "n_rounds": _ROUND_COUNTS[experiment_id],
        }
        job["run_id"] = _run_id(experiment_id, config_id, policy, int(seed))
        validate_job(job)
        selected.append(job)
    return selected
def validate_job(job: Mapping[str, Any]) -> None:
    """Reject malformed or out-of-matrix jobs before allocating output paths."""
    required = (
        "experiment_id", "platform", "config_id", "scenario", "beta", "N", "n_clients",
        "cross_mbps", "policy", "seed", "n_rounds", "run_id",
    )
    missing = [key for key in required if key not in job]
    if missing:
        raise GridError(f"job missing required field(s): {missing}")
    key = (str(job["experiment_id"]), str(job["config_id"]), str(job["policy"]), int(job["seed"]))
    if key not in _matrix_keys():
        raise GridError(f"job is not in the canonical experiment matrix: {key}")
    expected_platform = "mininet" if key[0] in {"M1", "M2"} else "sim"
    if job["platform"] != expected_platform:
        raise GridError(f"{key[0]} must use platform {expected_platform!r}")
    if isinstance(job["seed"], bool) or not isinstance(job["seed"], int) or job["seed"] < 0:
        raise GridError("seed must be a non-negative integer")
    if isinstance(job["n_rounds"], bool) or not isinstance(job["n_rounds"], int):
        raise GridError("n_rounds must be an integer")
    for name in ("beta", "cross_mbps"):
        value = job[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or float(value) < 0:
            raise GridError(f"{name} must be a finite non-negative number")
    scenario, beta, n_clients, cross = _parse_config_id(key[1])
    if job["n_rounds"] != _ROUND_COUNTS[key[0]]:
        raise GridError(f"{key[0]} requires {_ROUND_COUNTS[key[0]]} rounds, got {job['n_rounds']}")
    if job.get("scenario") != scenario or float(job.get("beta", math.nan)) != beta:
        raise GridError(f"job parameters disagree with config_id {key[1]!r}")
    if isinstance(job["N"], bool) or not isinstance(job["N"], int) or job["N"] != n_clients:
        raise GridError(f"job client count disagrees with config_id {key[1]!r}")
    if isinstance(job["n_clients"], bool) or not isinstance(job["n_clients"], int) or job["n_clients"] != n_clients:
        raise GridError(f"job n_clients disagrees with config_id {key[1]!r}")
    if float(job.get("cross_mbps", math.nan)) != cross:
        raise GridError(f"job cross traffic disagrees with config_id {key[1]!r}")
    if key[2] not in POLICIES:
        raise GridError(f"unsupported policy {key[2]!r}")
    if job["run_id"] != _run_id(*key):
        raise GridError("run_id is not the stable identifier for this matrix key")
def partition_jobs(
    jobs: list[dict[str, Any]],
    machine_id: int,
    num_machines: int,
    *,
    worker_id: int = 0,
    num_workers: int = 1,
) -> list[dict[str, Any]]:
    """Stable, disjoint machine/worker partition using global modulo assignment."""
    for name, value in (("machine_id", machine_id), ("worker_id", worker_id)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name, value in (("num_machines", num_machines), ("num_workers", num_workers)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if machine_id >= num_machines or worker_id >= num_workers:
        raise ValueError("machine_id and worker_id must be within their configured counts")
    slot = machine_id * num_workers + worker_id
    slots = num_machines * num_workers
    return [job for index, job in enumerate(jobs) if index % slots == slot]
def network_round_seed(master_seed: int, round_id: int) -> int:
    """Match ``fedgrant.fl.server._network_round_seed`` exactly, without importing FL/Torch."""
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise ValueError("master_seed must be a non-negative integer")
    if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 1:
        raise ValueError("round_id must be a positive integer")
    return int(np.random.SeedSequence([master_seed, round_id - 1]).generate_state(1)[0])
def _adapter_block_reasons(job: Mapping[str, Any], trace_dir: Path) -> list[str]:
    experiment, policy = str(job["experiment_id"]), str(job["policy"])
    reasons: list[str] = []
    if experiment == "M1":
        reasons.append("the Mininet launcher has a gateway-factory hook but no production client/baseline adapter; its client process uses synthetic updates")
    elif experiment == "M2":
        reasons.extend((
            "the Mininet launcher has no production client/baseline adapter and its client process uses synthetic updates",
            "the FL server/model-update/bitmap-aggregation lifecycle is not integrated with the Mininet run directory",
        ))
    elif experiment in FL_EXPERIMENTS and policy in {"B2", "B3"}:
        reasons.append(
            f"{policy} FL transport adapter is not integrated with the FEMNIST/FedAvg entry point"
        )
    elif policy in {"B2", "B3"} and experiment not in {
        "N-main", "N-X1", "N-X2", "N-abl", "M-sim",
    }:
        reasons.append(f"simulator baseline adapter for {policy!r} is unavailable in {experiment!r}")
    elif policy not in _SIMULATOR_POLICY_CONFIGS:
        if policy not in {"B2", "B3"}:
            reasons.append(f"simulator transfer-mode adapter for policy {policy!r} is unavailable")
    if job["scenario"] in {"L4", "L5"}:
        names = POOL_OOD if job["scenario"] == "L5" else POOL_ID
        split = "ood" if job["scenario"] == "L5" else "test"
        missing = [f"{name}_{split}.csv" for name in names if not (trace_dir / f"{name}_{split}.csv").is_file()]
        if missing:
            reasons.append(f"required trace input(s) unavailable under {trace_dir}: {', '.join(missing)}")
    if policy in {"FG-GRU", "FG-GRU-A2", "FG-GRU-A3", "FG-GRU-A4"} and not Path("models/gru_prod.npz").is_file():
        reasons.append("production GRU checkpoint models/gru_prod.npz is unavailable")
    if policy == "FG-MLP" and not Path("models/mlp_prod.npz").is_file():
        reasons.append("production MLP checkpoint models/mlp_prod.npz is unavailable")
    return reasons
def _input_hashes(job: Mapping[str, Any], trace_dir: Path) -> dict[str, str]:
    """Hash configuration, trace, and model assets that affect reproducibility."""
    candidates: dict[str, Path] = {}
    scenario_path = SCENARIO_DIR / f"{job['scenario']}.yaml"
    if scenario_path.is_file():
        candidates[f"scenario:{scenario_path.name}"] = scenario_path
    if job["experiment_id"] in {"M1", "M2", "M-sim"}:
        grid_name = "mn_m1.yaml" if job["experiment_id"] in {"M1", "M2"} else "mn_sim.yaml"
        grid_path = GRID_DIR / grid_name
        if grid_path.is_file():
            candidates[f"grid:{grid_path.name}"] = grid_path
    if job["scenario"] in {"L4", "L5"}:
        names = POOL_OOD if job["scenario"] == "L5" else POOL_ID
        split = "ood" if job["scenario"] == "L5" else "test"
        candidates.update({f"trace:{name}_{split}.csv": trace_dir / f"{name}_{split}.csv" for name in names})
    if job["policy"] in {"FG-GRU", "FG-GRU-A2", "FG-GRU-A3", "FG-GRU-A4"}:
        candidates["model:gru_prod.npz"] = Path("models/gru_prod.npz")
    if job["policy"] == "FG-MLP":
        candidates["model:mlp_prod.npz"] = Path("models/mlp_prod.npz")
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sorted(candidates.items())
        if path.is_file()
    }
def _config_fingerprint(job: Mapping[str, Any], input_hashes: Mapping[str, str]) -> str:
    fields = {key: job[key] for key in ("experiment_id", "platform", "config_id", "scenario", "beta", "N", "n_clients", "cross_mbps", "policy", "seed", "n_rounds")}
    fields["input_sha256"] = dict(sorted(input_hashes.items()))
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
def _identity(job: Mapping[str, Any]) -> dict[str, Any]:
    return {key: job[key] for key in ("experiment_id", "config_id", "policy", "seed")}
def _metadata_base(job: Mapping[str, Any], trace_dir: Path) -> dict[str, Any]:
    input_hashes = _input_hashes(job, trace_dir)
    trace_hashes = {name.removeprefix("trace:"): digest for name, digest in input_hashes.items() if name.startswith("trace:")}
    return {
        "run_id": job["run_id"],
        "identity": _identity(job),
        "config_sha256": _config_fingerprint(job, input_hashes),
        "input_sha256": input_hashes,
        "experiment_id": job["experiment_id"],
        "platform": job["platform"],
        "config_id": job["config_id"],
        "scenario": job["scenario"],
        "beta": job["beta"],
        "N": job["N"],
        "policy": job["policy"],
        "seed": job["seed"],
        "master_seed": job["seed"],
        "n_rounds": job["n_rounds"],
        "cross_mbps": job["cross_mbps"],
        "seed_convention": "network round seed = SeedSequence([master_seed, round_id - 1]).generate_state(1)[0]",
        "trace_dir": str(trace_dir),
        "trace_sha256": trace_hashes,
        "cpu_roles": None,
        "effective_qdisc_settings": None,
    }
def _row_for_missing(job: Mapping[str, Any]) -> dict[str, Any]:
    return make_row(
        str(job["experiment_id"]), str(job["scenario"]), float(job["beta"]), int(job["N"]),
        str(job["policy"]), int(job["seed"]), int(job["n_rounds"]),
        cross_mbps=float(job["cross_mbps"]), status="missing",
    )
def _log_row(schema: str, **values: Any) -> dict[str, Any]:
    return {
        column: "" if values.get(column) is None else values.get(column, "")
        for column in SCHEMAS[schema]
    }
def _initialize_logs(directory: Path, job: Mapping[str, Any], *, status: str = "running") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, schema in _LOG_FILES.items():
        path = directory / filename
        if schema == "run":
            run_record = {
                "run_id": job["run_id"],
                "experiment_id": job["experiment_id"],
                "config_id": job["config_id"],
                "policy": job["policy"],
                "seed": job["seed"],
                "scenario": job["scenario"],
                "beta": job["beta"],
                "n_clients": job["N"],
                "cross_mbps": job["cross_mbps"],
                "status": status,
                "started_mono_ns": time.monotonic_ns(),
                "ended_mono_ns": "",
            }
            write_rows(path, schema, [run_record])
        else:
            with path.open("w", newline="", encoding="utf-8") as stream:
                csv_writer(stream, schema)
def _ensure_log_headers(directory: Path, job: Mapping[str, Any]) -> None:
    """Make the documented log set discoverable without overwriting prior data."""
    directory.mkdir(parents=True, exist_ok=True)
    for filename, schema in _LOG_FILES.items():
        path = directory / filename
        if path.exists():
            continue
        if schema == "run":
            record = {
                "run_id": job["run_id"],
                "experiment_id": job["experiment_id"],
                "config_id": job["config_id"],
                "policy": job["policy"],
                "seed": job["seed"],
                "scenario": job["scenario"],
                "beta": job["beta"],
                "n_clients": job["N"],
                "cross_mbps": job["cross_mbps"],
                "status": "missing",
                "started_mono_ns": "",
                "ended_mono_ns": "",
            }
            write_rows(path, schema, [record])
        else:
            with path.open("w", newline="", encoding="utf-8") as stream:
                csv_writer(stream, schema)
def _write_simulator_run(
    job: Mapping[str, Any],
    stage: Path,
    *,
    trace_dir: Path,
    simulation_runner: Callable[..., Mapping[str, Any]],
    baseline_runner: Callable[..., Mapping[str, Any]],
) -> list[dict[str, Any]]:
    _initialize_logs(stage, job)
    round_rows: list[dict[str, Any]] = []
    rtt_rows: list[dict[str, Any]] = []
    round_event_rows: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    gateway = None
    policy_name = str(job["policy"])
    is_baseline = policy_name in {"B2", "B3"}
    policy = _SIMULATOR_POLICY_CONFIGS.get(policy_name)
    simulation_scenario = "X2" if float(job["cross_mbps"]) > 0 else str(job["scenario"])
    for round_id in range(1, int(job["n_rounds"]) + 1):
        grant_rtt_count = 0

        def sample_hook(record: Mapping[str, Any]) -> None:
            nonlocal grant_rtt_count
            if record.get("event") == "grant_resolved" and record.get("y") == 1 and record.get("rtt_ns") is not None:
                rtt_ns = int(record["rtt_ns"])
                grant_rtt_count += 1
                rtt_rows.append(_log_row(
                    "rtt_samples",
                    t_mono_ns=int(record.get("abs_t_ns", record["t_ns"])),
                    round_id=round_id,
                    rtt_ms=rtt_ns / 1_000_000.0,
                    rtt_type="grant_resolution",
                    client_id=record.get("client_id", ""),
                ))
        if is_baseline:
            result = baseline_runner(
                round_id=round_id,
                num_clients=int(job["N"]),
                beta=float(job["beta"]),
                scenario=simulation_scenario,
                policy=policy_name,
                seed=network_round_seed(int(job["seed"]), round_id),
                gateway=gateway,
                trace_dir=str(trace_dir),
            )
        else:
            if policy is None:
                raise GridError(f"simulator policy adapter for {policy_name!r} is unavailable")
            result = simulation_runner(
                round_id=round_id,
                num_clients=int(job["N"]),
                beta=float(job["beta"]),
                scenario=simulation_scenario,
                policy_config=policy,
                seed=network_round_seed(int(job["seed"]), round_id),
                gateway=gateway,
                trace_dir=str(trace_dir),
                sample_hook=sample_hook,
            )
        gateway = result["gateway"]
        metrics = result["metrics"]
        late_metrics = result.get("late_metrics", {})
        baseline_rtts = result.get("rtt_samples_ns", {})
        for client_id, rtt_ns_value in baseline_rtts.items():
            rtt_ns = int(rtt_ns_value)
            rtt_rows.append(_log_row(
                "rtt_samples",
                t_mono_ns=result.get("rtt_sample_times_ns", {}).get(client_id, ""),
                round_id=round_id,
                rtt_ms=rtt_ns / 1_000_000.0,
                rtt_type="plan_data",
                client_id=client_id,
            ))
        for record in result.get("round_events", ()):
            round_event_rows.append(_log_row(
                "rounds",
                t_mono_ns=record["t_ns"],
                round_id=record["round_id"],
                event=record["event"],
                client_id=record["client_id"],
                detail=record.get("detail", ""),
            ))
        for record in result.get("packet_events", ()):
            packet_rows.append(_log_row(
                "packets",
                t_mono_ns=record["t_ns"],
                round_id=record["round_id"],
                client_id=record["client_id"],
                packet_seq=record["packet_seq"],
                packet_type=record["packet_type"],
                bytes=record["bytes"],
                result=record["result"],
            ))
        for record in result.get("queue_samples", ()):
            queue_rows.append(_log_row(
                "queues",
                t_mono_ns=record["t_ns"],
                interface="bottleneck",
                qdisc="drop-tail",
                backlog_bytes=record["backlog_bytes"],
                drops=record["drops"],
            ))
        round_rows.append(_log_row(
            "round_metrics",
            round_id=round_id,
            eta=float(result["eta"]),
            delivered_blocks=float(sum(result["delivered_counts"])),
            clients_represented=metrics.get("clients_represented"),
            deadline_completion=metrics.get("deadline_completion"),
            goodput_bps=metrics.get("goodput_bps"),
            utilization=metrics.get("utilization"),
            queue_drops_fg=metrics.get("drops_queue_bn_client"),
            queue_drops_cross=metrics.get(
                "drops_queue_bn_cross",
                metrics.get("drops_queue_cross"),
            ),
            wasted_grants=metrics.get("wasted_grants_ratio"),
            late_after_timeout=metrics.get("late_after_timeout"),
            late_after_deadline=late_metrics.get("late_after_deadline", metrics.get("late_after_deadline")),
            protocol_overhead=metrics.get("protocol_overhead"),
            planned_blocks=metrics.get("planned_blocks"),
            plan_messages_sent=metrics.get("plan_messages_sent"),
            data_packets_sent=metrics.get("data_packets_sent"),
            data_bytes_sent=metrics.get("data_bytes_sent"),
            grants_issued=metrics.get("grants_issued"),
            grants_delivered=metrics.get("grants_delivered"),
            grants_timed_out=metrics.get("grants_timed_out"),
            grant_rtt_samples=grant_rtt_count if not is_baseline else None,
            plan_data_rtt_samples=len(baseline_rtts) if is_baseline and policy_name == "B3" else None,
            queue_drops_data=metrics.get("drops_queue_data"),
            queueing_delay_mean_ms=metrics.get("queueing_delay_mean_ms"),
            queueing_delay_p99_ms=metrics.get("queueing_delay_p99_ms"),
        ))
    write_rows(stage / "rounds.csv", "rounds", round_event_rows)
    write_rows(stage / "packets.csv", "packets", packet_rows)
    write_rows(stage / "queues.csv", "queues", queue_rows)
    write_rows(stage / "round_metrics.csv", "round_metrics", round_rows)
    write_rows(stage / "rtt_samples.csv", "rtt_samples", rtt_rows)
    run_csv_path = stage / "run.csv"
    with run_csv_path.open("r", newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    records[0]["ended_mono_ns"] = time.monotonic_ns()
    records[0]["status"] = "ok"
    write_rows(run_csv_path, "run", records)
    return round_rows
def _complete_artifacts_exist(run_dir: Path) -> bool:
    if not all((run_dir / filename).is_file() for filename in _LOG_FILES):
        return False
    for filename, schema in _LOG_FILES.items():
        with (run_dir / filename).open("r", newline="", encoding="utf-8") as stream:
            if tuple(next(csv.reader(stream), ())) != SCHEMAS[schema]:
                return False
    try:
        raw_metrics.run_row(run_dir)
    except (OSError, ValueError, KeyError, TypeError, RunsSchemaError, json.JSONDecodeError):
        return False
    return True
def _load_metadata(path: Path, job: Mapping[str, Any], trace_dir: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GridError(f"cannot safely resume from malformed metadata {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise GridError(f"cannot safely resume from non-object metadata {path}")
    if metadata.get("identity") != _identity(job):
        raise GridError(f"run directory identity mismatch at {path.parent}")
    if metadata.get("config_sha256") != _config_fingerprint(job, _input_hashes(job, trace_dir)):
        raise GridError(f"existing run configuration differs from requested job at {path.parent}")
    return metadata
def _blocked_metadata(base: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    now = time.monotonic_ns()
    return {
        **base,
        "status": "missing",
        "execution_state": "blocked",
        "availability": "blocked",
        "block_reasons": reasons,
        "started_mono_ns": now,
        "ended_mono_ns": now,
        "attempt_count": 0,
        "attempts": [],
        "summary_row": _row_for_missing(base),
    }
def execute_job(
    job: dict[str, Any],
    raw_dir: str | Path,
    *,
    trace_dir: str | Path = "data/traces",
    simulation_runner: Callable[..., Mapping[str, Any]] = run_network_round,
    baseline_runner: Callable[..., Mapping[str, Any]] = run_baseline_network_round,
) -> dict[str, Any]:
    """Run/resume one schema job under a per-run lock; return its summary row."""
    validate_job(job)
    raw_root, traces = Path(raw_dir), Path(trace_dir)
    if str(job["experiment_id"]) in FL_EXPERIMENTS:
        # FL writes its established raw JSON format and summary metrics. Keep
        # it out of the simulator CSV-log pipeline; use the same per-run lock
        # to prevent two grid workers from launching the same FL job together.
        fl_lock = raw_root / ".locks" / f"{job['run_id']}.lock"
        with _exclusive_lock(fl_lock):
            return _execute_fl_job(job, raw_root, trace_dir=traces)
    run_dir = raw_root / str(job["experiment_id"]) / str(job["run_id"])
    metadata_path = run_dir / "meta.json"
    lock_path = raw_root / ".locks" / f"{job['run_id']}.lock"
    with _exclusive_lock(lock_path):
        if run_dir.exists() and not metadata_path.exists() and any(run_dir.iterdir()):
            raise GridError(f"refusing to reuse non-empty run directory without metadata: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        existing = _load_metadata(metadata_path, job, traces)
        if existing is not None and existing.get("execution_state") == "complete":
            if _complete_artifacts_exist(run_dir):
                return raw_metrics.run_row(run_dir)
            existing["execution_state"] = "incomplete"
            existing["status"] = "missing"
            existing.setdefault("attempts", []).append({
                "attempt": existing.get("attempt_count", 0),
                "state": "incomplete",
                "error": "completed metadata exists but raw artifacts are incomplete or invalid",
            })
            _atomic_json(metadata_path, existing)
        _ensure_log_headers(run_dir, job)
        reasons = _adapter_block_reasons(job, traces)
        if reasons:
            if existing is None or existing.get("execution_state") != "blocked":
                blocked = _blocked_metadata(_metadata_base(job, traces), reasons)
                _atomic_json(metadata_path, blocked)
            else:
                blocked = existing
            return dict(blocked["summary_row"])
        base = _metadata_base(job, traces)
        attempts = list(existing.get("attempts", [])) if existing else []
        attempt_count = int(existing.get("attempt_count", 0)) if existing else 0
        state = existing.get("execution_state") if existing else None
        if state == "missing" and attempt_count >= 2:
            return dict(existing.get("summary_row", _row_for_missing(job)))
        if state == "running":
            attempts.append({"attempt": attempt_count, "state": "interrupted", "error": "previous process ended while running"})
        max_attempt_count = 2
        while attempt_count < max_attempt_count:
            attempt_count += 1
            started_ns = time.monotonic_ns()
            current = {
                **base,
                "status": "missing",
                "execution_state": "running",
                "availability": "available",
                "block_reasons": [],
                "started_mono_ns": existing.get("started_mono_ns", started_ns) if existing else started_ns,
                "ended_mono_ns": None,
                "attempt_count": attempt_count,
                "attempts": attempts,
                "active_attempt": attempt_count,
            }
            _atomic_json(metadata_path, current)
            stage = run_dir / "attempts" / f"attempt-{attempt_count:02d}-{os.getpid()}"
            if stage.exists():
                raise GridError(f"attempt staging directory already exists: {stage}")
            try:
                _write_simulator_run(
                    job,
                    stage,
                    trace_dir=traces,
                    simulation_runner=simulation_runner,
                    baseline_runner=baseline_runner,
                )
                stage_meta = {
                    **base,
                    "status": "ok" if attempt_count == 1 else "rerun",
                    "execution_state": "complete",
                    "availability": "available",
                    "attempt_count": attempt_count,
                    "started_mono_ns": started_ns,
                    "ended_mono_ns": time.monotonic_ns(),
                    "n_rounds_completed": int(job["n_rounds"]),
                }
                _atomic_json(stage / "meta.json", stage_meta)
                row = raw_metrics.run_row(stage)
                row["status"] = "ok" if attempt_count == 1 else "rerun"
                parse_row(row)
                run_csv_path = stage / "run.csv"
                with run_csv_path.open("r", newline="", encoding="utf-8") as stream:
                    run_records = list(csv.DictReader(stream))
                run_records[0]["status"] = row["status"]
                write_rows(run_csv_path, "run", run_records)
                _archive_existing_logs(run_dir, attempt_count)
                for filename in (*_LOG_FILES,):
                    os.replace(stage / filename, run_dir / filename)
                finished_ns = time.monotonic_ns()
                attempts.append({"attempt": attempt_count, "state": "complete" if attempt_count == 1 else "rerun", "started_mono_ns": started_ns, "ended_mono_ns": finished_ns})
                current.update({
                    "status": row["status"],
                    "execution_state": "complete",
                    "ended_mono_ns": finished_ns,
                    "attempts": attempts,
                    "n_rounds_completed": int(job["n_rounds"]),
                    "summary_row": row,
                })
                _atomic_json(metadata_path, current)
                return row
            except KeyboardInterrupt:
                ended_ns = time.monotonic_ns()
                attempts.append({"attempt": attempt_count, "state": "interrupted", "started_mono_ns": started_ns, "ended_mono_ns": ended_ns})
                current.update({"execution_state": "interrupted", "ended_mono_ns": ended_ns, "attempts": attempts})
                _atomic_json(metadata_path, current)
                raise
            except Exception as exc:
                ended_ns = time.monotonic_ns()
                attempts.append({
                    "attempt": attempt_count,
                    "state": "failed",
                    "started_mono_ns": started_ns,
                    "ended_mono_ns": ended_ns,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                current.update({"execution_state": "failed", "ended_mono_ns": ended_ns, "attempts": attempts})
                _atomic_json(metadata_path, current)
                existing = current
        missing = {
            **base,
            "status": "missing",
            "execution_state": "missing",
            "availability": "available",
            "block_reasons": [],
            "started_mono_ns": existing.get("started_mono_ns", time.monotonic_ns()) if existing else time.monotonic_ns(),
            "ended_mono_ns": time.monotonic_ns(),
            "attempt_count": attempt_count,
            "attempts": attempts,
            "summary_row": _row_for_missing(job),
        }
        _atomic_json(metadata_path, missing)
        return dict(missing["summary_row"])
def _archive_existing_logs(run_dir: Path, attempt: int) -> None:
    existing = [run_dir / name for name in (*_LOG_FILES,)]
    present = [path for path in existing if path.exists()]
    if not present:
        return
    archive = run_dir / "attempts" / f"previous-{attempt:02d}"
    archive.mkdir(parents=True, exist_ok=False)
    for path in present:
        os.replace(path, archive / path.name)
def upsert_summary_row(row: Mapping[str, Any], output_csv: str | Path) -> None:
    """Atomically upsert one run row, serialized across machines/workers."""
    path = Path(output_csv)
    parse_row(row)
    lock = path.with_name(path.name + ".lock")
    key = (row["experiment_id"], row["config_id"], row["policy"], int(row["seed"]))
    with _exclusive_lock(lock):
        existing: list[dict[str, Any]] = []
        if path.is_file():
            for run in read_runs_csv(path):
                if run.key != key:
                    existing.append(make_row(
                        run.experiment_id, run.scenario, run.beta, run.N, run.policy, run.seed, run.n_rounds,
                        cross_mbps=run.cross_mbps, status=run.status,
                        tta_censored=run.tta_censored, **run.metrics,
                    ))
        existing.append(dict(row))
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            write_runs_csv(existing, temp_path)
            with temp_path.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
def merge_summary_files(inputs: list[str | Path], output_csv: str | Path) -> Path:
    """Merge per-machine summary shards; duplicate schema keys are rejected."""
    rows: list[dict[str, Any]] = []
    for source in inputs:
        for run in read_runs_csv(source):
            rows.append(make_row(
                run.experiment_id, run.scenario, run.beta, run.N, run.policy, run.seed, run.n_rounds,
                cross_mbps=run.cross_mbps, status=run.status,
                tta_censored=run.tta_censored, **run.metrics,
            ))
    path = Path(output_csv)
    lock = path.with_name(path.name + ".lock")
    with _exclusive_lock(lock):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            write_runs_csv(rows, temp_path)
            with temp_path.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    return path
def _print_plan(jobs: list[dict[str, Any]], trace_dir: Path) -> None:
    available = 0
    blocked = 0
    for job in jobs:
        reasons = _adapter_block_reasons(job, trace_dir)
        if reasons:
            blocked += 1
        else:
            available += 1
    print(f"Expanded jobs: {len(jobs)} ({available} executable by an available adapter; {blocked} blocked/unavailable)")
    for job in jobs[:8]:
        reasons = _adapter_block_reasons(job, trace_dir)
        label = "available" if not reasons else "blocked: " + "; ".join(reasons)
        print(f"  {job['run_id']} | {job['n_rounds']} rounds | {label}")
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
def _write_fl_artifacts(
    job: Mapping[str, Any],
    raw_root: Path,
    result: Mapping[str, Any],
    *,
    trace_dir: Path,
) -> None:
    """Persist FL-native artifacts without fabricating network telemetry.

    FL runs have model/evaluation history, not gateway packet/queue samples.
    Keep those artifacts in the same per-run directory/metadata convention as
    simulator runs, but use fl_history.csv rather than empty network CSVs.
    The legacy flat JSON is retained separately for compatibility with the
    existing FEMNIST reference loader.
    """
    run_dir = raw_root / str(job["experiment_id"]) / str(job["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    now = time.monotonic_ns()
    metadata = {
        **_metadata_base(job, trace_dir),
        "status": str(result.get("status", "ok")),
        "execution_state": "complete" if result.get("status") == "ok" else "missing",
        "availability": "available",
        "artifact_type": "federated_learning",
        "artifacts": ["meta.json", "run.csv", "fl_history.csv", "fl_result.json"],
        "started_mono_ns": result.get("started_mono_ns", now),
        "ended_mono_ns": result.get("ended_mono_ns", now),
        "n_rounds_completed": len(result.get("history", [])),
        "block_reasons": [],
    }
    _atomic_json(run_dir / "meta.json", metadata)
    run_record = {
        "run_id": job["run_id"],
        "experiment_id": job["experiment_id"],
        "config_id": job["config_id"],
        "policy": job["policy"],
        "seed": job["seed"],
        "scenario": job["scenario"],
        "beta": job["beta"],
        "n_clients": job["N"],
        "cross_mbps": job["cross_mbps"],
        "status": result.get("status", "ok"),
        "started_mono_ns": metadata["started_mono_ns"],
        "ended_mono_ns": metadata["ended_mono_ns"],
    }
    write_rows(run_dir / "run.csv", "run", [run_record])
    history = list(result.get("history", []))
    columns = list(dict.fromkeys(key for row in history for key in row))
    history_path = run_dir / "fl_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        if columns:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(history)
        else:
            stream.write("\n")
    _atomic_json(run_dir / "fl_result.json", result)


def _write_full_reference(job: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    """Publish F-ref in the exact location consumed by F-main/F-abl."""
    if job["experiment_id"] != "F-ref":
        return
    seed = int(job["seed"])
    target = Path("data/runs") / f"femnist_full_s{seed}.json"
    _atomic_json(target, {"history": list(result.get("history", [])), "seed": seed,
                          "experiment_id": "F-ref", "run_id": job["run_id"]})


def _execute_fl_job(
    job: dict[str, Any], raw_dir: Path, *, trace_dir: Path
) -> dict[str, Any]:
    """Run FL experiments through the FEMNIST/FedAvg path, never network-only."""
    exp, pol, seed = job["experiment_id"], job["policy"], int(job["seed"])
    scen, beta, n_clients = job["scenario"], float(job["beta"]), int(job["N"])
    rounds, cross = int(job["n_rounds"]), float(job["cross_mbps"])
    raw_file = raw_dir / f"{exp}_{job['config_id']}_{pol}_s{seed}.json"
    # B2 push and B3 plan-then-send require Prajin's client/gateway adapters.
    # Ignore any old raw record produced by the former incorrect dispatch.
    if pol in {"B2", "B3"}:
        reason = f"{pol} FL transport adapter is not integrated with the FEMNIST/FedAvg entry point"
        blocked = {"job": job, "status": "missing", "reason": reason,
                   "history": [], "started_mono_ns": time.monotonic_ns(),
                   "ended_mono_ns": time.monotonic_ns()}
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(raw_file, blocked)
        _write_fl_artifacts(job, raw_dir, blocked, trace_dir=trace_dir)
        run_dir = raw_dir / exp / str(job["run_id"])
        metadata_path = run_dir / "meta.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update({"execution_state": "blocked", "availability": "blocked",
                         "block_reasons": [reason]})
        _atomic_json(metadata_path, metadata)
        return make_row(exp, scen, beta, n_clients, pol, seed, rounds, cross_mbps=cross, status="missing")
    if raw_file.is_file():
        try:
            data = json.loads(raw_file.read_text(encoding="utf-8"))
            run_dir = raw_dir / exp / str(job["run_id"])
            metadata_path = run_dir / "meta.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_config = _metadata_base(job, trace_dir)
            required_artifacts = ("run.csv", "fl_history.csv", "fl_result.json")
            cache_is_current = (
                data.get("status") == "ok"
                and data.get("job") == job
                and isinstance(data.get("summary_metrics"), Mapping)
                and "final_acc" in data["summary_metrics"]
                and metadata.get("identity") == _identity(job)
                and metadata.get("config_sha256") == expected_config["config_sha256"]
                and metadata.get("status") == "ok"
                and metadata.get("execution_state") == "complete"
                and metadata.get("artifact_type") == "federated_learning"
                and all((run_dir / name).is_file() for name in required_artifacts)
            )
            if cache_is_current:
                _write_full_reference(job, data)
                return make_row(exp, scen, beta, n_clients, pol, seed, rounds, cross_mbps=cross,
                                status="ok", tta_censored=data.get("tta_censored"),
                                **data["summary_metrics"])
        except Exception:
            pass
    started_ns = time.monotonic_ns()
    try:
        full_final = None if exp == "F-ref" else _load_full_reference_final_accuracies()
        # Lazy import keeps the non-ML simulator/test commands independent of torch.
        from fedgrant.fl.run_fl import run_experiment
        bank_path = "data/bank/femnist_full_s0.npz" if exp == "F-ref" and seed == 0 else None
        no_ef = exp == "F-abl" and pol == "FG-GRU-A1"
        history = run_experiment(seed=seed, rounds=rounds, policy="FULL" if exp == "F-ref" else pol,
                                 beta=beta, scenario=scen, no_ef=no_ef, bank_path=bank_path,
                                 trace_dir=str(trace_dir))
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
        result_record = {"job": job, "status": "ok", "tta_censored": censored,
                         "summary_metrics": summary, "history": history,
                         "started_mono_ns": started_ns, "ended_mono_ns": time.monotonic_ns()}
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(raw_file, result_record)
        _write_full_reference(job, result_record)
        _write_fl_artifacts(job, raw_dir, result_record, trace_dir=trace_dir)
        return make_row(exp, scen, beta, n_clients, pol, seed, rounds, cross_mbps=cross,
                        status="ok", tta_censored=censored, **summary)
    except Exception as exc:
        failure = {"job": job, "status": "missing", "error": repr(exc),
                   "started_mono_ns": started_ns, "ended_mono_ns": time.monotonic_ns(),
                   "history": []}
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(raw_file, failure)
        _write_fl_artifacts(job, raw_dir, failure, trace_dir=trace_dir)
        return make_row(exp, scen, beta, n_clients, pol, seed, rounds, cross_mbps=cross, status="missing")
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", nargs="+", choices=EXPERIMENTS, default=None, help="Filter experiment IDs")
    parser.add_argument("--policy", nargs="+", choices=POLICIES, default=None, help="Filter policies")
    parser.add_argument("--scenario", nargs="+", choices=SCENARIOS, default=None, help="Filter scenarios")
    parser.add_argument("--machine-id", type=int, default=0, help="Machine index (0-based)")
    parser.add_argument("--num-machines", type=int, default=1, help="Number of machines")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker index on this machine (0-based)")
    parser.add_argument("--num-workers", type=int, default=1, help="Workers per machine")
    parser.add_argument("--dry-run", action="store_true", help="Print this deterministic partition without execution")
    parser.add_argument("--run", action="store_true", help="Execute this partition")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit this partition for controlled runs")
    parser.add_argument("--merge-csv", nargs="+", type=Path, default=None,
                        help="Merge per-machine runs.csv shards into --output-csv, rejecting duplicate run keys")
    parser.add_argument("--trace-dir", type=Path, default=Path("data/traces"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/summary/runs.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    args = parser.parse_args()
    if args.merge_csv:
        try:
            merge_summary_files(args.merge_csv, args.output_csv)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(f"Merged {len(args.merge_csv)} summary shard(s) into {args.output_csv}")
        return
    if args.run and args.dry_run:
        parser.error("--run and --dry-run are mutually exclusive")
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be positive")
    try:
        jobs = expand_grid(args.exp, policies=args.policy, scenarios=args.scenario)
        assigned = partition_jobs(jobs, args.machine_id, args.num_machines,
                                  worker_id=args.worker_id, num_workers=args.num_workers)
    except (GridError, ValueError) as exc:
        parser.error(str(exc))
    if args.max_runs is not None:
        assigned = assigned[:args.max_runs]
    print(f"Machine {args.machine_id + 1}/{args.num_machines}, worker {args.worker_id + 1}/{args.num_workers}")
    print(f"Summary: {args.output_csv}; run directories: {args.raw_dir}")
    _print_plan(assigned, args.trace_dir)
    if not args.run:
        print("Pass --run to execute available simulator and FL adapters. Blocked jobs will be recorded as missing with reasons.")
        return
    started = time.monotonic_ns()
    for index, job in enumerate(assigned, 1):
        print(f"[{index}/{len(assigned)}] {job['run_id']}", flush=True)
        row = execute_job(job, args.raw_dir, trace_dir=args.trace_dir)
        upsert_summary_row(row, args.output_csv)
    elapsed_s = (time.monotonic_ns() - started) / 1e9
    print(f"Finished {len(assigned)} assigned jobs in {elapsed_s:.2f}s; summary: {args.output_csv}")
if __name__ == "__main__":
    main()