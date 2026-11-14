"""Deterministic experiment-grid expansion, execution, resume, and summary.
The schema's ``expected_runs`` remains the canonical experiment matrix. This
runner executes grant-based and B2/B3 simulator configurations through the
Task 16 run-directory pipeline, and FEMNIST/FedAvg FL jobs (including F-main
B2/B3 through the FL server's push/plan transport adapters) through the
existing FL entry point. FL runs receive per-run metadata, run.csv,
fl_history.csv, and fl_result.json; network-only packet/queue logs are
deliberately not fabricated. Every network-only policy uses the frozen FULL
seed-0 value bank. A crashed run is retried once with the same configuration
and seed (``rerun``); a second crash is ``missing``. Jobs whose adapter or
inputs are unavailable are *blocked*: they are written to a separate tracking
file with reasons and never to runs.csv. M1/M2 remain blocked until production
Mininet (FL) integration exists. TTA uses Tc = 1.0 s unless overridden; any
override is recorded in the run metadata.
"""

from __future__ import annotations
import argparse
import csv
import datetime as _dt
import fcntl
import functools
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import shutil
import socket
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
import numpy as np
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from fedgrant.analysis import raw_metrics
from fedgrant.analysis.metrics import compute_final_accuracy, compute_target_accuracy, compute_time_to_target
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
from fedgrant.sim.value_bank import load_value_bank

FULL_REFERENCE_DIR = Path("data/runs")
# Where fedgrant.fl.run_fl.run_experiment always writes its FULL record.
FL_RECORD_DIR = Path("data/runs")
VALUE_BANK_PATH = Path("data/bank/femnist_full_s0.npz")
FEMNIST_SHARD_ROOT = Path("data/femnist/reproduced/clients")
DEFAULT_TC_S = 1.0  # Section 11 common default; overrides are recorded per run.
FULL_EVAL_ROUNDS = list(range(5, 301, 5))
WARMUP_ROUNDS = 20  # Section 12: rounds 1..20 are excluded for every policy.
NETWORK_ONLY_EXPERIMENTS = frozenset({"N-main", "N-X1", "N-X2", "N-abl", "M-sim"})
FL_GRID_EXPERIMENTS = frozenset({"F-main", "F-ref", "F-abl"})
MAX_ATTEMPTS = 2
BLOCKED_COLUMNS: tuple[str, ...] = (
    "run_id", "experiment_id", "config_id", "scenario", "beta", "N", "cross_mbps",
    "policy", "seed", "n_rounds", "availability", "reasons_json",
    "recorded_at_utc", "host", "pid",
)
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
def validate_job(job: Mapping[str, Any], *, allow_custom: bool = False) -> None:
    """Reject malformed or out-of-matrix jobs before allocating output paths."""
    required = (
        "experiment_id", "platform", "config_id", "scenario", "beta", "N", "n_clients",
        "cross_mbps", "policy", "seed", "n_rounds", "run_id",
    )
    missing = [key for key in required if key not in job]
    if missing:
        raise GridError(f"job missing required field(s): {missing}")
    key = (str(job["experiment_id"]), str(job["config_id"]), str(job["policy"]), int(job["seed"]))
    if not allow_custom and key not in _matrix_keys():
        raise GridError(f"job is not in the canonical experiment matrix: {key}")
    if allow_custom and (key[0] not in EXPERIMENTS or key[2] not in POLICIES):
        raise GridError(f"job has an unknown experiment or policy: {key}")
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
    required_rounds = _ROUND_COUNTS[key[0]]
    if job["n_rounds"] != required_rounds and not (
        allow_custom and 1 <= job["n_rounds"] < required_rounds
    ):
        raise GridError(f"{key[0]} requires {required_rounds} rounds, got {job['n_rounds']}")
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


def _normalize_execution_job(job: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Fill stable identity fields for the older direct execute_job input shape."""
    normalized = dict(job)
    allow_custom = "run_id" not in normalized
    if allow_custom:
        scenario, _beta, n_clients, _cross = _parse_config_id(str(normalized["config_id"]))
        normalized.setdefault(
            "platform", "mininet" if normalized["experiment_id"] in {"M1", "M2"} else "sim"
        )
        normalized.setdefault("n_clients", normalized.get("N", n_clients))
        normalized.setdefault("scenario", scenario)
        normalized["run_id"] = _run_id(
            str(normalized["experiment_id"]), str(normalized["config_id"]),
            str(normalized["policy"]), int(normalized["seed"]),
        )
    return normalized, allow_custom
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
def _fl_policy_name(job: Mapping[str, Any]) -> str:
    """Policy name passed to the FL entry point (A1 is FG-GRU without EF)."""
    if job["experiment_id"] == "F-ref":
        return "FULL"
    if job["experiment_id"] == "F-abl" and job["policy"] == "FG-GRU-A1":
        return "FG-GRU"
    return str(job["policy"])


def _fl_no_ef(job: Mapping[str, Any]) -> bool:
    return job["experiment_id"] == "F-abl" and job["policy"] == "FG-GRU-A1"


@functools.lru_cache(maxsize=None)
def _fl_adapter_reasons(fl_policy: str) -> tuple[str, ...]:
    """Check that the FL entry point and the policy's adapter actually exist."""
    if importlib.util.find_spec("torch") is None:
        return ("PyTorch is not installed; the FEMNIST/FedAvg FL entry point cannot run",)
    try:
        from fedgrant.fl import server as fl_server
        from fedgrant.fl.run_fl import fl_policy_settings
    except Exception as exc:  # pragma: no cover - depends on the environment
        return (f"FL entry point import failed: {type(exc).__name__}: {exc}",)
    if fl_policy == "FULL":
        return ()
    try:
        _config, transport = fl_policy_settings(fl_policy)
    except ValueError as exc:
        return (f"no FL policy adapter for {fl_policy!r}: {exc}",)
    if transport is not None:
        parameters = inspect.signature(fl_server.run_fl_simulated).parameters
        if "transport_baseline" not in parameters or not callable(getattr(fl_server, "run_baseline_round", None)):
            return (f"{transport} FL transport adapter is unavailable in fedgrant.fl.server",)
    return ()


def _femnist_reasons() -> list[str]:
    manifest = FEMNIST_SHARD_ROOT / "manifest.json"
    if not manifest.is_file():
        return [f"FEMNIST client shards are unavailable: {manifest} not found (run scripts/get_femnist.sh)"]
    return []


def _fl_job_reasons(job: Mapping[str, Any]) -> list[str]:
    experiment = str(job["experiment_id"])
    if experiment == "F-ref":
        try:
            _load_full_reference(int(job["seed"]))
            return []  # a validated frozen FULL history is reused; no training needed
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass  # rejection reason is recorded in the run metadata at execution
    reasons = list(_fl_adapter_reasons(_fl_policy_name(job)))
    reasons.extend(_femnist_reasons())
    if experiment != "F-ref":
        for seed in range(3):
            try:
                _load_full_reference(seed)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                reasons.append(f"FULL reference seed {seed} (needed for a*) is unavailable or invalid: {exc}")
    return reasons


def _value_bank_reasons(job: Mapping[str, Any], cache: dict[Any, list[str]] | None = None) -> list[str]:
    key = (str(VALUE_BANK_PATH), int(job["N"]))
    if cache is not None and key in cache:
        return list(cache[key])
    try:
        load_value_bank(VALUE_BANK_PATH, int(job["N"]))
        reasons: list[str] = []
    except (OSError, ValueError) as exc:
        reasons = [f"frozen FEMNIST value bank {VALUE_BANK_PATH} is unusable for N={job['N']}: {exc}"]
    if cache is not None:
        cache[key] = reasons
    return list(reasons)


def _adapter_block_reasons(
    job: Mapping[str, Any],
    trace_dir: Path,
    *,
    check_value_bank: bool = True,
    bank_cache: dict[Any, list[str]] | None = None,
) -> list[str]:
    """Reasons a job cannot run now; dry-run and dispatch both use this."""
    experiment, policy = str(job["experiment_id"]), str(job["policy"])
    reasons: list[str] = []
    if experiment == "M1":
        reasons.append("the Mininet launcher has a gateway-factory hook but no production client/baseline adapter; its client process uses synthetic updates")
    elif experiment == "M2":
        reasons.extend((
            "the Mininet launcher has no production client/baseline adapter and its client process uses synthetic updates",
            "the FL server/model-update/bitmap-aggregation lifecycle is not integrated with the Mininet run directory",
            "M2 requires production Mininet FL integration; the network-only Mininet runner and the simulator are not substitutes",
        ))
    elif experiment in FL_GRID_EXPERIMENTS:
        reasons.extend(_fl_job_reasons(job))
    elif policy not in _SIMULATOR_POLICY_CONFIGS and policy not in {"B2", "B3"}:
        reasons.append(f"simulator transfer-mode adapter for policy {policy!r} is unavailable")
    if experiment in NETWORK_ONLY_EXPERIMENTS and check_value_bank:
        reasons.extend(_value_bank_reasons(job, bank_cache))
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
    if job["experiment_id"] in NETWORK_ONLY_EXPERIMENTS:
        # Every network-only policy, including B2/B3, uses the frozen bank.
        if VALUE_BANK_PATH.is_file():
            candidates[f"value_bank:{VALUE_BANK_PATH.name}"] = VALUE_BANK_PATH
    if job["experiment_id"] in FL_GRID_EXPERIMENTS:
        manifest = FEMNIST_SHARD_ROOT / "manifest.json"
        if manifest.is_file():
            candidates["femnist:clients_manifest.json"] = manifest
        if job["experiment_id"] != "F-ref":
            for seed in range(3):
                reference = FULL_REFERENCE_DIR / f"femnist_full_s{seed}.json"
                if reference.is_file():
                    candidates[f"full_reference:{reference.name}"] = reference
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
def _config_fingerprint(
    job: Mapping[str, Any], input_hashes: Mapping[str, str], *, tc_s: float = DEFAULT_TC_S,
) -> str:
    fields = {key: job[key] for key in ("experiment_id", "platform", "config_id", "scenario", "beta", "N", "n_clients", "cross_mbps", "policy", "seed", "n_rounds")}
    fields["input_sha256"] = dict(sorted(input_hashes.items()))
    if float(tc_s) != DEFAULT_TC_S and job["experiment_id"] in FL_GRID_EXPERIMENTS - {"F-ref"}:
        # A Tc override changes TTA seconds, so it is part of the run identity.
        fields["tc_s"] = float(tc_s)
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
def _metadata_base(job: Mapping[str, Any], trace_dir: Path, *, tc_s: float = DEFAULT_TC_S) -> dict[str, Any]:
    input_hashes = _input_hashes(job, trace_dir)
    trace_hashes = {name.removeprefix("trace:"): digest for name, digest in input_hashes.items() if name.startswith("trace:")}
    return {
        "run_id": job["run_id"],
        "identity": _identity(job),
        "config_sha256": _config_fingerprint(job, input_hashes, tc_s=tc_s),
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
        "value_bank": (
            {
                "path": str(VALUE_BANK_PATH),
                "sha256": input_hashes.get(f"value_bank:{VALUE_BANK_PATH.name}"),
                "client_weights": "bank client_weights",
                "sampling": "per (round, client): SeedSequence([master_seed, round_id - 1, 1]) uniform profile index",
            }
            if job["experiment_id"] in NETWORK_ONLY_EXPERIMENTS else None
        ),
        "cpu_roles": None,
        "effective_qdisc_settings": None,
    }
def _row_for_missing(job: Mapping[str, Any]) -> dict[str, Any]:
    return make_row(
        str(job["experiment_id"]), str(job["scenario"]), float(job["beta"]), int(job["N"]),
        str(job["policy"]), int(job["seed"]), int(job["n_rounds"]),
        cross_mbps=float(job["cross_mbps"]), status="missing",
    )
def _round_probe_grants(result: Mapping[str, Any], metrics: Mapping[str, Any]) -> int | None:
    """Probe grants issued in this round by a grant policy (rule (v) bounded probe).

    Prefers the simulator's ``probe_grants`` metric; otherwise reads the round's
    GatewayCore counter (reset at ROUND_START, kept through the drain, so it holds
    this round's count when the round returns). None (blank) when neither exists.
    """
    if metrics.get("probe_grants") is not None:
        return int(metrics["probe_grants"])
    core = getattr(result.get("gateway"), "core", None)
    value = getattr(core, "round_probe_grants", None)
    return None if value is None else int(value)


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
    value_bank: Any | None = None,
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
    # Pass the real loss/trace scenario plus the configured cross-traffic rate.
    # (The legacy "X2" alias treated N-X2 L4 jobs as constant-capacity L2.)
    simulation_scenario = str(job["scenario"])
    cross_mbps = float(job["cross_mbps"])
    for round_id in range(1, int(job["n_rounds"]) + 1):
        # Identical bank inputs and sampling for grant policies and B2/B3.
        bank_inputs: dict[str, Any] = {}
        if value_bank is not None:
            bank_inputs = {
                "client_weights": value_bank.client_weights,
                "client_energies": value_bank.energies_for_run_round(int(job["seed"]), round_id),
            }
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
                cross_mbps=cross_mbps,
                **bank_inputs,
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
                cross_mbps=cross_mbps,
                **bank_inputs,
            )
        gateway = result["gateway"]
        metrics = result["metrics"]
        late_metrics = result.get("late_metrics", {})
        baseline_rtts = result.get("rtt_samples_ns", {})
        if is_baseline:
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
            delivered_blocks=float(metrics.get("delivered_blocks", sum(result.get("delivered_counts", ())))),
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
            probe_grants=None if is_baseline else _round_probe_grants(result, metrics),
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
def _load_metadata(
    path: Path, job: Mapping[str, Any], trace_dir: Path, *, tc_s: float = DEFAULT_TC_S,
) -> dict[str, Any] | None:
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
    if metadata.get("execution_state") == "blocked":
        # A blocked record holds no results; availability is re-evaluated and
        # inputs that have since appeared legitimately change the fingerprint.
        return None
    if metadata.get("config_sha256") != _config_fingerprint(job, _input_hashes(job, trace_dir), tc_s=tc_s):
        raise GridError(f"existing run configuration differs from requested job at {path.parent}")
    return metadata


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _blocked_record(job: Mapping[str, Any], reasons: list[str]) -> dict[str, Any]:
    """Machine-readable blocked/unavailable record; never a runs.csv row."""
    return {
        "run_id": job["run_id"],
        "experiment_id": job["experiment_id"],
        "config_id": job["config_id"],
        "scenario": job["scenario"],
        "beta": job["beta"],
        "N": job["N"],
        "cross_mbps": job["cross_mbps"],
        "policy": job["policy"],
        "seed": job["seed"],
        "n_rounds": job["n_rounds"],
        "availability": "blocked",
        "reasons_json": json.dumps(list(reasons), separators=(",", ":")),
        "recorded_at_utc": _utc_now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        # Not a runs.csv status: callers route this to the blocked tracker.
        "status": "blocked",
        "block_reasons": list(reasons),
    }


def _record_blocked_run(
    job: Mapping[str, Any], raw_root: Path, trace_dir: Path, reasons: list[str], *, tc_s: float,
) -> dict[str, Any]:
    """Write blocked metadata (never results, never ``missing``) and return the record."""
    record = _blocked_record(job, reasons)
    run_dir = raw_root / str(job["experiment_id"]) / str(job["run_id"])
    metadata_path = run_dir / "meta.json"
    with _exclusive_lock(raw_root / ".locks" / f"{job['run_id']}.lock"):
        existing_state = None
        if metadata_path.is_file():
            try:
                existing_state = json.loads(metadata_path.read_text(encoding="utf-8")).get("execution_state")
            except (OSError, json.JSONDecodeError, AttributeError):
                existing_state = "unreadable"
        if existing_state in (None, "blocked"):
            # Do not overwrite metadata of earlier attempts; the tracker still
            # reports that the job cannot run with the current inputs.
            _atomic_json(metadata_path, {
                **_metadata_base(job, trace_dir, tc_s=tc_s),
                "status": "blocked",
                "execution_state": "blocked",
                "availability": "blocked",
                "block_reasons": list(reasons),
                "checked_at_utc": record["recorded_at_utc"],
                "attempt_count": 0,
                "attempts": [],
            })
    return record


def execute_job(
    job: dict[str, Any],
    raw_dir: str | Path,
    *,
    trace_dir: str | Path = "data/traces",
    simulation_runner: Callable[..., Mapping[str, Any]] | None = None,
    baseline_runner: Callable[..., Mapping[str, Any]] | None = None,
    tc_s: float = DEFAULT_TC_S,
) -> dict[str, Any]:
    """Run/resume one schema job under a per-run lock.

    Returns a runs.csv row (status ok/rerun/missing), or, when the adapter or
    an input is unavailable, a blocked record with ``status == "blocked"``
    that callers must send to the blocked-job tracker instead of runs.csv.
    Injected runners are a test seam and skip the frozen-bank requirement.
    """
    job, allow_custom = _normalize_execution_job(job)
    validate_job(job, allow_custom=allow_custom)
    if isinstance(tc_s, bool) or not isinstance(tc_s, Real) or not math.isfinite(float(tc_s)) or float(tc_s) <= 0:
        raise GridError("tc_s must be a finite positive number of seconds")
    tc_s = float(tc_s)
    injected_runner = simulation_runner is not None or baseline_runner is not None
    simulation_runner = run_network_round if simulation_runner is None else simulation_runner
    baseline_runner = run_baseline_network_round if baseline_runner is None else baseline_runner
    raw_root, traces = Path(raw_dir), Path(trace_dir)
    reasons = _adapter_block_reasons(job, traces, check_value_bank=not injected_runner)
    if reasons:
        return _record_blocked_run(job, raw_root, traces, reasons, tc_s=tc_s)
    if str(job["experiment_id"]) in FL_GRID_EXPERIMENTS:
        # FL writes its own artifacts (no fabricated network CSVs). The same
        # per-run lock prevents two workers from launching one FL job together.
        fl_lock = raw_root / ".locks" / f"{job['run_id']}.lock"
        with _exclusive_lock(fl_lock):
            return _execute_fl_job(job, raw_root, trace_dir=traces, tc_s=tc_s)
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
        base = _metadata_base(job, traces)
        attempts = list(existing.get("attempts", [])) if existing else []
        attempt_count = int(existing.get("attempt_count", 0)) if existing else 0
        state = existing.get("execution_state") if existing else None
        if state == "missing" and attempt_count >= 2:
            return dict(existing.get("summary_row", _row_for_missing(job)))
        if state == "running":
            attempts.append({"attempt": attempt_count, "state": "interrupted", "error": "previous process ended while running"})
        while attempt_count < MAX_ATTEMPTS:
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
                value_bank = None
                if not injected_runner:
                    # Same frozen bank, weights, and sampling for every
                    # network-only policy, B2/B3 included.
                    value_bank = load_value_bank(VALUE_BANK_PATH, int(job["N"]))
                _write_simulator_run(
                    job,
                    stage,
                    trace_dir=traces,
                    simulation_runner=simulation_runner,
                    baseline_runner=baseline_runner,
                    value_bank=value_bank,
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
def _read_blocked(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != BLOCKED_COLUMNS:
            raise GridError(f"{path}: unexpected blocked-job columns {reader.fieldnames}")
        return list(reader)


def _write_blocked(rows: list[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=BLOCKED_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: str(row["run_id"])))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def record_blocked_job(record: Mapping[str, Any], blocked_csv: str | Path) -> None:
    """Upsert one blocked/unavailable job (keyed by run_id) in the tracker."""
    if record.get("availability") != "blocked":
        raise GridError("only blocked records can be written to the blocked-job tracker")
    path = Path(blocked_csv)
    with _exclusive_lock(path.with_name(path.name + ".lock")):
        rows = [row for row in _read_blocked(path) if row["run_id"] != record["run_id"]]
        rows.append({column: record[column] for column in BLOCKED_COLUMNS})
        _write_blocked(rows, path)


def clear_blocked_job(run_id: str, blocked_csv: str | Path) -> None:
    """Remove a job from the tracker once it has produced a runs.csv row."""
    path = Path(blocked_csv)
    if not path.is_file():
        return
    with _exclusive_lock(path.with_name(path.name + ".lock")):
        rows = _read_blocked(path)
        kept = [row for row in rows if row["run_id"] != run_id]
        if len(kept) != len(rows):
            _write_blocked(kept, path)


def plan_availability(jobs: list[dict[str, Any]], trace_dir: Path) -> list[tuple[dict[str, Any], list[str]]]:
    """Availability exactly as ``execute_job`` will evaluate it with default runners."""
    bank_cache: dict[Any, list[str]] = {}
    return [(job, _adapter_block_reasons(job, trace_dir, bank_cache=bank_cache)) for job in jobs]


def _print_plan(jobs: list[dict[str, Any]], trace_dir: Path) -> None:
    planned = plan_availability(jobs, trace_dir)
    blocked = [(job, reasons) for job, reasons in planned if reasons]
    available = len(planned) - len(blocked)
    print(f"Expanded jobs: {len(jobs)} ({available} executable by an available adapter; {len(blocked)} blocked/unavailable)")
    by_reason: dict[str, int] = {}
    for _job, reasons in blocked:
        for reason in reasons:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    for reason, count in sorted(by_reason.items(), key=lambda item: (-item[1], item[0])):
        print(f"  blocked x{count}: {reason}")
    for job, reasons in planned[:8]:
        label = "available" if not reasons else "blocked: " + "; ".join(reasons)
        print(f"  {job['run_id']} | {job['n_rounds']} rounds | {label}")
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_full_record(record: Any, seed: int, path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate a frozen FULL record; return its evaluations and what was checked.

    Checks the seed, the FULL/no-network policy with error feedback, the
    FEMNIST client count and verified dataset provenance (cross-checked
    against the local provenance file when present), the 300-round history,
    and the 5-round evaluation schedule 5..300.
    """
    if not isinstance(record, Mapping):
        raise ValueError(f"FULL reference {path} is not a JSON object")
    if "seed" not in record or int(record["seed"]) != seed:
        raise ValueError(f"FULL reference seed mismatch in {path}: expected {seed}, found {record.get('seed')!r}")
    if int(record.get("rounds", -1)) != 300:
        raise ValueError(f"FULL reference {path} does not record a 300-round run (rounds={record.get('rounds')!r})")
    policy = str(record.get("policy", ""))
    if policy.lower() != "full":
        raise ValueError(f"FULL reference {path} was produced by policy {policy!r}, not FULL")
    if record.get("error_feedback", True) is not True:
        raise ValueError(f"FULL reference {path} was produced without error feedback")
    if "num_clients" in record and int(record["num_clients"]) != 20:
        raise ValueError(f"FULL reference {path} has {record['num_clients']} FL clients, expected 20")
    dataset = record.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError(f"FULL reference {path} has no dataset provenance metadata")
    if not dataset.get("source") or dataset.get("source") == "unverified" or not dataset.get("provenance_sha256"):
        raise ValueError(f"FULL reference {path} has unverified FEMNIST provenance")
    local_provenance = FEMNIST_SHARD_ROOT.parent / "provenance.json"
    provenance_check = "local provenance file unavailable"
    if local_provenance.is_file():
        if _sha256_file(local_provenance) != str(dataset["provenance_sha256"]):
            raise ValueError(f"FULL reference {path} provenance does not match {local_provenance}")
        provenance_check = "matches local provenance file"
    history = record.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"FULL reference history is empty or invalid: {path}")
    if max((int(row["round"]) for row in history if "round" in row), default=0) != 300:
        raise ValueError(f"FULL reference {path} history does not reach round 300")
    evaluations = sorted(
        (row for row in history if "test_accuracy" in row and "round" in row),
        key=lambda row: int(row["round"]),
    )
    if [int(row["round"]) for row in evaluations] != FULL_EVAL_ROUNDS:
        raise ValueError(f"FULL reference history in {path} must evaluate exactly every 5 rounds from 5 through 300")
    for row in evaluations:
        accuracy = float(row["test_accuracy"])
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"FULL reference {path} has invalid test_accuracy at round {row['round']}")
    checks = {
        "seed": seed,
        "rounds": 300,
        "policy": policy,
        "error_feedback": record.get("error_feedback", "not recorded"),
        "num_clients": record.get("num_clients", "not recorded"),
        "dataset_source": dataset.get("source"),
        "dataset_provenance_sha256": dataset.get("provenance_sha256"),
        "provenance_check": provenance_check,
        "evaluation_rounds": "5..300 step 5",
        "model_architecture": "not recorded in FULL records; not verifiable",
    }
    return evaluations, checks


def _load_full_reference(
    seed: int, reference_dir: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], Path]:
    root = FULL_REFERENCE_DIR if reference_dir is None else Path(reference_dir)
    path = root / f"femnist_full_s{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(f"FULL reference record is missing: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    evaluations, checks = _validate_full_record(record, seed, path)
    return record, evaluations, checks, path


def _load_full_history(seed: int, reference_dir: Path | None = None) -> list[dict[str, Any]]:
    """Validated FULL evaluations (rounds 5..300) for one seed."""
    return _load_full_reference(seed, reference_dir)[1]


def _full_target_accuracy(reference_dir: Path | None = None) -> float:
    finals = [
        compute_final_accuracy(
            [float(row["test_accuracy"]) for row in _load_full_history(seed, reference_dir)]
        )
        for seed in range(3)
    ]
    return compute_target_accuracy(finals)


def _post_warmup(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rounds if int(row.get("round_id", row.get("round", 0))) > WARMUP_ROUNDS]


def _pooled_percentiles(rounds: list[dict[str, Any]], key: str) -> tuple[float | None, float | None]:
    """Pool individual samples from rounds 21..last, then take percentiles (ms)."""
    samples = [float(sample) for row in _post_warmup(rounds) for sample in row.get(key) or ()]
    if not samples:
        return None, None
    return float(np.percentile(samples, 50) / 1e6), float(np.percentile(samples, 99) / 1e6)


def _pooled_rtt_percentiles(rounds: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """Grant-resolution RTT percentiles from pooled post-warm-up samples."""
    return _pooled_percentiles(rounds, "rtt_samples_ns")


def _mean_present(rows: list[dict[str, Any]], *keys: str) -> float | None:
    """Mean of the first present, finite key per row; None when never measured."""
    values = []
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value is not None and not isinstance(value, bool) and math.isfinite(float(value)):
                values.append(float(value))
                break
    return float(np.mean(values)) if values else None


def _network_summary(rounds: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    """Summarize post-warm-up (rounds 21..last) network metrics and pooled RTTs."""
    eligible = _post_warmup(rounds)
    if not eligible:
        return {}
    summary: dict[str, Any] = {
        "eta": float(np.mean([row["eta"] for row in eligible])),
        "utilization": float(np.mean([row.get("utilization", 0.0) for row in eligible])),
        "goodput_bps": float(np.mean([row.get("goodput_bps", 0.0) for row in eligible])),
        "queue_drops_fg": float(np.mean([row.get("drops_queue_bn_client", 0.0) for row in eligible])),
        "queue_drops_cross": float(np.mean([row.get("drops_queue_bn_cross", 0.0) for row in eligible])),
        "delivered_blocks": float(np.mean([
            row.get("delivered_blocks", row.get("grants_delivered", 0.0)) for row in eligible
        ])),
        "clients_represented": float(np.mean([row.get("clients_represented", 0.0) for row in eligible])),
        "late_after_deadline": float(np.mean([row.get("late_after_deadline", 0.0) for row in eligible])),
    }
    if policy not in {"B2", "B3"}:
        summary["wasted_grants"] = float(np.mean([row.get("wasted_grants_ratio", 0.0) for row in eligible]))
        summary["deadline_completion"] = float(np.mean([row.get("deadline_completion", 0.0) for row in eligible]))
        summary["late_after_timeout"] = float(np.mean([row.get("late_after_timeout", 0.0) for row in eligible]))
        summary["rtt_p50_ms"], summary["rtt_p99_ms"] = _pooled_rtt_percentiles(rounds)
        return summary
    summary["data_packets_sent"] = _mean_present(eligible, "data_packets_sent")
    summary["data_bytes_sent"] = _mean_present(eligible, "data_bytes_sent")
    if policy == "B3":
        summary["planned_blocks"] = _mean_present(eligible, "planned_blocks", "blocks_planned")
        summary["plan_messages_sent"] = _mean_present(eligible, "plan_messages_sent", "plans_issued")
        # PLAN->first-DATA RTT is never reported as grant RTT (rtt_p50/p99_ms).
        measured = [row for row in eligible if row.get("plan_data_rtt_samples_ns") is not None]
        if measured:
            summary["plan_data_rtt_samples"] = float(np.mean([
                len(row["plan_data_rtt_samples_ns"]) for row in measured
            ]))
            summary["plan_data_rtt_p50_ms"], summary["plan_data_rtt_p99_ms"] = _pooled_percentiles(
                measured, "plan_data_rtt_samples_ns"
            )
    return summary


def _fl_round_metrics(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rounds = []
    for record in history:
        metrics = dict(record.get("metrics", {}))
        late = record.get("late_metrics", {})
        row = {
            "round_id": int(record["round"]),
            "eta": float(record["eta"]),
            **metrics,
            "late_after_deadline": late.get("late_after_deadline", 0.0),
            "rtt_samples_ns": record.get("rtt_samples_ns", []),
        }
        if "plan_data_rtt_samples_ns" in record:
            row["plan_data_rtt_samples_ns"] = list(record["plan_data_rtt_samples_ns"])
        rounds.append(row)
    return rounds


def _tta_config(job: Mapping[str, Any], tc_s: float) -> dict[str, Any] | None:
    if job["experiment_id"] == "F-ref":
        return None
    d_s = compute_deadline_ns(float(job["beta"]), int(job["N"])) / 1e9
    return {
        "tc_s": float(tc_s),
        "tc_default_s": DEFAULT_TC_S,
        "tc_overridden": float(tc_s) != DEFAULT_TC_S,
        "d_s": d_s,
        "formula": "tta_seconds = tta_rounds * (Tc + D)",
        "target": "a* = 0.9 * mean final accuracy of FULL seeds 0-2",
    }


def _fl_evaluation_metrics(
    job: Mapping[str, Any], history: list[dict[str, Any]], *,
    reference_dir: Path | None = None,
    tc_s: float = DEFAULT_TC_S,
) -> dict[str, Any]:
    evaluations = sorted(
        (row for row in history if "test_accuracy" in row and "round" in row),
        key=lambda row: int(row["round"]),
    )
    if not evaluations:
        raise ValueError("FL run returned no test-accuracy evaluations")
    if int(evaluations[-1]["round"]) != int(job["n_rounds"]):
        raise ValueError("FL evaluation history does not reach the configured final round")
    if int(job["n_rounds"]) == 300 and [
        int(row["round"]) for row in evaluations[-10:]
    ] != list(range(255, 301, 5)):
        raise ValueError("300-round FL history must evaluate every 5 rounds through 300")
    target = _full_target_accuracy(reference_dir)
    final_acc = compute_final_accuracy([float(row["test_accuracy"]) for row in evaluations])
    tta_round, tta_seconds = compute_time_to_target(
        evaluations,
        target,
        max_round=int(job["n_rounds"]),
        t_c_s=float(tc_s),
        d_s=compute_deadline_ns(float(job["beta"]), int(job["N"])) / 1e9,
    )
    result: dict[str, Any] = {"final_acc": final_acc, "tta_censored": tta_round is None}
    if tta_round is not None:
        result["tta_rounds"] = float(tta_round)
        result["tta_seconds"] = float(tta_seconds)
    return result


def _fl_row(job: Mapping[str, Any], status: str, summary: Mapping[str, Any] | None = None,
            censored: bool | None = None) -> dict[str, Any]:
    return make_row(
        str(job["experiment_id"]), str(job["scenario"]), float(job["beta"]), int(job["N"]),
        str(job["policy"]), int(job["seed"]), int(job["n_rounds"]),
        cross_mbps=float(job["cross_mbps"]), status=status, tta_censored=censored,
        **dict(summary or {}),
    )


def _full_reference_row(job: Mapping[str, Any], reference_dir: Path | None = None) -> dict[str, Any]:
    """Summarize a validated F-ref FULL record; F-ref has no TTA fields."""
    history = _load_full_history(int(job["seed"]), reference_dir)
    final_acc = compute_final_accuracy([float(row["test_accuracy"]) for row in history])
    return _fl_row(job, "ok", {"final_acc": final_acc})


_FL_ARTIFACTS = ("run.csv", "fl_history.csv", "fl_result.json")


def _legacy_fl_raw_file(job: Mapping[str, Any], raw_dir: Path) -> Path:
    return raw_dir / f"{job['experiment_id']}_{job['config_id']}_{job['policy']}_s{job['seed']}.json"


def _write_fl_run_files(
    job: Mapping[str, Any], run_dir: Path, metadata: Mapping[str, Any], result: Mapping[str, Any],
) -> None:
    """Write run.csv, fl_history.csv and fl_result.json, then commit meta.json.

    FL runs have model/evaluation history, not gateway packet/queue samples,
    so no network CSVs are fabricated.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    write_rows(run_dir / "run.csv", "run", [{
        "run_id": job["run_id"],
        "experiment_id": job["experiment_id"],
        "config_id": job["config_id"],
        "policy": job["policy"],
        "seed": job["seed"],
        "scenario": job["scenario"],
        "beta": job["beta"],
        "n_clients": job["N"],
        "cross_mbps": job["cross_mbps"],
        "status": metadata["status"],
        "started_mono_ns": metadata.get("started_mono_ns") or "",
        "ended_mono_ns": metadata.get("ended_mono_ns") or "",
    }])
    history = list(result.get("history", []))
    columns = list(dict.fromkeys(key for row in history for key in row))
    with (run_dir / "fl_history.csv").open("w", newline="", encoding="utf-8") as stream:
        if columns:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(history)
        else:
            stream.write("\n")
    _atomic_json(run_dir / "fl_result.json", result)
    _atomic_json(run_dir / "meta.json", metadata)


def _fl_artifacts_valid(run_dir: Path, metadata: Mapping[str, Any]) -> bool:
    required = list(_FL_ARTIFACTS)
    reuse = metadata.get("reference_reuse") or {}
    if reuse.get("status") == "reused":
        required.append("full_reference.json")
    if not all((run_dir / name).is_file() for name in required):
        return False
    try:
        result = json.loads((run_dir / "fl_result.json").read_text(encoding="utf-8"))
        row = metadata["summary_row"]
        parse_row(row)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return result.get("status") == row.get("status") and row.get("status") in {"ok", "rerun"}


def _try_reuse_full_reference(
    job: Mapping[str, Any], run_dir: Path, base: Mapping[str, Any], attempts: list[dict[str, Any]],
    attempt_count: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Reuse a validated frozen FULL history as this F-ref run, or record why not."""
    seed = int(job["seed"])
    source = FULL_REFERENCE_DIR / f"femnist_full_s{seed}.json"
    checked_at = _utc_now()
    try:
        record, evaluations, checks, source = _load_full_reference(seed)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        rejection = {
            "status": "rejected" if source.is_file() else "unavailable",
            "source": str(source),
            "reason": f"{type(exc).__name__}: {exc}",
            "checked_at_utc": checked_at,
        }
        if source.is_file():
            # Keep the rejected file for audit before retraining replaces it.
            archived = run_dir / "rejected_full_reference.json"
            shutil.copyfile(source, archived)
            rejection["source_sha256"] = _sha256_file(source)
            rejection["archived_copy"] = archived.name
        return None, rejection
    copied = run_dir / "full_reference.json"
    shutil.copyfile(source, copied)
    source_sha = _sha256_file(source)
    if _sha256_file(copied) != source_sha:
        raise GridError(f"copy of FULL reference {source} into {copied} is not byte-identical")
    final_acc = compute_final_accuracy([float(row["test_accuracy"]) for row in evaluations])
    row = _fl_row(job, "ok", {"final_acc": final_acc})
    reuse = {
        "status": "reused",
        "source": str(source),
        "source_sha256": source_sha,
        "copied_to": copied.name,
        "checked_at_utc": checked_at,
        "validated": checks,
    }
    now = time.monotonic_ns()
    result = {
        "job": dict(job),
        "status": "ok",
        "tta_censored": None,
        "summary_metrics": {"final_acc": final_acc},
        "history": list(record["history"]),
        "rounds": [],
        "reference_reuse": reuse,
    }
    metadata = {
        **base,
        "status": "ok",
        "execution_state": "complete",
        "availability": "available",
        "block_reasons": [],
        "artifacts": ["meta.json", *_FL_ARTIFACTS, "full_reference.json"],
        "started_mono_ns": now,
        "ended_mono_ns": now,
        "attempt_count": attempt_count,
        "attempts": [*attempts, {"attempt": attempt_count, "state": "reused_reference", "checked_at_utc": checked_at}],
        "n_rounds_completed": 300,
        "reference_reuse": reuse,
        "summary_row": row,
    }
    _write_fl_run_files(job, run_dir, metadata, result)
    return row, reuse


def _publish_full_reference(job: Mapping[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    """Make the trained FULL record available where F-main/F-abl read it, validated."""
    name = f"femnist_full_s{int(job['seed'])}.json"
    produced, target = FL_RECORD_DIR / name, FULL_REFERENCE_DIR / name
    if produced.is_file() and produced.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=target.parent)
        os.close(fd)
        try:
            shutil.copyfile(produced, temp_name)
            os.replace(temp_name, target)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise
    record, _evaluations, checks, path = _load_full_reference(int(job["seed"]))
    if record["history"] != history:
        raise ValueError(f"published FULL reference {path} does not match the completed run history")
    return {"path": str(path), "sha256": _sha256_file(path), "validated": checks}


def _fl_attempt(job: Mapping[str, Any], trace_dir: Path, tc_s: float) -> dict[str, Any]:
    """One FL attempt with the job's exact configuration and seed."""
    from fedgrant.fl.run_fl import run_experiment  # lazy: simulator runs need no PyTorch

    experiment = str(job["experiment_id"])
    call = {
        "seed": int(job["seed"]),
        "rounds": int(job["n_rounds"]),
        "policy": _fl_policy_name(job),
        "beta": float(job["beta"]),
        "scenario": str(job["scenario"]),
        "no_ef": _fl_no_ef(job),
        "trace_dir": str(trace_dir),
    }
    if experiment == "F-ref" and int(job["seed"]) == 0 and not VALUE_BANK_PATH.is_file():
        # Only F-ref seed 0 creates the bank, and never over a frozen one.
        call["bank_path"] = str(VALUE_BANK_PATH)
    if experiment != "F-ref":
        _full_target_accuracy()  # fail before training if a* cannot be computed
    history = run_experiment(**call)
    if not isinstance(history, list):
        raise TypeError("run_experiment must return the per-round history list")
    published = None
    if experiment == "F-ref":
        published = _publish_full_reference(job, history)
        round_metrics: list[dict[str, Any]] = []
        summary: dict[str, Any] = {
            "final_acc": compute_final_accuracy(
                [float(row["test_accuracy"]) for row in history if "test_accuracy" in row]
            )
        }
        censored = None
    else:
        round_metrics = _fl_round_metrics(history)
        summary = _network_summary(round_metrics, str(job["policy"]))
        fl_summary = _fl_evaluation_metrics(job, history, tc_s=tc_s)
        censored = fl_summary.pop("tta_censored")
        summary.update(fl_summary)
    return {
        "call": call,
        "history": history,
        "rounds": round_metrics,
        "summary": summary,
        "censored": censored,
        "published_reference": published,
    }


def _execute_fl_job(
    job: dict[str, Any], raw_dir: Path, *, trace_dir: Path, tc_s: float = DEFAULT_TC_S,
) -> dict[str, Any]:
    """Run/resume an F-main/F-ref/F-abl job through the FEMNIST/FedAvg entry point.

    A crash is retried once with the same configuration and seed; success on
    the retry is ``rerun`` and a second crash is ``missing``. Every attempt is
    logged under ``attempts/`` with its failure reason.
    """
    experiment = str(job["experiment_id"])
    run_dir = raw_dir / experiment / str(job["run_id"])
    metadata_path = run_dir / "meta.json"
    if run_dir.exists() and not metadata_path.exists() and any(run_dir.iterdir()):
        raise GridError(f"refusing to reuse non-empty run directory without metadata: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = _load_metadata(metadata_path, job, trace_dir, tc_s=tc_s)
    base = {
        **_metadata_base(job, trace_dir, tc_s=tc_s),
        "artifact_type": "federated_learning",
        "fl_policy": _fl_policy_name(job),
        "error_feedback": not _fl_no_ef(job),
        "tta": _tta_config(job, tc_s),
    }
    if existing is not None and existing.get("execution_state") == "complete":
        if _fl_artifacts_valid(run_dir, existing):
            return dict(existing["summary_row"])
        existing["execution_state"] = "incomplete"
        existing["status"] = "missing"
        existing.setdefault("attempts", []).append({
            "attempt": existing.get("attempt_count", 0),
            "state": "incomplete",
            "error": "completed metadata exists but FL artifacts are incomplete or invalid",
        })
        _atomic_json(metadata_path, existing)
    attempts = list(existing.get("attempts", [])) if existing else []
    attempt_count = int(existing.get("attempt_count", 0)) if existing else 0
    state = existing.get("execution_state") if existing else None
    if state == "missing" and attempt_count >= MAX_ATTEMPTS:
        return dict(existing.get("summary_row", _row_for_missing(job)))
    if state == "running":
        attempts.append({"attempt": attempt_count, "state": "interrupted", "error": "previous process ended while running"})

    reference_reuse = existing.get("reference_reuse") if existing else None
    if experiment == "F-ref":
        row, reference_reuse = _try_reuse_full_reference(job, run_dir, base, attempts, attempt_count)
        if row is not None:
            return row

    raw_file = _legacy_fl_raw_file(job, raw_dir)
    started_first = existing.get("started_mono_ns") if existing else None
    while attempt_count < MAX_ATTEMPTS:
        attempt_count += 1
        started_ns = time.monotonic_ns()
        current = {
            **base,
            "status": "missing",
            "execution_state": "running",
            "availability": "available",
            "block_reasons": [],
            "started_mono_ns": started_first or started_ns,
            "ended_mono_ns": None,
            "attempt_count": attempt_count,
            "attempts": attempts,
            "active_attempt": attempt_count,
            "reference_reuse": reference_reuse,
        }
        _atomic_json(metadata_path, current)
        attempt_log = run_dir / "attempts" / f"attempt-{attempt_count:02d}.json"
        try:
            outcome = _fl_attempt(job, trace_dir, tc_s)
            status = "ok" if attempt_count == 1 else "rerun"
            row = _fl_row(job, status, outcome["summary"], outcome["censored"])
            ended_ns = time.monotonic_ns()
            _atomic_json(attempt_log, {
                "attempt": attempt_count, "state": "complete" if status == "ok" else "rerun",
                "started_mono_ns": started_ns, "ended_mono_ns": ended_ns,
                "finished_at_utc": _utc_now(), "call": outcome["call"],
            })
            attempts.append({
                "attempt": attempt_count, "state": "complete" if status == "ok" else "rerun",
                "started_mono_ns": started_ns, "ended_mono_ns": ended_ns, "log": attempt_log.name,
            })
            result = {
                "job": dict(job),
                "status": status,
                "tta_censored": outcome["censored"],
                "summary_metrics": outcome["summary"],
                "history": outcome["history"],
                "rounds": outcome["rounds"],
                "started_mono_ns": started_ns,
                "ended_mono_ns": ended_ns,
                "call": outcome["call"],
                "tta": base["tta"],
                "published_reference": outcome["published_reference"],
            }
            raw_file.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(raw_file, result)  # legacy flat record, kept for compatibility
            current.update({
                "status": status,
                "execution_state": "complete",
                "ended_mono_ns": ended_ns,
                "attempts": attempts,
                "artifacts": ["meta.json", *_FL_ARTIFACTS],
                "n_rounds_completed": len(outcome["history"]),
                "published_reference": outcome["published_reference"],
                "summary_row": row,
            })
            current.pop("active_attempt", None)
            _write_fl_run_files(job, run_dir, current, result)
            return row
        except KeyboardInterrupt:
            ended_ns = time.monotonic_ns()
            attempts.append({"attempt": attempt_count, "state": "interrupted", "started_mono_ns": started_ns, "ended_mono_ns": ended_ns})
            current.update({"execution_state": "interrupted", "ended_mono_ns": ended_ns, "attempts": attempts})
            _atomic_json(metadata_path, current)
            raise
        except Exception as exc:
            ended_ns = time.monotonic_ns()
            error = f"{type(exc).__name__}: {exc}"
            _atomic_json(attempt_log, {
                "attempt": attempt_count, "state": "failed",
                "started_mono_ns": started_ns, "ended_mono_ns": ended_ns,
                "failed_at_utc": _utc_now(), "error": error,
                "traceback": traceback.format_exc(),
            })
            attempts.append({
                "attempt": attempt_count, "state": "failed", "started_mono_ns": started_ns,
                "ended_mono_ns": ended_ns, "error": error, "log": attempt_log.name,
            })
            current.update({"execution_state": "failed", "ended_mono_ns": ended_ns, "attempts": attempts})
            _atomic_json(metadata_path, current)

    row = _row_for_missing(job)
    ended_ns = time.monotonic_ns()
    missing = {
        **base,
        "status": "missing",
        "execution_state": "missing",
        "availability": "available",
        "block_reasons": [],
        "started_mono_ns": started_first or ended_ns,
        "ended_mono_ns": ended_ns,
        "attempt_count": attempt_count,
        "attempts": attempts,
        "reference_reuse": reference_reuse,
        "artifacts": ["meta.json", *_FL_ARTIFACTS],
        "summary_row": row,
    }
    failure = {
        "job": dict(job),
        "status": "missing",
        "errors": [attempt.get("error") for attempt in attempts if attempt.get("state") == "failed"],
        "history": [],
    }
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(raw_file, failure)
    _write_fl_run_files(job, run_dir, missing, failure)
    return row


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
    parser.add_argument("--blocked-csv", type=Path, default=None,
                        help="Blocked/unavailable job tracker (default: blocked_jobs.csv beside --output-csv)")
    parser.add_argument("--tc-seconds", type=float, default=DEFAULT_TC_S,
                        help="Local-training time Tc for TTA seconds = rounds * (Tc + D); "
                             "default 1.0 s per the specification; overrides are recorded per run")
    args = parser.parse_args()
    if not math.isfinite(args.tc_seconds) or args.tc_seconds <= 0:
        parser.error("--tc-seconds must be a finite positive number")
    blocked_csv = args.blocked_csv or args.output_csv.with_name("blocked_jobs.csv")
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
    print(f"Summary: {args.output_csv}; blocked tracker: {blocked_csv}; run directories: {args.raw_dir}")
    if args.tc_seconds != DEFAULT_TC_S:
        print(f"Tc override: {args.tc_seconds} s (default {DEFAULT_TC_S} s); recorded in each FL run's meta.json")
    _print_plan(assigned, args.trace_dir)
    if not args.run:
        print("Pass --run to execute available simulator and FL adapters. Blocked jobs are written to the "
              "blocked tracker with reasons, never to runs.csv.")
        return
    started = time.monotonic_ns()
    counts = {"ok": 0, "rerun": 0, "missing": 0, "blocked": 0}
    for index, job in enumerate(assigned, 1):
        print(f"[{index}/{len(assigned)}] {job['run_id']}", flush=True)
        row = execute_job(job, args.raw_dir, trace_dir=args.trace_dir, tc_s=args.tc_seconds)
        if row.get("status") == "blocked":
            record_blocked_job(row, blocked_csv)
            print(f"  blocked: {'; '.join(row['block_reasons'])}", flush=True)
        else:
            upsert_summary_row(row, args.output_csv)
            clear_blocked_job(str(job["run_id"]), blocked_csv)
        counts[str(row["status"])] += 1
    elapsed_s = (time.monotonic_ns() - started) / 1e9
    print(f"Finished {len(assigned)} assigned jobs in {elapsed_s:.2f}s: "
          f"{counts['ok']} ok, {counts['rerun']} rerun, {counts['missing']} missing, "
          f"{counts['blocked']} blocked (not completed); summary: {args.output_csv}; blocked: {blocked_csv}")
if __name__ == "__main__":
    main()
