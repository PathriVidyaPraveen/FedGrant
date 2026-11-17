"""Generate the four ML datasets from the discrete-event network simulator.

Run with ``python -m fedgrant.ml.datagen``. By default this processes every
seed in each split's assigned range and writes bounded-size NPZ shards beneath
``data/mldata/{train,val,test_id,test_ood}/``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from fedgrant.ml.schema import SCHEMA_VERSION, SPLITS, SplitRecorder, load_split, meta
from fedgrant.policy.constants import POLICY_REVISION
from fedgrant.policy.types import PolicyConfig
from fedgrant.sim.run_net import compute_deadline_ns, run_network_round
from fedgrant.sim.traces import ID_TRACE_NAMES, OOD_TRACE_NAMES
from fedgrant.sim.seeding import network_round_seed
from fedgrant.sim.value_bank import load_value_bank

N_CLIENTS = 20
N_ROUNDS = 50
RUNS_PER_SHARD = 10
SAMPLE_TARGETS = {"train": 4_000_000, "val": 500_000, "test_id": 500_000, "test_ood": 500_000}
APPROVED_SPLIT_YIELDS = {"train": 3_998_395, "val": 483_486, "test_id": 451_785, "test_ood": 457_579}
APPROVED_RUN_COUNTS = {"train": 951, "val": 100, "test_id": 100, "test_ood": 100}
BETA_VALUES = (0.05, 0.10, 0.20)
ID_PI_VALUES = (0.02, 0.05, 0.10)
ID_BURST_VALUES = (1, 8, 32)
OOD_PI_VALUES = (0.05, 0.15)
OOD_BURST_VALUES = (8, 32)
TRACE_SPLIT_BY_DATA_SPLIT = {
    "train": "train",
    "val": "val",
    "test_id": "test",
    "test_ood": "ood",
}


def _run_settings(split: str, seed: int) -> dict[str, Any]:
    """Return a deterministic, balanced draw from the specified run conditions.

    Cycling a seeded permutation of the complete condition grid keeps every
    factor uniform and guarantees coverage in small validation/test splits.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLITS)}")
    seed_range = SPLITS[split]
    if seed not in seed_range:
        raise ValueError(f"seed {seed} outside the {split} range {seed_range}")

    if split == "test_ood":
        cases = list(itertools.product(("trace-OOD",), OOD_PI_VALUES, OOD_BURST_VALUES, BETA_VALUES))
        capacities = {"trace-OOD": "L5"}
    else:
        cases = list(itertools.product(("constant", "trace-ID"), ID_PI_VALUES, ID_BURST_VALUES, BETA_VALUES))
        capacities = {"constant": "L2", "trace-ID": "L4"}

    permutation = np.random.default_rng(seed_range.start).permutation(len(cases))
    # Apply the fixed permutation to the cycle index, independent of the
    # per-run seed, so all runs in a split cover a balanced condition grid.
    case_index = int(permutation[int(seed - seed_range.start) % len(cases)])
    capacity, pi, burst, beta = cases[case_index]
    return {
        "scenario": capacities[capacity],
        "capacity": capacity,
        "pi": float(pi),
        "burst": int(burst),
        "beta": float(beta),
        "trace_split": TRACE_SPLIT_BY_DATA_SPLIT[split],
    }


def _condition_key(settings: dict[str, Any]) -> tuple[str, float, int, float]:
    return (
        str(settings["capacity"]), float(settings["pi"]),
        int(settings["burst"]), float(settings["beta"]),
    )


def _condition_count(split: str) -> int:
    if split == "test_ood":
        return len(OOD_PI_VALUES) * len(OOD_BURST_VALUES) * len(BETA_VALUES)
    return 2 * len(ID_PI_VALUES) * len(ID_BURST_VALUES) * len(BETA_VALUES)


def _load_energy_bank(path: str | Path, n_clients: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Load validated FULL profiles, normalized weights and their digest."""
    bank = load_value_bank(path, n_clients)
    return bank.energies_sorted, bank.client_weights, bank.sha256


def _validate_traces(split: str, trace_dir: Path) -> None:
    trace_names = OOD_TRACE_NAMES if split == "test_ood" else ID_TRACE_NAMES
    trace_split = TRACE_SPLIT_BY_DATA_SPLIT[split]
    missing = [
        trace_dir / f"{name}_{trace_split}.csv"
        for name in trace_names
        if not (trace_dir / f"{name}_{trace_split}.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "required simulator trace files are missing: "
            + ", ".join(str(path) for path in missing)
        )


def _simulate_run(
    recorder: SplitRecorder,
    seed: int,
    split: str,
    trace_dir: Path,
    n_clients: int,
    n_rounds: int,
    energy_bank: np.ndarray,
    client_weights: np.ndarray,
) -> int:
    settings = _run_settings(split, seed)
    scenario = str(settings["scenario"])
    beta = float(settings["beta"])
    pi = float(settings["pi"])
    burst = int(settings["burst"])
    deadline_ns = compute_deadline_ns(beta, n_clients)
    weights = np.asarray(client_weights, dtype=np.float32)
    config = PolicyConfig(selector="value", predictor="ewma", explore_eps=0.1)
    gateway = None
    sample_count = 0
    recorder.begin_run(
        seed, seed, burst=burst, pi=pi, beta=beta,
        D_ns=deadline_ns, capacity=str(settings["capacity"]),
    )

    for round_index in range(n_rounds):
        round_id = round_index + 1
        round_seed = network_round_seed(seed, round_id)
        energy_rng = np.random.default_rng(np.random.SeedSequence([seed, round_index, 1]))
        bank_rounds = energy_rng.integers(0, energy_bank.shape[0], size=n_clients)
        energies = energy_bank[bank_rounds, np.arange(n_clients)].copy()
        round_start_ns = round_index * (1_000_000_000 + deadline_ns)
        handles: dict[int, int] = {}

        def sample_hook(entry: dict[str, Any]) -> None:
            nonlocal sample_count
            grant_id = int(entry["grant_id"])
            if entry["event"] == "grant_issued":
                handle = recorder.add_sample(
                    seed,
                    int(entry["client_id"]),
                    int(entry["abs_t_ns"]),
                    entry["x48"],
                    float(entry["p_ewma"]),
                    float(entry["p_hmm"]),
                    float(entry["p_oracle"]),
                )
                handles[grant_id] = handle
                sample_count += 1
            elif entry["event"] == "grant_resolved":
                if grant_id not in handles:
                    return  # The grant was issued after reaching the sample target.
                recorder.add_event(
                    handles.pop(grant_id),
                    int(entry["y"]),
                    int(entry["abs_t_ns"]),
                    entry["obs"],
                )
            else:
                raise ValueError(f"unknown simulator hook event {entry['event']!r}")

        result = run_network_round(
            round_id=round_id,
            num_clients=n_clients,
            client_weights=weights,
            client_energies=energies,
            beta=beta,
            scenario=scenario,
            split=str(settings["trace_split"]),
            trace_dir=trace_dir,
            policy_config=config,
            seed=round_seed,
            sample_hook=sample_hook,
            gateway=gateway,
            round_start_abs_ns=round_start_ns,
            loss_rate=pi,
            mean_burst_length=burst,
        )
        gateway = result["gateway"]
        if handles:
            raise ValueError(
                f"simulator left {len(handles)} issued grants unresolved in run {seed}, "
                f"round {round_id}"
            )

    recorder.end_run(seed)
    return sample_count


def _require_policy_wired() -> None:
    """Refuse to stamp POLICY_REVISION on data whose simulator does not run that policy.

    The revision names the bounded probe and min-RTT pacing; both are checked
    functionally (as run_mininet.probe_wired does for the probe), so data can
    never again carry a revision its grants did not follow.
    """
    from fedgrant.mininet.run_mininet import probe_wired

    if not probe_wired():
        raise RuntimeError(
            f"POLICY_REVISION {POLICY_REVISION!r} includes the bounded probe, but the simulator "
            "issues no probe grants (CandidateView.probe not wired); refusing to generate data"
        )
    if not min_rtt_wired():
        raise RuntimeError(
            f"POLICY_REVISION {POLICY_REVISION!r} includes min-RTT pacing (Eq. 12), but the "
            "simulator's P_target does not follow each client's minimum RTT; refusing to generate data"
        )


def min_rtt_wired() -> bool:
    """Whether GatewaySim's P_target uses the min-RTT RTT~ of Eq. 12 (functional check)."""
    try:
        from fedgrant.core.estimators import rtt_update
        from fedgrant.policy.pacing import p_target_min_rtt
        from fedgrant.predictors.ewma import PEWMA
        from fedgrant.sim.run_net import GatewaySim

        sim = GatewaySim(1, np.array([1.0], np.float32), PEWMA(), D_ns=1_000_000_000, K=2)
        state = sim.client_states[0]
        rtt_update(state, 20_000_000)
        rtt_update(state, 200_000_000)      # srtt rises; the minimum stays 20 ms
        return abs(sim.get_p_target() - p_target_min_rtt([state])) < 1e-9 and state.rtt_min_ns == 20_000_000
    except Exception:
        return False


def generate_split(
    split: str,
    out: str | Path = "data/mldata",
    trace_dir: str | Path = "data/traces",
    energy_bank_path: str | Path = "data/bank/femnist_full_s0.npz",
    runs: int | None = None,
    runs_per_shard: int = RUNS_PER_SHARD,
    n_clients: int = N_CLIENTS,
    n_rounds: int = N_ROUNDS,
    target_samples: int | None = None,
    verbose: bool = True,
) -> list[Path]:
    """Generate all assigned seeds, or the first ``runs`` seeds, for one split."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected {sorted(SPLITS)}")
    if runs_per_shard < 1 or n_clients < 1 or n_rounds < 1:
        raise ValueError("runs_per_shard, n_clients and n_rounds must be >= 1")
    seed_range = SPLITS[split]
    approved_runs = APPROVED_RUN_COUNTS[split]
    run_count = approved_runs if runs is None else int(runs)
    if run_count < 1 or run_count > approved_runs:
        raise ValueError(f"runs must be in [1, {approved_runs}] for approved {split} data")

    _require_policy_wired()
    trace_path = Path(trace_dir)
    _validate_traces(split, trace_path)
    energy_bank, client_weights, energy_bank_sha256 = _load_energy_bank(energy_bank_path, n_clients)
    destination = Path(out) / split
    destination.mkdir(parents=True, exist_ok=True)
    target = SAMPLE_TARGETS[split] if target_samples is None else int(target_samples)
    if target < 1:
        raise ValueError("target_samples must be >= 1")
    trace_names = OOD_TRACE_NAMES if split == "test_ood" else ID_TRACE_NAMES
    trace_split = TRACE_SPLIT_BY_DATA_SPLIT[split]
    trace_manifest = {
        f"{name}_{trace_split}.csv": hashlib.sha256(
            (trace_path / f"{name}_{trace_split}.csv").read_bytes()
        ).hexdigest()
        for name in trace_names
    }
    generator = (
        f"fedgrant.ml.datagen/v3;policy={POLICY_REVISION};conditions=balanced-grid;split={split};"
        f"clients={n_clients};rounds={n_rounds};energy_bank_sha256={energy_bank_sha256};"
        "weights=full-bank;round_seed=SeedSequence(master_seed,round_id-1)"
    )
    settings = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "seed_range": [seed_range.start, seed_range.stop],
        "generator": generator,
        "n_clients": n_clients,
        "n_rounds": n_rounds,
        "runs_per_shard": runs_per_shard,
        "target_samples": target,
        "full_grid_target": target == SAMPLE_TARGETS[split],
        "condition_count": _condition_count(split),
        "samples_per_condition": (target + _condition_count(split) - 1) // _condition_count(split),
        "trace_split": trace_split,
        "trace_sha256": trace_manifest,
        "bank_path": str(energy_bank_path),
        "bank_sha256": energy_bank_sha256,
        "energy_profile_draw": "uniform FULL round profile index independently for each (run, round, client)",
        "client_weights": client_weights.astype(float).tolist(),
        "scaler": {
            "applied_to_features": False,
            "stored_feature_domain": "raw",
            "fit_split": "train",
            "statistics": "fit by downstream model training from train only",
        },
        "policy": {
            "selector": "value", "predictor": "ewma", "explore_eps": 0.1,
            "revision": POLICY_REVISION, "probe_wired": True, "pacing_rtt": "min",
        },
        "beta_values": list(BETA_VALUES),
        "loss_and_burst_values": (
            {"pi": list(OOD_PI_VALUES), "mean_burst": list(OOD_BURST_VALUES)}
            if split == "test_ood"
            else {"pi": list(ID_PI_VALUES), "mean_burst": list(ID_BURST_VALUES)}
        ),
    }
    manifest_path = destination / "manifest.json"
    shard_paths = sorted(destination.glob("shard_*.npz"))
    shard_indices = []
    for path in shard_paths:
        suffix = path.stem.removeprefix("shard_")
        if not suffix.isdigit():
            raise ValueError(f"invalid shard name: {path.name}")
        shard_indices.append(int(suffix))
    if shard_indices != list(range(len(shard_indices))):
        raise ValueError("existing shards must be consecutive from shard_000.npz")
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, expected in settings.items():
            if manifest.get(key) != expected:
                raise ValueError(f"existing manifest setting {key!r} does not match this generation")

    next_seed = seed_range.start
    total_samples = 0
    condition_samples: dict[tuple[str, float, int, float], int] = {}
    shard_records: list[dict[str, Any]] = []
    for path in shard_paths:
        data = load_split(path)
        shard_meta = meta(data)
        if shard_meta.get("generator") != generator:
            raise ValueError(f"existing shard {path} was generated with different settings")
        seeds = data["run_seed"].astype(int).tolist()
        if not seeds or seeds != list(range(next_seed, next_seed + len(seeds))):
            raise ValueError(f"existing shard {path} has unexpected run seeds")
        if len(seeds) > runs_per_shard or next_seed + len(seeds) > seed_range.start + run_count:
            raise ValueError(f"existing shard {path} exceeds the configured run range")
        total_samples += int(data["y"].size)
        for run_seed in seeds:
            run_mask = data["sample_run"] == run_seed
            key = _condition_key(_run_settings(split, run_seed))
            condition_samples[key] = condition_samples.get(key, 0) + int(np.count_nonzero(run_mask))
        next_seed += len(seeds)
        shard_records.append({
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_runs": len(seeds),
            "n_samples": int(data["y"].size),
            "seed_range": [seeds[0], seeds[-1] + 1],
        })
    if manifest is not None and manifest.get("shards", []) != shard_records:
        raise ValueError("existing manifest shard hashes/counts do not match shard files")
    if next_seed >= seed_range.start + run_count:
        # The approved run prefix is complete even when its whole-run sample
        # yield falls below the nominal target. Leave the accepted manifest
        # byte-for-byte untouched on this no-op resume.
        return []

    created: list[Path] = []
    started = time.time()
    while next_seed < seed_range.start + run_count and total_samples < target:
        shard_index = len(shard_records)
        shard_runs = min(runs_per_shard, seed_range.start + run_count - next_seed)
        recorder = SplitRecorder(split, generator=generator)
        for run_offset in range(shard_runs):
            if total_samples >= target:
                break
            seed = next_seed + run_offset
            condition = _condition_key(_run_settings(split, seed))
            samples = _simulate_run(
                recorder,
                seed,
                split,
                trace_path,
                n_clients,
                n_rounds,
                energy_bank,
                client_weights,
            )
            total_samples += samples
            condition_samples[condition] = condition_samples.get(condition, 0) + samples
            if verbose:
                print(
                    f"[{split}] seed={seed} samples={samples} "
                    f"elapsed={time.time() - started:.0f}s",
                    flush=True,
                )
        shard_path = destination / f"shard_{shard_index:03d}.npz"
        if shard_path.exists():
            raise FileExistsError(f"refusing to overwrite existing shard: {shard_path}")
        recorder.save(shard_path)
        created.append(shard_path)
        shard_data = load_split(shard_path)
        seeds = shard_data["run_seed"].astype(int).tolist()
        next_seed += len(seeds)
        shard_records.append({
            "file": shard_path.name,
            "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
            "n_runs": len(seeds),
            "n_samples": int(shard_data["y"].size),
            "seed_range": [seeds[0], seeds[-1] + 1],
        })
        manifest_data = {
            **settings,
            "n_runs": sum(item["n_runs"] for item in shard_records),
            "n_events": total_samples,
            "n_samples": total_samples,
            "shards": shard_records,
            "complete": total_samples >= target,
            "target_met": total_samples >= target,
        }
        temp_manifest = manifest_path.with_suffix(".json.tmp")
        temp_manifest.write_text(json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp_manifest, manifest_path)
        if verbose:
            print(f"[{split}] wrote {shard_path}", flush=True)
    manifest_data = {
        **settings,
        "n_runs": sum(item["n_runs"] for item in shard_records),
        "n_events": total_samples,
        "n_samples": total_samples,
        "shards": shard_records,
        "complete": total_samples >= target,
        "target_met": total_samples >= target,
    }
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    temp_manifest.write_text(json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_manifest, manifest_path)
    return created


def validate_generated_split(
    split_dir: str | Path, *, expected_samples: int | None = None,
) -> dict[str, Any]:
    """Validate a generated split shard-by-shard without concatenating large arrays."""
    directory = Path(split_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"generation manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = str(manifest.get("split", ""))
    if split not in SPLITS or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("manifest has an unsupported split or schema version")
    declared_target = int(manifest.get("target_samples", SAMPLE_TARGETS[split]))
    target = declared_target
    if expected_samples is not None:
        target = int(expected_samples)
    shards = sorted(directory.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shard files in {directory}")
    if len(shards) != len(manifest.get("shards", [])):
        raise ValueError("manifest shard list does not match the directory")

    seen_seeds: set[int] = set()
    cases_seen: set[tuple[str, float, int, float]] = set()
    total_samples = total_events = total_runs = 0
    shard_summary = []
    for index, path in enumerate(shards):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        record = manifest["shards"][index]
        if record.get("file") != path.name or record.get("sha256") != digest:
            raise ValueError(f"manifest checksum does not match {path.name}")
        data = load_split(path)
        shard_meta = meta(data)
        if shard_meta.get("split") != split or shard_meta.get("generator") != manifest.get("generator"):
            raise ValueError(f"{path.name} metadata does not match manifest")
        seeds = [int(seed) for seed in data["run_seed"]]
        if len(set(seeds)) != len(seeds) or seen_seeds.intersection(seeds):
            raise ValueError("duplicate run seeds found across split shards")
        seen_seeds.update(seeds)
        if not np.array_equal(data["run_id"], data["run_seed"].astype(np.int32)):
            raise ValueError(f"{path.name} run_id must equal run_seed")
        if seeds != list(range(seeds[0], seeds[-1] + 1)):
            raise ValueError(f"{path.name} run seeds are not contiguous")
        if (
            int(record.get("n_runs", -1)) != len(seeds)
            or int(record.get("n_samples", -1)) != int(data["y"].size)
            or record.get("seed_range") != [seeds[0], seeds[-1] + 1]
        ):
            raise ValueError(f"manifest counts or seed range do not match {path.name}")
        if (
            int(shard_meta.get("n_runs", -1)) != len(seeds)
            or int(shard_meta.get("n_samples", -1)) != int(data["y"].size)
            or int(shard_meta.get("n_events", -1)) != int(data["events"].shape[0])
        ):
            raise ValueError(f"{path.name} embedded metadata counts do not match arrays")
        if not set(data["sample_run"].astype(int)).issubset(seeds):
            raise ValueError(f"{path.name} has samples without a complete run record")
        if not set(data["event_run"].astype(int)).issubset(seeds):
            raise ValueError(f"{path.name} has events without a complete run record")
        for seed in seeds:
            if seed not in SPLITS[split]:
                raise ValueError(f"seed {seed} lies outside the {split} seed range")
            settings = _run_settings(split, seed)
            case_key = _condition_key(settings)
            sample_total_for_run = int(np.count_nonzero(data["sample_run"] == seed))
            if sample_total_for_run:
                cases_seen.add(case_key)
            run_idx = int(np.flatnonzero(data["run_seed"] == seed)[0])
            expected_capacity = str(settings["capacity"])
            if (
                str(data["run_capacity"][run_idx]) != expected_capacity
                or not np.isclose(data["run_pi"][run_idx], settings["pi"])
                or int(data["run_burst"][run_idx]) != settings["burst"]
                or not np.isclose(data["run_beta"][run_idx], settings["beta"])
                or int(data["run_D_ns"][run_idx]) != compute_deadline_ns(settings["beta"], int(manifest["n_clients"]))
            ):
                raise ValueError(f"{path.name} run metadata disagrees with simulated settings for seed {seed}")
        event_keys = data["event_run"].astype(np.int64) * 65536 + data["event_client"].astype(np.int64)
        sample_keys = data["sample_run"].astype(np.int64) * 65536 + data["sample_client"].astype(np.int64)
        all_keys, inverse = np.unique(np.concatenate([event_keys, sample_keys]), return_inverse=True)
        event_inverse, sample_inverse = inverse[:event_keys.size], inverse[event_keys.size:]
        event_count = np.bincount(event_inverse, minlength=all_keys.size)
        sample_count = np.bincount(sample_inverse, minlength=all_keys.size)
        if not np.array_equal(event_count, sample_count):
            raise ValueError(f"{path.name} does not have one resolution event per recorded sample")
        event_y = np.bincount(event_inverse, weights=data["events"][:, 0], minlength=all_keys.size)
        sample_y = np.bincount(sample_inverse, weights=data["y"], minlength=all_keys.size)
        if not np.array_equal(event_y, sample_y):
            raise ValueError(f"{path.name} event labels disagree with sample labels")
        total_samples += int(data["y"].size)
        total_events += int(data["events"].shape[0])
        total_runs += len(seeds)
        shard_summary.append({"file": path.name, "sha256": digest})
    if total_samples != int(manifest.get("n_samples", -1)) or total_runs != int(manifest.get("n_runs", -1)):
        raise ValueError("manifest total counts do not match shard arrays")
    if "n_events" in manifest and total_events != int(manifest["n_events"]):
        raise ValueError("manifest event count does not match shard arrays")
    ordered_seeds = sorted(seen_seeds)
    if ordered_seeds and ordered_seeds != list(range(ordered_seeds[0], ordered_seeds[-1] + 1)):
        raise ValueError("run seeds across shards are not contiguous")
    if manifest.get("scaler") != {
        "applied_to_features": False,
        "stored_feature_domain": "raw",
        "fit_split": "train",
        "statistics": "fit by downstream model training from train only",
    }:
        raise ValueError("unexpected feature scaler metadata")
    if not manifest.get("bank_sha256") or not manifest.get("trace_sha256"):
        raise ValueError("manifest is missing bank or trace provenance")
    if bool(manifest.get("target_met")) != (total_samples >= declared_target):
        raise ValueError("manifest target_met does not agree with the actual sample count")
    if expected_samples is not None and total_samples != target:
        raise ValueError(f"expected {target} samples, found {total_samples}")
    if manifest.get("full_grid_target") and manifest.get("target_met") and len(cases_seen) != _condition_count(split):
        raise ValueError("target-complete split does not cover every required condition")
    return {
        "split": split,
        "n_runs": total_runs,
        "n_events": total_events,
        "n_samples": total_samples,
        "target_samples": target,
        "target_met": bool(manifest.get("target_met")),
        "conditions_covered": len(cases_seen),
        "conditions_required": _condition_count(split),
        "shards": shard_summary,
    }


def validate_candidate_splits(root: str | Path) -> dict[str, dict[str, Any]]:
    """Require all four complete split directories with disjoint run seeds."""
    base = Path(root)
    summaries = {
        split: validate_generated_split(base / split, expected_samples=APPROVED_SPLIT_YIELDS[split])
        for split in SPLITS
    }
    all_seeds: dict[int, str] = {}
    for split in SPLITS:
        directory = base / split
        for shard in sorted(directory.glob("shard_*.npz")):
            data = load_split(shard)
            for seed in data["run_seed"].astype(int).tolist():
                previous = all_seeds.setdefault(seed, split)
                if previous != split:
                    raise ValueError(f"run seed {seed} appears in both {previous} and {split}")
    return summaries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", nargs="+", choices=sorted(SPLITS), default=None)
    parser.add_argument(
        "--runs-per-split", type=int, default=None,
        help="limit each split to its first N assigned seeds (default: full range)",
    )
    parser.add_argument("--runs-per-shard", type=int, default=RUNS_PER_SHARD)
    parser.add_argument("--target-samples", type=int, default=None)
    parser.add_argument("--clients", type=int, default=N_CLIENTS)
    parser.add_argument("--rounds", type=int, default=N_ROUNDS)
    parser.add_argument("--trace-dir", default="data/traces")
    parser.add_argument("--energy-bank", default="data/bank/femnist_full_s0.npz")
    parser.add_argument("--out", default="data/mldata")
    args = parser.parse_args(argv)
    splits = args.splits or list(SPLITS)
    trace_dir = Path(args.trace_dir)
    for split in splits:
        _validate_traces(split, trace_dir)
    for split in splits:
        generate_split(
            split,
            out=args.out,
            trace_dir=args.trace_dir,
            energy_bank_path=args.energy_bank,
            runs=args.runs_per_split,
            runs_per_shard=args.runs_per_shard,
            n_clients=args.clients,
            n_rounds=args.rounds,
            target_samples=args.target_samples,
        )


if __name__ == "__main__":
    main()
