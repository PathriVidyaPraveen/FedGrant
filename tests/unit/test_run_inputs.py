from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import scripts.run_grid as run_grid
import fedgrant.ml.datagen as datagen
from fedgrant.analysis.runs_schema import make_config_id
from fedgrant.ml.schema import SplitRecorder, validate_split
from fedgrant.ml.datagen import _condition_count, _condition_key, _run_settings
from fedgrant.sim.seeding import network_round_seed
from fedgrant.sim.value_bank import load_value_bank
import fedgrant.sim.run_baselines as run_baselines
import fedgrant.sim.run_net as run_net


def _write_bank(path, n_clients: int = 20) -> tuple[np.ndarray, np.ndarray]:
    energies = np.tile(np.arange(169, 0, -1, dtype=np.float32), (3, n_clients, 1))
    energies[1] *= 2
    weights = np.arange(1, n_clients + 1, dtype=np.float32)
    weights /= weights.sum()
    provenance_path = path.parent / "provenance.json"
    manifest_path = path.parent / "manifest.json"
    provenance_path.write_text('{"fixture": true}\n')
    manifest_path.write_text('{"fixture": true}\n')
    np.savez_compressed(
        path,
        energies_sorted=energies,
        client_weights=weights,
        round_ids=np.arange(1, 4, dtype=np.int32),
        seed=np.asarray(0, dtype=np.int64),
        num_rounds=np.asarray(3, dtype=np.int32),
        dataset_metadata_json=np.asarray(json.dumps({
            "source": "test-fixture",
            "provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            "provenance_path": str(provenance_path),
            "client_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "client_manifest": str(manifest_path),
        })),
    )
    return energies, weights


def test_value_bank_loader_preserves_weights_profiles_and_hash(tmp_path) -> None:
    path = tmp_path / "bank.npz"
    energies, weights = _write_bank(path)

    bank = load_value_bank(path, 20)

    np.testing.assert_array_equal(bank.client_weights, weights)
    np.testing.assert_array_equal(bank.energies_for_round(2), energies[1])
    sampled = bank.energies_for_run_round(5, 2)
    rng = np.random.default_rng(np.random.SeedSequence([5, 1, 1]))
    ids = rng.integers(0, energies.shape[0], size=weights.size)
    np.testing.assert_array_equal(sampled, energies[ids, np.arange(weights.size)])
    assert bank.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="shape"):
        load_value_bank(path, 50)


@pytest.mark.parametrize("split, expected", [("train", 54), ("val", 54), ("test_id", 54), ("test_ood", 12)])
def test_datagen_run_settings_cover_the_full_required_grid(split: str, expected: int) -> None:
    first = min(__import__("fedgrant.ml.schema", fromlist=["SPLITS"]).SPLITS[split])
    cases = {_condition_key(_run_settings(split, seed)) for seed in range(first, first + expected)}
    assert len(cases) == expected == _condition_count(split)


def test_datagen_records_whole_run_and_uses_bank_weights_and_independent_profiles(tmp_path, monkeypatch) -> None:
    energies = np.tile(np.arange(169, 0, -1, dtype=np.float32), (5, 20, 1))
    energies *= np.arange(1, 6, dtype=np.float32)[:, None, None]
    weights = np.arange(1, 21, dtype=np.float32)
    weights /= weights.sum()
    calls = []

    def fake_round(**kwargs):
        calls.append(kwargs)
        hook = kwargs["sample_hook"]
        hook({
            "event": "grant_issued", "grant_id": 1, "client_id": 0,
            "abs_t_ns": kwargs["round_start_abs_ns"], "x48": np.zeros(48, np.float32),
            "p_ewma": .5, "p_hmm": .5, "p_oracle": .5,
        })
        hook({
            "event": "grant_resolved", "grant_id": 1, "y": 1,
            "abs_t_ns": kwargs["round_start_abs_ns"] + 1,
            "obs": np.asarray([1, 0, 0, 0, 0, 0, 0, 0], np.float32),
        })
        return {"gateway": kwargs["gateway"]}

    monkeypatch.setattr(datagen, "run_network_round", fake_round)
    recorder = SplitRecorder("train", generator="test")
    count = datagen._simulate_run(
        recorder, 1000, "train", tmp_path, 20, 2, energies, weights,
    )
    data = recorder.arrays()
    validate_split(data)

    # A sample target may stop generation between runs, never inside this run.
    assert count == data["y"].size == 2
    assert len(calls) == 2
    for round_index, call in enumerate(calls):
        np.testing.assert_array_equal(call["client_weights"], weights)
        rng = np.random.default_rng(np.random.SeedSequence([1000, round_index, 1]))
        profile_ids = rng.integers(0, len(energies), size=20)
        np.testing.assert_array_equal(
            call["client_energies"], energies[profile_ids, np.arange(20)]
        )


def test_datagen_resume_at_approved_train_boundary_does_not_simulate_or_add_shard(tmp_path, monkeypatch) -> None:
    source = datagen.Path("data/mldata/train/manifest.json")
    approved = json.loads(source.read_text())
    assert (approved["n_runs"], approved["n_samples"], approved["shards"][-1]["seed_range"]) == (
        951, 3_998_395, [1950, 1951]
    )

    destination = tmp_path / "mldata"
    train_dir = destination / "train"
    train_dir.mkdir(parents=True)
    fake_arrays = {}
    for record in approved["shards"]:
        path = train_dir / record["file"]
        path.write_bytes(record["file"].encode())
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        lo, hi = record["seed_range"]
        fake_arrays[path.name] = {
            "run_seed": np.arange(lo, hi, dtype=np.int64),
            "sample_run": np.empty(0, dtype=np.int32),
            "y": np.empty(record["n_samples"], dtype=np.uint8),
            "meta_json": np.asarray(json.dumps({
                "split": "train", "generator": approved["generator"],
                "n_runs": record["n_runs"], "n_events": record["n_samples"],
                "n_samples": record["n_samples"],
            })),
        }
    assert fake_arrays["shard_095.npz"]["run_seed"].tolist() == [1950]
    (train_dir / "manifest.json").write_text(json.dumps(approved))
    manifest_before = (train_dir / "manifest.json").read_bytes()

    monkeypatch.setattr(datagen, "load_split", lambda path: fake_arrays[datagen.Path(path).name])
    monkeypatch.setattr(
        datagen, "_load_energy_bank",
        lambda _path, _n: (
            np.zeros((1, 20, 169), dtype=np.float32),
            np.asarray(approved["client_weights"], dtype=np.float32),
            approved["bank_sha256"],
        ),
    )
    monkeypatch.setattr(datagen, "_simulate_run", lambda *args, **kwargs: pytest.fail("resume simulated past seed 1950"))

    before = sorted(path.name for path in train_dir.glob("shard_*.npz"))
    datagen.generate_split("train", out=destination, trace_dir="data/traces", verbose=False)

    after = sorted(path.name for path in train_dir.glob("shard_*.npz"))
    resumed = json.loads((train_dir / "manifest.json").read_text())
    assert after == before
    assert (train_dir / "manifest.json").read_bytes() == manifest_before
    assert resumed["n_runs"] == 951
    assert resumed["n_samples"] == 3_998_395
    assert resumed["shards"][-1]["seed_range"] == [1950, 1951]
    assert resumed["shards"][-1]["n_runs"] == 1
    assert resumed["shards"][-1]["n_samples"] == 0


def test_fresh_default_generation_stops_at_approved_run_prefix(tmp_path, monkeypatch) -> None:
    weights = np.full(20, 1 / 20, dtype=np.float32)
    energy_bank = np.zeros((1, 20, 169), dtype=np.float32)
    monkeypatch.setattr(datagen, "_validate_traces", lambda *_args: None)
    monkeypatch.setattr(datagen, "_load_energy_bank", lambda *_args: (energy_bank, weights, "bank-hash"))
    started_seeds = []

    def zero_sample_run(recorder, seed, split, _trace_dir, _n_clients, _rounds, _bank, _weights):
        started_seeds.append(seed)
        settings = datagen._run_settings(split, seed)
        recorder.begin_run(
            seed, seed, burst=settings["burst"], pi=settings["pi"], beta=settings["beta"],
            D_ns=datagen.compute_deadline_ns(settings["beta"]), capacity=settings["capacity"],
        )
        recorder.end_run(seed)
        return 0

    monkeypatch.setattr(datagen, "_simulate_run", zero_sample_run)
    datagen.generate_split(
        "train", out=tmp_path, trace_dir="data/traces", target_samples=1_000_000_000,
        verbose=False,
    )

    manifest = json.loads((tmp_path / "train" / "manifest.json").read_text())
    assert len(started_seeds) == manifest["n_runs"] == datagen.APPROVED_RUN_COUNTS["train"] == 951
    assert started_seeds == list(range(1000, 1951))
    assert started_seeds[-1] == 1950


def test_datagen_commits_an_over_target_run_whole(tmp_path, monkeypatch) -> None:
    weights = np.full(20, 1 / 20, dtype=np.float32)
    energy_bank = np.zeros((1, 20, 169), dtype=np.float32)
    monkeypatch.setattr(datagen, "_validate_traces", lambda *_args: None)
    monkeypatch.setattr(datagen, "_load_energy_bank", lambda *_args: (energy_bank, weights, "bank-hash"))
    calls = []

    def fake_run(recorder, seed, split, _trace_dir, n_clients, _rounds, _bank, _weights):
        calls.append(seed)
        settings = datagen._run_settings(split, seed)
        recorder.begin_run(
            seed, seed, burst=settings["burst"], pi=settings["pi"], beta=settings["beta"],
            D_ns=datagen.compute_deadline_ns(settings["beta"], n_clients),
            capacity=settings["capacity"],
        )
        for index in range(3):
            handle = recorder.add_sample(seed, 0, index * 2, np.zeros(48, np.float32), .5, .5, .5)
            obs = np.zeros(8, np.float32)
            obs[0] = 1
            recorder.add_event(handle, 1, index * 2 + 1, obs)
        recorder.end_run(seed)
        return 3

    monkeypatch.setattr(datagen, "_simulate_run", fake_run)
    paths = datagen.generate_split(
        "train", out=tmp_path, trace_dir="data/traces", runs=1, runs_per_shard=1,
        target_samples=1, verbose=False,
    )

    assert calls == [1000]
    assert len(paths) == 1
    stored = datagen.load_split(paths[0])
    manifest = json.loads((tmp_path / "train" / "manifest.json").read_text())
    assert stored["run_seed"].tolist() == [1000]
    assert stored["y"].size == stored["events"].shape[0] == 3
    assert manifest["n_runs"] == 1 and manifest["n_samples"] == 3
    assert manifest["target_met"] is True


def test_grid_network_jobs_use_bank_inputs_round_seeds_and_cross_traffic(tmp_path, monkeypatch) -> None:
    bank_path = tmp_path / "bank.npz"
    energies, weights = _write_bank(bank_path)
    monkeypatch.setattr(run_grid, "load_value_bank", lambda _path, _n: load_value_bank(bank_path, _n))
    captured = []

    def fake_round(**kwargs):
        captured.append(kwargs)
        return {
            "eta": 0.5,
            "metrics": {
                "goodput_bps": 1.0, "utilization": 0.5,
                "drops_queue_bn_client": 0, "drops_queue_bn_cross": 0,
                "delivered_blocks": 1, "clients_represented": 1,
                "wasted_grants_ratio": 0.1, "deadline_completion": 0.9,
                "late_after_timeout": 0, "late_after_deadline": 0,
            },
            "late_metrics": {},
            "rtt_samples_ns": [],
            "gateway": None,
        }

    monkeypatch.setattr(run_grid, "run_network_round", fake_round)
    job = {
        "experiment_id": "N-X2", "config_id": make_config_id("L2", .1, 20, 3),
        "scenario": "L2", "beta": .1, "N": 20, "cross_mbps": 3.0,
        "policy": "B6", "seed": 17, "n_rounds": 2,
    }
    row = run_grid.execute_job(job, tmp_path / "raw")

    assert row["status"] == "ok"
    assert len(captured) == 2
    for index, call in enumerate(captured):
        np.testing.assert_array_equal(call["client_weights"], weights)
        rng = np.random.default_rng(np.random.SeedSequence([17, index, 1]))
        bank_rounds = rng.integers(0, energies.shape[0], size=weights.size)
        np.testing.assert_array_equal(
            call["client_energies"], energies[bank_rounds, np.arange(weights.size)]
        )
        assert call["seed"] == network_round_seed(17, index + 1)
        assert call["cross_mbps"] == 3.0


def test_network_round_seed_changes_by_round_and_repeats_by_master_seed() -> None:
    assert network_round_seed(9, 1) == network_round_seed(9, 1)
    assert network_round_seed(9, 1) != network_round_seed(9, 2)


@pytest.mark.parametrize("module, invoke", [
    (run_net, lambda: run_net.run_network_round(num_clients=1, beta=.01, cross_mbps=3.0)),
    (run_baselines, lambda: run_baselines.run_baseline_round("B2", num_clients=1, beta=.01, cross_mbps=3.0)),
])
def test_configured_cross_traffic_rate_reaches_both_simulators(module, invoke, monkeypatch) -> None:
    observed = []

    class CrossTrafficSpy:
        def __init__(self, engine=None, bottleneck=None, *, rate_bps, **_kwargs):
            observed.append(rate_bps)

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(module, "CrossTrafficGenerator", CrossTrafficSpy)
    invoke()
    assert observed == [3_000_000.0]
