"""Focused tests for deterministic grid planning, execution, and resume."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fedgrant.analysis.raw_metrics import run_row
from fedgrant.analysis.runs_schema import expected_runs, make_config_id, make_row, read_runs_csv, write_runs_csv
from fedgrant.common.logschema import ROUND_METRIC_COLUMNS
from scripts.run_grid import (
    GridError,
    _run_id,
    execute_job,
    expand_grid,
    network_round_seed,
    merge_summary_files,
    partition_jobs,
    upsert_summary_row,
    validate_job,
)


def _supported_job() -> dict[str, Any]:
    return next(
        job for job in expand_grid(["N-main"])
        if job["scenario"] == "L1" and job["policy"] == "B4" and job["seed"] == 0
    )


def _fake_runner(*, round_id, sample_hook, **kwargs):
    if round_id > 20:
        for rtt in (1_000_000, 2_000_000, 3_000_000):
            sample_hook({
                "event": "grant_resolved",
                "y": 1,
                "rtt_ns": rtt,
                "t_ns": round_id * 1_000_000_000,
                "abs_t_ns": round_id * 1_000_000_000,
            })
    return {
        "gateway": object(),
        "eta": round_id / 300.0,
        "delivered_counts": [1] * int(kwargs["num_clients"]),
        "metrics": {
            "clients_represented": int(kwargs["num_clients"]),
            "deadline_completion": 1.0,
            "goodput_bps": 100.0,
            "utilization": 0.2,
            "drops_queue_bn_client": 2,
            "drops_queue_bn_cross": 0,
            "wasted_grants_ratio": 0.1,
            "late_after_timeout": 0,
        },
        "late_metrics": {"late_after_deadline": 0},
    }


def test_expansion_uses_schema_ids_and_validates_repository_yaml() -> None:
    jobs = expand_grid()
    assert len(jobs) == len(expected_runs()) == 1074
    assert len({job["run_id"] for job in jobs}) == len(jobs)
    assert all(job["config_id"] == make_config_id(
        job["scenario"], job["beta"], job["N"], job["cross_mbps"]
    ) for job in jobs)
    assert all(job["run_id"] == _run_id(
        job["experiment_id"], job["config_id"], job["policy"], job["seed"]
    ) for job in jobs)
    x2 = next(job for job in jobs if job["experiment_id"] == "N-X2")
    assert x2["scenario"] == "L2" and x2["cross_mbps"] == 3.0
    assert x2["config_id"] == "L2-b0.10-N20-c3"
    filtered = expand_grid(["N-main"], policies=["B4"], scenarios=["L1"])
    assert filtered and all(job["policy"] == "B4" and job["scenario"] == "L1" for job in filtered)


def test_validation_rejects_malformed_job_parameters() -> None:
    job = _supported_job()
    validate_job(job)
    bad = dict(job, beta=0.10)
    with pytest.raises(GridError, match="disagree"):
        validate_job(bad)
    bad = dict(job, run_id="some-other-output")
    with pytest.raises(GridError, match="run_id"):
        validate_job(bad)
    with pytest.raises(GridError, match="unknown experiment"):
        expand_grid(["not-an-experiment"])
    with pytest.raises(GridError, match="unknown policy"):
        expand_grid(policies=["not-a-policy"])


def test_machine_and_worker_partitions_cover_each_job_once_and_keep_ids_stable() -> None:
    jobs = expand_grid(["N-X1"])
    parts = [
        partition_jobs(jobs, machine, 3, worker_id=worker, num_workers=2)
        for machine in range(3)
        for worker in range(2)
    ]
    assigned = [job for part in parts for job in part]
    assert len(assigned) == len(jobs)
    assert len({job["run_id"] for job in assigned}) == len(jobs)
    assert {job["run_id"] for job in assigned} == {job["run_id"] for job in jobs}
    assert partition_jobs(jobs, 0, 1) == jobs
    with pytest.raises(ValueError, match="within"):
        partition_jobs(jobs, 3, 3)


def test_round_seed_matches_fl_server_seedsequence_convention() -> None:
    seed = network_round_seed(17, 9)
    expected = int(np.random.SeedSequence([17, 8]).generate_state(1)[0])
    assert seed == expected
    assert seed == network_round_seed(17, 9)
    assert seed != network_round_seed(17, 10)


def test_unsupported_modes_are_recorded_blocked_not_as_success(tmp_path: Path) -> None:
    job = next(job for job in expand_grid(["M1"]) if job["policy"] == "B4")

    def must_not_run(**kwargs):
        raise AssertionError("blocked mode reached the simulator")

    row = execute_job(job, tmp_path, simulation_runner=must_not_run)
    assert row["status"] == "missing"
    run_dir = tmp_path / job["experiment_id"] / job["run_id"]
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["execution_state"] == "blocked"
    assert "client" in meta["block_reasons"][0]
    assert "N" in meta and "n_clients" not in meta
    assert set(name for name in (
        "round_metrics.csv", "packets.csv", "queues.csv", "rtt_samples.csv",
        "scheduler_samples.csv", "processes.csv",
    ) if (run_dir / name).is_file()) == {
        "round_metrics.csv", "packets.csv", "queues.csv", "rtt_samples.csv",
        "scheduler_samples.csv", "processes.csv",
    }


def test_baseline_transfer_and_fl_jobs_are_not_misrepresented_as_grant_runs(tmp_path: Path) -> None:
    jobs = expand_grid(["N-main", "F-main", "M2"])
    selected = [
        next(job for job in jobs if job["experiment_id"] == "N-main" and job["policy"] == "B2"),
        next(job for job in jobs if job["experiment_id"] == "N-main" and job["policy"] == "B3"),
        next(job for job in jobs if job["experiment_id"] == "F-main" and job["policy"] == "FG-GRU"),
        next(job for job in jobs if job["experiment_id"] == "M2"),
    ]
    for job in selected:
        row = execute_job(job, tmp_path, simulation_runner=lambda **kwargs: pytest.fail("blocked adapter executed"))
        assert row["status"] == "missing"
        meta = json.loads((tmp_path / job["experiment_id"] / job["run_id"] / "meta.json").read_text())
        assert meta["execution_state"] == "blocked"
        assert meta["block_reasons"]


def test_simulator_run_writes_schema_logs_and_resumes_completed_result(tmp_path: Path) -> None:
    job = _supported_job()
    calls = 0

    def runner(**kwargs):
        nonlocal calls
        calls += 1
        return _fake_runner(**kwargs)

    first = execute_job(job, tmp_path, simulation_runner=runner)
    assert first["status"] == "ok"
    assert first["eta"] == pytest.approx(np.mean(np.arange(21, 201) / 300.0))
    assert first["rtt_p50_ms"] == pytest.approx(2.0)
    assert calls == job["n_rounds"]
    run_dir = tmp_path / job["experiment_id"] / job["run_id"]
    assert json.loads((run_dir / "meta.json").read_text())["execution_state"] == "complete"
    with (run_dir / "round_metrics.csv").open(newline="") as stream:
        assert tuple(next(csv.reader(stream))) == ROUND_METRIC_COLUMNS
    with (run_dir / "run.csv").open(newline="") as stream:
        assert "n_clients" in next(csv.reader(stream))

    second = execute_job(job, tmp_path, simulation_runner=runner)
    assert second == first
    assert calls == job["n_rounds"]


def test_failed_attempt_retries_once_then_resumes_without_rerunning(tmp_path: Path) -> None:
    job = _supported_job()
    calls = 0

    def fail_once(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected transient failure")
        return _fake_runner(**kwargs)

    row = execute_job(job, tmp_path, simulation_runner=fail_once)
    assert row["status"] == "rerun"
    assert calls == job["n_rounds"] + 1
    run_dir = tmp_path / job["experiment_id"] / job["run_id"]
    meta = json.loads((run_dir / "meta.json").read_text())
    assert [attempt["state"] for attempt in meta["attempts"]] == ["failed", "rerun"]

    execute_job(job, tmp_path, simulation_runner=fail_once)
    assert calls == job["n_rounds"] + 1


def test_partial_completed_artifacts_are_not_skipped_as_success(tmp_path: Path) -> None:
    job = _supported_job()
    calls = 0

    def counted(**kwargs):
        nonlocal calls
        calls += 1
        return _fake_runner(**kwargs)

    execute_job(job, tmp_path, simulation_runner=counted)
    run_dir = tmp_path / job["experiment_id"] / job["run_id"]
    (run_dir / "round_metrics.csv").unlink()

    row = execute_job(job, tmp_path, simulation_runner=counted)

    assert row["status"] == "rerun"
    assert calls == job["n_rounds"] * 2
    meta = json.loads((run_dir / "meta.json").read_text())
    assert [attempt["state"] for attempt in meta["attempts"]] == ["complete", "incomplete", "rerun"]


def test_two_failed_attempts_are_terminal_missing(tmp_path: Path) -> None:
    job = _supported_job()
    calls = 0

    def always_fails(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("persistent failure")

    row = execute_job(job, tmp_path, simulation_runner=always_fails)
    assert row["status"] == "missing"
    assert calls == 2
    run_dir = tmp_path / job["experiment_id"] / job["run_id"]
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["execution_state"] == "missing"
    assert [attempt["state"] for attempt in meta["attempts"]] == ["failed", "failed"]

    execute_job(job, tmp_path, simulation_runner=always_fails)
    assert calls == 2


def test_interrupted_run_can_resume_with_remaining_attempt(tmp_path: Path) -> None:
    job = _supported_job()

    def interrupted(**kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_job(job, tmp_path, simulation_runner=interrupted)
    run_dir = tmp_path / job["experiment_id"] / job["run_id"]
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["execution_state"] == "interrupted" and meta["attempt_count"] == 1

    row = execute_job(job, tmp_path, simulation_runner=_fake_runner)
    assert row["status"] == "rerun"
    meta = json.loads((run_dir / "meta.json").read_text())
    assert [attempt["state"] for attempt in meta["attempts"]] == ["interrupted", "rerun"]


def test_configuration_mismatch_in_existing_directory_is_never_reused(tmp_path: Path) -> None:
    job = _supported_job()
    execute_job(job, tmp_path, simulation_runner=_fake_runner)
    run_dir = tmp_path / job["experiment_id"] / job["run_id"]
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["config_sha256"] = "wrong-configuration"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(GridError, match="configuration differs"):
        execute_job(job, tmp_path, simulation_runner=_fake_runner)


def test_concurrent_same_job_execution_is_serialized(tmp_path: Path) -> None:
    job = _supported_job()
    calls = 0

    def counted(**kwargs):
        nonlocal calls
        calls += 1
        return _fake_runner(**kwargs)

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda _: execute_job(job, tmp_path, simulation_runner=counted), range(2)))
    assert rows[0] == rows[1]
    assert calls == job["n_rounds"]


def test_atomic_summary_upsert_preserves_other_runs(tmp_path: Path) -> None:
    jobs = expand_grid(["N-X1"])
    first_job, second_job = jobs[:2]
    first = make_row(first_job["experiment_id"], first_job["scenario"], first_job["beta"], first_job["N"],
                     first_job["policy"], first_job["seed"], first_job["n_rounds"],
                     cross_mbps=first_job["cross_mbps"], status="missing")
    second = make_row(second_job["experiment_id"], second_job["scenario"], second_job["beta"], second_job["N"],
                      second_job["policy"], second_job["seed"], second_job["n_rounds"],
                      cross_mbps=second_job["cross_mbps"], status="missing")
    path = tmp_path / "runs.csv"
    upsert_summary_row(first, path)
    upsert_summary_row(second, path)
    upsert_summary_row(first, path)
    rows = read_runs_csv(path)
    assert len(rows) == 2
    assert len({row.key for row in rows}) == 2


def test_machine_summary_shards_merge_and_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    jobs = expand_grid(["N-X1"])
    rows = [make_row(
        job["experiment_id"], job["scenario"], job["beta"], job["N"], job["policy"],
        job["seed"], job["n_rounds"], cross_mbps=job["cross_mbps"], status="missing",
    ) for job in jobs[:2]]
    shard_a, shard_b = tmp_path / "machine-0.csv", tmp_path / "machine-1.csv"
    write_runs_csv([rows[0]], shard_a)
    write_runs_csv([rows[1]], shard_b)
    merged = tmp_path / "merged.csv"
    merge_summary_files([shard_a, shard_b], merged)
    assert len(read_runs_csv(merged)) == 2
    with pytest.raises(ValueError, match="duplicate run"):
        merge_summary_files([shard_a, shard_a], tmp_path / "duplicate.csv")


def test_raw_metrics_accepts_legacy_n_clients_and_pools_rtt_samples(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text(json.dumps({
        "experiment_id": "M1", "scenario": "L2", "beta": 0.1,
        "n_clients": 20, "policy": "B4", "seed": 0, "n_rounds": 22,
        "cross_mbps": 0, "status": "ok",
    }))
    with (tmp_path / "round_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROUND_METRIC_COLUMNS)
        writer.writeheader()
        writer.writerow({"round_id": 1, "eta": 999})
        writer.writerow({"round_id": 21, "eta": 0.2})
        writer.writerow({"round_id": 22, "eta": 0.8})
    with (tmp_path / "rtt_samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("t_mono_ns", "round_id", "rtt_ms"))
        writer.writeheader()
        for index, value in enumerate((1, 2, 3), start=1):
            writer.writerow({"t_mono_ns": index, "round_id": 21, "rtt_ms": value})
        for index in range(4, 13):
            writer.writerow({"t_mono_ns": index, "round_id": 22, "rtt_ms": 100})

    row = run_row(tmp_path)
    assert row["N"] == 20
    assert row["eta"] == pytest.approx(0.5)  # excludes warm-up round 1
    assert row["rtt_p50_ms"] == pytest.approx(100.0)  # pooled samples, not mean per-round p50 (=51)
    assert row["rtt_p99_ms"] == pytest.approx(100.0)


def test_raw_metrics_rejects_conflicting_client_count_aliases(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text(json.dumps({"N": 20, "n_clients": 50}))
    with pytest.raises(ValueError, match="disagree"):
        run_row(tmp_path)
