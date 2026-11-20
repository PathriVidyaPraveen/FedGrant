"""Focused tests for deterministic grid planning, execution, and resume."""

from __future__ import annotations

import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fedgrant.analysis.raw_metrics import run_row
from fedgrant.analysis.runs_schema import (
    LEGACY_COLUMNS,
    expected_runs,
    make_config_id,
    make_row,
    read_runs_csv,
    write_runs_csv,
)
from fedgrant.common.logschema import ROUND_METRIC_COLUMNS
from scripts import run_grid as run_grid_module
from scripts.run_grid import (
    GridError,
    _run_id,
    _adapter_block_reasons,
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
            "grants_issued": 2,
            "grants_delivered": 1,
            "grants_timed_out": 1,
        },
        "late_metrics": {"late_after_deadline": 0},
    }


def _fake_baseline_runner(*, round_id, policy, num_clients, gateway, **kwargs):
    b3 = policy == "B3"
    rtt_ns = {0: 2_000_000} if b3 else {}
    return {
        "gateway": gateway if gateway is not None else object(),
        "eta": 0.25,
        "delivered_counts": ([1] + [0] * (num_clients - 1)) if b3 else [0] * num_clients,
        "metrics": {
            "clients_represented": 1 if b3 else 0,
            "deadline_completion": None,
            "goodput_bps": 50.0 if b3 else 0.0,
            "utilization": 0.1,
            "drops_queue_cross": 0,
            "drops_queue_data": 0,
            "wasted_grants_ratio": None,
            "late_after_timeout": None,
            "late_after_deadline": 0,
            "grants_issued": None,
            "grants_delivered": None,
            "grants_timed_out": None,
            "data_packets_sent": 2 if b3 else 0,
            "data_bytes_sent": 2864 if b3 else 0,
            "planned_blocks": 10 if b3 else None,
            "plan_messages_sent": num_clients if b3 else None,
            "queueing_delay_mean_ms": 0.0,
            "queueing_delay_p99_ms": 0.0,
            "protocol_overhead": None,
        },
        "late_metrics": {"late_after_deadline": 0},
        "rtt_samples_ns": rtt_ns,
        "rtt_sample_times_ns": {0: round_id * 1_000_000_000 + 2_000_000} if b3 else {},
        "round_events": [{
            "t_ns": round_id * 1_000_000_000,
            "round_id": round_id,
            "event": "client_transfer_end",
            "client_id": 0,
            "detail": "stop_reason=all_blocks_sent;packets_sent=2;application_bytes_sent=2864",
        }],
        "packet_events": ([{
            "t_ns": round_id * 1_000_000_000,
            "round_id": round_id,
            "client_id": 0,
            "packet_seq": 0,
            "packet_type": "PLAN",
            "bytes": 28,
            "result": "sent",
        }, {
            "t_ns": round_id * 1_000_000_000 + 2_000_000,
            "round_id": round_id,
            "client_id": 0,
            "packet_seq": 1,
            "packet_type": "DATA",
            "bytes": 1432,
            "result": "sent",
        }] if b3 else []),
        "queue_samples": [{
            "t_ns": round_id * 1_000_000_000 + 10_000_000,
            "backlog_bytes": 0,
            "drops": 0,
        }],
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


def _write_fl_inputs(root: Path, monkeypatch) -> None:
    """Valid FEMNIST manifest/provenance and FULL seeds 0-2 under ``root``."""
    shards = root / "femnist" / "clients"
    shards.mkdir(parents=True)
    (shards / "manifest.json").write_text('{"fixture": true}\n')
    provenance = root / "femnist" / "provenance.json"
    provenance.write_text('{"fixture": true}\n')
    full_dir = root / "full"
    full_dir.mkdir()
    for seed in range(3):
        (full_dir / f"femnist_full_s{seed}.json").write_text(json.dumps({
            "seed": seed, "rounds": 300, "policy": "full", "error_feedback": True, "num_clients": 20,
            "dataset": {"source": "fixture", "provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest()},
            "history": [{"round": r, "test_accuracy": 0.8} for r in range(5, 301, 5)],
        }))
    monkeypatch.setattr(run_grid_module, "FEMNIST_SHARD_ROOT", shards)
    monkeypatch.setattr(run_grid_module, "FULL_REFERENCE_DIR", full_dir)
    monkeypatch.setattr(run_grid_module, "FL_RECORD_DIR", full_dir)


def test_unsupported_modes_are_recorded_blocked_not_as_missing(tmp_path: Path) -> None:
    job = next(job for job in expand_grid(["M1"]) if job["policy"] == "B4")

    def must_not_run(**kwargs):
        raise AssertionError("blocked mode reached the simulator")

    record = execute_job(job, tmp_path, simulation_runner=must_not_run)
    assert record["status"] == "blocked" and record["availability"] == "blocked"
    assert record["run_id"] == job["run_id"] and record["config_id"] == job["config_id"]
    assert json.loads(record["reasons_json"]) == record["block_reasons"]
    assert record["recorded_at_utc"]
    run_dir = tmp_path / job["experiment_id"] / job["run_id"]
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["execution_state"] == "blocked" and meta["status"] == "blocked"
    assert "summary_row" not in meta
    assert "client" in meta["block_reasons"][0]
    assert "N" in meta and "n_clients" not in meta
    # Blocked jobs get no fabricated (header-only) result logs.
    assert not any((run_dir / name).exists() for name in ("run.csv", "round_metrics.csv", "packets.csv"))


def test_unsupported_baseline_jobs_are_recorded_blocked_not_as_grant_runs(tmp_path: Path) -> None:
    jobs = expand_grid(["M1", "M2"])
    selected = [
        next(job for job in jobs if job["experiment_id"] == "M1" and job["policy"] == "B2"),
        next(job for job in jobs if job["experiment_id"] == "M1" and job["policy"] == "B3"),
        next(job for job in jobs if job["experiment_id"] == "M2"),
    ]
    for job in selected:
        row = execute_job(job, tmp_path, simulation_runner=lambda **kwargs: pytest.fail("blocked adapter executed"))
        assert row["status"] == "blocked"
        meta = json.loads((tmp_path / job["experiment_id"] / job["run_id"] / "meta.json").read_text())
        assert meta["execution_state"] == "blocked"
        assert meta["block_reasons"]
    m2_reasons = _adapter_block_reasons(selected[2], tmp_path)
    assert any("not substitutes" in reason for reason in m2_reasons)


def test_fl_jobs_stay_on_the_fl_entrypoint(tmp_path: Path, monkeypatch) -> None:
    _write_fl_inputs(tmp_path / "inputs", monkeypatch)
    job = next(
        job for job in expand_grid(["F-main"])
        if job["policy"] == "FG-GRU" and job["seed"] == 0 and job["scenario"] == "L2"
    )
    called = []

    def fake_fl_job(received_job, raw_dir, *, trace_dir, tc_s):
        called.append((received_job, raw_dir, trace_dir, tc_s))
        return {"execution": "fl"}

    monkeypatch.setattr(run_grid_module, "_execute_fl_job", fake_fl_job)
    result = execute_job(
        job,
        tmp_path,
        simulation_runner=lambda **kwargs: pytest.fail("FL job reached simulator"),
    )
    assert result == {"execution": "fl"}
    assert called == [(job, tmp_path, Path("data/traces"), 1.0)]


@pytest.mark.parametrize("policy", ["B2", "B3"])
def test_f_main_baselines_are_available_through_fl_transport_adapters(policy, tmp_path: Path, monkeypatch) -> None:
    job = next(
        job for job in expand_grid(["F-main"])
        if job["policy"] == policy and job["seed"] == 0 and job["scenario"] == "L2"
    )
    # Without FEMNIST shards / FULL references the job is blocked, not missing.
    monkeypatch.setattr(run_grid_module, "FEMNIST_SHARD_ROOT", tmp_path / "absent")
    monkeypatch.setattr(run_grid_module, "FULL_REFERENCE_DIR", tmp_path / "absent")
    blocked = execute_job(job, tmp_path / "raw", simulation_runner=lambda **kwargs: pytest.fail("reached simulator"))
    assert blocked["status"] == "blocked"
    assert any("FEMNIST client shards" in reason for reason in blocked["block_reasons"])
    assert not any("transport adapter" in reason for reason in blocked["block_reasons"])

    _write_fl_inputs(tmp_path / "inputs", monkeypatch)
    assert _adapter_block_reasons(job, tmp_path) == []
    calls = []
    monkeypatch.setattr(run_grid_module, "_execute_fl_job", lambda j, *_a, **_k: calls.append(j) or {"execution": "fl"})
    assert execute_job(job, tmp_path / "raw") == {"execution": "fl"}
    assert calls == [job]


def test_fl_transport_adapter_unavailability_is_reported(monkeypatch) -> None:
    from fedgrant.fl import server as fl_server

    run_grid_module._fl_adapter_reasons.cache_clear()
    monkeypatch.delattr(fl_server, "run_baseline_round")
    try:
        reasons = run_grid_module._fl_adapter_reasons("B3")
    finally:
        run_grid_module._fl_adapter_reasons.cache_clear()
    assert reasons == ("B3 FL transport adapter is unavailable in fedgrant.fl.server",)


def test_dry_run_and_dispatch_report_identical_availability(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_grid_module, "VALUE_BANK_PATH", tmp_path / "no-bank.npz")
    monkeypatch.setattr(run_grid_module, "FEMNIST_SHARD_ROOT", tmp_path / "absent")
    monkeypatch.setattr(run_grid_module, "FULL_REFERENCE_DIR", tmp_path / "absent")
    jobs = [
        next(job for job in expand_grid(["N-main"]) if job["policy"] == "B4"),
        next(job for job in expand_grid(["N-main"]) if job["policy"] == "B3"),
        next(job for job in expand_grid(["F-main"]) if job["policy"] == "B2"),
        next(job for job in expand_grid(["F-ref"])),
        next(job for job in expand_grid(["M2"])),
    ]
    planned = run_grid_module.plan_availability(jobs, tmp_path)
    for job, reasons in planned:
        assert reasons, job["run_id"]
        record = execute_job(job, tmp_path / "raw", trace_dir=tmp_path)
        assert record["status"] == "blocked"
        assert record["block_reasons"] == reasons
    assert any("value bank" in reason for reason in planned[0][1])
    assert any("value bank" in reason for reason in planned[1][1])


def test_blocked_tracker_upserts_and_clears_and_main_keeps_blocked_out_of_runs_csv(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    out = tmp_path / "summary" / "runs.csv"
    monkeypatch.setattr("sys.argv", [
        "run_grid.py", "--exp", "M2", "--max-runs", "2", "--run",
        "--raw-dir", str(tmp_path / "raw"), "--output-csv", str(out),
    ])
    run_grid_module.main()
    blocked_csv = out.with_name("blocked_jobs.csv")
    assert not out.exists()
    with blocked_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and {row["availability"] for row in rows} == {"blocked"}
    assert all(row["experiment_id"] == "M2" and json.loads(row["reasons_json"]) for row in rows)
    assert "2 blocked (not completed)" in capsys.readouterr().out

    # Re-recording replaces, clearing removes; other entries are preserved.
    first = json.loads(json.dumps(rows[0]))
    first["reasons_json"] = json.dumps(["changed"])
    run_grid_module.record_blocked_job(first, blocked_csv)
    with blocked_csv.open(newline="") as stream:
        updated = list(csv.DictReader(stream))
    assert len(updated) == 2
    assert next(row for row in updated if row["run_id"] == first["run_id"])["reasons_json"] == '["changed"]'
    run_grid_module.clear_blocked_job(first["run_id"], blocked_csv)
    with blocked_csv.open(newline="") as stream:
        assert [row["run_id"] for row in csv.DictReader(stream)] == [rows[1]["run_id"]]
    with pytest.raises(GridError, match="only blocked"):
        run_grid_module.record_blocked_job({**first, "availability": "available"}, blocked_csv)


def test_baseline_protocol_metrics_and_typed_rtt_reach_run_outputs(tmp_path: Path) -> None:
    jobs = expand_grid(["N-main"], scenarios=["L1"])
    b2_job = next(job for job in jobs if job["policy"] == "B2" and job["seed"] == 0)
    b3_job = next(job for job in jobs if job["policy"] == "B3" and job["seed"] == 0)

    b2_row = execute_job(b2_job, tmp_path, baseline_runner=_fake_baseline_runner)
    b3_row = execute_job(b3_job, tmp_path, baseline_runner=_fake_baseline_runner)
    assert b2_row["status"] == b3_row["status"] == "ok"

    b2_dir = tmp_path / b2_job["experiment_id"] / b2_job["run_id"]
    with (b2_dir / "round_metrics.csv").open(newline="") as stream:
        b2_metrics = list(csv.DictReader(stream))[20]
    assert b2_metrics["data_packets_sent"] == "0"
    assert b2_metrics["data_bytes_sent"] == "0"
    assert b2_metrics["planned_blocks"] == ""
    assert b2_metrics["plan_messages_sent"] == ""
    assert b2_metrics["grants_issued"] == ""
    assert b2_metrics["grant_rtt_samples"] == ""
    assert b2_metrics["plan_data_rtt_samples"] == ""
    assert b2_row["data_packets_sent"] == 0
    assert b2_row["grants_issued"] == ""
    assert b2_row["rtt_p50_ms"] == ""
    assert b2_row["plan_data_rtt_p50_ms"] == ""

    b3_dir = tmp_path / b3_job["experiment_id"] / b3_job["run_id"]
    with (b3_dir / "round_metrics.csv").open(newline="") as stream:
        b3_metrics = list(csv.DictReader(stream))[20]
    assert b3_metrics["planned_blocks"] == "10"
    assert b3_metrics["plan_messages_sent"] == "20"
    assert b3_metrics["data_packets_sent"] == "2"
    assert b3_metrics["data_bytes_sent"] == "2864"
    assert b3_metrics["grants_issued"] == ""
    assert b3_metrics["grant_rtt_samples"] == ""
    assert b3_metrics["plan_data_rtt_samples"] == "1"
    assert b3_row["planned_blocks"] == 10
    assert b3_row["plan_messages_sent"] == 20
    assert b3_row["data_packets_sent"] == 2
    assert b3_row["data_bytes_sent"] == 2864
    assert b3_row["grants_issued"] == ""
    assert b3_row["rtt_p50_ms"] == ""
    assert b3_row["plan_data_rtt_p50_ms"] == pytest.approx(2.0)
    with (b3_dir / "rtt_samples.csv").open(newline="") as stream:
        samples = list(csv.DictReader(stream))
    assert samples[20]["rtt_type"] == "plan_data"
    assert samples[20]["client_id"] == "0"
    with (b3_dir / "packets.csv").open(newline="") as stream:
        packets = list(csv.DictReader(stream))
    assert {row["packet_type"] for row in packets} == {"PLAN", "DATA"}
    with (b3_dir / "rounds.csv").open(newline="") as stream:
        events = list(csv.DictReader(stream))
    assert "stop_reason=all_blocks_sent" in events[0]["detail"]


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
    assert first["grants_issued"] == 2.0
    assert first["grants_delivered"] == 1.0
    assert first["grants_timed_out"] == 1.0
    assert first["grant_rtt_samples"] == 3.0
    assert calls == job["n_rounds"]
    run_dir = tmp_path / job["experiment_id"] / job["run_id"]
    assert json.loads((run_dir / "meta.json").read_text())["execution_state"] == "complete"
    with (run_dir / "round_metrics.csv").open(newline="") as stream:
        assert tuple(next(csv.reader(stream))) == ROUND_METRIC_COLUMNS
    with (run_dir / "rtt_samples.csv").open(newline="") as stream:
        sample = next(csv.DictReader(stream))
        assert sample["rtt_type"] == "grant_resolution"
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


def test_summary_reader_accepts_schema_v1_columns(tmp_path: Path) -> None:
    row = make_row(
        "N-main", "L1", 0.1, 20, "B4", 0, 200, status="missing",
    )
    path = tmp_path / "legacy-runs.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEGACY_COLUMNS)
        writer.writeheader()
        writer.writerow({column: row[column] for column in LEGACY_COLUMNS})

    run = read_runs_csv(path)[0]
    assert run.status == "missing"
    assert np.isnan(run.metrics["planned_blocks"])


# -- probe_grants in round_metrics.csv (rule (v) bounded probe) ------------------------


class _Core:
    def __init__(self, probes: int) -> None:
        self.round_probe_grants = probes


class _Gateway:
    def __init__(self, probes: int) -> None:
        self.core = _Core(probes)


def _round_metric_rows(tmp_path: Path, job: dict) -> list[dict]:
    with (tmp_path / job["experiment_id"] / job["run_id"] / "round_metrics.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_grant_round_metrics_record_the_rounds_probe_grants(tmp_path: Path) -> None:
    job = _supported_job()

    def runner(**kwargs):
        result = _fake_runner(**kwargs)
        result["gateway"] = _Gateway(kwargs["round_id"] % 3)  # this round's GatewayCore count
        return result

    assert execute_job(job, tmp_path, simulation_runner=runner)["status"] == "ok"
    rows = _round_metric_rows(tmp_path, job)
    assert [row["probe_grants"] for row in rows[:4]] == ["1", "2", "0", "1"]
    assert len(rows) == job["n_rounds"]


def test_simulator_probe_metric_takes_precedence_over_the_core_counter(tmp_path: Path) -> None:
    job = _supported_job()

    def runner(**kwargs):
        result = _fake_runner(**kwargs)
        result["gateway"] = _Gateway(99)
        result["metrics"]["probe_grants"] = 4
        return result

    execute_job(job, tmp_path, simulation_runner=runner)
    assert {row["probe_grants"] for row in _round_metric_rows(tmp_path, job)} == {"4"}


def test_probe_grants_blank_for_b2_b3_and_without_a_counter(tmp_path: Path) -> None:
    jobs = expand_grid(["N-main"], scenarios=["L1"])
    for policy in ("B2", "B3"):
        job = next(j for j in jobs if j["policy"] == policy and j["seed"] == 0)

        def baseline(**kwargs):
            result = _fake_baseline_runner(**kwargs)
            result["gateway"] = _Gateway(7)  # even with a counter, B2/B3 have no grants
            return result

        execute_job(job, tmp_path, baseline_runner=baseline)
        assert {row["probe_grants"] for row in _round_metric_rows(tmp_path, job)} == {""}
    grant_job = _supported_job()
    execute_job(grant_job, tmp_path / "plain", simulation_runner=_fake_runner)  # gateway=object(): no counter
    assert {row["probe_grants"] for row in _round_metric_rows(tmp_path / "plain", grant_job)} == {""}


def test_real_simulator_round_exposes_the_probe_counter() -> None:
    from fedgrant.policy.constants import K
    from fedgrant.sim.run_net import run_network_round

    result = run_network_round(round_id=1, num_clients=4, beta=0.1, scenario="L1", seed=3,
                               client_energies=np.tile(np.linspace(1.0, 0.01, K, dtype=np.float32), (4, 1)))
    # The core's real per-round counter, not a blank (a nonzero count is checked
    # end to end in tests/integration/test_probe_end_to_end.py).
    assert run_grid_module._round_probe_grants(result, result["metrics"]) == result["gateway"].core.round_probe_grants
