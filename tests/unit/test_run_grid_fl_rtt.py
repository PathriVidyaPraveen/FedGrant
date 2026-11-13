from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fedgrant.analysis.runs_schema import make_config_id, parse_row
from fedgrant.encoder.ef import ErrorFeedback
from fedgrant.fl.run_fl import fl_policy_settings
from fedgrant.fl import server as fl_server
from scripts import run_grid


def _job(exp: str, policy: str, *, scenario: str = "L2", beta: float = 0.05, seed: int = 0):
    return {
        "experiment_id": exp,
        "config_id": make_config_id(scenario, beta, 20, 0.0),
        "scenario": scenario,
        "beta": beta,
        "N": 20,
        "cross_mbps": 0.0,
        "policy": policy,
        "seed": seed,
        "n_rounds": 300,
    }


def _eval_history(final_accuracy: float = 0.8, *, b3: bool = False) -> list[dict]:
    history = []
    for round_id in range(1, 301):
        if b3:
            metrics = {
                "utilization": 0.4, "goodput_bps": 4_000_000.0,
                "drops_queue_bn_client": 2, "drops_queue_bn_cross": 0,
                "delivered_blocks": 50, "clients_represented": 20,
                "data_packets_sent": 60, "data_bytes_sent": 85_920,
                "plans_issued": 20, "blocks_planned": 70,
                "grants_issued": None, "wasted_grants_ratio": None,
                "deadline_completion": None, "late_after_timeout": None,
            }
        else:
            metrics = {
                "utilization": 0.4,
                "goodput_bps": 4_000_000.0,
                "drops_queue_bn_client": 2,
                "drops_queue_bn_cross": 0,
                "wasted_grants_ratio": 0.1,
                "deadline_completion": 0.9,
                "late_after_timeout": 1,
            }
        row = {
            "round": round_id,
            "eta": 0.6,
            "metrics": metrics,
            "late_metrics": {"late_after_deadline": 0},
            "rtt_samples_ns": [] if b3 else ([10_000_000, 20_000_000] if round_id >= 21 else []),
        }
        if b3:
            # Warm-up samples are extreme so any leakage changes the result.
            row["plan_data_rtt_samples_ns"] = (
                [10_000_000, 30_000_000] if round_id >= 21 else [999_000_000]
            )
        if round_id % 5 == 0:
            row["test_accuracy"] = 0.2 if round_id <= 15 else final_accuracy
        history.append(row)
    return history


def _full_record(record_seed: int, provenance_sha: str, **overrides) -> dict:
    record = {
        "seed": record_seed,
        "rounds": 300,
        "policy": "full",
        "error_feedback": True,
        "num_clients": 20,
        "dataset": {"source": "fixture", "provenance_sha256": provenance_sha},
        "history": [
            {"round": r, "test_loss": 1.0, "test_accuracy": 0.8 if r >= 255 else 0.4}
            for r in range(5, 301, 5)
        ],
    }
    record.update(overrides)
    return record


@pytest.fixture
def fl_env(tmp_path, monkeypatch):
    """FEMNIST manifest/provenance and valid FULL seeds 0-2 in a temp tree."""
    shards = tmp_path / "femnist" / "clients"
    shards.mkdir(parents=True)
    (shards / "manifest.json").write_text('{"fixture": true}\n')
    provenance = tmp_path / "femnist" / "provenance.json"
    provenance.write_text('{"fixture": true}\n')
    sha = hashlib.sha256(provenance.read_bytes()).hexdigest()
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    for seed in range(3):
        (full_dir / f"femnist_full_s{seed}.json").write_text(json.dumps(_full_record(seed, sha)))
    monkeypatch.setattr(run_grid, "FEMNIST_SHARD_ROOT", shards)
    monkeypatch.setattr(run_grid, "FULL_REFERENCE_DIR", full_dir)
    monkeypatch.setattr(run_grid, "FL_RECORD_DIR", full_dir)
    monkeypatch.setattr(run_grid, "VALUE_BANK_PATH", tmp_path / "bank" / "femnist_full_s0.npz")
    return SimpleNamespace(root=tmp_path, full_dir=full_dir, provenance_sha=sha, raw=tmp_path / "raw")


def _run_dir(env, job) -> Path:
    normalized, _ = run_grid._normalize_execution_job(job)
    return env.raw / job["experiment_id"] / normalized["run_id"]


def test_f_main_dispatches_to_end_to_end_fl_and_writes_valid_fl_row(fl_env, monkeypatch) -> None:
    calls = []

    def fake_run_experiment(**kwargs):
        calls.append(kwargs)
        return _eval_history()

    monkeypatch.setattr("fedgrant.fl.run_fl.run_experiment", fake_run_experiment)
    job = _job("F-main", "B4", scenario="L2", beta=0.05)
    row = run_grid.execute_job(job, fl_env.raw)

    assert calls == [{
        "seed": 0,
        "rounds": 300,
        "policy": "B4",
        "beta": 0.05,
        "scenario": "L2",
        "no_ef": False,
        "trace_dir": "data/traces",
    }]
    parsed = parse_row(row)
    assert parsed.status == "ok"
    assert parsed.metrics["final_acc"] == pytest.approx(0.8)
    assert parsed.tta_censored is False
    assert parsed.metrics["tta_rounds"] == 30
    assert parsed.metrics["tta_seconds"] == pytest.approx(30 * (1 + 0.199))
    raw = json.loads(next(fl_env.raw.glob("F-main_*.json")).read_text())
    assert raw["rounds"][20]["rtt_samples_ns"] == [10_000_000, 20_000_000]
    meta = json.loads((_run_dir(fl_env, job) / "meta.json").read_text())
    assert meta["execution_state"] == "complete" and meta["status"] == "ok"
    assert meta["tta"] == {
        "tc_s": 1.0, "tc_default_s": 1.0, "tc_overridden": False, "d_s": 0.199,
        "formula": "tta_seconds = tta_rounds * (Tc + D)",
        "target": "a* = 0.9 * mean final accuracy of FULL seeds 0-2",
    }
    assert [attempt["state"] for attempt in meta["attempts"]] == ["complete"]

    # Resume does not retrain a valid completed run.
    assert run_grid.execute_job(job, fl_env.raw) == row
    assert len(calls) == 1


def test_tc_override_is_recorded_in_manifest_and_used_for_tta_seconds(fl_env, monkeypatch) -> None:
    monkeypatch.setattr("fedgrant.fl.run_fl.run_experiment", lambda **_kwargs: _eval_history())
    job = _job("F-main", "B6", beta=0.10)
    row = run_grid.execute_job(job, fl_env.raw, tc_s=2.5)

    assert row["tta_rounds"] == 30.0
    assert row["tta_seconds"] == pytest.approx(30 * (2.5 + 0.398))
    meta = json.loads((_run_dir(fl_env, job) / "meta.json").read_text())
    assert meta["tta"]["tc_s"] == 2.5 and meta["tta"]["tc_overridden"] is True
    assert meta["tta"]["d_s"] == pytest.approx(0.398)
    # The override is part of the run identity: a default-Tc request cannot
    # silently reuse the overridden result.
    with pytest.raises(run_grid.GridError, match="configuration differs"):
        run_grid.execute_job(job, fl_env.raw)
    with pytest.raises(run_grid.GridError, match="tc_s"):
        run_grid.execute_job(job, fl_env.raw, tc_s=0.0)


def test_f_abl_maps_a1_to_gru_without_error_feedback(fl_env, monkeypatch) -> None:
    calls = []

    def fake_run_experiment(**kwargs):
        calls.append(kwargs)
        return _eval_history()

    monkeypatch.setattr("fedgrant.fl.run_fl.run_experiment", fake_run_experiment)
    job = _job("F-abl", "FG-GRU-A1", scenario="L2", beta=0.10)
    row = run_grid.execute_job(job, fl_env.raw)

    assert calls[0]["policy"] == "FG-GRU"
    assert calls[0]["no_ef"] is True
    assert row["status"] == "ok"
    assert row["final_acc"] == pytest.approx(0.8)
    assert row["tta_censored"] == "false"
    meta = json.loads((_run_dir(fl_env, job) / "meta.json").read_text())
    assert meta["fl_policy"] == "FG-GRU" and meta["error_feedback"] is False


def test_fl_summary_marks_tta_censored_without_round_or_seconds(fl_env, monkeypatch) -> None:
    monkeypatch.setattr("fedgrant.fl.run_fl.run_experiment", lambda **_kwargs: _eval_history(final_accuracy=0.1))
    row = run_grid.execute_job(_job("F-main", "FG-GRU"), fl_env.raw)

    parsed = parse_row(row)
    assert parsed.status == "ok"
    assert parsed.tta_censored is True
    assert parsed.metrics["tta_rounds"] != parsed.metrics["tta_rounds"]
    assert parsed.metrics["tta_seconds"] != parsed.metrics["tta_seconds"]


def test_fl_crash_is_retried_once_with_same_config_and_seed_then_marked_rerun(fl_env, monkeypatch) -> None:
    calls = []

    def flaky(**kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise RuntimeError("injected FL crash")
        return _eval_history()

    monkeypatch.setattr("fedgrant.fl.run_fl.run_experiment", flaky)
    job = _job("F-main", "B5")
    row = run_grid.execute_job(job, fl_env.raw)

    assert row["status"] == "rerun"
    assert len(calls) == 2 and calls[0] == calls[1]
    run_dir = _run_dir(fl_env, job)
    meta = json.loads((run_dir / "meta.json").read_text())
    assert [attempt["state"] for attempt in meta["attempts"]] == ["failed", "rerun"]
    assert meta["attempts"][0]["error"] == "RuntimeError: injected FL crash"
    first_log = json.loads((run_dir / "attempts" / "attempt-01.json").read_text())
    assert first_log["state"] == "failed" and "injected FL crash" in first_log["traceback"]
    assert json.loads((run_dir / "attempts" / "attempt-02.json").read_text())["state"] == "rerun"
    assert json.loads((run_dir / "fl_result.json").read_text())["status"] == "rerun"

    assert run_grid.execute_job(job, fl_env.raw) == row
    assert len(calls) == 2


def test_fl_second_crash_is_missing_and_terminal(fl_env, monkeypatch) -> None:
    calls = []

    def always_crashes(**kwargs):
        calls.append(kwargs)
        raise RuntimeError(f"crash {len(calls)}")

    monkeypatch.setattr("fedgrant.fl.run_fl.run_experiment", always_crashes)
    job = _job("F-main", "B7")
    row = run_grid.execute_job(job, fl_env.raw)

    assert row["status"] == "missing"
    assert len(calls) == 2
    run_dir = _run_dir(fl_env, job)
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["execution_state"] == "missing" and meta["attempt_count"] == 2
    assert [attempt["error"] for attempt in meta["attempts"]] == ["RuntimeError: crash 1", "RuntimeError: crash 2"]
    for attempt in (1, 2):
        log = json.loads((run_dir / "attempts" / f"attempt-{attempt:02d}.json").read_text())
        assert log["state"] == "failed" and f"crash {attempt}" in log["error"]

    assert run_grid.execute_job(job, fl_env.raw)["status"] == "missing"
    assert len(calls) == 2


def test_f_main_b3_pools_plan_data_rtt_after_warmup_and_not_as_grant_rtt(fl_env, monkeypatch) -> None:
    calls = []

    def fake_run_experiment(**kwargs):
        calls.append(kwargs)
        return _eval_history(b3=True)

    monkeypatch.setattr("fedgrant.fl.run_fl.run_experiment", fake_run_experiment)
    row = run_grid.execute_job(_job("F-main", "B3"), fl_env.raw)

    assert calls[0]["policy"] == "B3"
    assert row["status"] == "ok"
    assert row["plan_data_rtt_p50_ms"] == pytest.approx(20.0)
    assert row["plan_data_rtt_p99_ms"] == pytest.approx(30.0)
    assert row["plan_data_rtt_samples"] == pytest.approx(2.0)
    assert row["rtt_p50_ms"] == "" and row["rtt_p99_ms"] == ""
    assert row["grant_rtt_samples"] == "" and row["grants_issued"] == ""
    assert row["planned_blocks"] == pytest.approx(70.0)
    assert row["plan_messages_sent"] == pytest.approx(20.0)
    assert row["data_packets_sent"] == pytest.approx(60.0)


def test_f_ref_reuses_valid_frozen_history_with_full_run_identity(fl_env, monkeypatch) -> None:
    monkeypatch.setattr(
        "fedgrant.fl.run_fl.run_experiment",
        lambda **_kwargs: pytest.fail("valid F-ref must not retrain"),
    )
    job = _job("F-ref", "FULL", scenario="none", beta=0.0)
    row = run_grid.execute_job(job, fl_env.raw)

    assert row["status"] == "ok"
    assert row["final_acc"] == pytest.approx(0.8)
    assert row["tta_censored"] == ""
    run_dir = _run_dir(fl_env, job)
    source = fl_env.full_dir / "femnist_full_s0.json"
    assert (run_dir / "full_reference.json").read_bytes() == source.read_bytes()
    for name in ("run.csv", "fl_history.csv", "fl_result.json"):
        assert (run_dir / name).is_file()
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["identity"] == {"experiment_id": "F-ref", "config_id": "none-b0.00-N20-c0", "policy": "FULL", "seed": 0}
    assert meta["config_sha256"] and meta["execution_state"] == "complete"
    reuse = meta["reference_reuse"]
    assert reuse["status"] == "reused"
    assert reuse["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert reuse["validated"]["seed"] == 0 and reuse["validated"]["provenance_check"] == "matches local provenance file"

    assert run_grid.execute_job(job, fl_env.raw) == row


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"seed": 1}, "seed mismatch"),
        ({"rounds": 200}, "300-round"),
        ({"policy": "B4"}, "not FULL"),
        ({"dataset": {"source": "unverified"}}, "unverified"),
        ({"dataset": {"source": "fixture", "provenance_sha256": "0" * 64}}, "provenance does not match"),
        ({"history": [{"round": r, "test_accuracy": 0.8} for r in range(10, 301, 10)]}, "every 5 rounds"),
    ],
)
def test_f_ref_rejects_invalid_history_records_reason_and_retrains(fl_env, monkeypatch, override, reason) -> None:
    source = fl_env.full_dir / "femnist_full_s0.json"
    source.write_text(json.dumps(_full_record(0, fl_env.provenance_sha, **override)))
    calls = []
    trained = _full_record(0, fl_env.provenance_sha)

    def fake_full_training(**kwargs):
        calls.append(kwargs)
        # run_experiment writes the complete FULL record (with provenance) itself.
        source.write_text(json.dumps(trained))
        return trained["history"]

    monkeypatch.setattr("fedgrant.fl.run_fl.run_experiment", fake_full_training)
    job = _job("F-ref", "FULL", scenario="none", beta=0.0)
    row = run_grid.execute_job(job, fl_env.raw)

    assert row["status"] == "ok"
    assert calls and calls[0]["policy"] == "FULL"
    assert calls[0]["bank_path"] == str(run_grid.VALUE_BANK_PATH)  # bank absent -> F-ref s0 creates it
    meta = json.loads((_run_dir(fl_env, job) / "meta.json").read_text())
    assert meta["reference_reuse"]["status"] == "rejected"
    assert reason in meta["reference_reuse"]["reason"]
    assert (_run_dir(fl_env, job) / "rejected_full_reference.json").is_file()
    assert meta["published_reference"]["validated"]["seed"] == 0


def test_f_ref_never_overwrites_frozen_value_bank(fl_env, monkeypatch) -> None:
    (fl_env.full_dir / "femnist_full_s0.json").unlink()
    run_grid.VALUE_BANK_PATH.parent.mkdir(parents=True)
    run_grid.VALUE_BANK_PATH.write_bytes(b"frozen")
    calls = []

    def fake_full_training(**kwargs):
        calls.append(kwargs)
        record = _full_record(0, fl_env.provenance_sha)
        (fl_env.full_dir / "femnist_full_s0.json").write_text(json.dumps(record))
        return record["history"]

    monkeypatch.setattr("fedgrant.fl.run_fl.run_experiment", fake_full_training)
    row = run_grid.execute_job(_job("F-ref", "FULL", scenario="none", beta=0.0), fl_env.raw)
    assert row["status"] == "ok"
    assert "bank_path" not in calls[0]
    assert run_grid.VALUE_BANK_PATH.read_bytes() == b"frozen"


@pytest.mark.parametrize(
    ("policy", "selector", "predictor", "transport"),
    [
        ("B2", None, None, "B2"),
        ("B3", None, None, "B3"),
        ("B4", "rr", "ewma", None),
        ("B5", "srpt", "ewma", None),
        ("B6", "value", "ewma", None),
        ("B7", "value", "hmm", None),
        ("FG-GRU", "value", "gru", None),
        ("ORACLE", "value", "oracle", None),
    ],
)
def test_fl_policy_settings_follow_experiment_policy(policy, selector, predictor, transport) -> None:
    config, mode = fl_policy_settings(policy)
    assert mode == transport
    if config is None:
        assert selector is None and predictor is None
    else:
        assert (config.selector, config.predictor) == (selector, predictor)


def test_rtt_percentiles_pool_unequal_samples_and_ignore_round_20() -> None:
    rounds = [
        {"round_id": 20, "rtt_samples_ns": [999_000_000]},
        {"round_id": 21, "rtt_samples_ns": [10_000_000, 20_000_000, 30_000_000]},
        {"round_id": 22, "rtt_samples_ns": [100_000_000]},
    ]
    summary = run_grid._network_summary(
        [
            {
                "round_id": 20,
                "eta": 0.5,
                "rtt_samples_ns": [999_000_000],
            },
            *[
                {"round_id": round_id, "eta": 0.5, **{
                    "rtt_samples_ns": row["rtt_samples_ns"]
                }}
                for round_id, row in ((21, rounds[1]), (22, rounds[2]))
            ],
        ],
        "B4",
    )
    assert summary["rtt_p50_ms"] == pytest.approx(25.0)
    assert summary["rtt_p99_ms"] == pytest.approx(97.9)
    assert summary["rtt_p50_ms"] != pytest.approx((20.0 + 100.0) / 2)


def test_m2_is_blocked_not_missing_and_never_dispatched(tmp_path, monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("Mininet-only job called a substitute runner")

    monkeypatch.setattr(run_grid, "run_network_round", unexpected)
    monkeypatch.setattr(run_grid, "_execute_fl_job", unexpected)
    record = run_grid.execute_job(_job("M2", "FG-GRU", beta=0.10), tmp_path / "raw")
    assert record["status"] == "blocked"
    assert any("production Mininet FL integration" in reason for reason in record["block_reasons"])


@pytest.mark.parametrize("mode", ["B2", "B3"])
def test_fl_server_routes_push_and_plan_through_dedicated_transport(mode, monkeypatch) -> None:
    class FakeModel:
        def to(self, _device):
            return self

    persistent_state = object()
    calls = []

    def fake_baseline_round(received_mode, **kwargs):
        calls.append((received_mode, kwargs))
        return {
            "baseline_state": persistent_state,
            "round_result": SimpleNamespace(
                client_results={0: SimpleNamespace(delivered_indices=())}
            ),
            "delivered_counts": [0],
            "eta": 0.0,
            "metrics": {"delivered_blocks": 0},
            "late_metrics": {"late_after_deadline": 0},
            "b3_rtt_samples_ns": {0: 7_000_000} if received_mode == "B3" else {},
        }

    monkeypatch.setattr(fl_server, "train_client", lambda *args, **kwargs: np.zeros(117694, dtype=np.float32))
    monkeypatch.setattr(fl_server, "run_baseline_round", fake_baseline_round)
    monkeypatch.setattr(fl_server, "run_network_round", lambda *a, **k: (_ for _ in ()).throw(AssertionError("grant path used")))
    monkeypatch.setattr(fl_server, "get_flat_params", lambda _model: np.zeros(117694, dtype=np.float32))
    monkeypatch.setattr(fl_server, "set_flat_params", lambda *_args: None)
    monkeypatch.setattr(fl_server, "evaluate", lambda *_args, **_kwargs: (0.0, 0.5))

    history = fl_server.run_fl_simulated(
        model=FakeModel(),
        train_clients=[(None, None)],
        test_client_loader=lambda: [],
        client_weights=np.asarray([1.0], dtype=np.float32),
        loaders=[object()],
        error_feedback=[ErrorFeedback(117694)],
        num_rounds=2,
        seed=3,
        transport_baseline=mode,
        eval_every=2,
        client_workers=1,
    )

    assert len(calls) == 2
    assert [call[0] for call in calls] == [mode, mode]
    assert calls[0][1]["state"] is None
    assert calls[1][1]["state"] is persistent_state
    assert len(calls[0][1]["encoded_updates"]) == 1
    assert all(record["rtt_samples_ns"] == [] for record in history)
    if mode == "B3":
        assert [record["plan_data_rtt_samples_ns"] for record in history] == [[7_000_000], [7_000_000]]
    else:
        assert all("plan_data_rtt_samples_ns" not in record for record in history)
