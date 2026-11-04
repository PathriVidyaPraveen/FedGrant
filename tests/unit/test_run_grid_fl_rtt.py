from __future__ import annotations

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


def _eval_history(final_accuracy: float = 0.8) -> list[dict]:
    history = []
    for round_id in range(1, 301):
        row = {
            "round": round_id,
            "eta": 0.6,
            "metrics": {
                "utilization": 0.4,
                "goodput_bps": 4_000_000.0,
                "drops_queue_bn_client": 2,
                "drops_queue_bn_cross": 0,
                "wasted_grants_ratio": 0.1,
                "deadline_completion": 0.9,
                "late_after_timeout": 1,
            },
            "late_metrics": {"late_after_deadline": 0},
            "rtt_samples_ns": [10_000_000, 20_000_000] if round_id >= 21 else [],
        }
        if round_id % 5 == 0:
            row["test_accuracy"] = 0.2 if round_id <= 15 else final_accuracy
        history.append(row)
    return history


def _write_full_records(directory: Path) -> None:
    directory.mkdir(parents=True)
    for seed in range(3):
        record = {
            "seed": seed,
            "rounds": 300,
            "history": [
                {"round": r, "test_accuracy": 0.8 if r >= 255 else 0.4}
                for r in range(5, 301, 5)
            ],
        }
        (directory / f"femnist_full_s{seed}.json").write_text(json.dumps(record))


def test_f_main_dispatches_to_end_to_end_fl_and_writes_valid_fl_row(tmp_path, monkeypatch) -> None:
    full_dir = tmp_path / "full"
    _write_full_records(full_dir)
    monkeypatch.setattr(run_grid, "FULL_REFERENCE_DIR", full_dir)
    calls = []

    def fake_run_experiment(**kwargs):
        calls.append(kwargs)
        return _eval_history()

    monkeypatch.setattr("fedgrant.fl.run_fl.run_experiment", fake_run_experiment)
    job = _job("F-main", "B4", scenario="L2", beta=0.05)
    row = run_grid.execute_job(job, tmp_path / "raw")

    assert calls == [{
        "seed": 0,
        "rounds": 300,
        "policy": "B4",
        "beta": 0.05,
        "scenario": "L2",
        "no_ef": False,
    }]
    parsed = parse_row(row)
    assert parsed.status == "ok"
    assert parsed.metrics["final_acc"] == pytest.approx(0.8)
    assert parsed.tta_censored is False
    assert parsed.metrics["tta_rounds"] == 30
    assert parsed.metrics["tta_seconds"] == pytest.approx(30 * (1 + 0.199))
    raw = json.loads(next((tmp_path / "raw").glob("F-main_*.json")).read_text())
    assert raw["rounds"][20]["rtt_samples_ns"] == [10_000_000, 20_000_000]


def test_f_abl_maps_a1_to_gru_without_error_feedback(tmp_path, monkeypatch) -> None:
    full_dir = tmp_path / "full"
    _write_full_records(full_dir)
    monkeypatch.setattr(run_grid, "FULL_REFERENCE_DIR", full_dir)
    calls = []

    def fake_run_experiment(**kwargs):
        calls.append(kwargs)
        return _eval_history()

    monkeypatch.setattr("fedgrant.fl.run_fl.run_experiment", fake_run_experiment)
    job = _job("F-abl", "FG-GRU-A1", scenario="L4", beta=0.10)
    row = run_grid.execute_job(job, tmp_path / "raw")

    assert calls[0]["policy"] == "FG-GRU"
    assert calls[0]["no_ef"] is True
    assert row["status"] == "ok"
    assert row["final_acc"] == pytest.approx(0.8)
    assert row["tta_censored"] == "false"


def test_fl_summary_marks_tta_censored_without_round_or_seconds(tmp_path, monkeypatch) -> None:
    full_dir = tmp_path / "full"
    _write_full_records(full_dir)
    monkeypatch.setattr(run_grid, "FULL_REFERENCE_DIR", full_dir)

    def fake_run_experiment(**_kwargs):
        return _eval_history(final_accuracy=0.1)

    monkeypatch.setattr("fedgrant.fl.run_fl.run_experiment", fake_run_experiment)
    row = run_grid.execute_job(_job("F-main", "FG-GRU"), tmp_path / "raw")

    parsed = parse_row(row)
    assert parsed.status == "ok"
    assert parsed.tta_censored is True
    assert parsed.metrics["tta_rounds"] != parsed.metrics["tta_rounds"]
    assert parsed.metrics["tta_seconds"] != parsed.metrics["tta_seconds"]


def test_f_ref_uses_existing_full_record_without_invoking_training(tmp_path, monkeypatch) -> None:
    full_dir = tmp_path / "full"
    _write_full_records(full_dir)
    monkeypatch.setattr(run_grid, "FULL_REFERENCE_DIR", full_dir)
    row = run_grid.execute_job(_job("F-ref", "FULL", scenario="none", beta=0.0), tmp_path / "raw")

    assert row["status"] == "ok"
    assert row["final_acc"] == pytest.approx(0.8)
    assert row["tta_censored"] == "false"
    assert not (tmp_path / "raw").exists()


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


def test_mininet_only_jobs_remain_missing_without_simulator_dispatch(tmp_path, monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("Mininet-only job called simulator")

    monkeypatch.setattr(run_grid, "run_network_round", unexpected)
    row = run_grid.execute_job(_job("M2", "FG-GRU"), tmp_path / "raw")
    assert row["status"] == "missing"


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
        }

    monkeypatch.setattr(fl_server, "train_client", lambda *args, **kwargs: np.zeros(117694, dtype=np.float32))
    monkeypatch.setattr(fl_server, "run_baseline_round", fake_baseline_round)
    monkeypatch.setattr(fl_server, "run_network_round", lambda *a, **k: (_ for _ in ()).throw(AssertionError("grant path used")))
    monkeypatch.setattr(fl_server, "get_flat_params", lambda _model: np.zeros(117694, dtype=np.float32))
    monkeypatch.setattr(fl_server, "set_flat_params", lambda *_args: None)
    monkeypatch.setattr(fl_server, "evaluate", lambda *_args, **_kwargs: (0.0, 0.5))

    fl_server.run_fl_simulated(
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
