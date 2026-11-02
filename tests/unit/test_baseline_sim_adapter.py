"""Focused tests for B2/B3 execution through the discrete-event network."""

from __future__ import annotations

import numpy as np
import pytest

from fedgrant.baselines.b2_udp import B2RoundResult
from fedgrant.baselines.b3_plan import B3RoundResult
from fedgrant.baselines.sim_adapter import run_baseline_network_round
from fedgrant.sim.ge import GilbertElliottModel


def _set_channel(monkeypatch, *, drop: bool) -> None:
    def make_channel(cls, scenario: str, seed: int | None = None):
        return GilbertElliottModel(1.0, 0.0, seed=seed) if drop else GilbertElliottModel(0.0, 1.0, seed=seed)

    monkeypatch.setattr(GilbertElliottModel, "for_scenario", classmethod(make_channel))


def test_b2_separates_sent_packets_from_unique_deliveries(monkeypatch) -> None:
    _set_channel(monkeypatch, drop=False)

    result = run_baseline_network_round(
        round_id=1,
        num_clients=1,
        beta=0.1,
        scenario="L1",
        policy="B2",
        seed=7,
        d=1,
        deadline_ns=1_000_000_000,
    )

    transfer = result["transfers"][0]
    assert transfer.sent_indices == (0,)
    assert len(transfer.send_times_ns) == transfer.packets_sent == 1
    assert transfer.application_bytes_sent == result["metrics"]["data_bytes_sent"] == 1432
    assert result["metrics"]["data_packets_sent"] == 1
    assert result["delivered_counts"] == [1]
    assert result["protocol_result"].delivered_blocks == 1
    assert result["metrics"]["planned_blocks"] is None
    assert result["metrics"]["plan_messages_sent"] is None
    assert result["metrics"]["grants_issued"] is None
    assert result["metrics"]["drops_queue_cross"] == 0
    assert result["metrics"]["late_after_deadline"] == 0
    assert transfer.stop_reason == "all_blocks_sent"
    assert isinstance(result["protocol_result"], B2RoundResult)


def test_b2_reports_sent_but_not_delivered_when_access_packets_are_lost(monkeypatch) -> None:
    _set_channel(monkeypatch, drop=True)

    result = run_baseline_network_round(
        round_id=1,
        num_clients=1,
        beta=0.1,
        scenario="L1",
        policy="B2",
        seed=7,
        d=1,
        deadline_ns=1_000_000_000,
    )

    assert result["metrics"]["data_packets_sent"] == 1
    assert result["metrics"]["data_bytes_sent"] == 1432
    assert result["delivered_counts"] == [0]
    assert result["protocol_result"].delivered_blocks == 0


def test_b3_uses_plan_result_and_records_plan_data_rtt(monkeypatch) -> None:
    _set_channel(monkeypatch, drop=False)

    result = run_baseline_network_round(
        round_id=1,
        num_clients=1,
        beta=0.1,
        scenario="L1",
        policy="B3",
        seed=7,
        d=1,
        deadline_ns=3_000_000_000,
    )

    protocol_result = result["protocol_result"]
    assert isinstance(protocol_result, B3RoundResult)
    assert result["metrics"]["plan_messages_sent"] == 1
    assert result["metrics"]["planned_blocks"] == sum(
        plan.blocks for plan in protocol_result.plans.values()
    )
    assert result["metrics"]["data_packets_sent"] == result["transfers"][0].packets_sent
    assert result["metrics"]["data_bytes_sent"] == result["transfers"][0].application_bytes_sent
    assert result["delivered_counts"] == [1]
    assert set(result["rtt_samples_ns"]) == {0}
    assert result["rtt_sample_times_ns"][0] >= protocol_result.plans[0].sent_ns
    assert result["metrics"]["grants_issued"] is None
    assert result["metrics"]["wasted_grants_ratio"] is None
    assert result["transfers"][0].stop_reason == "all_blocks_sent"
    assert result["plans"] == protocol_result.plans


def test_b3_lost_data_keeps_zero_delivery_separate_from_plan_and_updates_ewma(monkeypatch) -> None:
    _set_channel(monkeypatch, drop=True)

    first = run_baseline_network_round(
        round_id=1,
        num_clients=1,
        beta=0.1,
        scenario="L1",
        policy="B3",
        seed=7,
        d=1,
        deadline_ns=3_000_000_000,
    )
    assert first["metrics"]["planned_blocks"] == 1
    assert first["metrics"]["plan_messages_sent"] == 1
    assert first["metrics"]["data_packets_sent"] == 1
    assert first["delivered_counts"] == [0]
    assert first["rtt_samples_ns"] == {}
    assert first["gateway"].delivery_ewma.get(0) == pytest.approx(0.09)

    second = run_baseline_network_round(
        round_id=2,
        num_clients=1,
        beta=0.1,
        scenario="L1",
        policy="B3",
        seed=8,
        gateway=first["gateway"],
        d=1,
        deadline_ns=3_000_000_000,
    )
    assert second["protocol_result"].round_id == 2
    assert np.isfinite(second["gateway"].delivery_ewma.get(0))
