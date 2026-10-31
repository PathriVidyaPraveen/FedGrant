"""Regression checks for simulator correctness fixes."""

import numpy as np
import pytest

from fedgrant.analysis.metrics import (
    compute_accuracy_vs_wallclock,
    compute_ece,
    compute_time_to_target,
    summarize_run_history,
)
from fedgrant.fl.server import _network_round_seed
from fedgrant.predictors.ewma import PEWMA
from fedgrant.policy.types import PolicyConfig
from fedgrant.sim.engine import EventType, SimulationEngine
from fedgrant.sim.run_net import GatewaySim, energies_by_block, run_network_round
from fedgrant.sim.traces import TraceManager


def test_ready_energy_is_weight_adjusted_before_value_scheduling() -> None:
    gateway = GatewaySim(
        num_clients=2,
        client_weights=np.array([0.25, 0.75], dtype=np.float32),
        predictor=PEWMA(),
        K=3,
    )
    gateway.on_round_start(1, round_start_abs_ns=1)
    energies = np.array([3.0, 2.0, 1.0], dtype=np.float32)

    gateway.on_ready(1, energies, t_ns=0)

    np.testing.assert_allclose(gateway.client_states[1].val, energies * 0.75**2)


def test_energy_lookup_uses_original_block_indices() -> None:
    # Send order is [block 2, block 0, block 1], with energies in send order.
    mapped = energies_by_block(
        np.array([[30.0, 10.0, 20.0]], dtype=np.float32),
        [np.array([2, 0, 1])],
        num_clients=1,
        K=3,
    )
    np.testing.assert_array_equal(mapped, [[10.0, 20.0, 30.0]])


def test_network_round_seeds_are_distinct_and_reproducible() -> None:
    first = _network_round_seed(17, 1)
    assert first == _network_round_seed(17, 1)
    assert first != _network_round_seed(17, 2)


def test_gateway_diagnostics_reset_each_round() -> None:
    gateway = GatewaySim(1, np.array([1.0], dtype=np.float32), PEWMA(), K=1)
    gateway.on_round_start(1, round_start_abs_ns=1)
    gateway.grants_issued = 4
    gateway.grants_delivered = 3
    gateway.grants_timed_out = 1
    gateway.duplicate_packets = 2
    gateway.late_after_timeout = 5
    gateway.late_after_deadline = 6

    gateway.on_round_start(2, round_start_abs_ns=2)

    assert (gateway.grants_issued, gateway.grants_delivered, gateway.grants_timed_out) == (0, 0, 0)
    assert (gateway.duplicate_packets, gateway.late_after_timeout, gateway.late_after_deadline) == (0, 0, 0)


def test_late_data_is_counted_without_entering_deadline_state() -> None:
    gateway = GatewaySim(1, np.array([1.0], dtype=np.float32), PEWMA(), D_ns=10, K=1)
    gateway.on_round_start(1, round_start_abs_ns=1)
    gateway.on_data(0, 1, 1, 0, np.zeros(1, dtype=np.float16), t_ns=11)
    assert gateway.late_after_deadline == 1
    assert gateway.arrivals.horizon_ns is None


def test_empty_heap_is_rebuilt_when_a_client_becomes_eligible() -> None:
    class DelayedPredictor:
        def reset_round(self, _ctx) -> None:
            pass

        def observe(self, _client, _y, _t_ns, _obs) -> None:
            pass

        def predict(self, inp):
            return np.full(inp.n, float(inp.t_ns >= 5_000_000), dtype=np.float32)

    gateway = GatewaySim(
        1,
        np.array([1.0], dtype=np.float32),
        DelayedPredictor(),
        D_ns=1_000_000_000,
        K=1,
    )
    gateway.on_round_start(1, round_start_abs_ns=1)
    gateway.on_ready(0, np.array([1.0], dtype=np.float32), t_ns=0)
    assert gateway.heap == []

    grants = gateway.on_tick(5_000_000)

    assert len(grants) == 1
    assert grants[0].client_id == 0


def test_round_robin_pointer_does_not_advance_when_pacing_blocks_grant() -> None:
    from fedgrant.policy.types import PolicyConfig

    gateway = GatewaySim(
        2,
        np.array([0.5, 0.5], dtype=np.float32),
        PEWMA(),
        selector="rr",
        policy_config=PolicyConfig(selector="rr", predictor="ewma"),
        D_ns=1_000_000_000,
        K=1,
    )
    gateway.on_round_start(1, round_start_abs_ns=1)
    for grant_id in range(100, 200):
        gateway.inflight.add(grant_id, 1.0)

    assert gateway.on_ready(0, np.array([1.0], dtype=np.float32), t_ns=0) == []
    assert gateway.rr_pointer == 0

    ready_gateway = GatewaySim(
        2,
        np.array([0.5, 0.5], dtype=np.float32),
        PEWMA(),
        selector="rr",
        policy_config=PolicyConfig(selector="rr", predictor="ewma"),
        D_ns=1_000_000_000,
        K=1,
    )
    ready_gateway.on_round_start(1, round_start_abs_ns=1)
    assert ready_gateway.on_ready(0, np.array([1.0], dtype=np.float32), t_ns=0)
    assert ready_gateway.rr_pointer == 1


def test_duplicate_data_packets_contribute_to_delivered_rate() -> None:
    from fedgrant.policy.constants import S

    gateway = GatewaySim(1, np.array([1.0], dtype=np.float32), PEWMA(), D_ns=1_000_000_000, K=1)
    gateway.on_round_start(1, round_start_abs_ns=1)
    gateway.on_ready(0, np.array([1.0], dtype=np.float32), t_ns=0)
    payload = np.zeros(1, dtype=np.float16)

    gateway.on_data(0, 1, 1, 0, payload, t_ns=10)
    gateway.on_data(0, 1, 1, 0, payload, t_ns=20)

    assert gateway.duplicate_packets == 1
    assert gateway.rate_windows[0].bits == 2 * 8 * S


def test_l4_and_l5_require_their_trace_files(tmp_path) -> None:
    for scenario in ("L4", "L5"):
        with pytest.raises(FileNotFoundError, match=f"scenario {scenario} requires"):
            TraceManager(num_clients=1, scenario=scenario, trace_dir=tmp_path)


def test_full_round_counts_late_packets_outside_deadline_metrics(monkeypatch) -> None:
    import fedgrant.sim.run_net as run_net
    from fedgrant.sim.links import BottleneckLink as RealBottleneckLink

    bottlenecks = []

    def track_bottleneck(**kwargs):
        link = RealBottleneckLink(**kwargs)
        bottlenecks.append(link)
        return link

    monkeypatch.setattr(run_net, "BottleneckLink", track_bottleneck)
    result = run_network_round(num_clients=20, seed=42)

    assert result["late_metrics"]["late_after_deadline"] > 0
    assert result["metrics"]["frames_sent"] < bottlenecks[0].frames_sent
    assert "late_after_deadline" not in result["metrics"]


def test_oracle_predictor_rebinds_to_each_round_links() -> None:
    config = PolicyConfig(selector="value", predictor="oracle")
    first = run_network_round(num_clients=1, policy_config=config, seed=31)
    old_links = first["gateway"].predictor.access_links
    second = run_network_round(
        round_id=2,
        num_clients=1,
        policy_config=config,
        seed=32,
        gateway=first["gateway"],
    )
    assert second["gateway"].predictor is second["gateway"].shadow_oracle
    assert second["gateway"].predictor.access_links is not old_links


def test_engine_can_drain_already_scheduled_late_arrivals() -> None:
    engine = SimulationEngine()
    seen: list[int] = []
    engine.schedule(11, EventType.GW_ARRIVE, lambda _event: seen.append(engine.now_ns))

    engine.run_until(10)
    assert seen == []
    engine.run_all()
    assert seen == [11]


def test_unsupported_predictor_does_not_silently_become_ewma() -> None:
    with pytest.raises(ValueError, match="predictor must be one of"):
        run_network_round(
            num_clients=1,
            policy_config=PolicyConfig(selector="value", predictor="unsupported"),
        )


def test_round_timing_ece_and_summary_defaults_match_spec() -> None:
    records = [
        {"round": 1, "test_accuracy": 0.5},
        {"round": 2, "test_accuracy": 0.6},
        {"round": 3, "test_accuracy": 0.7},
    ]
    assert compute_time_to_target(records, target_acc=0.6)[1] == pytest.approx(3 * (1.0 + 0.398))
    assert compute_accuracy_vs_wallclock(records)[-1][0] == pytest.approx(3 * (1.0 + 0.398))

    # At the 15-bin default, these probabilities occupy separate bins.
    assert compute_ece(np.array([0.32, 0.34]), np.array([0.0, 1.0])) == pytest.approx(0.49)

    # Every measured round contributes network metrics, including non-evaluation rounds.
    history = [
        {
            "round": 21,
            "eta": 0.2,
            "metrics": {"goodput_bps": 100.0},
            "late_metrics": {"late_after_deadline": 3},
        },
        {
            "round": 22,
            "eta": 0.8,
            "metrics": {"goodput_bps": 300.0},
            "late_metrics": {"late_after_deadline": 5},
        },
    ]
    summary = summarize_run_history(history)
    assert summary["mean_eta"] == pytest.approx(0.5)
    assert summary["mean_goodput_bps"] == pytest.approx(200.0)
    assert summary["mean_late_after_deadline"] == pytest.approx(4.0)
