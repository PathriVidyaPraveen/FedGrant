"""The simulator and gateway feature entry point must remain the same implementation."""

from __future__ import annotations

import numpy as np
import pytest

from fedgrant.core.state import ClientState, OutstandingGrant
import fedgrant.policy.features as gateway_features
import fedgrant.sim.run_net as simulator
from fedgrant.policy.features import GatewayView
from fedgrant.policy.constants import S
from fedgrant.policy.types import PolicyConfig
from fedgrant.predictors.ewma import PEWMA
from fedgrant.core.gateway import GatewayCore, SendGrant as CoreSendGrant
from fedgrant.sim.run_net import GatewaySim


def test_simulator_uses_gateway_feature_implementation() -> None:
    # This repository has no separate gateway-daemon source; both sides must
    # import the canonical gateway implementation rather than fork its logic.
    assert simulator.features48 is gateway_features.features48


def test_equivalent_simulator_and_gateway_states_have_equal_feature_values() -> None:
    """Compare the numeric features from separately built but equivalent states."""
    sim_state = ClientState(client_id=3, w=0.2)
    gateway_state = ClientState(client_id=3, w=0.2)
    for state in (sim_state, gateway_state):
        state.hist.extend((1, 0, 1, 1))
        state.ewma_fast = 0.72
        state.ewma_slow = 0.61
        state.n_resolved = 4
        state.srtt_ns = 40_000_000
        state.rttvar_ns = 5_000_000
        state.rtt_last8.extend((35_000_000, 45_000_000))
        state.t_last_del_ns = 25_000_000
        state.t_last_loss_ns = 10_000_000
        state.consec_loss = 1
        state.m = 12
        state.outstanding[77] = OutstandingGrant(
            pos=2, t_issue_ns=12_000_000, p_issue=0.4, timeout_ns=100_000_000
        )

    sim_view = GatewayView(t_ns=80_000_000, D_ns=200_000_000, P_target=1.5)
    gateway_view = GatewayView(t_ns=80_000_000, D_ns=200_000_000, P_target=1.5)
    for view in (sim_view, gateway_view):
        view.inflight.add(77, 0.4)
        view.arrivals.record(65_000_000)
        view.arrivals.record(75_000_000)

    sim_features = simulator.features48(sim_state, sim_view)
    gateway_features_value = gateway_features.features48(gateway_state, gateway_view)

    assert sim_features.shape == gateway_features_value.shape == (48,)
    assert sim_features.dtype == gateway_features_value.dtype == np.float32
    np.testing.assert_array_equal(sim_features, gateway_features_value)
    assert np.isclose(sim_features[34], 0.04)  # four resolved grants (feature 35)
    assert sim_features[39] > 0.0  # one outstanding grant (feature 40)


def test_replayed_event_log_feature_parity() -> None:
    """Replay an identical sequence of event actions through simulator and gateway state.
    
    Verifies that state transitions and features48 output match exactly step-by-step.
    """
    st_sim = ClientState(client_id=0, w=0.1)
    st_gw = ClientState(client_id=0, w=0.1)
    view_sim = GatewayView(t_ns=0, D_ns=400_000_000, P_target=4.0)
    view_gw = GatewayView(t_ns=0, D_ns=400_000_000, P_target=4.0)

    # Replay sequence: issue g1, issue g2, resolve g1 (y=1), resolve g2 (y=0)
    # Step 1: Issue g1 at t=10ms
    t1 = 10_000_000
    view_sim.t_ns = view_gw.t_ns = t1
    for st, v in [(st_sim, view_sim), (st_gw, view_gw)]:
        pos = st.next_pos()
        st.take_position(pos)
        st.outstanding[1] = OutstandingGrant(pos=pos, t_issue_ns=t1, p_issue=0.8, timeout_ns=t1 + 100_000_000)
        v.inflight.add(1, 0.8)

    f_sim_1 = simulator.features48(st_sim, view_sim)
    f_gw_1 = gateway_features.features48(st_gw, view_gw)
    np.testing.assert_array_equal(f_sim_1, f_gw_1)

    # Step 2: DATA arrives for g1 at t=35ms
    t2 = 35_000_000
    view_sim.t_ns = view_gw.t_ns = t2
    view_sim.arrivals.record(t2)
    view_gw.arrivals.record(t2)

    o_sim_1 = simulator.resolve_event(st_sim, view_sim, 1, 1, t2)
    o_gw_1 = gateway_features.resolve_event(st_gw, view_gw, 1, 1, t2)
    np.testing.assert_array_equal(o_sim_1, o_gw_1)

    f_sim_2 = simulator.features48(st_sim, view_sim)
    f_gw_2 = gateway_features.features48(st_gw, view_gw)
    np.testing.assert_array_equal(f_sim_2, f_gw_2)


@pytest.mark.parametrize("outcome", ["data", "timeout"])
def test_gatewaycore_simulator_replay_parity_and_arrival_scope(outcome: str) -> None:
    """The simulator adapter and direct GatewayCore replay share one event path."""
    records = [[], []]
    simulators = [
        GatewaySim(
            1, np.asarray([1.0], dtype=np.float32), PEWMA(), selector="value",
            policy_config=PolicyConfig(selector="value", predictor="ewma"),
            D_ns=400_000_000, K=1, L=1, d_model=1,
            rng=np.random.default_rng(17), sample_hook=records[index].append,
        )
        for index in range(2)
    ]
    for sim in simulators:
        assert isinstance(sim.core, GatewayCore)
        sim.on_round_start(1, t_ns=0, round_start_abs_ns=0)

    adapter_grants = simulators[0].on_ready(0, np.asarray([1.0], np.float32), 0)
    direct_grants = simulators[1].core.on_ready(0, np.asarray([1.0], np.float32), 0)
    assert len(adapter_grants) == len(direct_grants) == 1
    assert isinstance(direct_grants[0], CoreSendGrant)
    assert (adapter_grants[0].grant_id, adapter_grants[0].pos) == (
        direct_grants[0].grant_id, direct_grants[0].send_pos
    )

    if outcome == "data":
        t_event = 20_000_000
        simulators[0].on_data(0, adapter_grants[0].grant_id, adapter_grants[0].pos, 0,
                              np.asarray([0.25], np.float32), t_event)
        simulators[1].core.on_data(0, direct_grants[0].grant_id, direct_grants[0].send_pos, 0,
                                   np.asarray([0.25], np.float32), t_event)
        assert simulators[0].arrivals.bits_last_20ms(t_event + 1) == 8 * S
    else:
        timeout_ns = simulators[0].client_states[0].outstanding[adapter_grants[0].grant_id].timeout_ns
        simulators[0].on_tick(timeout_ns)
        simulators[1].core.on_tick(timeout_ns)
        assert simulators[0].arrivals.bits_last_20ms(timeout_ns + 1) == 0

    issued = [next(row for row in batch if row["event"] == "grant_issued") for batch in records]
    resolved = [next(row for row in batch if row["event"] == "grant_resolved") for batch in records]
    np.testing.assert_array_equal(issued[0]["x48"], issued[1]["x48"])
    assert issued[0]["x48"].shape == (48,)
    assert issued[0]["p_hat"] == issued[1]["p_hat"]
    np.testing.assert_array_equal(resolved[0]["obs"], resolved[1]["obs"])
    assert resolved[0]["obs"].shape == (8,)
    assert resolved[0]["y"] == resolved[1]["y"] == (1 if outcome == "data" else 0)
    for sim in simulators:
        assert sim.Phi == 0.0
        assert sim.client_states[0].outstanding == {}
        assert len(sim.arrivals._q) == (1 if outcome == "data" else 0)

    # A new round clears the FedGrant-only arrivals log while keeping core state alive.
    for sim in simulators:
        sim.on_deadline(400_000_000)
        sim.on_round_start(2, t_ns=0, round_start_abs_ns=1_000_000_000)
        assert len(sim.arrivals._q) == 0


def test_gatewaycore_batches_predictions_for_all_ready_clients_on_five_ms_tick() -> None:
    class BatchSpy:
        def __init__(self) -> None:
            self.batches: list[tuple[int, ...]] = []

        def reset_round(self, _ctx) -> None:
            pass

        def observe(self, _client, _y, _t_ns, _obs) -> None:
            pass

        def predict(self, inp):
            self.batches.append(tuple(int(cid) for cid in inp.client_ids))
            return np.full(inp.n, 0.5, dtype=np.float32)

    predictor = BatchSpy()
    sim = GatewaySim(
        2, np.asarray([0.5, 0.5], dtype=np.float32), predictor, selector="value",
        policy_config=PolicyConfig(selector="value", predictor="ewma"),
        D_ns=400_000_000, K=1, L=1, d_model=1, rng=np.random.default_rng(9),
    )
    sim.on_round_start(1, t_ns=0, round_start_abs_ns=0)
    sim.on_ready(0, np.asarray([1.0], np.float32), 0)
    sim.on_ready(1, np.asarray([1.0], np.float32), 0)
    predictor.batches.clear()

    sim.on_tick(5_000_000)

    assert predictor.batches == [(0, 1)]
