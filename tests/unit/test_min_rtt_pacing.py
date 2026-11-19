"""Eq. 12 min-RTT amendment (CHANGELOG 2026-11-17): RTT~ = median of per-client minimum RTT."""

from __future__ import annotations

import numpy as np
import pytest

from fedgrant.core.estimators import rtt_update
from fedgrant.core.state import ClientState
from fedgrant.policy.constants import C, S, SRTT0_NS, TAU_Q_NS
from fedgrant.policy.pacing import p_target, p_target_min_rtt, pacing_rtt_ns
from fedgrant.predictors.ewma import PEWMA
from fedgrant.sim.run_net import GatewaySim

MS = 1_000_000


def _state(cid: int, *samples_ns: int) -> ClientState:
    st = ClientState(cid, 0.5)
    for r in samples_ns:
        rtt_update(st, r)
    return st


def test_rtt_min_tracks_the_smallest_sample_and_persists_across_rounds() -> None:
    st = _state(0)
    assert st.rtt_min_ns is None
    rtt_update(st, 50 * MS)
    rtt_update(st, 30 * MS)
    rtt_update(st, 90 * MS)
    assert st.rtt_min_ns == 30 * MS
    assert st.srtt_ns > 30 * MS                       # srtt, unlike the minimum, carries the larger samples
    st.reset_round(st.K, round_start_ns=1)
    st.reset_round(st.K, round_start_ns=2)
    assert st.rtt_min_ns == 30 * MS


def test_pacing_rtt_is_the_median_of_minimums_over_clients_with_a_sample() -> None:
    a, b, c = _state(0, 40 * MS, 20 * MS), _state(1, 60 * MS), _state(2, 100 * MS, 80 * MS)
    assert pacing_rtt_ns([a, b, c]) == 60 * MS
    assert pacing_rtt_ns([a, b]) == 40 * MS           # even count: mean of the middle two
    # A client without a sample is left out, not counted as 100 ms.
    assert pacing_rtt_ns([a, _state(3)]) == 20 * MS


def test_pacing_rtt_falls_back_to_srtt0_only_when_no_client_has_a_sample() -> None:
    assert pacing_rtt_ns([_state(0), _state(1)]) == SRTT0_NS
    assert pacing_rtt_ns([]) == SRTT0_NS
    # Before any sample the amended P_target equals the unamended one at srtt0.
    assert p_target_min_rtt([_state(0)]) == pytest.approx(p_target([SRTT0_NS]))


def test_p_target_min_rtt_formula() -> None:
    st = _state(0, 30 * MS)
    assert p_target_min_rtt([st]) == pytest.approx(C * (30 * MS + TAU_Q_NS) / (8 * S * 1e9))


def test_simulator_p_target_uses_min_rtt_of_ready_clients() -> None:
    sim = GatewaySim(2, np.array([0.5, 0.5], np.float32), PEWMA(), D_ns=1_000 * MS, K=2)
    s0, s1 = sim.client_states
    rtt_update(s0, 20 * MS)
    rtt_update(s0, 200 * MS)
    rtt_update(s1, 80 * MS)
    # No client ready: every client counts.
    assert sim.get_p_target() == pytest.approx(p_target_min_rtt([s0, s1]))
    s1.set_ready(np.array([1.0, 0.5], np.float32))
    assert sim.get_p_target() == pytest.approx(p_target_min_rtt([s1]))
