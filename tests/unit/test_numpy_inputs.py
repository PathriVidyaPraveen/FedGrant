"""Every public entry point must accept NumPy scalars (np.int64 / np.int32 / np.float64 /
np.float32), as a simulator produces them, and store plain Python ints for times."""

from __future__ import annotations

import numpy as np
import pytest

from fedgrant.core.estimators import (
    RateWindow,
    check_y,
    compute_cap,
    grant_timeout,
    rate_update,
    rtt_update,
    update_outcome_stats,
)
from fedgrant.core.state import ClientState, OutstandingGrant
from fedgrant.encoder.blocks import block_payload_f16, encode
from fedgrant.encoder.ef import ErrorFeedback, bitmap_from_indices
from fedgrant.policy.constants import K
from fedgrant.policy.eligibility import is_eligible
from fedgrant.policy.features import ArrivalLog, GatewayView, features48, resolve_event
from fedgrant.policy.pacing import InflightTracker, p_target
from fedgrant.policy.plan import DeliveryRatioEWMA, plan_allocate, plan_budget, plan_caps, plan_time_ns
from fedgrant.policy.selectors import value_key
from fedgrant.policy.types import CandidateView, PolicyConfig
from fedgrant.predictors.base import RoundCtx

MS = 1_000_000
I64, I32 = np.int64, np.int32


def _ready() -> ClientState:
    st = ClientState(I64(3), np.float64(0.25))
    st.reset_round(I64(K), np.linspace(1, 0, K).astype(np.float32), round_start_ns=I64(0))
    return st


def test_estimators_accept_numpy_ints() -> None:
    st = _ready()
    rtt_update(st, I64(60 * MS))
    assert type(st.srtt_ns) is int and type(st.rttvar_ns) is int and st.srtt_ns == 95 * MS
    assert type(st.rtt_last8[-1]) is int
    tau = grant_timeout(I64(5 * MS), st, I32(398) * I64(MS))
    assert type(tau) is int
    assert type(compute_cap(st)) is int
    rate_update(st, I64(20 * 11_792), np.bool_(True))
    for y in (I64(1), I32(0), np.uint8(1)):
        update_outcome_stats(st, y, I64(9 * MS))
    assert all(type(v) is int for v in st.hist) and type(st.t_last_del_ns) is int
    assert type(st.t_last_loss_ns) is int


def test_y_rejects_bools_of_both_kinds() -> None:
    for bad in (True, np.bool_(True), 2, I64(2), 1.0, np.float64(1.0)):
        with pytest.raises(ValueError):
            check_y(bad)
    assert check_y(I64(1)) == 1 and type(check_y(I64(1))) is int


def test_rate_window_numpy_times() -> None:
    st = _ready()
    w = RateWindow(I64(0), I64(3))
    w.set_outstanding(I64(10 * MS), I32(5))
    w.add_delivery(I64(20 * MS), I64(11_792))
    assert w.advance(st, I64(100 * MS)) == 1
    assert type(w.t_start) is int


def test_state_numpy_positions() -> None:
    st = _ready()
    s = st.next_pos()
    st.take_position(I64(s))
    assert st.mark_timeout(I32(s)) is True
    st.take_position(I64(st.next_pos()))
    assert st.mark_delivered(I64(s)) is True
    st.check_invariants()
    with pytest.raises(TypeError):
        st.take_position(np.bool_(True))  # type: ignore[arg-type]


def test_features_and_resolve_numpy() -> None:
    st = _ready()
    view = GatewayView(t_ns=I64(50 * MS), D_ns=I64(400 * MS), P_target=np.float64(6.0))
    pos = st.next_pos()
    st.take_position(pos)
    st.outstanding[I64(7)] = OutstandingGrant(pos, I64(10 * MS), np.float32(0.5), I64(300 * MS))
    view.inflight.add(I64(7), np.float32(0.5))
    view.arrivals.record(I64(45 * MS), I64(11_792))
    x = features48(st, view)
    assert x.dtype == np.float32 and np.all(np.isfinite(x))
    o = resolve_event(st, view, I64(7), I64(1), I64(50 * MS))
    assert o[0] == 1.0 and not st.outstanding and view.Phi == 0.0
    assert type(st.t_prev_event_ns) is int and type(st.t_last_del_ns) is int
    assert list(st.hist) == [1] and type(st.hist[0]) is int


def test_arrival_log_numpy() -> None:
    a = ArrivalLog()
    a.record(I64(5 * MS))
    assert a.bits_last_20ms(I64(6 * MS)) == 11_792


def test_policy_modules_numpy() -> None:
    st = _ready()
    cfg = PolicyConfig("value", "gru", pred_in_pacing=np.bool_(False), explore_eps=np.float32(0.1))
    assert cfg.pred_in_pacing is False and type(cfg.explore_eps) is float
    ok, _ = is_eligible(st, I64(0), I64(398 * MS), np.float32(0.9), cfg)
    assert ok
    assert value_key(st) == 1.0
    assert p_target(np.array([100 * MS, 120 * MS], dtype=np.int64)) > 0
    tr = InflightTracker()
    tr.add(I64(1), np.float32(0.5))
    assert tr.remove(1) == 0.5                      # same key as int
    cv = CandidateView(np.array([1], np.int32), np.array([0.5]), np.array([3], np.int32), I32(2))
    assert type(cv.rr_pointer) is int


def test_plan_numpy() -> None:
    tp = plan_time_ns(I64(398 * MS))
    assert type(tp) is int and tp == 39_800_000
    assert plan_budget(I64(10_000_000), I64(398 * MS), tp, np.float64(100 * MS)) == 218
    assert plan_caps(np.float64(1.0), np.float64(4e6), I64(398 * MS), tp, I64(100 * MS)) == 87
    v = {I64(0): np.array([5, 3], np.float32), I64(1): np.array([4, 1], np.float32)}
    k = plan_allocate(v, {I64(0): I64(2), I64(1): I32(2)}, I64(3))
    assert k == {0: 2, 1: 1}
    e = DeliveryRatioEWMA()
    assert e.update(I64(0), I64(5), I64(10)) == pytest.approx(0.95)


def test_encoder_numpy() -> None:
    u = (np.random.default_rng(0).standard_normal(1500) * 1e-2).astype(np.float32)
    enc = encode(u)
    p, n = block_payload_f16(enc, I64(2))
    assert n == 100 and type(n) is int
    ef = ErrorFeedback(I64(1500))
    uu = ef.prepare(u)
    ef.finish_round(uu, bitmap_from_indices(np.array([0, 2], np.int64), I64(3)))
    assert np.array_equal(ef.reconstruct(uu, np.array([0, 2])) + ef.e, uu)


def test_roundctx_numpy() -> None:
    r = RoundCtx(I64(1), I64(398 * MS), I32(20), I64(169))
    assert type(r.D_ns) is int