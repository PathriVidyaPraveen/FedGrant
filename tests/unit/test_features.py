"""Tests for fedgrant/policy/features.py (Sections 4.2, 4.3)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from fedgrant.core.state import ClientState, OutstandingGrant
from fedgrant.policy.constants import C, K, S
from fedgrant.policy.features import (
    ARRIVAL_WINDOW_NS,
    FEATURE_NAMES,
    OBS_NAMES,
    ArrivalLog,
    GatewayView,
    build_pred_input,
    context16,
    features48,
    observation8,
    resolve_event,
)

MS = 1_000_000
FB = 8 * S                      # bits per DATA frame, 11 792
LOAD = 0.02 * C                 # 200 000 bits


def f32(x: float) -> np.float32:
    return np.float32(x)


def _view(t: int = 100 * MS, D: int = 400 * MS, P: float = 6.0) -> GatewayView:
    return GatewayView(t_ns=t, D_ns=D, P_target=P)


def _issue(st: ClientState, view: GatewayView, gid: int, t: int, p: float) -> None:
    pos = st.next_pos()
    st.take_position(pos)
    st.outstanding[gid] = OutstandingGrant(pos=pos, t_issue_ns=t, p_issue=p, timeout_ns=t + 300 * MS)
    view.inflight.add(gid, p)


# --------------------------------------------------------------------------- names


def test_names() -> None:
    assert len(FEATURE_NAMES) == 48 and len(set(FEATURE_NAMES)) == 48
    assert FEATURE_NAMES[0] == "y_lag1" and FEATURE_NAMES[31] == "y_lag32"
    assert FEATURE_NAMES[32] == "ewma_fast" and FEATURE_NAMES[47] == "delivered_frac"
    assert len(OBS_NAMES) == 8


# --------------------------------------------------------------------------- ArrivalLog


def test_arrival_log_half_open_and_prune() -> None:
    a = ArrivalLog()
    t = 100 * MS
    a.record(t - ARRIVAL_WINDOW_NS - 1)       # pruned once t is recorded
    a.record(t - ARRIVAL_WINDOW_NS)           # included in [t-20ms, t)
    a.record(t - 5 * MS)
    a.record(t - 5 * MS, bits=100)
    a.record(t)                               # excluded: interval is half-open
    assert a.bits_last_20ms(t) == 2 * FB + 100
    assert a.bits_in(t - 5 * MS, t + 1) == 2 * FB + 100
    assert a.bits_in(t, t) == 0
    assert a.horizon_ns == t - ARRIVAL_WINDOW_NS
    with pytest.raises(ValueError):
        a.bits_in(t - ARRIVAL_WINDOW_NS - 1, t)   # before horizon
    with pytest.raises(ValueError):
        a.record(t - 1)                           # time went backwards
    with pytest.raises(ValueError):
        a.bits_in(t, t - 1)


def test_arrival_log_empty_and_reset() -> None:
    a = ArrivalLog()
    assert a.horizon_ns is None and a.bits_last_20ms(0) == 0
    a.record(5 * MS)
    a.reset()
    assert a.bits_last_20ms(5 * MS) == 0


def test_arrival_log_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    a, log = ArrivalLog(), []
    t = 0
    for _ in range(3000):
        t += int(rng.integers(0, 3 * MS))
        b = int(rng.integers(0, 20000))
        a.record(t, b)
        log.append((t, b))
        q = t + int(rng.integers(0, 2 * MS))
        lo = q - ARRIVAL_WINDOW_NS
        if lo >= a.horizon_ns:
            assert a.bits_in(lo, q) == sum(bb for tt, bb in log if lo <= tt < q)
    assert a._sum == sum(b for _, b in a._q)


# --------------------------------------------------------------------------- features48


def _hand_state() -> tuple[ClientState, GatewayView]:
    st = ClientState(client_id=4, w=0.2)                    # K = 169, cap 45
    for y in (1, 1, 0):
        st.hist.append(y)
    st.ewma_fast, st.ewma_slow = 0.5, 0.8
    st.n_resolved = 150
    st.srtt_ns, st.rttvar_ns = 80 * MS, 20 * MS
    st.cap = 12
    for g in (1, 2, 3):
        st.outstanding[g] = OutstandingGrant(g, 0, 0.5, 1)
    st.rtt_last8.extend([40 * MS, 60 * MS, 200 * MS])
    st.t_last_del_ns, st.t_last_loss_ns = 20 * MS, None
    st.consec_loss = 40
    st.m = 13
    view = _view(t=100 * MS, D=400 * MS, P=6.0)
    view.inflight.add(1, 1.0)
    view.inflight.add(2, 0.5)
    t = view.t_ns
    view.arrivals.record(t - 20 * MS - 1)
    view.arrivals.record(t - 20 * MS)
    for k in range(3):
        view.arrivals.record(t - 10 * MS + k)
    view.arrivals.record(t)
    return st, view


def test_features48_hand_computed() -> None:
    st, view = _hand_state()
    x = features48(st, view)
    assert x.dtype == np.float32 and x.shape == (48,)
    exp = np.full(48, np.nan)
    exp[:32] = 0.5
    exp[:3] = [0, 1, 1]                       # most recent first
    exp[32], exp[33] = 0.5, 0.8               # 33, 34
    exp[34] = 1.0                             # 35: min(150,100)/100
    exp[35], exp[36] = 0.2, 0.05              # 36, 37: 80/400, 20/400
    exp[37] = 0.75                            # 38: 300/400
    exp[38] = (300 / 80) / 20                 # 39
    exp[39] = 3 / 12                          # 40
    exp[40] = 1.5 / 6.0                       # 41
    exp[41] = 4 * FB / LOAD                   # 42: 4 frames in [t-20ms, t)
    exp[42] = (200 / 80) / 5                  # 43: last RTT
    exp[43] = (100 / 80) / 5                  # 44: mean of 40, 60, 200
    exp[44] = (80 / 80) / 20                  # 45: t - t_last_del = 80 ms
    exp[45] = 1.0                             # 46: no loss yet
    exp[46] = 1.0                             # 47: min(40,32)/32
    exp[47] = 13 / 169                        # 48
    np.testing.assert_array_equal(x, exp.astype(np.float32))


def test_features48_clipping() -> None:
    st, view = _hand_state()
    view.inflight.add(9, 1.0)
    for g in range(10, 20):
        view.inflight.add(g, 1.0)             # Phi = 12.5, P_target 6 -> 2
    for k in range(40):
        view.arrivals.record(view.t_ns)       # heavy load at t ...
    view.t_ns += 1                            # ... counted in [t+1-20ms, t+1) -> 2
    st.rtt_last8.clear()
    st.rtt_last8.extend([10_000 * MS] * 8)    # ratios > 5 -> 1
    st.t_last_del_ns = -10_000 * MS           # long ago -> 1
    st.consec_loss = 7
    x = features48(st, view)
    assert x[40] == 2.0 and x[41] == 2.0
    assert x[42] == 1.0 and x[43] == 1.0 and x[44] == 1.0
    assert x[46] == f32(7 / 32)
    view.arrivals.reset()                     # new time origin: fresh arrival log
    view.t_ns = 0
    view.D_ns = 10_000 * MS                   # (D - t)/srtt huge -> 1
    assert features48(st, view)[38] == 1.0


def test_features48_no_history_defaults() -> None:
    st = ClientState(client_id=0, w=0.1)
    view = _view(t=0, D=398 * MS)
    x = features48(st, view)
    assert np.all(x[:32] == 0.5)
    assert x[32] == f32(0.9) and x[33] == f32(0.9)
    assert x[34] == 0.0
    assert x[42] == 1.0 and x[43] == 1.0      # SPEC-NOTE: no RTT sample -> 1
    assert x[44] == 1.0 and x[45] == 1.0      # "1 if none"
    assert x[39] == 0.0 and x[40] == 0.0 and x[41] == 0.0 and x[46] == 0.0 and x[47] == 0.0
    assert x[37] == 1.0


def test_hist_full_and_partial_padding() -> None:
    st = ClientState(client_id=0, w=0.1)
    for k in range(40):
        st.hist.append(k % 2)                 # last appended = 1
    x = features48(st, _view())
    assert x[0] == 1.0 and x[1] == 0.0 and np.all(np.isin(x[:32], [0.0, 1.0]))
    st.hist.clear()
    st.hist.append(0)
    x = features48(st, _view())
    assert x[0] == 0.0 and np.all(x[1:32] == 0.5)


def test_feature_ranges_random() -> None:
    rng = np.random.default_rng(3)
    for _ in range(300):
        st = ClientState(client_id=0, w=0.1)
        for _ in range(int(rng.integers(0, 40))):
            st.hist.append(int(rng.integers(0, 2)))
        st.ewma_fast, st.ewma_slow = float(rng.random()), float(rng.random())
        st.n_resolved = int(rng.integers(0, 500))
        st.srtt_ns, st.rttvar_ns = int(rng.integers(1, 500 * MS)), int(rng.integers(0, 200 * MS))
        st.cap = int(rng.integers(4, 100))
        for g in range(int(rng.integers(0, st.cap))):
            st.outstanding[g] = OutstandingGrant(1, 0, 0.5, 1)
        for _ in range(int(rng.integers(0, 12))):
            st.rtt_last8.append(int(rng.integers(0, 2000 * MS)))
        D = int(rng.integers(100, 1000)) * MS
        t = int(rng.integers(0, D + 1))
        if rng.random() < 0.5:
            st.t_last_del_ns = t - int(rng.integers(0, 5000 * MS))
        if rng.random() < 0.5:
            st.t_last_loss_ns = t - int(rng.integers(0, 5000 * MS))
        st.consec_loss = int(rng.integers(0, 100))
        st.m = int(rng.integers(0, K + 1))
        view = _view(t=t, D=D, P=float(rng.uniform(1, 100)))
        for g in range(int(rng.integers(0, 60))):
            view.inflight.add(g, float(rng.random()))
        for ta in np.sort(rng.integers(t - 20 * MS, t + 1, size=int(rng.integers(0, 50)))):
            view.arrivals.record(int(ta))
        x = features48(st, view)
        assert np.all(np.isfinite(x))
        assert np.all(np.isin(x[:32], [0.0, 0.5, 1.0]))
        for k in (32, 33, 34, 37, 38, 42, 43, 44, 45, 46, 47):
            assert 0.0 <= x[k] <= 1.0, (k, x[k])
        assert 0.0 <= x[40] <= 2.0 and 0.0 <= x[41] <= 2.0
        assert x[35] >= 0 and x[36] >= 0 and 0 <= x[39] < 1.0


def test_features48_deterministic_bitwise() -> None:
    st, view = _hand_state()
    a = features48(st, view)
    b = features48(copy.deepcopy(st), copy.deepcopy(view))
    assert a.tobytes() == b.tobytes() == features48(st, view).tobytes()


def test_context16_and_build_pred_input() -> None:
    st, view = _hand_state()
    st2 = ClientState(client_id=9, w=0.1)
    x = features48(st, view)
    np.testing.assert_array_equal(context16(x), x[32:48])
    inp = build_pred_input([st2, st], view)
    assert inp.client_ids.tolist() == [9, 4] and inp.t_ns == view.t_ns
    np.testing.assert_array_equal(inp.x48[1], x)
    np.testing.assert_array_equal(inp.ctx, context16(inp.x48))
    assert context16(inp.x48).shape == (2, 16)
    with pytest.raises(ValueError):
        context16(np.zeros(47, np.float32))


def test_view_validation() -> None:
    st = ClientState(client_id=0, w=0.1)
    with pytest.raises(ValueError):
        features48(st, _view(P=0.0))
    with pytest.raises(ValueError):
        features48(st, _view(D=0))


# --------------------------------------------------------------------------- observations / resolve_event


def _obs_setup() -> tuple[ClientState, GatewayView]:
    st = ClientState(client_id=2, w=0.3)
    st.reset_round(K, np.linspace(1, 0, K).astype(np.float32), round_start_ns=0)
    st.srtt_ns, st.rttvar_ns = 80 * MS, 20 * MS
    st.cap = 12
    st.t_prev_event_ns = 30 * MS
    view = _view(t=50 * MS, D=400 * MS, P=5.0)
    _issue(st, view, 7, 10 * MS, 0.5)
    _issue(st, view, 8, 12 * MS, 0.25)
    for k in range(3):
        view.arrivals.record(35 * MS + k)
    view.arrivals.record(50 * MS)                    # at t_e: excluded from o7
    return st, view


def test_resolve_delivery_hand_computed_and_order() -> None:
    st, view = _obs_setup()
    st.h_stale = False                               # as if P-GRU had just refreshed h
    o = resolve_event(st, view, 7, 1, 50 * MS)       # RTT = 50 - 10 = 40 ms
    exp = np.array([
        1.0,
        (40 / 80) / 5,                 # o2 uses srtt BEFORE the update (80 ms, not 75 ms)
        (20 / 80) / 20,                # o3: t_e - t_prev = 20 ms
        350 / 400,                     # o4
        1 / 12,                        # o5: grant 7 already removed, grant 8 remains
        0.25 / 5.0,                    # o6: Phi after removing 0.5
        3 * FB / LOAD,                 # o7
        1.0,                           # o8: first event of the round
    ]).astype(np.float32)
    np.testing.assert_array_equal(o, exp)
    # state after (c)
    assert st.srtt_ns == 75 * MS and st.rttvar_ns == 25 * MS       # (7*80+40)/8, (3*20+40)/4
    assert list(st.rtt_last8) == [40 * MS]
    assert list(st.hist) == [1] and st.n_resolved == 1 and st.consec_loss == 0
    assert st.t_last_del_ns == 50 * MS and st.t_prev_event_ns == 50 * MS
    assert st.first_event_in_round is False and st.h_stale is True
    assert len(st.evbuf) == 1 and st.evbuf[0].tobytes() == o.tobytes()
    assert set(st.outstanding) == {8} and view.Phi == 0.25 and 7 not in view.inflight


def test_resolve_timeout_after_delivery() -> None:
    st, view = _obs_setup()
    resolve_event(st, view, 7, 1, 50 * MS)
    srtt_before = st.srtt_ns                          # 75 ms
    o = resolve_event(st, view, 8, 0, 65 * MS)
    assert o[0] == 0.0 and o[1] == 1.0
    assert o[2] == f32((15 / 75) / 20)
    assert o[4] == 0.0 and o[5] == 0.0 and o[7] == 0.0
    assert st.srtt_ns == srtt_before and list(st.rtt_last8) == [40 * MS]
    assert st.consec_loss == 1 and st.t_last_loss_ns == 65 * MS and st.n_resolved == 2
    assert view.Phi == 0.0 and not st.outstanding and len(st.evbuf) == 2


def test_o3_first_event_ever_is_1() -> None:
    st, view = _obs_setup()
    st.t_prev_event_ns = None
    assert resolve_event(st, view, 7, 1, 50 * MS)[2] == 1.0


def test_o5_o6_are_after_removal() -> None:
    st, view = _obs_setup()
    o = resolve_event(st, view, 8, 0, 50 * MS)
    assert o[4] == f32(1 / 12) and o[5] == f32(0.5 / 5.0)   # grant 7 remains


def test_o4_clipped_and_o2_clipped() -> None:
    st, view = _obs_setup()
    o = resolve_event(st, view, 7, 1, 450 * MS, rtt_ns=10_000 * MS)
    assert o[3] == 0.0 and o[1] == 1.0


def test_explicit_rtt_is_used() -> None:
    st, view = _obs_setup()
    o = resolve_event(st, view, 7, 1, 50 * MS, rtt_ns=16 * MS)
    assert o[1] == f32((16 / 80) / 5) and list(st.rtt_last8) == [16 * MS]


def test_observation8_direct_matches_resolve() -> None:
    st, view = _obs_setup()
    st2, view2 = copy.deepcopy(st), copy.deepcopy(view)
    del st2.outstanding[7]
    view2.inflight.remove(7)
    direct = observation8(st2, view2, 1, 40 * MS, 50 * MS)
    assert direct.tobytes() == resolve_event(st, view, 7, 1, 50 * MS).tobytes()


def test_resolve_rejects_bad_calls_without_mutating() -> None:
    st, view = _obs_setup()
    snap = (dict(st.outstanding), view.Phi, st.srtt_ns, list(st.hist))
    with pytest.raises(KeyError):
        resolve_event(st, view, 99, 1, 50 * MS)
    with pytest.raises(ValueError):
        resolve_event(st, view, 7, 0, 50 * MS, rtt_ns=5)
    with pytest.raises(ValueError):
        resolve_event(st, view, 7, 2, 50 * MS)
    with pytest.raises(ValueError):
        resolve_event(st, view, 7, 1, 5 * MS)            # negative RTT
    with pytest.raises(ValueError):
        resolve_event(st, view, 7, 1, 30 * MS)           # behind the arrival log
    view.inflight.remove(8)
    with pytest.raises(KeyError):
        resolve_event(st, view, 8, 1, 50 * MS)           # missing from Phi
    view.inflight.add(8, 0.25)
    assert (dict(st.outstanding), view.Phi, st.srtt_ns, list(st.hist)) == snap


def test_resolve_deterministic_bitwise() -> None:
    a_st, a_v = _obs_setup()
    b_st, b_v = _obs_setup()
    oa = [resolve_event(a_st, a_v, 7, 1, 50 * MS), resolve_event(a_st, a_v, 8, 0, 70 * MS)]
    ob = [resolve_event(b_st, b_v, 7, 1, 50 * MS), resolve_event(b_st, b_v, 8, 0, 70 * MS)]
    assert all(x.tobytes() == y.tobytes() for x, y in zip(oa, ob))
    assert features48(a_st, a_v).tobytes() == features48(b_st, b_v).tobytes()


def test_obs_ranges_random() -> None:
    rng = np.random.default_rng(5)
    for trial in range(200):
        st = ClientState(client_id=0, w=0.1)
        st.reset_round(K, np.linspace(1, 0, K).astype(np.float32), round_start_ns=0)
        st.cap = int(rng.integers(4, 60))
        view = _view(t=0, D=int(rng.integers(100, 1000)) * MS, P=float(rng.uniform(1, 90)))
        t = 0
        n_first = 0
        for g in range(1, int(rng.integers(2, st.cap)) + 1):
            _issue(st, view, g, t, float(rng.random()))
        for g in list(st.outstanding):
            t += int(rng.integers(0, 40 * MS))
            if rng.random() < 0.5:
                view.arrivals.record(t)
            y = int(rng.integers(0, 2))
            o = resolve_event(st, view, g, y, t)
            assert np.all(np.isfinite(o)) and o.dtype == np.float32
            assert o[0] in (0.0, 1.0) and o[7] in (0.0, 1.0)
            for k in (1, 2, 3, 4):
                assert 0.0 <= o[k] <= 1.0
            assert 0.0 <= o[5] <= 2.0 and 0.0 <= o[6] <= 2.0
            n_first += int(o[7])
        assert view.Phi == 0.0 and not st.outstanding
        assert n_first == 1                                        # exactly one first-in-round
        assert len(st.evbuf) == min(16, st.n_resolved)             # ring buffer of T = 16


def test_first_event_flag_resets_each_round_and_timestamps_shift() -> None:
    st, view = _obs_setup()
    resolve_event(st, view, 7, 1, 50 * MS)
    resolve_event(st, view, 8, 0, 60 * MS)
    st.reset_round(K, np.linspace(1, 0, K).astype(np.float32), round_start_ns=1_400 * MS)  # 1.4 s later
    assert st.t_prev_event_ns == 60 * MS - 1_400 * MS
    view2 = _view(t=0, D=400 * MS, P=5.0)
    _issue(st, view2, 1, 0, 0.9)
    o = resolve_event(st, view2, 1, 0, 40 * MS)
    assert o[7] == 1.0
    srtt = st.srtt_ns
    assert o[2] == f32(min((40 * MS - (60 * MS - 1_400 * MS)) / srtt, 20) / 20)


# --------------------------------------------------------------------------- multi-round integration


def test_multi_round_timestamps_match_absolute_clock_reference() -> None:
    """Several rounds with idle gaps between them (T_c + D): features 45/46 and o3,
    computed from round-relative times after reset_round(round_start_ns=...), must
    equal an independent reference computed on one absolute clock."""
    rng = np.random.default_rng(11)
    st = ClientState(client_id=1, w=0.2)
    D = 398 * MS
    start = 7_000 * MS                                        # absolute clock, arbitrary origin
    ref_last_del = ref_last_loss = ref_prev = None             # absolute times
    gid = 0
    for rnd in range(6):
        st.reset_round(K, np.linspace(1, 0, K).astype(np.float32), round_start_ns=start)
        view = _view(t=0, D=D, P=8.0)
        t = 0
        for _ in range(int(rng.integers(0 if rnd == 2 else 3, 12))):   # round 2 has no events
            t += int(rng.integers(1 * MS, 30 * MS))
            if t >= D:
                break
            view.t_ns = t
            # features at a prediction time t vs. the absolute-clock reference
            x = features48(st, view)
            srtt = max(st.srtt_ns, 1)
            abs_t = start + t
            exp45 = 1.0 if ref_last_del is None else min((abs_t - ref_last_del) / srtt, 20) / 20
            exp46 = 1.0 if ref_last_loss is None else min((abs_t - ref_last_loss) / srtt, 20) / 20
            assert x[44] == np.float32(exp45) and x[45] == np.float32(exp46)
            # one grant resolved at t
            gid += 1
            _issue(st, view, gid, max(t - 5 * MS, 0), 0.5)
            y = int(rng.integers(0, 2))
            srtt_before = max(st.srtt_ns, 1)
            o = resolve_event(st, view, gid, y, t)
            exp_o3 = 1.0 if ref_prev is None else min((abs_t - ref_prev) / srtt_before, 20) / 20
            assert o[2] == np.float32(exp_o3)
            ref_prev = abs_t
            if y == 1:
                ref_last_del = abs_t
            else:
                ref_last_loss = abs_t
        start += 1_000 * MS + D + int(rng.integers(0, 50 * MS))    # T_c + D + jitter
    assert ref_prev is not None and st.n_resolved > 10