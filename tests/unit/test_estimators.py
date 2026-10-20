"""Tests for fedgrant/core/estimators.py (Section 3.7, Algorithm 1)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from fedgrant.core.estimators import (
    RateWindow,
    _div_round,
    compute_cap,
    grant_timeout,
    rate_update,
    rebase_timestamps,
    rtt_update,
    update_outcome_stats,
)
from fedgrant.core.state import ClientState
from fedgrant.policy.constants import S

MS = 1_000_000
FRAME_BITS = 8 * S                        # 11 792
# With srtt = 100 ms, Eq. (11) inner term is bhat / 94 336 exactly.
UNIT_BPS = 94_336


def _st() -> ClientState:
    return ClientState(client_id=0, w=0.1, K=4)


# --------------------------------------------------------------------------- RFC 6298


def test_rtt_hand_computed_sequence() -> None:
    st = _st()
    # 1) R = 60 ms: rttvar = (3*50 + |100-60|)/4 = 47.5 ms; srtt = (7*100 + 60)/8 = 95 ms
    rtt_update(st, 60 * MS)
    assert (st.srtt_ns, st.rttvar_ns) == (95_000_000, 47_500_000)
    # 2) R = 200 ms: rttvar = (3*47.5 + 105)/4 = 61.875 ms; srtt = (7*95 + 200)/8 = 108.125 ms
    rtt_update(st, 200 * MS)
    assert (st.srtt_ns, st.rttvar_ns) == (108_125_000, 61_875_000)
    # 3) R = 108 125 003 ns: rttvar = 185 625 003 / 4 = 46 406 250.75 -> 46 406 251
    #                        srtt = 865 000 003 / 8 = 108 125 000.375 -> 108 125 000
    rtt_update(st, 108_125_003)
    assert (st.srtt_ns, st.rttvar_ns) == (108_125_000, 46_406_251)
    assert list(st.rtt_last8) == [60 * MS, 200 * MS, 108_125_003]
    assert isinstance(st.srtt_ns, int) and isinstance(st.rttvar_ns, int)


def test_rtt_first_sample_does_not_use_rfc_initialisation() -> None:
    st = _st()
    rtt_update(st, 20 * MS)
    # RFC first-sample rule would give srtt = 20 ms, rttvar = 10 ms
    assert st.srtt_ns == 90 * MS and st.rttvar_ns == 57_500_000


def test_rtt_rttvar_uses_old_srtt() -> None:
    st = _st()
    rtt_update(st, 180 * MS)
    # old srtt (100): |100-180| = 80 -> rttvar = (150+80)/4 = 57.5 ms
    # new srtt would be 110 -> (150+70)/4 = 55 ms (wrong order)
    assert st.rttvar_ns == 57_500_000 and st.srtt_ns == 110 * MS


@pytest.mark.parametrize("seed", range(5))
def test_rtt_matches_exact_rational_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    st = _st()
    srtt, rttvar = Fraction(100 * MS), Fraction(50 * MS)
    worst_s = Fraction(0)
    for r in rng.integers(1 * MS, 400 * MS, size=200).tolist():
        rtt_update(st, r)
        rttvar = Fraction(3, 4) * rttvar + Fraction(1, 4) * abs(srtt - r)
        srtt = Fraction(7, 8) * srtt + Fraction(1, 8) * r
        # rounding error is bounded: srtt <= 0.5/(1/8) = 4 ns; rttvar <= (0.5 + 4/4)/(1/4) = 6 ns
        assert abs(st.srtt_ns - srtt) <= 4 and abs(st.rttvar_ns - rttvar) <= 6
        worst_s = max(worst_s, abs(st.srtt_ns - srtt))
    assert worst_s <= 4
    assert len(st.rtt_last8) == 8


def test_div_round_ties_to_even() -> None:
    assert [_div_round(n, 4) for n in (2, 6, 10, 5, 7)] == [0, 2, 2, 1, 2]
    assert [_div_round(n, 8) for n in (4, 12, 3, 13)] == [0, 2, 0, 2]


def test_rtt_rejects_bad_sample() -> None:
    st = _st()
    with pytest.raises(ValueError):
        rtt_update(st, -1)
    with pytest.raises(TypeError):
        rtt_update(st, 0.06)  # type: ignore[arg-type]
    assert st.srtt_ns == 100 * MS and len(st.rtt_last8) == 0


# --------------------------------------------------------------------------- Eq. (10)


def test_timeout_initial_values() -> None:
    st = _st()                                     # srtt 100 ms, rttvar 50 ms
    assert grant_timeout(5 * MS, st, 1_000 * MS) == 305 * MS


def test_timeout_clipped_at_deadline() -> None:
    st = _st()
    assert grant_timeout(5 * MS, st, 250 * MS) == 250 * MS
    assert grant_timeout(0, st, 300 * MS) == 300 * MS          # exactly D
    assert grant_timeout(0, st, 300 * MS + 1) == 300 * MS


def test_timeout_10ms_floor() -> None:
    st = _st()
    st.srtt_ns, st.rttvar_ns = 20 * MS, 1 * MS                 # 4*rttvar = 4 ms < 10 ms
    assert grant_timeout(7 * MS, st, 1_000 * MS) == 37 * MS
    st.rttvar_ns = 2_500_000                                   # 4*rttvar = 10 ms exactly
    assert grant_timeout(7 * MS, st, 1_000 * MS) == 37 * MS
    st.rttvar_ns = 2_500_001
    assert grant_timeout(7 * MS, st, 1_000 * MS) == 37 * MS + 4


# --------------------------------------------------------------------------- Eq. (11)


def test_cap_initial_is_45() -> None:
    assert compute_cap(_st()) == 45


def test_cap_floor_of_4() -> None:
    st = _st()
    st.bhat_bps = 0.0
    assert compute_cap(st) == 4                    # ceil(0) + 2 = 2 -> 4
    st.bhat_bps = float(UNIT_BPS)                  # inner term exactly 1 -> 3 -> 4
    assert compute_cap(st) == 4
    st.bhat_bps = float(2 * UNIT_BPS)              # exactly 2 -> 4
    assert compute_cap(st) == 4
    st.bhat_bps = float(2 * UNIT_BPS + 1)          # just above 2 -> 3 + 2 = 5
    assert compute_cap(st) == 5


def test_cap_exact_integer_boundary() -> None:
    st = _st()
    st.bhat_bps = float(42 * UNIT_BPS)             # inner term exactly 42
    assert compute_cap(st) == 44
    st.bhat_bps = float(42 * UNIT_BPS) + 1e-6
    assert compute_cap(st) == 45


def test_cap_matches_float_formula_off_boundaries() -> None:
    rng = np.random.default_rng(1)
    st = _st()
    for _ in range(500):
        st.bhat_bps = float(rng.uniform(1e5, 1e7))
        st.srtt_ns = int(rng.integers(1 * MS, 500 * MS))
        raw = 1.25 * st.bhat_bps * (st.srtt_ns / 1e9) / (8 * S)
        if abs(raw - round(raw)) < 1e-6:
            continue
        assert compute_cap(st) == max(4, int(np.ceil(raw)) + 2)


def test_cap_rejects_bad_bhat() -> None:
    st = _st()
    for bad in (float("nan"), float("inf"), -1.0):
        st.bhat_bps = bad
        with pytest.raises(ValueError):
            compute_cap(st)


# --------------------------------------------------------------------------- rate rule


def test_rate_rule_backlogged_branch() -> None:
    st = _st()
    R = rate_update(st, 20 * FRAME_BITS, backlogged=True)
    assert R == 2_358_400.0
    assert st.bhat_bps == pytest.approx(0.7 * 4e6 + 0.3 * 2_358_400, rel=0, abs=1e-6)  # 3 507 520
    assert st.cap == 40                            # ceil(37.18) + 2


def test_rate_rule_not_backlogged_takes_max() -> None:
    st = _st()
    rate_update(st, 10 * FRAME_BITS, backlogged=False)        # R = 1.1792 Mbit/s < bhat
    assert st.bhat_bps == 4e6 and st.cap == 45
    R = rate_update(st, 100 * FRAME_BITS, backlogged=False)   # R = 11.792 Mbit/s
    assert R == 11_792_000.0 and st.bhat_bps == 11_792_000.0
    assert st.cap == 127                                      # 125 exactly + 2


def test_rate_rule_rejects_bad_bits() -> None:
    with pytest.raises(ValueError):
        rate_update(_st(), -1, True)
    with pytest.raises(TypeError):
        rate_update(_st(), 1.5, True)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- RateWindow


def test_window_time_weighted_average() -> None:
    w = RateWindow(0, outstanding=0)
    w.set_outstanding(10 * MS, 30)        # 0 for 10 ms
    w.set_outstanding(40 * MS, 10)        # 30 for 30 ms
    w.set_outstanding(90 * MS, 0)         # 10 for 50 ms, then 0 for 10 ms
    # (0*10 + 30*30 + 10*50 + 0*10) / 100 = 14
    assert w.avg_outstanding() == pytest.approx(14.0)


def test_window_close_not_backlogged_uses_max() -> None:
    st = _st()                                                 # cap 45 -> threshold 22.5
    w = RateWindow(0, outstanding=30)
    w.set_outstanding(50 * MS, 0)                              # avg 15 < 22.5
    for k in range(100):
        w.add_delivery(50 * MS + k, FRAME_BITS)
    R = w.close(st)
    assert R == 11_792_000.0 and st.bhat_bps == 11_792_000.0 and st.cap == 127
    assert w.t_start == 100 * MS and w.bits == 0


def test_window_close_backlogged_uses_ewma() -> None:
    st = _st()
    w = RateWindow(0, outstanding=45)                          # avg 45 >= 22.5
    for k in range(20):
        w.add_delivery(k * MS, FRAME_BITS)
    w.close(st)
    assert st.bhat_bps == pytest.approx(3_507_520.0, abs=1e-6) and st.cap == 40


def test_window_threshold_exactly_half_cap() -> None:
    st = _st()
    w = RateWindow(0, outstanding=45)
    w.set_outstanding(50 * MS, 0)                              # avg = 22.5 = cap/2 -> backlogged
    w.close(st)
    assert st.bhat_bps == pytest.approx(0.7 * 4e6)

    st2 = _st()
    w2 = RateWindow(0, outstanding=45)
    w2.set_outstanding(50 * MS - 1, 0)                         # 1 ns short -> not backlogged
    w2.close(st2)
    assert st2.bhat_bps == 4e6


def test_window_uses_cap_in_force_and_carries_outstanding() -> None:
    st = _st()
    st.cap = 10                                                # threshold 5
    w = RateWindow(0, outstanding=6)
    w.close(st)                                                # backlogged: bhat = 0.7*4e6
    assert st.bhat_bps == pytest.approx(2.8e6)
    assert st.cap == compute_cap(st) == 32                     # ceil(29.68) + 2
    w.close(st)                                                # next window, still 6 < 16
    assert st.bhat_bps == pytest.approx(2.8e6)                 # max(bhat, 0)


def test_advance_closes_exact_boundaries_and_half_open() -> None:
    st = _st()
    w = RateWindow(0, outstanding=0)
    w.add_delivery(30 * MS, FRAME_BITS)
    assert w.advance(st, 99 * MS) == 0
    assert w.advance(st, 100 * MS) == 1                        # boundary closes window 0
    w.add_delivery(100 * MS, FRAME_BITS)                       # belongs to [100, 200)
    assert w.bits == FRAME_BITS
    assert w.advance(st, 350 * MS) == 2                        # closes [100,200) and [200,300)
    assert w.t_start == 300 * MS and w.bits == 0


def test_window_rejects_out_of_order_and_out_of_window() -> None:
    w = RateWindow(0, outstanding=1)
    w.set_outstanding(20 * MS, 2)
    with pytest.raises(ValueError):
        w.set_outstanding(10 * MS, 3)                          # backwards
    with pytest.raises(ValueError):
        w.add_delivery(100 * MS)                               # needs advance()
    with pytest.raises(ValueError):
        w.set_outstanding(30 * MS, -1)


def test_window_b3_override() -> None:
    st = _st()
    w = RateWindow(0, outstanding=0)                           # would be "not backlogged"
    for k in range(20):
        w.add_delivery(k * MS)                                 # default 8 S bits
    w.close(st, backlogged=True)
    assert st.bhat_bps == pytest.approx(3_507_520.0, abs=1e-6)

    st2 = _st()
    w2 = RateWindow(0, outstanding=45)                         # would be "backlogged"
    assert w2.advance(st2, 100 * MS, backlogged=False) == 1
    assert st2.bhat_bps == 4e6


# --------------------------------------------------------------------------- outcome stats


def test_update_outcome_stats() -> None:
    st = _st()
    update_outcome_stats(st, 0, 5 * MS)
    assert list(st.hist) == [0]
    assert st.ewma_fast == pytest.approx(0.7 * 0.9) and st.ewma_slow == pytest.approx(0.97 * 0.9)
    assert st.consec_loss == 1 and st.t_last_loss_ns == 5 * MS and st.t_last_del_ns is None
    assert st.n_resolved == 1 and st.first_event_in_round is False

    update_outcome_stats(st, 0, 6 * MS)
    assert st.consec_loss == 2
    update_outcome_stats(st, 1, 9 * MS)
    assert st.consec_loss == 0 and st.t_last_del_ns == 9 * MS and st.t_last_loss_ns == 6 * MS
    assert list(st.hist) == [0, 0, 1] and st.n_resolved == 3

    ef, es = 0.9, 0.9
    for y in (0, 0, 1):
        ef = 0.7 * ef + 0.3 * y
        es = 0.97 * es + 0.03 * y
    assert st.ewma_fast == pytest.approx(ef, abs=1e-15)
    assert st.ewma_slow == pytest.approx(es, abs=1e-15)


def test_update_outcome_stats_hist_ring_and_bad_y() -> None:
    st = _st()
    for k in range(40):
        update_outcome_stats(st, k % 2, k)
    assert len(st.hist) == 32 and st.hist[0] == 0 and st.hist[-1] == 1 and st.n_resolved == 40
    for bad in (2, -1, True, 1.0):
        with pytest.raises(ValueError):
            update_outcome_stats(st, bad, 0)  # type: ignore[arg-type]


def test_rebase_timestamps() -> None:
    st = _st()
    rebase_timestamps(st, 10)                                  # None stays None
    assert st.t_last_del_ns is None and st.t_last_loss_ns is None
    st.t_last_del_ns, st.t_last_loss_ns, st.t_prev_event_ns = 300 * MS, 350 * MS, 360 * MS
    rebase_timestamps(st, 1_400 * MS)                          # next round starts 1.4 s later
    assert (st.t_last_del_ns, st.t_last_loss_ns) == (-1_100 * MS, -1_050 * MS)
    assert st.t_prev_event_ns == -1_040 * MS