"""Tests for policy/selectors.py, pacing.py, eligibility.py, plan.py (Sections 3.4, 3.8-3.10, 10)."""

from __future__ import annotations

import heapq
import itertools
import math

import numpy as np
import pytest

from fedgrant.core.state import ClientState, OutstandingGrant
from fedgrant.policy.constants import C, S
from fedgrant.policy.eligibility import Reason, is_eligible
from fedgrant.policy.pacing import InflightTracker, median_srtt_ns, p_target, pacing_p
from fedgrant.policy.plan import (
    DeliveryRatioEWMA,
    plan_allocate,
    plan_budget,
    plan_caps,
    plan_time_ns,
)
from fedgrant.policy.selectors import (
    RoundRobinSelector,
    SRPTSelector,
    ValueSelector,
    heap_key,
    make_selector,
    value_key,
)
from fedgrant.policy.types import CandidateView, PolicyConfig

MS = 1_000_000


def _cv(ids, vals=None, und=None, rr=0) -> CandidateView:
    n = len(ids)
    return CandidateView(
        client_id=np.asarray(ids, np.int32),
        next_value=np.asarray(vals if vals is not None else [0.0] * n, np.float64),
        undelivered=np.asarray(und if und is not None else [0] * n, np.int32),
        rr_pointer=rr,
    )


def _ready_state(K: int = 5, cid: int = 0) -> ClientState:
    st = ClientState(client_id=cid, w=0.5, K=K)
    st.reset_round(K, np.arange(K, 0, -1, dtype=np.float32))     # val = K, K-1, ..., 1
    return st


# --------------------------------------------------------------------------- selectors


def test_value_selector_argmax_and_ties_lower_id() -> None:
    sel = ValueSelector()
    assert sel.select(_cv([3, 1, 2], [0.5, 0.9, 0.1])) == 1
    assert sel.select(_cv([7, 2, 5], [0.9, 0.9, 0.1])) == 2           # tie -> lower id
    assert sel.select(_cv([4, 9, 6], [0.2, math.inf, math.inf])) == 6  # +inf ties -> lower id
    with pytest.raises(ValueError):
        sel.select(_cv([]))
    with pytest.raises(ValueError):
        sel.select(_cv([1, 2], [math.nan, 0.1]))


def test_value_key_eq13_and_starvation() -> None:
    st = _ready_state(K=5)
    assert value_key(st) == 5.0                                  # val[s_next - 1], s_next = 1
    st.take_position(1)
    assert value_key(st) == 4.0
    st.mark_timeout(1)                                           # LOST pos 1 is re-requested first
    assert value_key(st) == 5.0
    st.starve = 5                                                # starve >= 5 and m == 0
    assert value_key(st) == math.inf
    st.starve = 4
    assert value_key(st) == 5.0
    st.starve = 7
    st.mark_delivered(1)                                         # m = 1 -> rule off
    assert st.m == 1 and value_key(st) == 4.0


def test_value_key_requires_pending() -> None:
    st = _ready_state(K=1)
    st.take_position(1)
    with pytest.raises(ValueError):
        value_key(st)


def test_heap_key_order() -> None:
    keys = [heap_key(0.5, 3), heap_key(0.9, 7), heap_key(0.9, 2), heap_key(math.inf, 9), heap_key(0.1, 0)]
    heapq.heapify(keys)
    order = [heapq.heappop(keys)[1] for _ in range(5)]
    assert order == [9, 2, 7, 3, 0]
    with pytest.raises(ValueError):
        heap_key(math.nan, 0)


def test_heap_and_value_selector_agree() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 12))
        ids = rng.permutation(40)[:n]
        vals = rng.integers(0, 4, n).astype(float)
        vals[rng.random(n) < 0.2] = math.inf
        heap = [heap_key(v, i) for v, i in zip(vals, ids)]
        heapq.heapify(heap)
        assert heap[0][1] == ValueSelector().select(_cv(ids, vals))


def test_round_robin_wraps_and_pointer() -> None:
    rr = RoundRobinSelector()
    assert rr.select(_cv([2, 5, 9], rr=0)) == 2 and rr.next_pointer == 3
    assert rr.select(_cv([2, 5, 9], rr=3)) == 5 and rr.next_pointer == 6
    assert rr.select(_cv([9, 2, 5], rr=5)) == 5                   # exactly the pointer
    assert rr.select(_cv([2, 5, 9], rr=10)) == 2 and rr.next_pointer == 3   # wrap
    assert rr.select(_cv([5, 2, 9], rr=9)) == 9 and rr.next_pointer == 10


def test_round_robin_cycles_through_all() -> None:
    rr, ptr, seen = RoundRobinSelector(), 0, []
    for _ in range(8):
        seen.append(rr.select(_cv([1, 3, 4, 7], rr=ptr)))
        ptr = rr.next_pointer
    assert seen == [1, 3, 4, 7, 1, 3, 4, 7]


def test_srpt_fewest_undelivered_then_lower_id() -> None:
    sel = SRPTSelector()
    assert sel.select(_cv([1, 2, 3], und=[10, 4, 7])) == 2
    assert sel.select(_cv([8, 3, 5], und=[4, 4, 9])) == 3
    assert sel.select(_cv([4], und=[0])) == 4


def test_make_selector() -> None:
    assert isinstance(make_selector("value"), ValueSelector)
    assert isinstance(make_selector("rr"), RoundRobinSelector)
    assert isinstance(make_selector("srpt"), SRPTSelector)
    assert make_selector("rr") is not make_selector("rr")
    with pytest.raises(ValueError):
        make_selector("greedy")


# --------------------------------------------------------------------------- pacing


def test_p_target_eq12() -> None:
    # C (100 ms + 5 ms) / (8 S) = 10e6 * 0.105 / 11792
    assert p_target([100 * MS] * 4) == pytest.approx(1_050_000 / 11_792, rel=1e-12)
    # median of 4 values is the mean of the two middle ones
    assert median_srtt_ns([10, 40, 20, 30]) == 25.0
    assert median_srtt_ns([30, 10, 20]) == 20.0
    assert median_srtt_ns([10, 20, 900]) == 20.0                 # skewed: median != mean
    assert median_srtt_ns([10, 20, 30, 900]) == 25.0
    assert p_target([50 * MS, 150 * MS, 100 * MS]) == pytest.approx(C * 0.105 / (8 * S))
    with pytest.raises(ValueError):
        p_target([])


def test_pacing_boundary_is_inclusive() -> None:
    tr = InflightTracker()
    tr.add(1, 0.5)
    tr.add(2, 0.25)
    assert tr.phi == 0.75
    assert tr.can_issue(0.25, 1.0) is True              # 0.75 + 0.25 == 1.0 -> allowed
    assert tr.can_issue(0.5, 1.0) is False
    assert InflightTracker().can_issue(1.0, 1.0) is True


def test_inflight_add_remove_and_exactness() -> None:
    tr = InflightTracker()
    rng = np.random.default_rng(1)
    ps = rng.random(2000)
    for g, p in enumerate(ps):
        tr.add(g, float(p))
    for g in range(0, 2000, 2):
        assert tr.remove(g) == float(ps[g])
    assert tr.phi == math.fsum(ps[1::2]) and len(tr) == 1000
    for g in range(1, 2000, 2):
        tr.remove(g)
    assert tr.phi == 0.0 and len(tr) == 0
    with pytest.raises(KeyError):
        tr.remove(5)
    tr.add(5, 0.1)
    with pytest.raises(KeyError):
        tr.add(5, 0.2)
    with pytest.raises(ValueError):
        tr.add(6, 1.5)
    tr.clear()
    assert tr.phi == 0.0 and 5 not in tr


def test_pacing_p_ablation_a3() -> None:
    assert pacing_p(0.3, PolicyConfig("value", "gru")) == 0.3
    assert pacing_p(0.3, PolicyConfig("value", "gru", pred_in_pacing=False)) == 1.0


# --------------------------------------------------------------------------- eligibility


D = 398 * MS
CFG = PolicyConfig("value", "gru")


def _eligible_state() -> ClientState:
    st = _ready_state(K=5)          # srtt 100 ms, cap 45
    return st


def test_eligible_baseline() -> None:
    assert is_eligible(_eligible_state(), 0, D, 0.9, CFG) == (True, Reason.ELIGIBLE)


def test_rule_i_ready() -> None:
    st = ClientState(0, 0.5, K=5)
    assert is_eligible(st, 0, D, 0.9, CFG) == (False, Reason.NOT_READY)


def test_rule_ii_pending_and_retransmit_ablation() -> None:
    st = _eligible_state()
    for s in range(1, 6):
        st.take_position(s)
    assert is_eligible(st, 0, D, 0.9, CFG) == (False, Reason.NO_PENDING)
    st.mark_timeout(3)                                          # LOST -> pending again
    assert is_eligible(st, 0, D, 0.9, CFG) == (True, Reason.ELIGIBLE)
    a2 = PolicyConfig("value", "gru", retransmit=False)
    assert is_eligible(st, 0, D, 0.9, a2) == (False, Reason.NO_PENDING)


def test_rule_iii_cap() -> None:
    st = _eligible_state()
    st.cap = 2
    st.outstanding[1] = OutstandingGrant(1, 0, 0.9, 10)
    assert is_eligible(st, 0, D, 0.9, CFG)[0] is True           # 1 < 2
    st.outstanding[2] = OutstandingGrant(2, 0, 0.9, 10)
    assert is_eligible(st, 0, D, 0.9, CFG) == (False, Reason.CAP_REACHED)   # 2 < 2 fails


def test_rule_iv_deadline_boundary() -> None:
    st = _eligible_state()
    t_last_ok = D - st.srtt_ns - 2 * MS                         # t + srtt + 2 ms == D
    assert is_eligible(st, t_last_ok, D, 0.9, CFG)[0] is True
    assert is_eligible(st, t_last_ok + 1, D, 0.9, CFG) == (False, Reason.TOO_LATE)


def test_rule_v_theta_and_ablation_a4() -> None:
    st = _eligible_state()
    assert is_eligible(st, 0, D, 0.05, CFG)[0] is True          # p >= theta, inclusive
    assert is_eligible(st, 0, D, 0.0499, CFG) == (False, Reason.LOW_PREDICTION)
    assert is_eligible(st, 0, D, math.nan, CFG) == (False, Reason.LOW_PREDICTION)
    a4 = PolicyConfig("value", "gru", pred_in_eligibility=False)
    assert is_eligible(st, 0, D, 0.0, a4) == (True, Reason.ELIGIBLE)


def test_rules_reported_in_order() -> None:
    st = ClientState(0, 0.5, K=5)                               # fails (i), and (v)
    assert is_eligible(st, D, D, 0.0, CFG)[1] == Reason.NOT_READY
    st = _eligible_state()
    st.cap = 0                                                  # fails (iii), (iv), (v)
    assert is_eligible(st, D, D, 0.0, CFG)[1] == Reason.CAP_REACHED


# --------------------------------------------------------------------------- plan (B3)


def test_plan_budget_and_caps_hand_computed() -> None:
    D_ = 398 * MS
    tp = plan_time_ns(D_)
    assert tp == 39_800_000
    # 10e6 * (398 - 39.8 - 100) ms / 11 792 = 218.96 -> 218
    assert plan_budget(C, D_, tp, 100 * MS) == 218
    # 1.0 * 4e6 * 0.2582 / 11 792 = 87.58 -> 87
    assert plan_caps(1.0, 4e6, D_, tp, 100 * MS) == 87
    assert plan_caps(0.5, 4e6, D_, tp, 100 * MS) == 43         # 43.79
    assert plan_budget(C, D_, tp, 400 * MS) == 0               # negative -> 0
    assert plan_caps(1.0, 4e6, D_, tp, 400 * MS) == 0
    assert plan_budget(C, D_, tp, 100.5 * MS) == 218           # half-integer median accepted


def test_plan_budget_exact_floor() -> None:
    # choose a window giving exactly 10 frames: 10 * 8 S / C s = 11.792 ms
    assert plan_budget(C, 11_792_000, 0, 0) == 10
    assert plan_budget(C, 11_791_999, 0, 0) == 9


def test_plan_allocate_simple_and_ties() -> None:
    v = {0: np.array([5, 3, 1], np.float32), 1: np.array([4, 3, 2], np.float32)}
    assert plan_allocate(v, {0: 3, 1: 3}, 3) == {0: 2, 1: 1}   # 5, 4, then 3 vs 3 -> lower id
    assert plan_allocate(v, {0: 1, 1: 3}, 4) == {0: 1, 1: 3}   # cap binds
    assert plan_allocate(v, {0: 10, 1: 10}, 100) == {0: 3, 1: 3}   # cap > K is clipped
    assert plan_allocate(v, {0: 3, 1: 3}, 0) == {0: 0, 1: 0}
    assert plan_allocate(v, {0: 0, 1: 0}, 5) == {0: 0, 1: 0}


def test_plan_allocate_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        plan_allocate({0: np.array([1, 2], np.float32)}, {0: 2}, 1)    # increasing
    with pytest.raises(ValueError):
        plan_allocate({0: np.array([2, 1], np.float32)}, {1: 2}, 1)    # client mismatch
    with pytest.raises(ValueError):
        plan_allocate({0: np.array([2, 1], np.float32)}, {0: 2}, -1)


def _brute_force(values: dict, caps: dict, M: int) -> float:
    ids = sorted(values)
    pref = {i: np.concatenate([[0.0], np.cumsum(values[i].astype(np.float64))]) for i in ids}
    ranges = [range(min(caps[i], len(values[i])) + 1) for i in ids]
    best = 0.0
    for ks in itertools.product(*ranges):
        if sum(ks) <= M:
            best = max(best, sum(pref[i][k] for i, k in zip(ids, ks)))
    return best


@pytest.mark.parametrize("seed", range(60))
def test_greedy_equals_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    N, K = int(rng.integers(1, 5)), int(rng.integers(1, 6))
    values, caps = {}, {}
    for i in range(N):
        cid = int(rng.integers(0, 50)) * 10 + i                  # distinct, unsorted ids
        raw = rng.integers(0, 5, K) if seed % 2 else rng.random(K)  # half the cases tie-heavy
        values[cid] = np.sort(raw.astype(np.float32))[::-1].copy()
        caps[cid] = int(rng.integers(0, K + 2))
    M = int(rng.integers(0, N * K + 2))
    k = plan_allocate(values, caps, M)
    assert sum(k.values()) == min(M, sum(min(caps[i], K) for i in values))
    assert all(0 <= k[i] <= min(caps[i], K) for i in values)
    got = sum(float(np.sum(values[i][: k[i]].astype(np.float64))) for i in values)
    assert got == pytest.approx(_brute_force(values, caps, M), rel=1e-12, abs=1e-12)


def test_delivery_ratio_ewma() -> None:
    e = DeliveryRatioEWMA()
    assert e.get(3) == 1.0
    assert e.update(3, 5, 10) == pytest.approx(0.9 * 1.0 + 0.1 * 0.5)
    assert e.update(3, 0, 0) == pytest.approx(0.95)            # k = 0 -> no update
    assert e.update(3, 10, 10) == pytest.approx(0.9 * 0.95 + 0.1)
    assert e.get(4) == 1.0
    with pytest.raises(ValueError):
        e.update(3, 11, 10)