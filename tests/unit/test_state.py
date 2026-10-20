"""Tests for fedgrant/core/state.py: Table 2 fields and Section 3.10 transitions."""

from __future__ import annotations

import heapq

import numpy as np
import pytest

from fedgrant.core.state import ClientState, OutstandingGrant, Status, values_from_energies
from fedgrant.policy import constants as c


def _ready(K: int = 5, w: float = 0.5) -> ClientState:
    st = ClientState(client_id=3, w=w, K=K)
    st.reset_round(K, np.arange(K, 0, -1, dtype=np.float32))
    return st


def _take_next(st: ClientState, retransmit: bool = True) -> int:
    s = st.next_pos(retransmit)
    st.take_position(s, retransmit)
    st.check_invariants()
    return s


# --------------------------------------------------------------------------- constants


def test_constants_section_3_1() -> None:
    assert (c.L, c.K, c.LAST_BLOCK_COORDS, c.D_MODEL) == (700, 169, 94, 117_694)
    assert (c.S, c.C, c.Q) == (1474, 10_000_000, 64)
    assert c.THETA == 0.05 and c.TAU_Q_NS == 5_000_000
    assert (c.SRTT0_NS, c.RTTVAR0_NS, c.BHAT0_BPS) == (100_000_000, 50_000_000, 4_000_000.0)
    assert (c.RFC6298_ALPHA, c.RFC6298_BETA) == (0.125, 0.25)
    assert (c.EWMA_FAST_W, c.EWMA_SLOW_W, c.EWMA_INIT) == (0.3, 0.03, 0.9)
    assert (c.HIST_LEN, c.RTT_LAST_LEN, c.T_EVBUF, c.GRU_HIDDEN) == (32, 8, 16, 32)


def test_cap0_matches_eq11_in_floats() -> None:
    raw = 1.25 * c.BHAT0_BPS * (c.SRTT0_NS / 1e9) / (8 * c.S)   # 42.40...
    assert c.CAP0 == max(4, int(np.ceil(raw)) + 2) == 45


# --------------------------------------------------------------------------- initial values


def test_initial_values_table2() -> None:
    st = ClientState(client_id=7, w=0.25)
    assert st.K == c.K == 169
    assert st.val.dtype == np.float32 and st.val.shape == (169,) and not st.val.any()
    assert st.status.dtype == np.uint8 and st.status.shape == (169,)
    assert np.all(st.status == Status.UNSENT)
    assert st.lost_heap == [] and st.unsent_ptr == 1 and st.outstanding == {}
    assert st.m == 0 and st.ready is False and st.first_event_in_round is True
    assert st.srtt_ns == 100_000_000 and st.rttvar_ns == 50_000_000
    assert st.bhat_bps == 4_000_000.0
    assert st.ewma_fast == 0.9 and st.ewma_slow == 0.9
    assert st.hist.maxlen == 32 and len(st.hist) == 0
    assert st.rtt_last8.maxlen == 8 and st.evbuf.maxlen == 16
    assert st.h is None and st.h_stale is True
    assert st.t_last_del_ns is None and st.t_last_loss_ns is None and st.t_prev_event_ns is None
    assert st.consec_loss == 0 and st.starve == 0 and st.n_resolved == 0
    assert st.cap == 45
    assert isinstance(st.srtt_ns, int) and isinstance(st.rttvar_ns, int)
    st.check_invariants()


def test_instances_do_not_share_buffers() -> None:
    a, b = ClientState(0, 0.5), ClientState(1, 0.5)
    a.hist.append(1)
    a.evbuf.append(np.zeros(8, np.float32))
    a.rtt_last8.append(5)
    assert len(b.hist) == len(b.evbuf) == len(b.rtt_last8) == 0


def test_no_attribute_typos_possible() -> None:
    st = ClientState(0, 0.5)
    with pytest.raises(AttributeError):
        st.srt_ns = 5  # type: ignore[attr-defined]


@pytest.mark.parametrize("w", [-0.1, 1.5, float("nan"), float("inf")])
def test_bad_weight_rejected(w: float) -> None:
    with pytest.raises(ValueError):
        ClientState(0, w)


# --------------------------------------------------------------------------- values / READY


def test_values_from_energies_eq2() -> None:
    e = np.array([4.0, 2.5, 1e-3, 0.0], dtype=np.float32)
    v = values_from_energies(e, 0.1)
    assert v.dtype == np.float32
    np.testing.assert_array_equal(v, (0.01 * e.astype(np.float64)).astype(np.float32))


def test_set_ready_validation() -> None:
    st = ClientState(0, 0.5, K=4)
    with pytest.raises(ValueError):
        st.set_ready(np.array([1, 2, 0, 0], np.float32))        # increasing
    with pytest.raises(ValueError):
        st.set_ready(np.array([3, 2, 1], np.float32))           # wrong length
    with pytest.raises(ValueError):
        st.set_ready(np.array([np.inf, 2, 1, 0], np.float32))   # non-finite
    with pytest.raises(ValueError):
        st.set_ready(np.array([3, 2, 1, -1], np.float32))       # negative
    assert st.ready is False
    st.set_ready(np.array([3, 3, 1, 0], np.float32))            # ties allowed
    assert st.ready is True and st.unsent_ptr == 1
    np.testing.assert_array_equal(st.val, [3, 3, 1, 0])


def test_reset_round_keeps_persistent_and_clears_per_round() -> None:
    st = _ready(K=4)
    _take_next(st)
    st.mark_delivered(1)
    st.outstanding[1] = OutstandingGrant(pos=2, t_issue_ns=0, p_issue=0.9, timeout_ns=10)
    st.srtt_ns, st.rttvar_ns, st.bhat_bps = 123, 45, 6e6
    st.ewma_fast, st.ewma_slow = 0.1, 0.2
    st.hist.append(1); st.rtt_last8.append(9); st.evbuf.append(np.ones(8, np.float32))
    st.h, st.h_stale = np.ones(32, np.float32), False
    st.t_last_del_ns, st.t_last_loss_ns, st.consec_loss = 11, 12, 3
    st.t_prev_event_ns = 13
    st.starve, st.n_resolved, st.cap = 2, 17, 9
    st.first_event_in_round = False

    st.reset_round(6)

    assert st.K == 6 and st.val.shape == (6,) and st.status.shape == (6,)
    assert np.all(st.status == Status.UNSENT) and st.lost_heap == []
    assert st.unsent_ptr == 1 and st.outstanding == {} and st.m == 0
    assert st.ready is False and st.first_event_in_round is True
    assert (st.srtt_ns, st.rttvar_ns, st.bhat_bps) == (123, 45, 6e6)
    assert (st.ewma_fast, st.ewma_slow) == (0.1, 0.2)
    assert list(st.hist) == [1] and list(st.rtt_last8) == [9] and len(st.evbuf) == 1
    assert st.h is not None and st.h_stale is False
    assert (st.t_last_del_ns, st.t_last_loss_ns, st.consec_loss) == (11, 12, 3)
    assert st.t_prev_event_ns == 13                     # persistent (event sequence persists)
    assert (st.starve, st.n_resolved, st.cap) == (2, 17, 9)
    st.check_invariants()


def test_reset_round_with_values_marks_ready() -> None:
    st = ClientState(0, 0.5, K=3)
    st.reset_round(3, np.array([2, 1, 0], np.float32))
    assert st.ready is True
    np.testing.assert_array_equal(st.val, [2, 1, 0])


# --------------------------------------------------------------------------- transitions


def test_unsent_progression_and_k_plus_1_sentinel() -> None:
    st = _ready(K=3)
    assert [_take_next(st) for _ in range(3)] == [1, 2, 3]
    assert st.unsent_ptr == 4 == st.K + 1
    assert st.next_pos() == 4 and not st.has_pending()
    assert np.all(st.status == Status.OUTSTANDING)


def test_take_position_rejects_anything_but_s_next() -> None:
    st = _ready(K=4)
    with pytest.raises(ValueError):
        st.take_position(2)
    with pytest.raises(ValueError):
        st.take_position(0)
    with pytest.raises(ValueError):
        st.take_position(5)
    with pytest.raises(TypeError):
        st.take_position(True)  # type: ignore[arg-type]
    _take_next(st)
    with pytest.raises(ValueError):
        st.take_position(1)                       # already taken
    st.check_invariants()


def test_lost_is_rerequested_first_and_unsent_ptr_unchanged() -> None:
    st = _ready(K=5)
    for _ in range(3):
        _take_next(st)                            # 1, 2, 3 outstanding; ptr = 4
    assert st.mark_timeout(2) is True
    assert st.status[1] == Status.LOST and st.lost_heap == [2]
    assert st.next_pos() == 2
    st.take_position(2)
    assert st.status[1] == Status.OUTSTANDING and st.unsent_ptr == 4 and st.lost_heap == []
    st.check_invariants()
    assert st.next_pos() == 4


def test_multiple_lost_served_in_position_order() -> None:
    st = _ready(K=6)
    for _ in range(5):
        _take_next(st)
    for p in (4, 1, 3):
        st.mark_timeout(p)
    st.check_invariants()
    assert [_take_next(st) for _ in range(4)] == [1, 3, 4, 6]


def test_every_lost_position_below_unsent_ptr() -> None:
    st = _ready(K=8)
    for _ in range(6):
        _take_next(st)
    for p in (6, 2, 5):
        st.mark_timeout(p)
    assert all(p < st.unsent_ptr for p in st.lost_heap)
    st.check_invariants()


def test_timeout_on_non_outstanding_is_noop() -> None:
    st = _ready(K=4)
    _take_next(st)
    st.mark_delivered(1)
    assert st.mark_timeout(1) is False and st.status[0] == Status.DELIVERED
    assert st.mark_timeout(3) is False and st.status[2] == Status.UNSENT   # UNSENT
    assert st.lost_heap == []
    st.check_invariants()


def test_late_delivery_of_lost_position_and_lazy_deletion() -> None:
    st = _ready(K=5)
    for _ in range(3):
        _take_next(st)
    st.mark_timeout(1)
    st.mark_timeout(2)
    assert sorted(st.lost_heap) == [1, 2]
    assert st.mark_delivered(1) is True                 # late DATA for a LOST position
    assert st.status[0] == Status.DELIVERED and st.m == 1
    assert 1 in st.lost_heap                            # entry discarded lazily, not eagerly
    st.check_invariants()
    assert st.next_pos() == 2                           # stale top popped and ignored
    assert st.lost_heap == [2]
    _take_next(st)
    assert st.lost_heap == [] and st.next_pos() == 4


def test_all_lost_entries_stale_falls_back_to_unsent_ptr() -> None:
    st = _ready(K=4)
    _take_next(st); _take_next(st)
    st.mark_timeout(1); st.mark_timeout(2)
    st.mark_delivered(1); st.mark_delivered(2)
    assert st.next_pos() == 3 and st.lost_heap == []
    st.check_invariants()


def test_retransmitted_position_late_original_then_duplicate() -> None:
    st = _ready(K=3)
    _take_next(st)                                      # grant g1 for pos 1
    st.mark_timeout(1)
    _take_next(st)                                      # grant g2 re-requests pos 1
    assert st.mark_delivered(1) is True                 # late DATA of g1
    assert st.mark_delivered(1) is False                # DATA of g2 is a duplicate
    assert st.m == 1
    assert st.mark_timeout(1) is False                  # timeout of g2: already DELIVERED
    st.check_invariants()


def test_m_counts_first_delivery_only() -> None:
    st = _ready(K=3)
    _take_next(st)
    assert st.mark_delivered(1) and not st.mark_delivered(1) and st.m == 1


def test_push_mode_delivery_of_unsent_positions_keeps_unsent_ptr_definition() -> None:
    st = _ready(K=5)
    assert st.mark_delivered(2) is True                 # beyond unsent_ptr
    assert st.unsent_ptr == 1
    assert st.mark_delivered(1) is True                 # at unsent_ptr -> skips delivered 2
    assert st.unsent_ptr == 3 and st.m == 2
    st.check_invariants()
    for p in (3, 4, 5):
        st.mark_delivered(p)
    assert st.unsent_ptr == 6 and not st.has_pending()
    st.check_invariants()


def test_retransmit_false_ablation_a2() -> None:
    st = _ready(K=3)
    _take_next(st, retransmit=False)
    st.mark_timeout(1)
    assert st.next_pos(retransmit=False) == 2
    assert st.next_pos(retransmit=True) == 1
    _take_next(st, retransmit=False)
    _take_next(st, retransmit=False)
    assert not st.has_pending(retransmit=False)
    assert st.has_pending(retransmit=True)              # pos 1 still LOST
    st.check_invariants()


def test_update_starve() -> None:
    st = _ready(K=2)
    for expected in (1, 2, 3):
        st.update_starve()
        assert st.starve == expected
    _take_next(st)
    st.mark_delivered(1)
    st.update_starve()
    assert st.starve == 0


def test_status_enum_values() -> None:
    assert [int(s) for s in Status] == [0, 1, 2, 3]
    assert [s.name for s in Status] == ["UNSENT", "OUTSTANDING", "DELIVERED", "LOST"]


def test_outstanding_grant_record() -> None:
    g = OutstandingGrant(pos=3, t_issue_ns=10, p_issue=0.8, timeout_ns=20)
    assert (g.pos, g.t_issue_ns, g.p_issue, g.timeout_ns) == (3, 10, 0.8, 20)
    with pytest.raises(AttributeError):
        g.pos = 4  # type: ignore[misc]


# --------------------------------------------------------------------------- randomized model check


def _reference_next(status: np.ndarray, retransmit: bool) -> int:
    allowed = (Status.UNSENT, Status.LOST) if retransmit else (Status.UNSENT,)
    cand = [i + 1 for i, s in enumerate(status) if s in allowed]
    return min(cand) if cand else len(status) + 1


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("retransmit", [True, False])
def test_randomized_against_reference(seed: int, retransmit: bool) -> None:
    """Random take/timeout/deliver sequences; s_next must equal the brute-force
    minimum over UNSENT (and LOST) positions, and all invariants must hold."""
    rng = np.random.default_rng(seed)
    K = int(rng.integers(1, 30))
    st = _ready(K=K)
    for _ in range(400):
        op = rng.integers(0, 3)
        if op == 0 and st.has_pending(retransmit):
            assert st.next_pos(retransmit) == _reference_next(st.status, retransmit)
            _take_next(st, retransmit)
        elif op == 1:
            p = int(rng.integers(1, K + 1))
            before = int(st.status[p - 1])
            changed = st.mark_timeout(p)
            assert changed == (before == Status.OUTSTANDING)
        else:
            p = int(rng.integers(1, K + 1))
            before = int(st.status[p - 1])
            first = st.mark_delivered(p)
            assert first == (before != Status.DELIVERED)
        st.check_invariants()
        assert st.next_pos(retransmit) == _reference_next(st.status, retransmit)
        # the heap's valid top is always the smallest LOST position
        lost = [i + 1 for i, s in enumerate(st.status) if s == Status.LOST]
        top = st._heap_top()
        assert top == (min(lost) if lost else None)
        assert heapq.nsmallest(1, st.lost_heap) == st.lost_heap[:1]