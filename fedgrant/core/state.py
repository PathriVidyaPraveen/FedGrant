"""Per-client gateway state (Table 2) and the position transitions of Section 3.10.

Send positions are 1-based (Section 8); arrays are indexed pos - 1.
Per-round fields are reset by reset_round(); estimator fields persist (Table 2).
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from fedgrant.encoder.blocks import values as _encoder_values
from fedgrant.policy.constants import (
    CAP0,
    EWMA_INIT,
    BHAT0_BPS,
    HIST_LEN,
    K as K_DEFAULT,
    RTT_LAST_LEN,
    RTTVAR0_NS,
    SRTT0_NS,
    T_EVBUF,
)


class Status(IntEnum):
    """Per-position status (Table 2)."""

    UNSENT = 0
    OUTSTANDING = 1
    DELIVERED = 2
    LOST = 3


_UNSENT = int(Status.UNSENT)
_OUTSTANDING = int(Status.OUTSTANDING)
_DELIVERED = int(Status.DELIVERED)
_LOST = int(Status.LOST)


@dataclass(frozen=True, slots=True)
class OutstandingGrant:
    """Entry of the `outstanding` map (Table 2): grant_id -> this record."""

    pos: int            # send position s (1-based)
    t_issue_ns: int     # t_g
    p_issue: float      # p_hat at issue, used in Phi (Section 3.8)
    timeout_ns: int     # tau_g, Eq. (10)


def values_from_energies(energies: np.ndarray, w: float) -> np.ndarray:
    """Eq. (2): v_i(s) = w_i^2 * eps_{i,pi_i(s)}, as float32 [K] (Table 2 `val`).

    `energies` are the READY energies in send order (float32 on the wire, Section 8).
    Same arithmetic as fedgrant.encoder.blocks.values (formed in float64, rounded
    once to float32), so client and gateway values are bit-identical.
    """
    return _encoder_values(energies, w)


@dataclass(slots=True, eq=False)
class ClientState:
    """All gateway state for one client (Table 2), plus n_resolved, ready, w,
    first_event_in_round, t_prev_event_ns and cap.

    Construct with ClientState(client_id, w[, K]); the per-round fields start at
    their initial values. Call reset_round(K, round_start_ns=...) at every ROUND_START.
    """

    client_id: int
    w: float                                   # w_i = n_i / sum_j n_j (Section 3.1)
    K: int = K_DEFAULT

    # ---- per-round (reset by reset_round) ----------------------------------
    val: np.ndarray = field(init=False)        # float32 [K], non-increasing
    status: np.ndarray = field(init=False)     # uint8 [K], Status values
    lost_heap: list[int] = field(init=False)   # heapq min-heap of LOST positions (lazy)
    unsent_ptr: int = field(init=False)        # smallest UNSENT position, or K+1
    outstanding: dict[int, OutstandingGrant] = field(init=False)
    m: int = field(init=False)                 # delivered blocks this round
    ready: bool = field(init=False)            # READY received this round, rule (i)
    first_event_in_round: bool = field(init=False)   # for observation o8 (Section 4.2)

    # ---- persistent (Table 2 initial values) --------------------------------
    srtt_ns: int = field(init=False, default=SRTT0_NS)
    rttvar_ns: int = field(init=False, default=RTTVAR0_NS)
    bhat_bps: float = field(init=False, default=BHAT0_BPS)
    hist: deque = field(init=False, default_factory=lambda: deque(maxlen=HIST_LEN))
    ewma_fast: float = field(init=False, default=EWMA_INIT)
    ewma_slow: float = field(init=False, default=EWMA_INIT)
    rtt_last8: deque = field(init=False, default_factory=lambda: deque(maxlen=RTT_LAST_LEN))
    evbuf: deque = field(init=False, default_factory=lambda: deque(maxlen=T_EVBUF))
    # SPEC-NOTE: Table 2 gives no initial value for h / h_stale. h=None with
    # h_stale=True means "not yet computed"; with no events Eq. (14) gives h = 0.
    h: np.ndarray | None = field(init=False, default=None)
    h_stale: bool = field(init=False, default=True)
    t_last_del_ns: int | None = field(init=False, default=None)    # None = no delivery yet
    t_last_loss_ns: int | None = field(init=False, default=None)   # None = no loss yet
    t_prev_event_ns: int | None = field(init=False, default=None)  # last resolution time (o3); None = no event yet
    round_start_ns: int | None = field(init=False, default=None)   # absolute start of the current round
    consec_loss: int = field(init=False, default=0)
    starve: int = field(init=False, default=0)
    n_resolved: int = field(init=False, default=0)                 # feature 35
    cap: int = field(init=False, default=CAP0)                     # Eq. (11) at initial estimates

    def __post_init__(self) -> None:
        if isinstance(self.client_id, bool) or not isinstance(self.client_id, (int, np.integer)):
            raise TypeError("client_id must be an int")
        self.client_id = int(self.client_id)
        w = float(self.w)
        if not np.isfinite(w) or not 0.0 <= w <= 1.0:
            raise ValueError(f"w must be in [0, 1], got {self.w!r}")
        self.w = w
        self._reset_per_round(self.K)

    # ------------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------------
    def reset_round(
        self, K: int, values: np.ndarray | None = None, *, round_start_ns: int
    ) -> None:
        """ROUND_START (Algorithm 1, RoundStart): reset every per-round field of Table 2
        and move the persistent timestamps onto the new round's clock.

        round_start_ns: absolute start time of this round on ONE clock shared by
        all rounds (simulator time, or CLOCK_MONOTONIC in Mininet). It must
        strictly increase from round to round.

        All times inside a round are relative to its start. t_last_del,
        t_last_loss and t_prev_event persist across rounds (Table 2; the event
        sequence persists, Section 4.2), so they are shifted by
        -(round_start - previous round_start) here, which keeps features 45/46
        and observation o3 correct. The argument is required so this cannot be
        skipped. Persistent estimator values are otherwise untouched.

        If `values` (val, float32 [K]) is given, the client is also marked ready,
        as by set_ready().
        """
        if isinstance(K, (bool, np.bool_)) or not isinstance(K, (int, np.integer)) or int(K) < 1:
            raise ValueError(f"K must be a positive int, got {K!r}")
        if isinstance(round_start_ns, (bool, np.bool_)) or not isinstance(round_start_ns, (int, np.integer)):
            raise TypeError("round_start_ns must be an int (ns)")
        start = int(round_start_ns)
        prev = self.round_start_ns
        if prev is not None and start <= prev:
            raise ValueError(
                f"round_start_ns must strictly increase across rounds: {start} <= previous {prev}"
            )
        if prev is not None:
            off = start - prev
            if self.t_last_del_ns is not None:
                self.t_last_del_ns -= off
            if self.t_last_loss_ns is not None:
                self.t_last_loss_ns -= off
            if self.t_prev_event_ns is not None:
                self.t_prev_event_ns -= off
        self.round_start_ns = start
        self._reset_per_round(int(K))
        if values is not None:
            self.set_ready(values)

    def _reset_per_round(self, K: int) -> None:
        """Per-round fields of Table 2 to their initial values."""
        self.K = K
        self.val = np.zeros(self.K, dtype=np.float32)
        self.status = np.zeros(self.K, dtype=np.uint8)       # all UNSENT
        self.lost_heap = []
        self.unsent_ptr = 1
        self.outstanding = {}
        self.m = 0
        self.ready = False
        self.first_event_in_round = True

    def set_ready(self, values: np.ndarray) -> None:
        """Algorithm 1, OnReady: val <- w^2 eps (pass values_from_energies(...)); mark ready.

        Raises ValueError if values are not K finite, non-negative, non-increasing
        numbers (Table 2 requires val non-increasing). The gateway should treat a
        rejected READY as a malformed message (Section 8: dropped and counted), and
        should ignore a duplicate READY in the same round.
        """
        v = np.asarray(values, dtype=np.float32)
        if v.shape != (self.K,):
            raise ValueError(f"values must have shape ({self.K},), got {v.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError("values must be finite")
        if np.any(v < 0):
            raise ValueError("values must be non-negative")
        if self.K > 1 and np.any(np.diff(v) > 0):
            raise ValueError("values must be non-increasing in send order (Table 2)")
        self.val = v.copy()
        # Algorithm 1 sets unsent_ptr <- 1 here. reset_round already did so, and
        # unsent_ptr can only have moved if DATA preceded READY (push/plan modes);
        # it is then the smallest UNSENT position and must not be reset.
        self.ready = True

    def update_starve(self) -> None:
        """Algorithm 1 line 34, at the deadline: starve <- (starve + 1) * 1{m = 0}."""
        self.starve = self.starve + 1 if self.m == 0 else 0

    # ------------------------------------------------------------------------
    # Position selection and transitions (Section 3.10)
    # ------------------------------------------------------------------------
    def _check_pos(self, pos: int) -> int:
        if isinstance(pos, bool) or not isinstance(pos, (int, np.integer)):
            raise TypeError(f"pos must be an int, got {type(pos).__name__}")
        p = int(pos)
        if not 1 <= p <= self.K:
            raise ValueError(f"pos must be in [1, {self.K}], got {p}")
        return p

    def _heap_top(self) -> int | None:
        """Valid top of lost_heap after lazy deletion, or None if empty.

        Entries whose status is not LOST are popped and ignored (Section 3.10).
        """
        heap, status = self.lost_heap, self.status
        while heap and status[heap[0] - 1] != _LOST:
            heapq.heappop(heap)
        return heap[0] if heap else None

    def next_pos(self, retransmit: bool = True) -> int:
        """s_next = min(valid top of lost_heap, unsent_ptr), or K+1 if none (Section 3.10).

        retransmit=False (ablation A2) ignores LOST positions: s_next = unsent_ptr.
        """
        if not retransmit:
            return self.unsent_ptr
        top = self._heap_top()
        return self.unsent_ptr if top is None else min(top, self.unsent_ptr)

    def has_pending(self, retransmit: bool = True) -> bool:
        """Eligibility rule (ii): a position with status UNSENT (or LOST, if retransmitting)."""
        return self.next_pos(retransmit) <= self.K

    def take_position(self, s: int, retransmit: bool = True) -> None:
        """Apply exactly one transition of Section 3.10 for a grant requesting s.

        s must equal next_pos(retransmit); otherwise ValueError (the spec requires s = s_next).
        - s is the top of lost_heap: pop it, status <- OUTSTANDING, unsent_ptr unchanged.
        - s == unsent_ptr: status <- OUTSTANDING, unsent_ptr <- smallest UNSENT s' > s, or K+1.
        """
        s = self._check_pos(s)
        expected = self.next_pos(retransmit)
        if s != expected:
            raise ValueError(f"take_position({s}) but s_next is {expected}")
        if retransmit and self.lost_heap and self.lost_heap[0] == s:
            heapq.heappop(self.lost_heap)
            self.status[s - 1] = _OUTSTANDING
            return
        # s == unsent_ptr (every LOST position is below unsent_ptr, so the cases are disjoint)
        self.status[s - 1] = _OUTSTANDING
        self._advance_unsent_ptr(s)

    def _advance_unsent_ptr(self, s: int) -> None:
        """unsent_ptr <- smallest position s' > s with status UNSENT, or K+1."""
        p, status, K = s + 1, self.status, self.K
        while p <= K and status[p - 1] != _UNSENT:
            p += 1
        self.unsent_ptr = p

    def mark_timeout(self, pos: int) -> bool:
        """Timeout of a grant for pos (Section 3.10).

        If status is OUTSTANDING: set LOST and push pos; return True.
        Otherwise (e.g. already DELIVERED by a late arrival) do nothing; return False.
        """
        p = self._check_pos(pos)
        if self.status[p - 1] != _OUTSTANDING:
            return False
        self.status[p - 1] = _LOST
        heapq.heappush(self.lost_heap, p)
        return True

    def mark_delivered(self, pos: int) -> bool:
        """DATA for pos arrived by D (Algorithm 1, OnData line 20).

        First delivery: status <- DELIVERED, m += 1, return True. A LOST position
        becomes DELIVERED and its heap entry is discarded lazily. A duplicate
        (already DELIVERED) changes nothing and returns False.
        """
        p = self._check_pos(pos)
        st = self.status[p - 1]
        if st == _DELIVERED:
            return False
        self.status[p - 1] = _DELIVERED
        self.m += 1
        # SPEC-NOTE: in push/plan modes (B2, B3) DATA arrives for UNSENT positions
        # with no grant. Keep the Table 2 definition "unsent_ptr = smallest UNSENT
        # position" true by advancing the pointer when it is delivered.
        if st == _UNSENT and p == self.unsent_ptr:
            self._advance_unsent_ptr(p)
        return True

    # ------------------------------------------------------------------------
    # Debug / test helper
    # ------------------------------------------------------------------------
    def check_invariants(self) -> None:
        """Assert the Section 3.10 / Table 2 invariants (O(K); for tests and debugging)."""
        st, K = self.status, self.K
        assert st.shape == (K,) and st.dtype == np.uint8
        assert np.all(st <= _LOST), "invalid status value"
        unsent = np.flatnonzero(st == _UNSENT) + 1
        expected_ptr = int(unsent[0]) if unsent.size else K + 1
        assert self.unsent_ptr == expected_ptr, (self.unsent_ptr, expected_ptr)
        lost = set((np.flatnonzero(st == _LOST) + 1).tolist())
        live = [p for p in self.lost_heap if st[p - 1] == _LOST]
        assert len(live) == len(set(live)), "duplicate live entry in lost_heap"
        assert set(live) == lost, "every LOST position must have exactly one heap entry"
        assert all(p < self.unsent_ptr for p in lost), "LOST position not below unsent_ptr"
        assert all(st[p - 1] in (_LOST, _DELIVERED) for p in self.lost_heap), "stale heap entry not DELIVERED"
        assert self.m == int(np.count_nonzero(st == _DELIVERED)), "m != #DELIVERED"
        assert self.val.shape == (K,) and self.val.dtype == np.float32