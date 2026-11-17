"""Per-client estimators operating on ClientState (Section 3.7, Table 2, Algorithm 1).

All times are integer ns. srtt/rttvar stay integers (rounded to the nearest ns);
bhat is a float in bit/s. Nothing here imports teammates' modules.
"""

from __future__ import annotations

from fractions import Fraction
from math import ceil

import numpy as np

from fedgrant.core.state import ClientState
from fedgrant.policy.constants import (
    CAP_HEADROOM,
    CAP_MIN,
    EWMA_FAST_W,
    EWMA_SLOW_W,
    RFC6298_ALPHA,
    RFC6298_BETA,
    S,
)

RTO_FLOOR_NS: int = 10_000_000            # 10 ms in Eq. (10)
RATE_WINDOW_NS: int = 100_000_000         # 100 ms rate window (Section 3.7)
RATE_OLD_W: float = 0.7                   # bhat <- 0.7 bhat + 0.3 R
RATE_NEW_W: float = 0.3

# alpha = 1/8 and beta = 1/4 are applied in exact integer arithmetic below.
assert RFC6298_ALPHA == 1 / 8 and RFC6298_BETA == 1 / 4


def _check_int(x: object, name: str) -> int:
    """Accept Python and NumPy integers (not bools); return a Python int."""
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(x).__name__}")
    return int(x)


def check_y(y: object) -> int:
    """Validate an outcome label y in {0, 1} (Python or NumPy int, not bool); return int."""
    if isinstance(y, (bool, np.bool_)) or not isinstance(y, (int, np.integer)) or int(y) not in (0, 1):
        raise ValueError(f"y must be 0 or 1, got {y!r}")
    return int(y)


def _div_round(num: int, den: int) -> int:
    """num/den rounded to the nearest integer, ties to even (num >= 0, den > 0)."""
    q, r = divmod(num, den)
    if 2 * r > den or (2 * r == den and q % 2 == 1):
        q += 1
    return q


# --------------------------------------------------------------------------- RTT (RFC 6298)


def rtt_update(state: ClientState, sample_ns: int) -> None:
    """RFC 6298 update with alpha = 1/8, beta = 1/4 (Section 3.7), append to rtt_last8,
    and fold the sample into rtt_min_ns (Eq. 12 min-RTT amendment).

    rttvar <- (1 - beta) rttvar + beta |srtt - R|   (uses the OLD srtt)
    srtt   <- (1 - alpha) srtt + alpha R
    Each result is rounded to the nearest ns (ties to even).

    SPEC-NOTE: the RFC 6298 first-sample rule (srtt <- R, rttvar <- R/2) is NOT
    applied; Table 2 gives initial values 100 ms / 50 ms, so every sample,
    including the first, uses the update rule above.
    SPEC-NOTE: srtt/rttvar are kept as integer ns (project rule). Rounding adds at
    most 0.5 ns per update; the accumulated deviation from exact arithmetic stays
    below 4 ns (srtt) and 6 ns (rttvar).
    """
    r = _check_int(sample_ns, "sample_ns")
    if r < 0:
        raise ValueError(f"RTT sample must be >= 0, got {r}")
    srtt, rttvar = state.srtt_ns, state.rttvar_ns
    state.rttvar_ns = _div_round(3 * rttvar + abs(srtt - r), 4)
    state.srtt_ns = _div_round(7 * srtt + r, 8)
    state.rtt_last8.append(r)
    if state.rtt_min_ns is None or r < state.rtt_min_ns:
        state.rtt_min_ns = r


def grant_timeout(t_g_ns: int, state: ClientState, D_ns: int) -> int:
    """Eq. (10): tau_g = min(D, t_g + srtt_i + max(4 rttvar_i, 10 ms)), in ns."""
    t_g = _check_int(t_g_ns, "t_g_ns")
    D = _check_int(D_ns, "D_ns")
    return min(D, t_g + state.srtt_ns + max(4 * state.rttvar_ns, RTO_FLOOR_NS))


# --------------------------------------------------------------------------- window (Eq. 11)


def compute_cap(state: ClientState) -> int:
    """Eq. (11): cap_i = max(4, ceil(1.25 * bhat_i * srtt_i / (8 S)) + 2), srtt in seconds.

    Evaluated in exact rational arithmetic on the float bhat, so the ceiling is
    never off by one at an exact integer boundary.
    """
    bhat = state.bhat_bps
    if not (bhat >= 0.0 and bhat != float("inf")):
        raise ValueError(f"bhat must be finite and >= 0, got {bhat!r}")
    srtt = _check_int(state.srtt_ns, "srtt_ns")
    # 1.25 * bhat * (srtt_ns / 1e9) / (8 S) = 5 * bhat * srtt_ns / (32 * S * 1e9)
    x = Fraction(5) * Fraction(bhat) * srtt / (32 * S * 1_000_000_000)
    return max(CAP_MIN, ceil(x) + CAP_HEADROOM)


# --------------------------------------------------------------------------- rate estimate


def rate_update(state: ClientState, bits: int, backlogged: bool) -> float:
    """Section 3.7 rate rule for one closed 100 ms window; also recomputes cap (Eq. 11).

    R = bits / 0.1 s. backlogged: bhat <- 0.7 bhat + 0.3 R; otherwise bhat <- max(bhat, R).
    Returns R in bit/s.
    """
    b = _check_int(bits, "bits")
    if b < 0:
        raise ValueError(f"bits must be >= 0, got {b}")
    R = b * 1_000_000_000 / RATE_WINDOW_NS          # = bits / 0.1 s
    if backlogged:
        state.bhat_bps = RATE_OLD_W * state.bhat_bps + RATE_NEW_W * R
    else:
        state.bhat_bps = max(state.bhat_bps, R)
    state.cap = compute_cap(state)
    return R


class RateWindow:
    """Bits delivered and time-integrated outstanding count of one client over the
    current 100 ms window [t_start, t_start + 100 ms) (Section 3.7).

    Usage (gateway, per client): start(t0, outstanding) at round start; before
    handling any event at time t call advance(state, t), which closes every
    window whose end is <= t at its exact boundary; then record the event with
    add_delivery / set_outstanding. Windows are half-open, so an event at
    exactly a boundary belongs to the new window.

    SPEC-NOTE: "on average at least cap_i/2 outstanding grants" is read as the
    time-weighted average of outstanding_i over the window, compared with the
    cap_i in force during it (cap only changes when a window closes, so it is
    constant within a window). Compared exactly: 2 * integral >= cap * 100 ms.

    SPEC-NOTE: "bits delivered" is whatever the caller passes to add_delivery.
    Recommended: 8 S per DATA frame received from the client by D (a frame that
    crossed the path counts even if it is a duplicate).

    B3 (plan mode, Section 10): pass backlogged=... to close()/advance(); the
    outstanding average is then ignored.
    """

    __slots__ = ("t_start", "_t_last", "_n_out", "_integral", "bits")

    def __init__(self, t_start_ns: int = 0, outstanding: int = 0) -> None:
        self.start(t_start_ns, outstanding)

    @property
    def t_end(self) -> int:
        return self.t_start + RATE_WINDOW_NS

    def start(self, t_start_ns: int, outstanding: int = 0) -> None:
        """Begin a fresh window at t_start with the given outstanding count."""
        self.t_start = _check_int(t_start_ns, "t_start_ns")
        self._t_last = self.t_start
        self._n_out = self._check_count(outstanding)
        self._integral = 0                       # sum of outstanding * dt, count * ns
        self.bits = 0

    @staticmethod
    def _check_count(n: object) -> int:
        n = _check_int(n, "outstanding")
        if n < 0:
            raise ValueError(f"outstanding must be >= 0, got {n}")
        return n

    def _integrate_to(self, t_ns: int) -> None:
        t = _check_int(t_ns, "t_ns")
        if t < self._t_last:
            raise ValueError(f"time went backwards: {t} < {self._t_last}")
        if t >= self.t_end:
            raise ValueError(f"t={t} is outside the window ending at {self.t_end}; call advance() first")
        self._integral += self._n_out * (t - self._t_last)
        self._t_last = t

    def set_outstanding(self, t_ns: int, outstanding: int) -> None:
        """The outstanding count changes to `outstanding` at time t (grant issued or resolved)."""
        n = self._check_count(outstanding)
        self._integrate_to(t_ns)
        self._n_out = n

    def add_delivery(self, t_ns: int, bits: int = 8 * S) -> None:
        """Count `bits` delivered by the client at time t."""
        b = _check_int(bits, "bits")
        if b < 0:
            raise ValueError(f"bits must be >= 0, got {b}")
        self._integrate_to(t_ns)
        self.bits += b

    def avg_outstanding(self) -> float:
        """Time-weighted average outstanding over the full window (valid at close)."""
        total = self._integral + self._n_out * (self.t_end - self._t_last)
        return total / RATE_WINDOW_NS

    def close(self, state: ClientState, backlogged: bool | None = None) -> float:
        """Close the window at its boundary t_end: update bhat and cap, then start
        the next window at t_end with the current outstanding count. Returns R (bit/s).

        backlogged=None uses the outstanding rule (2 * integral >= cap * 100 ms);
        a bool overrides it (B3).
        """
        if backlogged is None:
            total = self._integral + self._n_out * (self.t_end - self._t_last)
            backlogged = 2 * total >= state.cap * RATE_WINDOW_NS
        R = rate_update(state, self.bits, bool(backlogged))
        self.start(self.t_end, self._n_out)
        return R

    def advance(self, state: ClientState, t_ns: int, backlogged: bool | None = None) -> int:
        """Close every window whose end is <= t (Algorithm 1, line 27). Returns how many closed.

        For B3, `backlogged` applies to each window closed by this call.
        """
        t = _check_int(t_ns, "t_ns")
        n = 0
        while self.t_end <= t:
            self.close(state, backlogged)
            n += 1
        return n


# --------------------------------------------------------------------------- outcome statistics


def update_outcome_stats(state: ClientState, y: int, t_ns: int) -> None:
    """Algorithm 1 line 30: hist, EWMAs, consec_loss, timestamps, n_resolved.

    Call AFTER the observation o (Section 4.2) and anything else that needs the
    pre-event state has been computed. Also clears first_event_in_round (o8).
    evbuf, h/h_stale and the RTT sample are handled by their own callers.
    """
    y = check_y(y)
    t = _check_int(t_ns, "t_ns")
    state.hist.append(y)
    state.ewma_fast = (1.0 - EWMA_FAST_W) * state.ewma_fast + EWMA_FAST_W * y
    state.ewma_slow = (1.0 - EWMA_SLOW_W) * state.ewma_slow + EWMA_SLOW_W * y
    if y == 1:
        state.consec_loss = 0
        state.t_last_del_ns = t
    else:
        state.consec_loss += 1
        state.t_last_loss_ns = t
    state.n_resolved += 1
    state.first_event_in_round = False