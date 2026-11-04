"""Features and observations (Sections 4.2, 4.3). The ONE module used unchanged by
the simulator, the gateway and dataset generation.

Reads only ClientState plus a GatewayView filled in by the caller. Every value is
computed in float64 and rounded once to float32, so outputs are bitwise
deterministic. Spec feature indices are 1-based; array indices are 0-based
(feature k is x48[k - 1]).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from fedgrant.core.estimators import check_y, rtt_update, update_outcome_stats
from fedgrant.core.state import ClientState
from fedgrant.policy.constants import C, CTX_SLICE, HIST_LEN, N_OBS, N_X48, S
from fedgrant.policy.pacing import InflightTracker
from fedgrant.predictors.base import PredInput

ARRIVAL_WINDOW_NS: int = 20_000_000          # 20 ms, feature 42 and o7
DATA_BITS: int = 8 * S                       # wire bits of one DATA frame
_LOAD_NORM_BITS: float = 0.02 * C            # 0.02 C = bits the bottleneck carries in 20 ms

FEATURE_NAMES: tuple[str, ...] = (
    *(f"y_lag{k}" for k in range(1, HIST_LEN + 1)),   # 1-32, lag1 = most recent
    "ewma_fast",                 # 33
    "ewma_slow",                 # 34
    "n_resolved",                # 35
    "srtt_over_D",               # 36
    "rttvar_over_D",             # 37
    "time_left_frac",            # 38
    "time_left_over_srtt",       # 39
    "outstanding_over_cap",      # 40
    "phi_over_ptarget",          # 41
    "gateway_load_20ms",         # 42
    "last_rtt_over_srtt",        # 43
    "mean_rtt8_over_srtt",       # 44
    "since_last_del",            # 45
    "since_last_loss",           # 46
    "consec_loss",               # 47
    "delivered_frac",            # 48
)
OBS_NAMES: tuple[str, ...] = (
    "y", "rtt_over_srtt", "gap_over_srtt", "time_left_frac",
    "outstanding_over_cap", "phi_over_ptarget", "gateway_load_20ms", "first_in_round",
)
assert len(FEATURE_NAMES) == N_X48 and len(OBS_NAMES) == N_OBS


def _int(x: object, name: str) -> int:
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise TypeError(f"{name} must be an int (ns), got {type(x).__name__}")
    return int(x)


# --------------------------------------------------------------------------- arrivals


class ArrivalLog:
    """(t_ns, bits) of every FedGrant DATA frame arriving at the gateway, all clients.

    SPEC-NOTE: feature 42 and o7 count FedGrant DATA frames only. Cross traffic
    (scenario X2) also crosses the bottleneck but the gateway daemon never sees
    it, so the simulator must not record it here either, or simulator and
    gateway features differ in X2.

    Record times must be non-decreasing. Entries older than 20 ms before the
    latest record are pruned, so queries may start no earlier than
    horizon = latest record time - 20 ms. Call reset() at ROUND_START.
    """

    __slots__ = ("_q", "_sum", "_t_last")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._q: deque[tuple[int, int]] = deque()
        self._sum: int = 0                       # bits of all retained entries
        self._t_last: int | None = None

    @property
    def horizon_ns(self) -> int | None:
        """Earliest t0 a query may use; None if nothing was recorded."""
        return None if self._t_last is None else self._t_last - ARRIVAL_WINDOW_NS

    def record(self, t_ns: int, bits: int = DATA_BITS) -> None:
        """A DATA frame of `bits` wire bits arrived at time t."""
        t, b = _int(t_ns, "t_ns"), _int(bits, "bits")
        if b < 0:
            raise ValueError("bits must be >= 0")
        if self._t_last is not None and t < self._t_last:
            raise ValueError(f"arrival times must be non-decreasing: {t} < {self._t_last}")
        self._q.append((t, b))
        self._sum += b
        self._t_last = t
        cutoff = t - ARRIVAL_WINDOW_NS
        while self._q[0][0] < cutoff:
            self._sum -= self._q.popleft()[1]

    def bits_in(self, t0_ns: int, t1_ns: int) -> int:
        """Bits that arrived in [t0, t1). Raises if t0 is before the pruning horizon."""
        t0, t1 = _int(t0_ns, "t0_ns"), _int(t1_ns, "t1_ns")
        if t1 < t0:
            raise ValueError(f"empty interval [{t0}, {t1})")
        h = self.horizon_ns
        if h is None:
            return 0
        if t0 < h:
            raise ValueError(f"t0={t0} is before the retained horizon {h}")
        total = self._sum
        for t, b in self._q:                      # entries before t0 (front)
            if t >= t0:
                break
            total -= b
        for t, b in reversed(self._q):            # entries at or after t1 (back)
            if t < t1:
                break
            total -= b
        return total

    def bits_last_20ms(self, t_ns: int) -> int:
        """Bits that arrived in [t - 20 ms, t) (feature 42, o7)."""
        t = _int(t_ns, "t_ns")
        return self.bits_in(t - ARRIVAL_WINDOW_NS, t)


# --------------------------------------------------------------------------- gateway view


@dataclass(slots=True)
class GatewayView:
    """Gateway-wide state the features need; the simulator / gateway fill it in.

    t_ns: current time, relative to round start. D_ns: deadline.
    inflight: the gateway's Phi tracker (Section 3.8); resolve_event removes the
      grant from it, so Phi stays consistent everywhere.
    P_target: Eq. (12) value currently in force (recomputed by the caller).
    arrivals: ArrivalLog of all DATA arrivals.

    SPEC-NOTE / interface: Phi is exposed as the read-only property `Phi` backed
    by `inflight`, rather than a plain float, because resolve_event must remove
    the resolved grant's p from Phi (o6 is computed after removal).
    """

    t_ns: int
    D_ns: int
    P_target: float
    inflight: InflightTracker = field(default_factory=InflightTracker)
    arrivals: ArrivalLog = field(default_factory=ArrivalLog)

    @property
    def Phi(self) -> float:
        return self.inflight.phi

    def _check(self) -> None:
        _int(self.t_ns, "t_ns")
        if _int(self.D_ns, "D_ns") <= 0:
            raise ValueError("D_ns must be > 0")
        if not (self.P_target > 0.0):
            raise ValueError(f"P_target must be > 0, got {self.P_target!r}")


def _srtt(state: ClientState) -> float:
    # SPEC-NOTE: srtt can only reach 0 ns after a long run of 0-ns RTT samples;
    # it is floored at 1 ns to keep every ratio finite.
    return float(max(state.srtt_ns, 1))


def _clip_ratio(num: float, den: float, cap: float) -> float:
    """min(num / den, cap) / cap."""
    return min(num / den, cap) / cap


def _load(view: GatewayView, t_ns: int) -> float:
    """min(bits in [t - 20 ms, t) / (0.02 C), 2)."""
    return min(view.arrivals.bits_last_20ms(t_ns) / _LOAD_NORM_BITS, 2.0)


# --------------------------------------------------------------------------- features (Section 4.3)


def features48(state: ClientState, view: GatewayView) -> np.ndarray:
    """x_i(t), float32 [48], exactly the table of Section 4.3, at t = view.t_ns.

    SPEC-NOTE: features 43/44 are 1.0 when no RTT sample exists yet ("no
    information"), consistent with the "1 if none" rule of features 45/46.
    SPEC-NOTE: 38/39 are computed literally (no clipping at 0); they are only
    evaluated for t <= D.
    """
    view._check()
    t, D = int(view.t_ns), int(view.D_ns)
    srtt = _srtt(state)
    x = np.empty(N_X48, dtype=np.float64)

    # 1-32: last 32 outcomes, most recent first; 0.5 where fewer exist
    x[:HIST_LEN] = 0.5
    h = list(state.hist)
    if h:
        x[: len(h)] = h[::-1]

    x[32] = state.ewma_fast                                   # 33
    x[33] = state.ewma_slow                                   # 34
    x[34] = min(state.n_resolved, 100) / 100                  # 35
    x[35] = state.srtt_ns / D                                 # 36
    x[36] = state.rttvar_ns / D                               # 37
    x[37] = (D - t) / D                                       # 38
    x[38] = _clip_ratio(D - t, srtt, 20.0)                    # 39
    x[39] = len(state.outstanding) / state.cap                # 40
    x[40] = min(view.Phi / view.P_target, 2.0)                # 41
    x[41] = _load(view, t)                                    # 42
    if state.rtt_last8:
        x[42] = _clip_ratio(state.rtt_last8[-1], srtt, 5.0)   # 43
        x[43] = _clip_ratio(float(np.mean(np.asarray(state.rtt_last8, dtype=np.float64))), srtt, 5.0)  # 44
    else:
        x[42] = x[43] = 1.0
    x[44] = 1.0 if state.t_last_del_ns is None else _clip_ratio(t - state.t_last_del_ns, srtt, 20.0)    # 45
    x[45] = 1.0 if state.t_last_loss_ns is None else _clip_ratio(t - state.t_last_loss_ns, srtt, 20.0)  # 46
    x[46] = min(state.consec_loss, 32) / 32                   # 47
    x[47] = state.m / state.K                                 # 48
    return x.astype(np.float32)


def context16(x48: np.ndarray) -> np.ndarray:
    """c_i = features 33..48 = x48[32:48] (Section 4.2), as a float32 copy (or [n,16] for [n,48])."""
    x = np.asarray(x48, dtype=np.float32)
    if x.shape[-1] != N_X48:
        raise ValueError(f"last dimension must be {N_X48}, got {x.shape}")
    return x[..., CTX_SLICE].copy()


def build_pred_input(states: list[ClientState], view: GatewayView) -> PredInput:
    """PredInput for the given clients at t = view.t_ns (rows in the order given)."""
    x = np.zeros((len(states), N_X48), dtype=np.float32)
    for r, st in enumerate(states):
        x[r] = features48(st, view)
    ids = np.array([st.client_id for st in states], dtype=np.int32)
    return PredInput.from_x48(ids, x, int(view.t_ns))


# --------------------------------------------------------------------------- observations (Section 4.2)


def observation8(
    state: ClientState, view: GatewayView, y: int, rtt_ns: int | None, t_e_ns: int
) -> np.ndarray:
    """o_e, float32 [8], Section 4.2.

    Precondition (resolve_event guarantees it): the grant of event e has already
    been removed from state.outstanding and from view.inflight, and NOTHING else
    has been updated by e yet (srtt, t_prev_event, first_event_in_round, ...).

    o1 y; o2 min(RTT/srtt_before, 5)/5 if y = 1 else 1;
    o3 min((t_e - t_prev)/srtt_before, 20)/20, 1 for the client's first event ever;
    o4 max(0, (D - t_e)/D); o5 outstanding/cap after removal;
    o6 min(Phi/P_target, 2) after removal; o7 min(bits in [t_e - 20 ms, t_e)/(0.02 C), 2);
    o8 1 if e is the client's first resolved event in this round.
    """
    view._check()
    y = check_y(y)
    t_e, D = _int(t_e_ns, "t_e_ns"), int(view.D_ns)
    srtt = _srtt(state)
    o = np.empty(N_OBS, dtype=np.float64)
    o[0] = y
    if y == 1:
        if rtt_ns is None:
            raise ValueError("y = 1 needs an RTT sample")
        o[1] = _clip_ratio(_int(rtt_ns, "rtt_ns"), srtt, 5.0)
    else:
        o[1] = 1.0
    o[2] = 1.0 if state.t_prev_event_ns is None else _clip_ratio(t_e - state.t_prev_event_ns, srtt, 20.0)
    o[3] = max(0.0, (D - t_e) / D)
    o[4] = len(state.outstanding) / state.cap
    o[5] = min(view.Phi / view.P_target, 2.0)
    o[6] = _load(view, t_e)
    o[7] = 1.0 if state.first_event_in_round else 0.0
    return o.astype(np.float32)


def resolve_event(
    state: ClientState,
    view: GatewayView,
    grant_id: int,
    y: int,
    t_e_ns: int,
    rtt_ns: int | None = None,
) -> np.ndarray:
    """Resolve grant g of this client (Algorithm 1, Resolve). THE single entry point.

    Exact order:
      (a) remove g from state.outstanding and from Phi (view.inflight);
      (b) obs = observation8(...) from the state as it is now: srtt, t_prev_event,
          first_event_in_round, hist, ... still hold their pre-event values;
      (c) if y = 1: RFC 6298 update with the RTT sample (also appends rtt_last8);
          then update_outcome_stats (hist, EWMAs, consec_loss, t_last_del/loss,
          n_resolved, first_event_in_round <- False); then evbuf.append(obs),
          h_stale <- True, t_prev_event <- t_e;
      (d) return obs (the caller passes it to Predictor.observe(i, y, t_e, obs)).

    rtt_ns: for y = 1, the RTT sample t_e - t_g; if None it is computed from the
    grant's issue time. Must be None for y = 0.

    Persistent timestamps are kept on the round's clock by
    ClientState.reset_round(K, round_start_ns=...), which the caller runs at
    every ROUND_START.

    Not done here (caller, Algorithm 1): position status changes (mark_delivered
    before resolving on DATA; mark_timeout after resolving on timeout), the
    RateWindow outstanding update, Predictor.observe, recomputing p_hat / P_target.
    """
    gid = _int(grant_id, "grant_id")
    t_e = _int(t_e_ns, "t_e_ns")
    y = check_y(y)
    if gid not in state.outstanding:
        raise KeyError(f"grant {gid} is not outstanding for client {state.client_id}")
    if gid not in view.inflight:
        raise KeyError(f"grant {gid} is not in Phi (view.inflight)")
    if y == 0 and rtt_ns is not None:
        raise ValueError("rtt_ns must be None for y = 0")

    og = state.outstanding[gid]
    rtt: int | None = None
    if y == 1:
        rtt = t_e - og.t_issue_ns if rtt_ns is None else _int(rtt_ns, "rtt_ns")
        if rtt < 0:
            raise ValueError(f"negative RTT {rtt}")
    view._check()                       # validate before mutating anything
    view.arrivals.bits_last_20ms(t_e)   # raises now if t_e is behind the arrival log
    # (a)
    del state.outstanding[gid]
    view.inflight.remove(gid)
    # (b)
    obs = observation8(state, view, y, rtt, t_e)
    # (c)
    if rtt is not None:
        rtt_update(state, rtt)
    update_outcome_stats(state, y, t_e)
    state.evbuf.append(obs)
    state.h_stale = True
    state.t_prev_event_ns = t_e
    # (d)
    return obs


__all__ = [
    "ARRIVAL_WINDOW_NS",
    "ArrivalLog",
    "FEATURE_NAMES",
    "GatewayView",
    "OBS_NAMES",
    "build_pred_input",
    "context16",
    "features48",
    "observation8",
    "resolve_event",
]