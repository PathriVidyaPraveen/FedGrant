"""Pacing rule (Section 3.8, Eq. 12).

A grant to the selected client i* is issued only if Phi + p_hat_i* <= P_target,
where Phi is the sum of p_hat recorded at issue over outstanding grants.

Min-RTT amendment (CHANGELOG, PROPOSED 2026-11-17): RTT~ in Eq. 12 is the median
over ready clients of each client's minimum RTT sample, not the median srtt, so
P_target is not inflated by the queueing delay it is meant to bound
(p_target_min_rtt). B3's RTT~ (median srtt) is unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from fedgrant.policy.constants import C, S, SRTT0_NS, TAU_Q_NS
from fedgrant.policy.types import PolicyConfig


def median_srtt_ns(srtts_ns: Iterable[int]) -> float:
    """Median of srtt_i over ready clients (RTT~ in Eq. 12 and in B3), in ns.

    SPEC-NOTE: for an even number of clients the median is the mean of the two
    middle values (numpy convention), so it may be a half-integer.
    """
    arr = np.fromiter((int(x) for x in srtts_ns), dtype=np.int64)
    if arr.size == 0:
        raise ValueError("median of srtt needs at least one ready client")
    return float(np.median(arr))


def p_target(srtts_of_ready_clients_ns: Iterable[int]) -> float:
    """Eq. (12) with the unamended RTT~ = median srtt: C (RTT~ + tau_q) / (8 S), in packets.

    Kept for callers that pass RTT values directly; the grant policy uses
    p_target_min_rtt (min-RTT amendment).
    """
    rtt = median_srtt_ns(srtts_of_ready_clients_ns)
    return _p_target_from_rtt(rtt)


def _p_target_from_rtt(rtt_ns: float) -> float:
    return C * (rtt_ns + TAU_Q_NS) / (8 * S * 1e9)


def pacing_rtt_ns(states: Iterable[object]) -> float:
    """RTT~ of Eq. (12) under the min-RTT amendment, in ns.

    The median, over the given (ready) clients that have at least one RTT sample,
    of each client's minimum RTT sample (ClientState.rtt_min_ns, persistent across
    rounds). Clients without a sample are left out; if none has one, SRTT0 = 100 ms.
    Median of an even count is the mean of the two middle values (numpy).
    """
    samples = [int(s.rtt_min_ns) for s in states if getattr(s, "rtt_min_ns", None) is not None]
    if not samples:
        return float(SRTT0_NS)
    return float(np.median(np.asarray(samples, dtype=np.int64)))


def p_target_min_rtt(states_of_ready_clients: Iterable[object]) -> float:
    """Eq. (12), min-RTT amendment: P_target = C (RTT_min~ + tau_q) / (8 S), in packets."""
    return _p_target_from_rtt(pacing_rtt_ns(states_of_ready_clients))


def pacing_p(p_hat: float, cfg: PolicyConfig) -> float:
    """The p used for pacing: p_hat, or 1.0 under ablation A3 (pred_in_pacing=False).

    Record this same value in Phi when the grant is issued.
    """
    return float(p_hat) if cfg.pred_in_pacing else 1.0


class InflightTracker:
    """Phi = sum over outstanding grants of the p recorded at issue (Section 3.8).

    add() updates Phi incrementally; remove() recomputes it with math.fsum over
    the remaining grants, so rounding error never accumulates across removals,
    and Phi is exactly 0 when nothing is outstanding.
    """

    __slots__ = ("_p", "_phi")

    def __init__(self) -> None:
        self._p: dict[int, float] = {}
        self._phi: float = 0.0

    @property
    def phi(self) -> float:
        return self._phi

    def __len__(self) -> int:
        return len(self._p)

    def __contains__(self, grant_id: int) -> bool:
        return grant_id in self._p

    @staticmethod
    def _check_p(p: float) -> float:
        p = float(p)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p!r}")
        return p

    def add(self, grant_id: int, p: float) -> None:
        """Record grant g with its p at issue."""
        if grant_id in self._p:
            raise KeyError(f"grant {grant_id} already outstanding")
        p = self._check_p(p)
        self._p[grant_id] = p
        self._phi += p

    def remove(self, grant_id: int) -> float:
        """Resolve grant g (Algorithm 1 line 29: Phi <- Phi - p_g). Returns its p."""
        p = self._p.pop(grant_id)            # KeyError if unknown
        self._phi = math.fsum(self._p.values()) if self._p else 0.0
        return p

    def can_issue(self, p_hat: float, P_target: float) -> bool:
        """Pacing test: Phi + p_hat <= P_target (inclusive)."""
        return self._phi + self._check_p(p_hat) <= P_target

    def clear(self) -> None:
        """Drop all grants (round end)."""
        self._p.clear()
        self._phi = 0.0