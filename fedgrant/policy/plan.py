"""B3 plan-then-send allocation (Section 10, B3; optimality from Section 3.4)."""

from __future__ import annotations

import heapq
from collections.abc import Mapping
from fractions import Fraction
from math import floor

import numpy as np

from fedgrant.policy.constants import S

EBAR_WEIGHT: float = 0.1
# SPEC-NOTE: the spec gives no initial value for e-bar_i; 1.0 ("assume full
# delivery") is used so the first-round caps are not all zero. Needs team sign-off.
EBAR_INIT: float = 1.0


def plan_time_ns(D_ns: int) -> int:
    """t_plan = 0.1 D. D is a whole number of ms, so this is exact in ns.

    SPEC-NOTE: floors if D_ns is not divisible by 10.
    """
    return int(D_ns) // 10


def plan_budget(C: int, D_ns: int, t_plan_ns: int, rtt_median_ns: float) -> int:
    """M = floor(C (D - t_plan - RTT~) / (8 S)), clamped at 0. C in bit/s."""
    num = Fraction(C) * (Fraction(D_ns) - t_plan_ns - Fraction(rtt_median_ns))
    return max(0, floor(num / (8 * S * 1_000_000_000)))


def plan_caps(ebar: float, bhat_bps: float, D_ns: int, t_plan_ns: int, srtt_ns: int) -> int:
    """c_i = floor(e-bar_i * bhat_i * (D - t_plan - srtt_i) / (8 S)), clamped at 0.

    Exact rational arithmetic on the float inputs.
    """
    for name, v in (("ebar", ebar), ("bhat_bps", bhat_bps)):
        if not (v >= 0.0 and v != float("inf")):
            raise ValueError(f"{name} must be finite and >= 0, got {v!r}")
    x = Fraction(ebar) * Fraction(bhat_bps) * (D_ns - t_plan_ns - srtt_ns)
    return max(0, floor(x / (8 * S * 1_000_000_000)))


def plan_allocate(
    values: Mapping[int, np.ndarray], caps: Mapping[int, int], M: int
) -> dict[int, int]:
    """Greedy B3 allocation: repeatedly give one more block to the client with the
    largest next value v_i(k_i + 1) among clients with k_i < min(c_i, K), ties ->
    lower id, until sum k_i = M or nobody can take more.

    Optimal for the laminar matroid {k_i <= c_i, sum k_i <= M} because each
    client's values are non-increasing (Section 3.4). Returns k_i for every client.
    """
    M = int(M)
    if M < 0:
        raise ValueError(f"M must be >= 0, got {M}")
    if set(values) != set(caps):
        raise ValueError("values and caps must have the same clients")
    vals: dict[int, np.ndarray] = {}
    limit: dict[int, int] = {}
    for cid, v in values.items():
        a = np.asarray(v, dtype=np.float64)
        if a.ndim != 1:
            raise ValueError(f"values[{cid}] must be 1-D")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"values[{cid}] must be finite")
        if a.size > 1 and np.any(np.diff(a) > 0):
            raise ValueError(f"values[{cid}] must be non-increasing (Eq. 2)")
        c = int(caps[cid])
        if c < 0:
            raise ValueError(f"caps[{cid}] must be >= 0")
        vals[cid] = a
        limit[cid] = min(c, a.size)

    k = {cid: 0 for cid in values}
    heap = [(-float(vals[cid][0]), cid) for cid in sorted(values) if limit[cid] > 0]
    heapq.heapify(heap)
    total = 0
    while heap and total < M:
        _, cid = heapq.heappop(heap)
        k[cid] += 1
        total += 1
        if k[cid] < limit[cid]:
            heapq.heappush(heap, (-float(vals[cid][k[cid]]), cid))
    return k


class DeliveryRatioEWMA:
    """Per-client EWMA (weight 0.1) of the B3 delivery ratio over past rounds.

    SPEC-NOTE: ratio = (blocks of the client delivered by D) / k_i. A round with
    k_i = 0 carries no information and does not update the EWMA.
    """

    def __init__(self, weight: float = EBAR_WEIGHT, init: float = EBAR_INIT) -> None:
        if not 0.0 < weight <= 1.0:
            raise ValueError("weight must be in (0, 1]")
        self.weight = float(weight)
        self.init = float(init)
        self._e: dict[int, float] = {}

    def get(self, client: int) -> float:
        return self._e.get(client, self.init)

    def update(self, client: int, delivered: int, planned: int) -> float:
        """Fold one round's ratio delivered/planned into e-bar_i; returns the new value."""
        if planned < 0 or delivered < 0 or delivered > planned:
            raise ValueError(f"need 0 <= delivered <= planned, got {delivered}/{planned}")
        if planned == 0:
            return self.get(client)
        e = (1.0 - self.weight) * self.get(client) + self.weight * (delivered / planned)
        self._e[client] = e
        return e


__all__ = [
    "DeliveryRatioEWMA",
    "plan_allocate",
    "plan_budget",
    "plan_caps",
    "plan_time_ns",
]
