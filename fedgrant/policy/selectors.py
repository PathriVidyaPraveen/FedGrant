"""Client selectors (Sections 3.10, 10: value / rr / srpt) and the Eq. (13) heap key."""

from __future__ import annotations

import math

import numpy as np

from fedgrant.core.state import ClientState
from fedgrant.policy.types import CandidateView, Selector

STARVE_ROUNDS: int = 5          # Eq. (13): starve_i >= 5 and m_i = 0 -> +inf


def _nonempty(cand: CandidateView) -> None:
    if len(cand) == 0:
        raise ValueError("no eligible candidates")


def value_key(state: ClientState, retransmit: bool = True) -> float:
    """Eq. (13): v~_i = +inf if starve_i >= 5 and m_i = 0, else v_i(s_next_i) = val[s_next - 1].

    Requires a pending position (eligibility rule (ii)); raises ValueError otherwise.
    """
    s = state.next_pos(retransmit)
    if s > state.K:
        raise ValueError(f"client {state.client_id} has no pending position")
    if state.starve >= STARVE_ROUNDS and state.m == 0:
        return math.inf
    return float(state.val[s - 1])


def heap_key(v_tilde: float, client_id: int) -> tuple[float, int]:
    """Min-heap key: heapq pops the largest v~ first, then the lower client id."""
    if math.isnan(v_tilde):
        raise ValueError("v_tilde must not be NaN")
    return (-v_tilde, int(client_id))


class ValueSelector:
    """selector "value": argmax v~_i over eligible clients, ties -> lower id (Eq. 13)."""

    name = "value"

    def select(self, cand: CandidateView) -> int:
        _nonempty(cand)
        if np.isnan(cand.next_value).any():
            raise ValueError("next_value contains NaN")
        # lexsort: last key is primary -> sort by (-value, id)
        order = np.lexsort((cand.client_id, -cand.next_value))
        return int(cand.client_id[order[0]])


class RoundRobinSelector:
    """selector "rr" (B4): cyclic pointer over eligible clients by id.

    Returns the smallest eligible id >= rr_pointer, wrapping to the smallest
    eligible id. After select(), next_pointer = chosen + 1. The caller stores it
    and passes it back as CandidateView.rr_pointer, but only if the grant is
    actually issued: if pacing then blocks the grant (Algorithm 1 line 11), the
    pointer must stay where it was.
    """

    name = "rr"

    def __init__(self) -> None:
        self.next_pointer: int = 0

    @staticmethod
    def pick(client_ids: np.ndarray, rr_pointer: int) -> int:
        ids = np.asarray(client_ids)
        at_or_after = ids[ids >= rr_pointer]
        return int(at_or_after.min() if at_or_after.size else ids.min())

    def select(self, cand: CandidateView) -> int:
        _nonempty(cand)
        chosen = self.pick(cand.client_id, cand.rr_pointer)
        self.next_pointer = chosen + 1
        return chosen


class SRPTSelector:
    """selector "srpt" (B5): fewest undelivered blocks (K - m_i - outstanding_i), ties -> lower id."""

    name = "srpt"

    def select(self, cand: CandidateView) -> int:
        _nonempty(cand)
        order = np.lexsort((cand.client_id, cand.undelivered))
        return int(cand.client_id[order[0]])


_SELECTORS: dict[str, type] = {
    "value": ValueSelector,
    "rr": RoundRobinSelector,
    "srpt": SRPTSelector,
}


def make_selector(name: str) -> Selector:
    """Selector for a PolicyConfig.selector name ("value" | "rr" | "srpt")."""
    try:
        return _SELECTORS[name]()
    except KeyError:
        raise ValueError(f"unknown selector {name!r}; expected one of {sorted(_SELECTORS)}") from None