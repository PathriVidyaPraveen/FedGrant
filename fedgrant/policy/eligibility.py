"""Eligibility rule (Section 3.9): client i is eligible iff rules (i)-(v) all hold."""

from __future__ import annotations

from enum import IntEnum

from fedgrant.core.state import ClientState
from fedgrant.policy.constants import THETA
from fedgrant.policy.types import PolicyConfig

DEADLINE_MARGIN_NS: int = 2_000_000      # rule (iv): t + srtt_i + 2 ms <= D


class Reason(IntEnum):
    """First failing rule, for logging. ELIGIBLE means all rules hold."""

    ELIGIBLE = 0
    NOT_READY = 1        # (i)
    NO_PENDING = 2       # (ii)
    CAP_REACHED = 3      # (iii)
    TOO_LATE = 4         # (iv)
    LOW_PREDICTION = 5   # (v)


def is_eligible(
    state: ClientState, t_ns: int, D_ns: int, p_hat: float, cfg: PolicyConfig
) -> tuple[bool, Reason]:
    """Rules (i)-(v) of Section 3.9, checked in order; returns (eligible, reason).

    (i)   READY received.
    (ii)  a pending position: UNSENT or LOST (LOST only if cfg.retransmit; A2).
    (iii) outstanding_i < cap_i.
    (iv)  t + srtt_i + 2 ms <= D.
    (v)   p_hat_i(t) >= theta = 0.05; skipped when cfg.pred_in_eligibility is False (A4).
    """
    if not state.ready:
        return False, Reason.NOT_READY
    if not state.has_pending(cfg.retransmit):
        return False, Reason.NO_PENDING
    if len(state.outstanding) >= state.cap:
        return False, Reason.CAP_REACHED
    if t_ns + state.srtt_ns + DEADLINE_MARGIN_NS > D_ns:
        return False, Reason.TOO_LATE
    if cfg.pred_in_eligibility and not p_hat >= THETA:      # NaN fails too
        return False, Reason.LOW_PREDICTION
    return True, Reason.ELIGIBLE