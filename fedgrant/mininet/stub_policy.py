"""Self-contained first-eligible GatewayCore providers for Mininet smoke rounds."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fedgrant.core.gateway import GatewayCore
from fedgrant.core.state import ClientState
from fedgrant.policy.types import CandidateView


class FirstEligibleSelector:
    def select(self, candidates: CandidateView) -> int:
        if len(candidates) == 0:
            raise ValueError("no eligible candidates")
        return int(candidates.client_id.min())


class FirstEligible:
    """Simple READY/pending/cap/deadline eligibility, without predictor gating."""
    def candidates(self, clients: Mapping[int, ClientState], now_ns: int, deadline_ns: int, retransmit: bool) -> CandidateView:
        ids: list[int] = []
        values: list[float] = []
        pending: list[int] = []
        for client_id in sorted(clients):
            state = clients[client_id]
            pos = state.next_pos(retransmit)
            if (state.ready and pos <= state.K and len(state.outstanding) < state.cap
                    and now_ns + state.srtt_ns + 2_000_000 <= deadline_ns):
                ids.append(client_id)
                values.append(float(state.val[pos - 1]))
                pending.append(max(0, state.K - state.m - len(state.outstanding)))
        return CandidateView(
            client_id=np.asarray(ids, dtype=np.int32),
            next_value=np.asarray(values, dtype=np.float64),
            undelivered=np.asarray(pending, dtype=np.int32),
            rr_pointer=0,
        )


class AllowAllPacing:
    def allow(self, client_id: int, p_hat: float, now_ns: int, deadline_ns: int) -> bool:
        return now_ns < deadline_ns


class FixedPrediction:
    def probability(self, client_id: int, now_ns: int) -> float:
        return 0.95


class FixedTimeout:
    def __init__(self, duration_ns: int = 100_000_000):
        if duration_ns <= 0:
            raise ValueError("duration_ns must be positive")
        self.duration_ns = int(duration_ns)

    def timeout_ns(self, client_id: int, issue_ns: int) -> int:
        return issue_ns + self.duration_ns


def create_gateway(num_clients: int, *, timeout_ns: int = 100_000_000) -> GatewayCore:
    """Gateway factory for tests; each client has unit weight and one grant in flight."""
    if isinstance(num_clients, bool) or not 1 <= int(num_clients) <= 50:
        raise ValueError("num_clients must be in [1, 50]")
    n = int(num_clients)
    states = {i: ClientState(i, 1.0 / n) for i in range(n)}
    # Limit the smoke provider to one in-flight grant per client; this keeps the
    # protocol exercise compact while still testing grant/timeout/retransmit state.
    for state in states.values():
        state.cap = 1
    return GatewayCore(
        states,
        selector=FirstEligibleSelector(),
        eligibility=FirstEligible(),
        pacing=AllowAllPacing(),
        prediction=FixedPrediction(),
        timeout=FixedTimeout(timeout_ns),
        retransmit=True,
    )


__all__ = ["FirstEligibleSelector", "FirstEligible", "AllowAllPacing", "FixedPrediction", "FixedTimeout", "create_gateway"]
