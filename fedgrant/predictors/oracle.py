"""Simulator-only predictor using the simulator's true channel and queue state."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fedgrant.core.estimators import RTO_FLOOR_NS
from fedgrant.predictors.base import PredInput, RoundCtx
from fedgrant.policy.constants import S


class OraclePredictor:
    """Estimate success probability from current GE state and exact link backlogs.

    The oracle is conditional on the next frame using the current channel
    state. It assigns zero probability when the modeled grant round trip
    (gateway-to-client grant, client uplink, and queued serialization) cannot
    complete before the client's RFC 6298 timeout.
    """

    name = "oracle"

    def __init__(self, access_links: list[Any], bottleneck: Any, client_states: list[Any] | None = None):
        self.access_links = access_links
        self.bottleneck = bottleneck
        self.client_states = client_states
        self.ctx: RoundCtx | None = None

    def reset_round(self, ctx: RoundCtx) -> None:
        self.ctx = ctx

    def observe(self, client: int, y: int, t_ns: int, obs: np.ndarray) -> None:
        # True channel state is owned by the network simulator; no fitted
        # predictor state is needed here.
        if y not in (0, 1):
            raise ValueError("y must be 0 or 1")

    def predict(self, inp: PredInput) -> np.ndarray:
        if self.client_states is None:
            raise RuntimeError("oracle client states have not been attached to the gateway")
        probabilities = np.zeros(inp.n, dtype=np.float32)
        for index, raw_client_id in enumerate(inp.client_ids):
            cid = int(raw_client_id)
            if not 0 <= cid < len(self.access_links) or cid >= len(self.client_states):
                raise ValueError(f"unknown oracle client id {cid}")
            link = self.access_links[cid]
            state = self.client_states[cid]
            rate = max(1.0, float(link.rate_bps))
            bn_rate = max(1.0, float(self.bottleneck.capacity_bps))
            # Include one new frame in addition to the exact current FIFO and
            # in-service backlogs. Service times use the current link rates.
            access_service_ns = math.ceil((link.backlog_bits + S * 8) * 1e9 / rate)
            bottleneck_service_ns = math.ceil((self.bottleneck.backlog_bits + S * 8) * 1e9 / bn_rate)
            grant_return_ns = link.prop_delay_ns + 5_000_000
            data_forward_ns = link.prop_delay_ns + access_service_ns + 5_000_000 + bottleneck_service_ns
            predicted_rtt_ns = grant_return_ns + data_forward_ns
            rto_budget_ns = int(state.srtt_ns) + max(4 * int(state.rttvar_ns), RTO_FLOOR_NS)
            deadline_budget_ns = (self.ctx.D_ns - inp.t_ns) if self.ctx is not None else rto_budget_ns
            timeout_budget_ns = min(rto_budget_ns, max(0, deadline_budget_ns))
            can_meet_timeout = predicted_rtt_ns <= timeout_budget_ns
            has_queue_room = len(link.queue) + int(link.is_busy) < link.max_queue_frames
            p_good = float(link.ge_model.prob_next_good())
            probabilities[index] = p_good if can_meet_timeout and has_queue_room else 0.0
        return probabilities


__all__ = ["OraclePredictor"]
