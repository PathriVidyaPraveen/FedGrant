"""Network simulation driver executing round-by-round frame transmission (Section 6, Algorithm 1)."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from fedgrant.core.estimators import (
    RateWindow,
    grant_timeout,
)
from fedgrant.core.gateway import (
    EmitAggregate as CoreEmitAggregate,
    GatewayCore,
    SendGrant as CoreSendGrant,
    SendRoundEnd as CoreSendRoundEnd,
)
from fedgrant.core.state import ClientState, OutstandingGrant
from fedgrant.policy.constants import (
    C,
    D_MODEL,
    K as K_DEFAULT,
    L as L_DEFAULT,
    S,
)
from fedgrant.policy.eligibility import Reason, is_eligible
from fedgrant.policy.features import (
    ArrivalLog,
    GatewayView,
    build_pred_input,
    context16,
    features48,
    resolve_event,
)
from fedgrant.policy.pacing import InflightTracker, p_target, pacing_p
from fedgrant.policy.selectors import (
    RoundRobinSelector,
    SRPTSelector,
    ValueSelector,
    value_key,
)
from fedgrant.policy.types import CandidateView, PolicyConfig, Selector
from fedgrant.predictors.base import PredInput, Predictor, RoundCtx
from fedgrant.predictors.ewma import PEWMA
from fedgrant.predictors.factory import make_predictor
from fedgrant.predictors.fallback import FallbackPredictor
from fedgrant.predictors.hmm import PHMM
from fedgrant.predictors.oracle import OraclePredictor
from fedgrant.sim.crosstraffic import CrossTrafficGenerator
from fedgrant.sim.engine import EventType, SimulationEngine
from fedgrant.sim.ge import GilbertElliottModel
from fedgrant.sim.links import (
    AccessLink,
    BottleneckLink,
    Frame,
    access_delay_ns,
)
from fedgrant.sim.traces import TraceManager


@dataclass(frozen=True, slots=True)
class SendGrant:
    """Action emitted to transmit a GRANT message to client."""

    client_id: int
    grant_id: int
    pos: int


@dataclass(frozen=True, slots=True)
class SendRoundEnd:
    """Action emitted to send ROUND_END bitmap to client."""

    client_id: int
    bitmap: np.ndarray
    delivered_indices: list[int]


@dataclass(frozen=True, slots=True)
class EmitAggregate:
    """Action emitted when round aggregation finishes at deadline."""

    G: np.ndarray


def energies_by_block(
    client_energies: np.ndarray, send_orders: list[np.ndarray] | None, num_clients: int, K: int
) -> np.ndarray:
    """Return READY energies indexed by original block id instead of send position."""
    energies = np.asarray(client_energies)
    if energies.shape != (num_clients, K):
        raise ValueError(f"client_energies must have shape ({num_clients}, {K})")
    if send_orders is not None and len(send_orders) != num_clients:
        raise ValueError("send_orders must contain one order per client")
    mapped = np.empty_like(energies)
    for cid in range(num_clients):
        order = np.arange(K, dtype=np.int64) if send_orders is None else np.asarray(send_orders[cid])
        if order.shape != (K,) or not np.array_equal(np.sort(order), np.arange(K)):
            raise ValueError(f"send_orders[{cid}] must be a permutation of [0, {K})")
        mapped[cid, order] = energies[cid]
    return mapped


def _policy_config_signature(config: PolicyConfig) -> tuple[tuple[str, Any], ...]:
    """Snapshot every validated PolicyConfig field for safe gateway reuse."""
    return tuple((field.name, getattr(config, field.name)) for field in fields(PolicyConfig))


def _policy_config_mismatches(
    expected: tuple[tuple[str, Any], ...], config: PolicyConfig
) -> list[str]:
    actual = dict(_policy_config_signature(config))
    return [name for name, value in expected if actual.get(name) != value]


def _normalize_decision_event(event: dict[str, Any], round_start_abs_ns: int) -> dict[str, Any]:
    """Copy a decision record into the public relative/absolute time convention."""
    forwarded = dict(event)
    event_t_ns = int(event["t_ns"])
    if event.get("abs_t_ns") == round_start_abs_ns + event_t_ns:
        return forwarded  # Already normalized; do not convert twice.
    forwarded["abs_t_ns"] = event_t_ns
    forwarded["t_ns"] = event_t_ns - round_start_abs_ns
    return forwarded


class _GatewaySimState:
    """State/configuration holder retained as the GatewayCore simulator adapter base."""

    def __init__(
        self,
        num_clients: int,
        client_weights: np.ndarray,
        predictor: Predictor,
        selector: Selector | str = "value",
        policy_config: PolicyConfig | None = None,
        D_ns: int = 398_000_000,
        K: int = K_DEFAULT,
        L: int = L_DEFAULT,
        d_model: int = D_MODEL,
        rng: np.random.Generator | None = None,
        sample_hook: Callable[[dict[str, Any]], None] | None = None,
        shadow_oracle: Predictor | None = None,
        shadow_hmm: Predictor | None = None,
        decision_hook: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.N = int(num_clients)
        self.weights = np.ascontiguousarray(client_weights, dtype=np.float32)
        if len(self.weights) != self.N:
            raise ValueError(f"weights length ({len(self.weights)}) != num_clients ({self.N})")
        self.predictor = predictor
        configured = policy_config or PolicyConfig(selector="value", predictor="ewma")
        # Keep an internal validated copy: callers may mutate their dataclass
        # after construction, but that must not silently change this gateway.
        self.cfg = PolicyConfig.from_dict(dict(_policy_config_signature(configured)))
        self._policy_config_signature = _policy_config_signature(self.cfg)
        if isinstance(selector, str):
            selectors: dict[str, Selector] = {
                "value": ValueSelector(), "rr": RoundRobinSelector(), "srpt": SRPTSelector(),
            }
            try:
                self.selector = selectors[selector.lower()]
            except KeyError:
                raise ValueError(f"Unknown selector '{selector}'") from None
        else:
            self.selector = selector
        self.D_ns, self.K, self.L, self.d_model = int(D_ns), int(K), int(L), int(d_model)
        self.rng = rng or np.random.default_rng()
        self.sample_hook = sample_hook
        self.decision_hook = decision_hook
        self.shadow_oracle, self.shadow_hmm = shadow_oracle, shadow_hmm
        self.client_states = [ClientState(client_id=i, w=float(self.weights[i]), K=self.K) for i in range(self.N)]
        self.rate_windows = [RateWindow(t_start_ns=0, outstanding=0) for _ in range(self.N)]
        self.inflight = InflightTracker()
        self.arrivals = ArrivalLog()
        self.pewma = PEWMA()
        self.round_id = 1
        self.round_start_abs_ns = 0
        self.next_gid = 1
        self.rr_pointer = 0
        self.grants_issued = self.grants_delivered = self.grants_timed_out = 0
        self.duplicate_packets = self.late_after_timeout = self.late_after_deadline = 0
        self.p_hat: dict[int, float] = {}
        self._core_stats_baseline = (0, 0, 0)

class _CoreEligibility:
    def __init__(self, sim: "GatewaySim") -> None:
        self.sim = sim

    def candidates(self, clients, now_ns: int, deadline_ns: int, retransmit: bool) -> CandidateView:
        sim = self.sim
        sim._set_view_time(now_ns)
        ids: list[int] = []
        values: list[float] = []
        remaining: list[int] = []
        probe: list[bool] = []
        relative_t = now_ns - sim.round_start_abs_ns
        for cid, state in clients.items():
            if not state.ready:
                continue
            p_hat = sim.core._predict_probability(cid, now_ns)
            # Bounded probe (rule (v) amendment): the core owns the per-round allowance.
            ok, reason = is_eligible(state, relative_t, sim.D_ns, p_hat, sim.cfg,
                                     probe_available=sim.core.probe_available(cid))
            if not ok:
                continue
            ids.append(cid)
            values.append(value_key(state, retransmit))
            remaining.append(sim.K - state.m - len(state.outstanding))
            probe.append(reason == Reason.PROBE)
        return CandidateView(
            client_id=np.asarray(ids, dtype=np.int32),
            next_value=np.asarray(values, dtype=np.float64),
            undelivered=np.asarray(remaining, dtype=np.int32),
            rr_pointer=sim.rr_pointer,
            probe=np.asarray(probe, dtype=np.bool_),
        )


class _CoreSelector:
    def __init__(self, sim: "GatewaySim") -> None:
        self.sim = sim

    def select(self, candidates: CandidateView) -> int:
        sim = self.sim
        if sim.cfg.explore_eps > 0 and sim.rng.random() < sim.cfg.explore_eps:
            return int(sim.rng.choice(candidates.client_id))
        return int(sim.selector.select(candidates))


class _CorePacing:
    def __init__(self, sim: "GatewaySim") -> None:
        self.sim = sim

    def allow(self, client_id: int, p_hat: float, now_ns: int, deadline_ns: int) -> bool:
        self.sim._set_view_time(now_ns)
        return self.sim.inflight.can_issue(pacing_p(p_hat, self.sim.cfg), self.sim.view.P_target)

    def issue_probability(self, client_id: int, p_hat: float) -> float:
        return pacing_p(p_hat, self.sim.cfg)


class _CoreTimeout:
    def __init__(self, sim: "GatewaySim") -> None:
        self.sim = sim

    def timeout_ns(self, client_id: int, issue_ns: int) -> int:
        relative_issue = issue_ns - self.sim.round_start_abs_ns
        return self.sim.round_start_abs_ns + grant_timeout(
            relative_issue, self.sim.client_states[client_id], self.sim.D_ns
        )


class _CoreEstimator:
    """Translate GatewayCore estimator callbacks into the shared feature path."""

    def __init__(self, sim: "GatewaySim") -> None:
        self.sim = sim

    def reset_round(self, ctx: RoundCtx) -> None:
        self.sim.reset_round(ctx)

    def grant_issued(self, client_id: int, grant: OutstandingGrant, t_ns: int) -> None:
        self.sim.grant_issued(client_id, grant, t_ns)

    def data_arrival(self, client_id: int, t_ns: int) -> None:
        self.sim.data_arrival(client_id, t_ns)

    def resolve(self, client_id: int, grant: OutstandingGrant, delivered: bool, t_ns: int) -> None:
        self.sim.resolve(client_id, grant, delivered, t_ns)

    def on_tick(self, t_ns: int) -> None:
        self.sim.estimator_on_tick(t_ns)

    def on_deadline(self, t_ns: int) -> None:
        self.sim.estimator_on_deadline(t_ns)


class GatewaySim(_GatewaySimState):
    """Network adapter around the shared GatewayCore event and state machine.

    The core owns READY/GRANT/DATA/timeout/deadline ordering. This adapter supplies
    simulator-specific scheduling providers and translates network-relative time.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.view = GatewayView(t_ns=0, D_ns=self.D_ns, P_target=p_target([s.srtt_ns for s in self.client_states]))
        self.core = GatewayCore(
            clients={state.client_id: state for state in self.client_states},
            selector=_CoreSelector(self),
            eligibility=_CoreEligibility(self),
            pacing=_CorePacing(self),
            timeout=_CoreTimeout(self),
            predictor=self.predictor,
            estimator=_CoreEstimator(self),
            aggregate=self,
            view=self.view,
            decision_hook=self._decision_hook,
            retransmit=self.cfg.retransmit,
        )
        self.client_states = [self.core.clients[i] for i in range(self.N)]
        self._snapshot_core_stats()
        if self.shadow_oracle is not None:
            self.shadow_oracle.client_states = self.client_states

    @property
    def inflight(self) -> InflightTracker:
        return self.view.inflight if hasattr(self, "view") else self._legacy_inflight

    @inflight.setter
    def inflight(self, tracker: InflightTracker) -> None:
        self._legacy_inflight = tracker

    @property
    def arrivals(self) -> ArrivalLog:
        return self.view.arrivals if hasattr(self, "view") else self._legacy_arrivals

    @arrivals.setter
    def arrivals(self, log: ArrivalLog) -> None:
        self._legacy_arrivals = log

    @property
    def Phi(self) -> float:
        return self.inflight.phi

    def get_p_target(self) -> float:
        ready = [state.srtt_ns for state in self.client_states if state.ready]
        return float(p_target(ready or [state.srtt_ns for state in self.client_states]))

    def get_view(self, t_ns: int) -> GatewayView:
        self._set_view_time(self.round_start_abs_ns + int(t_ns))
        return self.view

    def _set_view_time(self, absolute_ns: int) -> None:
        self.view.t_ns = int(absolute_ns) - self.round_start_abs_ns
        self.view.D_ns = self.D_ns
        self.view.P_target = self.get_p_target()

    def on_round_start(self, round_id: int, t_ns: int = 0, round_start_abs_ns: int | None = None) -> None:
        self.round_id = int(round_id)
        if round_start_abs_ns is not None:
            self.round_start_abs_ns = int(round_start_abs_ns)
        self.core.on_round_start(
            self.round_start_abs_ns + int(t_ns),
            round_id=self.round_id,
            deadline_ns=self.round_start_abs_ns + int(t_ns) + self.D_ns,
            K=self.K,
            L=self.L,
        )

    def on_ready(self, client_id: int, energies: np.ndarray, t_ns: int) -> list[SendGrant]:
        return self._send_grants(self.core.on_ready(client_id, energies, self.round_start_abs_ns + int(t_ns)))

    def on_data(self, client_id: int, grant_id: int, pos: int, block_idx: int,
                payload: np.ndarray, t_ns: int) -> list[SendGrant]:
        actions = self.core.on_data(
            client_id, grant_id, pos, block_idx, payload,
            self.round_start_abs_ns + int(t_ns), n_coords=min(self.L, self.d_model - block_idx * self.L),
        )
        self._sync_core_stats()
        return self._send_grants(actions)

    def _send_grants(self, actions) -> list[SendGrant]:
        return [SendGrant(action.client_id, action.grant_id, action.send_pos)
                for action in actions if isinstance(action, CoreSendGrant)]

    def reset_round(self, ctx: RoundCtx) -> None:
        self._snapshot_core_stats()
        self.p_hat.clear()
        self.grants_issued = self.grants_delivered = self.grants_timed_out = 0
        self.duplicate_packets = self.late_after_timeout = self.late_after_deadline = 0
        for window in self.rate_windows:
            window.start(0, 0)
        self.pewma.reset_round(ctx)
        if self.shadow_hmm is not None:
            self.shadow_hmm.reset_round(ctx)
        if self.shadow_oracle is not None:
            self.shadow_oracle.reset_round(ctx)

    def _snapshot_core_stats(self) -> None:
        stats = self.core.stats
        self._core_stats_baseline = (
            stats.duplicate_data,
            stats.late_after_timeout,
            stats.late_after_deadline,
        )

    def _sync_core_stats(self) -> None:
        duplicate_base, timeout_base, deadline_base = self._core_stats_baseline
        stats = self.core.stats
        self.duplicate_packets = stats.duplicate_data - duplicate_base
        self.late_after_timeout = stats.late_after_timeout - timeout_base
        self.late_after_deadline = stats.late_after_deadline - deadline_base

    def grant_issued(self, client_id: int, grant: OutstandingGrant, t_ns: int) -> None:
        relative_t = t_ns - self.round_start_abs_ns
        self.rate_windows[client_id].advance(self.client_states[client_id], relative_t)
        self.rate_windows[client_id].set_outstanding(relative_t, len(self.client_states[client_id].outstanding))
        self.grants_issued += 1
        if isinstance(self.selector, RoundRobinSelector):
            self.rr_pointer = (client_id + 1) % self.N

    def data_arrival(self, client_id: int, t_ns: int) -> None:
        relative_t = t_ns - self.round_start_abs_ns
        state = self.client_states[client_id]
        window = self.rate_windows[client_id]
        window.advance(state, relative_t)
        window.add_delivery(relative_t, 8 * S)

    def resolve(self, client_id: int, grant: OutstandingGrant, delivered: bool, t_ns: int) -> None:
        state = self.client_states[client_id]
        grant_id = next((gid for gid, value in state.outstanding.items() if value is grant), None)
        if grant_id is None:
            raise KeyError("GatewayCore estimator callback ran after grant removal")
        relative_t = t_ns - self.round_start_abs_ns
        self._set_view_time(t_ns)
        window = self.rate_windows[client_id]
        window.advance(state, relative_t)
        rtt_ns = t_ns - grant.t_issue_ns if delivered else None
        obs = resolve_event(state, self.view, grant_id, int(delivered), relative_t, rtt_ns)
        window.set_outstanding(relative_t, len(state.outstanding))
        # GatewayCore wraps the configured model with FallbackPredictor. Route
        # outcomes through that wrapper so its P-EWMA shadow stays warm and
        # receives every resolution even after the primary trips fallback.
        self.core.predictor.observe(client_id, int(delivered), relative_t, obs)
        self.pewma.observe(client_id, int(delivered), relative_t, obs)
        if self.shadow_hmm is not None:
            self.shadow_hmm.observe(client_id, int(delivered), relative_t, obs)
        if self.shadow_oracle is not None:
            self.shadow_oracle.observe(client_id, int(delivered), relative_t, obs)
        self.p_hat.pop(client_id, None)
        if delivered:
            self.grants_delivered += 1
        else:
            self.grants_timed_out += 1
        if self.sample_hook is not None:
            self.sample_hook({
                "event": "grant_resolved", "client_id": client_id, "grant_id": grant_id,
                "y": int(delivered), "t_ns": relative_t,
                "abs_t_ns": self.round_start_abs_ns + relative_t, "obs": obs,
                "rtt_ns": rtt_ns,
            })

    def estimator_on_tick(self, t_ns: int) -> None:
        relative_t = t_ns - self.round_start_abs_ns
        for cid, state in enumerate(self.client_states):
            self.rate_windows[cid].advance(state, relative_t)
        self._set_view_time(t_ns)

    def estimator_on_deadline(self, t_ns: int) -> None:
        relative_t = t_ns - self.round_start_abs_ns
        for cid, state in enumerate(self.client_states):
            self.rate_windows[cid].advance(state, relative_t)

    def aggregate(self, clients) -> np.ndarray:
        result = np.zeros(self.d_model, dtype=np.float32)
        for cid, state in clients.items():
            update = np.zeros(self.d_model, dtype=np.float32)
            for block_idx, payload in state.delivered_payloads.items():
                start = block_idx * self.L
                end = min(start + self.L, self.d_model)
                update[start:end] = np.asarray(payload[: end - start], dtype=np.float32)
            result += float(self.weights[cid]) * update
        return result

    def _decision_hook(self, event: dict[str, Any]) -> None:
        if self.decision_hook is not None:
            record = dict(event)
            # Keep the simulator's public decision schema: GatewayCore names the
            # chosen client ``client_id``; pacing has just set view.P_target.
            record.setdefault("selection_method", self.cfg.selector)
            record.setdefault("selected_client_id", record.get("client_id"))
            record.setdefault("p_pacing", float(pacing_p(record["p_hat"], self.cfg)))
            record.setdefault("p_target", float(self.view.P_target))
            for key in ("grant_id", "send_pos", "timeout_ns"):
                record.setdefault(key, None)  # pacing-blocked decisions issue no grant
            self.decision_hook(record)
        if event.get("outcome") != "grant_issued":
            return
        client_id = int(event["client_id"])
        t_ns = int(event["t_ns"])
        self._set_view_time(t_ns)
        state = self.client_states[client_id]
        x48 = features48(state, self.view)
        inp = PredInput.from_x48(np.asarray([client_id], dtype=np.int32), x48[None, :], t_ns - self.round_start_abs_ns)
        p_hat = float(event["p_hat"])
        if self.sample_hook is not None:
            self.sample_hook({
                "event": "grant_issued", "client_id": client_id,
                "grant_id": int(event["grant_id"]), "pos": int(event["send_pos"]),
                "t_ns": t_ns - self.round_start_abs_ns,
                "abs_t_ns": t_ns, "tau_ns": int(event["timeout_ns"]),
                "p_hat": p_hat, "p_ewma": float(self.pewma.predict(inp)[0]),
                "p_hmm": float(self.shadow_hmm.predict(inp)[0]) if self.shadow_hmm else p_hat,
                "p_oracle": float(self.shadow_oracle.predict(inp)[0]) if self.shadow_oracle else p_hat,
                "x48": x48, "ctx": context16(x48),
            })

    def on_tick(self, t_ns: int) -> list[SendGrant]:
        relative_t = int(t_ns)
        for cid, state in enumerate(self.client_states):
            self.rate_windows[cid].advance(state, relative_t)
        self._set_view_time(self.round_start_abs_ns + relative_t)
        return self._send_grants(self.core.on_tick(self.round_start_abs_ns + relative_t))

    def on_deadline(self, t_ns: int) -> tuple[list[SendRoundEnd], EmitAggregate]:
        actions = self.core.on_deadline(self.round_start_abs_ns + int(t_ns))
        ends: list[SendRoundEnd] = []
        aggregate = np.zeros(self.d_model, dtype=np.float32)
        for action in actions:
            if isinstance(action, CoreSendRoundEnd):
                delivered = sorted(self.core.delivered_blocks.get(action.client_id, ()))
                ends.append(SendRoundEnd(
                    action.client_id, np.frombuffer(action.bitmap, dtype=np.uint8).copy(), delivered
                ))
            elif isinstance(action, CoreEmitAggregate):
                aggregate = np.asarray(action.aggregate, dtype=np.float32)
        return ends, EmitAggregate(G=aggregate)


def compute_deadline_ns(beta: float = 0.10, num_clients: int = 20, K: int = K_DEFAULT) -> int:
    """Compute round deadline D = round(beta * T_full) in nanoseconds (Section 3.1).

    T_full = 3.985 s for N=20, K=169.
    Values for N=20: beta=0.05 -> 199ms, beta=0.10 -> 398ms, beta=0.20 -> 797ms.
    Value for N=50 at beta=0.10 -> 996ms.
    """
    t_full_s = 3.985 * (num_clients / 20.0) * (K / 169.0)
    d_ms = int(round(beta * t_full_s * 1000.0))
    return d_ms * 1_000_000


def run_network_round(
    round_id: int = 1,
    num_clients: int = 20,
    client_weights: np.ndarray | None = None,
    client_energies: np.ndarray | None = None,
    client_payloads: list[dict[int, np.ndarray]] | None = None,
    send_orders: list[np.ndarray] | None = None,
    beta: float = 0.10,
    scenario: str = "L2",
    split: str = "test",
    trace_dir: str | Path = "data/traces",
    policy_config: PolicyConfig | None = None,
    predictor: Predictor | None = None,
    seed: int = 0,
    sample_hook: Callable[[dict[str, Any]], None] | None = None,
    gateway: GatewaySim | None = None,
    round_start_abs_ns: int = 0,
    decision_hook: Callable[[dict[str, Any]], None] | None = None,
    event_hook: Callable[[dict[str, Any]], None] | None = None,
    loss_rate: float | None = None,
    mean_burst_length: float | None = None,
    cross_mbps: float = 0.0,
) -> dict[str, Any]:
    """Execute one simulated round (Section 6).

    ``decision_hook`` receives pre-mutation grant decisions. ``event_hook``
    reports round start, received READY/DATA, scheduler ticks and deadline,
    interleaved in order with the same ``grant_decision`` records given to
    ``decision_hook`` (GatewayCore's record plus the simulator schema fields
    ``selection_method``, ``selected_client_id``, ``p_pacing``, ``p_target``;
    pacing-blocked decisions carry ``None`` grant fields). DATA is reported only at gateway arrival, never
    at link enqueue/drop; late DATA drained after the deadline is reported
    with ``t_ns > D``. Link/loss/queue internals are never reported. Every
    record carries round-relative ``t_ns`` and absolute ``abs_t_ns``.
    These diagnostic hooks do not feed back into simulation decisions.
    """
    cfg = policy_config or PolicyConfig(selector="value", predictor="ewma")
    if gateway is not None:
        mismatches = _policy_config_mismatches(gateway._policy_config_signature, cfg)
        if mismatches:
            changed = ", ".join(mismatches)
            raise ValueError(
                "cannot reuse GatewaySim with a different PolicyConfig; "
                f"changed field(s): {changed}"
            )
    if (loss_rate is None) != (mean_burst_length is None):
        raise ValueError("loss_rate and mean_burst_length must be provided together")
    if not np.isfinite(cross_mbps) or cross_mbps < 0:
        raise ValueError("cross_mbps must be finite and non-negative")
    K = K_DEFAULT
    L = L_DEFAULT
    D_ns = compute_deadline_ns(beta, num_clients, K)

    def _record_decision(record: dict[str, Any]) -> None:
        # GatewayCore decision timestamps are absolute. Normalize a copy at
        # this adapter boundary; its internal event and API remain unchanged.
        forwarded = _normalize_decision_event(record, round_start_abs_ns)
        if decision_hook is not None:
            decision_hook(dict(forwarded))
        # A single ordered observation stream lets test/replay tooling
        # interleave scheduler decisions with READY/DATA/tick inputs.
        if event_hook is not None:
            event_hook(dict(forwarded))

    if round_start_abs_ns == 0 and round_id > 1:
        round_start_abs_ns = int((round_id - 1) * (1_000_000_000 + D_ns))

    if client_weights is None:
        client_weights = np.full(num_clients, 1.0 / num_clients, dtype=np.float32)

    # 1. Spawn 6 child seed streams (Section 6)
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(6)
    rng_trace = np.random.default_rng(child_seeds[0])
    rng_ge = np.random.default_rng(child_seeds[1])
    rng_readiness = np.random.default_rng(child_seeds[2])
    rng_explore = np.random.default_rng(child_seeds[3])
    rng_crosstraffic = np.random.default_rng(child_seeds[5])

    # 2. Simulation engine & links
    engine = SimulationEngine()
    bottleneck = BottleneckLink(engine=engine, capacity_bps=C, prop_delay_ns=5_000_000, max_queue_frames=64)

    trace_mgr = TraceManager(
        num_clients=num_clients,
        scenario=scenario,
        split=split,
        trace_dir=trace_dir,
        rng=rng_trace,
    )

    access_links: list[AccessLink] = []
    for cid in range(num_clients):
        ge_seed = int(rng_ge.integers(0, 2**31))
        ge_model = (
            GilbertElliottModel.for_scenario(scenario, seed=ge_seed)
            if loss_rate is None
            else GilbertElliottModel.for_loss_burst(loss_rate, mean_burst_length, seed=ge_seed)
        )
        delta_ns = access_delay_ns(cid)
        initial_rate = trace_mgr.get_client_rate(cid, 0)

        link = AccessLink(
            client_id=cid,
            engine=engine,
            prop_delay_ns=delta_ns,
            ge_model=ge_model,
            max_queue_frames=32,
            initial_rate_bps=initial_rate,
        )
        link.next_hop_cb = bottleneck.route_from_access
        access_links.append(link)

    # 3. Predictor & GatewaySim
    shadow_oracle = OraclePredictor(access_links, bottleneck, None)
    shadow_hmm = PHMM()

    if gateway is None:
        if predictor is None:
            if cfg.predictor == "oracle":
                pred: Predictor = shadow_oracle
            else:
                pred = make_predictor(cfg.predictor, "models/")
        else:
            pred = predictor

        gateway = GatewaySim(
            num_clients=num_clients,
            client_weights=client_weights,
            predictor=pred,
            selector=cfg.selector,
            policy_config=cfg,
            D_ns=D_ns,
            K=K,
            L=L,
            d_model=D_MODEL,
            rng=rng_explore,
            sample_hook=sample_hook,
            shadow_oracle=shadow_oracle,
            shadow_hmm=shadow_hmm,
            decision_hook=_record_decision if (decision_hook is not None or event_hook is not None) else None,
        )
        shadow_oracle.client_states = gateway.client_states
    else:
        previous_predictor_name = gateway.cfg.predictor
        gateway.D_ns = D_ns
        gateway.rng = rng_explore
        gateway.sample_hook = sample_hook
        gateway.decision_hook = _record_decision if (decision_hook is not None or event_hook is not None) else None
        if predictor is not None:
            gateway.predictor = predictor
        gateway.shadow_oracle = shadow_oracle
        shadow_oracle.client_states = gateway.client_states
        if gateway.shadow_hmm is None:
            gateway.shadow_hmm = shadow_hmm
        if predictor is None and (cfg.predictor != previous_predictor_name or cfg.predictor == "oracle"):
            gateway.predictor = (
                shadow_oracle if cfg.predictor == "oracle" else make_predictor(cfg.predictor, "models/")
            )
        gateway.core.predictor = (
            gateway.predictor
            if isinstance(gateway.predictor, FallbackPredictor)
            else FallbackPredictor(gateway.predictor)
        )
        gateway.core.retransmit = cfg.retransmit
        gateway.core._p_hat.clear()

    # 4. Cross traffic is a configuration input (N-X2), independent of capacity.
    crosstraffic: CrossTrafficGenerator | None = None
    effective_cross_mbps = 3.0 if scenario.upper() == "X2" and cross_mbps == 0 else cross_mbps
    if effective_cross_mbps > 0:
        crosstraffic = CrossTrafficGenerator(
            engine=engine,
            bottleneck=bottleneck,
            rate_bps=effective_cross_mbps * 1_000_000.0,
            frame_size_bytes=S,
            rng=rng_crosstraffic,
        )
        crosstraffic.start()

    # 5. Client transmission state & preparation
    if client_energies is None:
        client_energies = np.linspace(1.0, 0.01, K, dtype=np.float32)
        client_energies = np.tile(client_energies, (num_clients, 1))

    rtt_samples: list[int] = []

    def _dispatch_grants(grants: Sequence[SendGrant]) -> None:
        for g in grants:
            cid = g.client_id
            delta_ns = access_links[cid].prop_delay_ns
            rev_delay_ns = 5_000_000 + delta_ns

            engine.schedule(
                delay_ns=rev_delay_ns,
                event_type=EventType.GRANT_ARRIVE_CLIENT,
                callback=lambda _evt, gr=g: _on_grant_arrive_client(gr),
            )

    def _on_grant_arrive_client(grant: SendGrant) -> None:
        engine.schedule(
            delay_ns=100_000,
            event_type=EventType.ACCESS_ENQUEUE,
            callback=lambda _evt, gr=grant: _emit_client_data(gr),
        )

    def _emit_client_data(grant: SendGrant) -> None:
        cid = grant.client_id
        pos_1based = grant.pos
        pos_0based = pos_1based - 1

        if send_orders is not None and cid < len(send_orders):
            block_idx = int(send_orders[cid][pos_0based])
        else:
            block_idx = pos_0based

        if client_payloads is not None and cid < len(client_payloads) and block_idx in client_payloads[cid]:
            payload = client_payloads[cid][block_idx]
        else:
            payload = np.zeros(L, dtype=np.float16)

        frame = Frame(
            frame_id=f"c{cid}_g{grant.grant_id}_pos{pos_1based}",
            client_id=cid,
            size_bytes=S,
            created_at_ns=engine.now_ns,
            payload={
                "grant_id": grant.grant_id,
                "pos": pos_1based,
                "block_idx": block_idx,
                "payload": payload,
            },
        )

        access_links[cid].enqueue(frame)

    def _on_gw_arrive(frame: Frame) -> None:
        if frame.client_id >= 0:
            cid = frame.client_id
            gid = frame.payload.get("grant_id", 0)
            pos = frame.payload.get("pos", 0)
            b_idx = frame.payload.get("block_idx", 0)
            pld = frame.payload.get("payload", np.zeros(L, dtype=np.float16))

            # This hook is emitted only after the frame traverses both simulated
            # links and reaches the gateway; link-internal GE/drop state is not
            # part of the gateway's observable event stream.
            if event_hook is not None:
                event_hook({
                    "event": "data_received",
                    "round_id": round_id,
                    "client_id": cid,
                    "grant_id": gid,
                    "send_pos": pos,
                    "block_idx": b_idx,
                    "n_coords": max(0, min(L, D_MODEL - b_idx * L)),
                    "payload": tuple(float(value) for value in pld),
                    "t_ns": engine.now_ns,
                    "abs_t_ns": round_start_abs_ns + engine.now_ns,
                })

            absolute_arrival_ns = round_start_abs_ns + engine.now_ns
            if gid in gateway.client_states[cid].outstanding:
                og = gateway.client_states[cid].outstanding[gid]
                # A late DATA may reach the gateway before the next scheduler
                # tick has removed its grant. Match GatewayCore's strict
                # timeout rule: arrivals after tau resolve as loss, not RTT.
                if absolute_arrival_ns <= og.timeout_ns:
                    rtt_ns = absolute_arrival_ns - og.t_issue_ns
                    if rtt_ns < 0:
                        raise RuntimeError("simulated DATA arrival precedes its absolute grant issue time")
                    rtt_samples.append(rtt_ns)

            new_grants = gateway.on_data(
                client_id=cid,
                grant_id=gid,
                pos=pos,
                block_idx=b_idx,
                payload=pld,
                t_ns=engine.now_ns,
            )
            _dispatch_grants(new_grants)

    bottleneck.dest_cb = _on_gw_arrive

    # 6. Schedule Round Start and Client Readiness
    if event_hook is not None:
        event_hook({"event": "round_start", "round_id": round_id,
                    "t_ns": 0, "abs_t_ns": round_start_abs_ns, "deadline_ns": D_ns})
    gateway.on_round_start(round_id=round_id, t_ns=0, round_start_abs_ns=round_start_abs_ns)

    for cid in range(num_clients):
        max_rho_ns = int(0.10 * D_ns)
        rho_ns = int(rng_readiness.integers(0, max_rho_ns + 1)) if max_rho_ns > 0 else 0
        delta_ns = access_links[cid].prop_delay_ns
        round_start_downlink_ns = delta_ns + 5_000_000
        ready_uplink_ns = delta_ns + 5_000_000
        ready_arrival_ns = round_start_downlink_ns + rho_ns + ready_uplink_ns

        engine.schedule(
            delay_ns=ready_arrival_ns,
            event_type=EventType.READY_ARRIVE,
            callback=lambda _evt, c=cid: _on_ready_arrive(c),
        )

    def _on_ready_arrive(cid: int) -> None:
        energies = client_energies[cid]
        if event_hook is not None:
            event_hook({"event": "ready_received", "round_id": round_id,
                        "client_id": cid, "t_ns": engine.now_ns,
                        "abs_t_ns": round_start_abs_ns + engine.now_ns,
                        "energies": tuple(float(v) for v in energies)})
        new_grants = gateway.on_ready(client_id=cid, energies=energies, t_ns=engine.now_ns)
        _dispatch_grants(new_grants)

    # 7. Schedule periodic 1 ms gateway ticks
    for tick_ns in range(1_000_000, D_ns, 1_000_000):
        engine.schedule(
            delay_ns=tick_ns,
            event_type=EventType.TICK_1MS,
            callback=lambda _evt, t=tick_ns: _on_tick(t),
        )

    def _on_tick(t_ns: int) -> None:
        if event_hook is not None:
            event_hook({"event": "tick", "round_id": round_id, "t_ns": t_ns,
                        "abs_t_ns": round_start_abs_ns + t_ns})
        new_grants = gateway.on_tick(t_ns=t_ns)
        _dispatch_grants(new_grants)

    # 8. Schedule trace rate updates every 100 ms
    for t_100ms in range(100_000_000, D_ns, 100_000_000):
        engine.schedule(
            delay_ns=t_100ms,
            event_type=EventType.TICK_1MS,
            callback=lambda _evt, t=t_100ms: _on_trace_update(t),
        )

    def _on_trace_update(t_ns: int) -> None:
        for cid, link in enumerate(access_links):
            new_rate = trace_mgr.get_client_rate(cid, t_ns)
            link.set_rate(new_rate)

    # Sample bottleneck backlog every 10 ms for queueing delay stats
    backlog_bits_samples: list[int] = []
    for t_10ms in range(10_000_000, D_ns, 10_000_000):
        engine.schedule(
            delay_ns=t_10ms,
            event_type=EventType.TICK_1MS,
            callback=lambda _evt: backlog_bits_samples.append(bottleneck.backlog_bits),
        )

    # 9. Run discrete-event simulation up to deadline D
    engine.run_until(D_ns)

    if crosstraffic is not None:
        crosstraffic.stop()

    # 10. Deadline finish & aggregation
    if event_hook is not None:
        event_hook({"event": "deadline", "round_id": round_id,
                    "t_ns": D_ns, "abs_t_ns": round_start_abs_ns + D_ns})
    round_ends, emit_agg = gateway.on_deadline(t_ns=D_ns)

    delivered_mask = [[False] * K for _ in range(num_clients)]
    delivered_indices_per_client = [re.delivered_indices for re in round_ends]
    for cid, indices in enumerate(delivered_indices_per_client):
        for idx in indices:
            if idx < K:
                delivered_mask[cid][idx] = True

    delivered_counts = [len(idx) for idx in delivered_indices_per_client]
    total_delivered_blocks = sum(delivered_counts)

    # READY energies are in send order; successful DATA frames carry original
    # block indices. Invert the send order for value accounting by block index.
    energy_by_block = energies_by_block(client_energies, send_orders, num_clients, K)

    total_round_energy = float(np.sum(client_weights[:, None] ** 2 * energy_by_block))
    delivered_energy_vr = 0.0
    for cid in range(num_clients):
        w_sq = float(client_weights[cid]) ** 2
        for idx in delivered_indices_per_client[cid]:
            if idx < K:
                delivered_energy_vr += w_sq * float(energy_by_block[cid][idx])

    eta_r = (delivered_energy_vr / total_round_energy) if total_round_energy > 0 else 0.0

    d_seconds = D_ns / 1_000_000_000.0
    # DATA always carries a fixed L-coordinate fp16 block. The final block is
    # zero-padded on wire; n_coords is decoder metadata, not a shorter packet.
    delivered_payload_bytes = total_delivered_blocks * (2 * L_DEFAULT)
    goodput_bps = (delivered_payload_bytes * 8) / d_seconds if d_seconds > 0 else 0.0
    frames_sent_at_deadline = bottleneck.frames_sent
    utilization = (frames_sent_at_deadline * S * 8) / (C * d_seconds) if d_seconds > 0 else 0.0

    total_ge_drops = sum(a.drops_ge for a in access_links)
    total_access_q_drops = sum(a.drops_queue for a in access_links)

    rtt_median_ms = (float(np.median(rtt_samples)) / 1e6) if rtt_samples else 0.0
    rtt_p99_ms = (float(np.percentile(rtt_samples, 99)) / 1e6) if rtt_samples else 0.0
    queueing_delays_ms = [(b / float(C)) * 1000.0 for b in backlog_bits_samples]
    queueing_delay_mean_ms = float(np.mean(queueing_delays_ms)) if queueing_delays_ms else 0.0
    queueing_delay_p99_ms = float(np.percentile(queueing_delays_ms, 99)) if queueing_delays_ms else 0.0
    wasted_grants_ratio = (gateway.grants_timed_out / gateway.grants_issued) if gateway.grants_issued > 0 else 0.0
    clients_represented = int(sum(1 for cnt in delivered_counts if cnt > 0))
    deadline_completion = 1.0 if gateway.grants_timed_out == 0 else 0.0

    # Freeze every deadline metric before draining packets already in flight.
    metrics = {
        "goodput_bps": goodput_bps,
        "utilization": utilization,
        "frames_sent": frames_sent_at_deadline,
        "grants_issued": gateway.grants_issued,
        # Bounded probe grants issued this round (rule (v) amendment); the core's own counter.
        "probe_grants": int(getattr(gateway.core, "round_probe_grants", 0)),
        "grants_delivered": gateway.grants_delivered,
        "grants_timed_out": gateway.grants_timed_out,
        "wasted_grants_ratio": wasted_grants_ratio,
        "clients_represented": clients_represented,
        "deadline_completion": deadline_completion,
        "rtt_median_ms": rtt_median_ms,
        "rtt_p99_ms": rtt_p99_ms,
        "queueing_delay_mean_ms": queueing_delay_mean_ms,
        "queueing_delay_p99_ms": queueing_delay_p99_ms,
        "drops_ge": total_ge_drops,
        "drops_queue_access": total_access_q_drops,
        "drops_queue_bn_client": bottleneck.drops_queue_client,
        "drops_queue_bn_cross": bottleneck.drops_queue_cross,
        "late_after_timeout": gateway.late_after_timeout,
        "duplicate_packets": gateway.duplicate_packets,
    }

    # Continue only to observe late packet arrivals. These events must not
    # change the round's aggregate or any metric snapshot taken at D.
    engine.run_all()
    gateway._sync_core_stats()
    late_metrics = {"late_after_deadline": gateway.late_after_deadline}

    return {
        "round_id": round_id,
        "deadline_ms": D_ns / 1_000_000.0,
        "delivered_mask": delivered_mask,
        "delivered_counts": delivered_counts,
        "delivered_energy_vr": delivered_energy_vr,
        "eta": eta_r,
        "aggregate_G": emit_agg.G,
        "round_ends": round_ends,
        "gateway": gateway,
        "metrics": metrics,
        "late_metrics": late_metrics,
        # Keep sample-level values so callers can pool RTTs across warm rounds;
        # averaging round percentiles is not statistically equivalent.
        "rtt_samples_ns": tuple(int(value) for value in rtt_samples),
    }


__all__ = [
    "run_network_round",
    "compute_deadline_ns",
    "GatewaySim",
    "SendGrant",
    "SendRoundEnd",
    "EmitAggregate",
]
