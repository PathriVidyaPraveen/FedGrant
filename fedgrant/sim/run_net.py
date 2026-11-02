"""Network simulation driver executing round-by-round frame transmission (Section 6, Algorithm 1)."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from fedgrant.core.estimators import (
    RateWindow,
    compute_cap,
    grant_timeout,
    rtt_update,
    update_outcome_stats,
)
from fedgrant.core.state import ClientState, OutstandingGrant, values_from_energies
from fedgrant.policy.constants import (
    C,
    D_MODEL,
    K as K_DEFAULT,
    L as L_DEFAULT,
    S,
)
from fedgrant.policy.eligibility import is_eligible
from fedgrant.policy.features import (
    ArrivalLog,
    GatewayView,
    build_pred_input,
    context16,
    features48,
    observation8,
    resolve_event,
)
from fedgrant.policy.pacing import InflightTracker, p_target, pacing_p
from fedgrant.policy.selectors import (
    RoundRobinSelector,
    SRPTSelector,
    ValueSelector,
    heap_key,
    value_key,
)
from fedgrant.policy.types import CandidateView, PolicyConfig, Selector
from fedgrant.predictors.base import PredInput, Predictor, RoundCtx
from fedgrant.predictors.ewma import PEWMA
from fedgrant.predictors.factory import make_predictor
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


class GatewaySim:
    """Gateway simulation controller executing Algorithm 1 inside the network simulator."""

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
    ) -> None:
        self.N = int(num_clients)
        self.weights = np.ascontiguousarray(client_weights, dtype=np.float32)
        if len(self.weights) != self.N:
            raise ValueError(f"weights length ({len(self.weights)}) != num_clients ({self.N})")

        self.predictor = predictor
        self.cfg = policy_config or PolicyConfig(
            selector="value" if isinstance(selector, str) else "custom",
            predictor="ewma",
        )

        if isinstance(selector, str):
            s_name = selector.lower()
            if s_name == "value":
                self.selector: Selector = ValueSelector()
            elif s_name == "rr":
                self.selector = RoundRobinSelector()
            elif s_name == "srpt":
                self.selector = SRPTSelector()
            else:
                raise ValueError(f"Unknown selector '{selector}'")
        else:
            self.selector = selector

        self.D_ns = int(D_ns)
        self.K = int(K)
        self.L = int(L)
        self.d_model = int(d_model)
        self.rng = rng or np.random.default_rng()
        self.sample_hook = sample_hook
        self.shadow_oracle = shadow_oracle
        self.shadow_hmm = shadow_hmm

        # Persistent estimator state across rounds (Table 2)
        self.client_states = [ClientState(client_id=i, w=float(self.weights[i]), K=self.K) for i in range(self.N)]
        self.rate_windows = [RateWindow(t_start_ns=0, outstanding=0) for _ in range(self.N)]

        # Network trackers
        self.inflight = InflightTracker()
        self.arrivals = ArrivalLog()
        self.pewma = PEWMA()

        # Per-round state
        self.round_id: int = 1
        self.round_start_abs_ns: int = 0
        self.next_gid: int = 1
        self.heap: list[tuple[tuple[float, int], int]] = []
        self.delivered_payloads: list[dict[int, np.ndarray]] = [{} for _ in range(self.N)]
        self.rr_pointer: int = 0

        # Diagnostics / logging
        self.grants_issued: int = 0
        self.grants_delivered: int = 0
        self.grants_timed_out: int = 0
        self.duplicate_packets: int = 0
        self.late_after_timeout: int = 0
        self.late_after_deadline: int = 0

    @property
    def Phi(self) -> float:
        return self.inflight.phi

    def get_p_target(self) -> float:
        ready = [st.srtt_ns for st in self.client_states if st.ready]
        if ready:
            return float(p_target(ready))
        return float(p_target([st.srtt_ns for st in self.client_states]))

    def get_view(self, t_ns: int) -> GatewayView:
        return GatewayView(
            t_ns=t_ns,
            D_ns=self.D_ns,
            P_target=self.get_p_target(),
            inflight=self.inflight,
            arrivals=self.arrivals,
        )

    def on_round_start(self, round_id: int, t_ns: int = 0, round_start_abs_ns: int | None = None) -> None:
        """Procedure RoundStart(t): reset per-round state; Phi <- 0; next_gid <- 1."""
        self.round_id = int(round_id)
        if round_start_abs_ns is not None:
            self.round_start_abs_ns = int(round_start_abs_ns)
        self.inflight.clear()
        self.arrivals.reset()
        self.next_gid = 1
        self.heap.clear()
        self.delivered_payloads = [{} for _ in range(self.N)]
        self.grants_issued = 0
        self.grants_delivered = 0
        self.grants_timed_out = 0
        self.duplicate_packets = 0
        self.late_after_timeout = 0
        self.late_after_deadline = 0

        ctx = RoundCtx(round_id=self.round_id, D_ns=self.D_ns, N=self.N, K=self.K)
        self.predictor.reset_round(ctx)
        self.pewma.reset_round(ctx)
        if self.shadow_hmm is not None:
            self.shadow_hmm.reset_round(ctx)
        if self.shadow_oracle is not None:
            self.shadow_oracle.reset_round(ctx)

        for st in self.client_states:
            st.reset_round(K=self.K, round_start_ns=self.round_start_abs_ns + t_ns)

        for rw in self.rate_windows:
            rw.start(t_start_ns=t_ns, outstanding=0)

    def on_ready(self, client_id: int, energies: np.ndarray, t_ns: int) -> list[SendGrant]:
        """Procedure OnReady(i, eps, t): register sorted energies, mark ready, push to heap, pump."""
        st = self.client_states[client_id]
        st.set_ready(values_from_energies(energies, st.w))

        v_tilde = value_key(st, self.cfg.retransmit)
        heapq.heappush(self.heap, (heap_key(v_tilde, client_id), client_id))

        return self.pump(t_ns)

    def pump(self, t_ns: int) -> list[SendGrant]:
        """Procedure Pump(t): select eligible clients, check pacing, issue grants."""
        ready_srtts = [st.srtt_ns for st in self.client_states if st.ready]
        if not ready_srtts:
            return []

        P_targ = self.get_p_target()
        grants: list[SendGrant] = []

        while True:
            # 1. Pop heap entries until the top client is valid and eligible (rules i-v)
            top_client: int | None = None
            while self.heap:
                h_key, cid = self.heap[0]
                st = self.client_states[cid]

                # Check if position is pending
                if not st.has_pending(self.cfg.retransmit):
                    heapq.heappop(self.heap)
                    continue

                # Query predictor for client
                p_hat = self._predict_one(cid, t_ns)
                eligible, _reason = is_eligible(st, t_ns, self.D_ns, p_hat, self.cfg)

                if not eligible:
                    heapq.heappop(self.heap)
                    continue

                # Verify current key matches heap key
                cur_v = value_key(st, self.cfg.retransmit)
                if heap_key(cur_v, cid) != h_key:
                    heapq.heappop(self.heap)
                    heapq.heappush(self.heap, (heap_key(cur_v, cid), cid))
                    continue

                top_client = cid
                break

            if top_client is None:
                break

            # 2. Select client (exploration or configured selector)
            if self.cfg.explore_eps > 0.0 and self.rng.random() < self.cfg.explore_eps:
                eligible_cids = self._get_eligible_clients(t_ns)
                if not eligible_cids:
                    break
                sel_cid = int(self.rng.choice(eligible_cids))
            elif isinstance(self.selector, ValueSelector):
                sel_cid = top_client
            else:
                cand_view = self._build_candidate_view(t_ns)
                if len(cand_view) == 0:
                    break
                sel_cid = self.selector.select(cand_view)

            st_sel = self.client_states[sel_cid]
            view = self.get_view(t_ns)
            x48 = features48(st_sel, view)
            inp_sel = PredInput.from_x48(np.array([sel_cid], dtype=np.int32), x48[None, :], t_ns)

            p_hat_sel = float(self.predictor.predict(inp_sel)[0])
            p_pacing = pacing_p(p_hat_sel, self.cfg)

            # 3. Pacing check: Phi + p_pacing <= P_target
            if not self.inflight.can_issue(p_pacing, P_targ):
                break

            # 4. Pop from heap if top_client was chosen
            if top_client == sel_cid and self.heap:
                heapq.heappop(self.heap)

            # Shadow predictions for ML logging
            p_ewma_val = float(self.pewma.predict(inp_sel)[0])
            p_oracle_val = (
                float(self.shadow_oracle.predict(inp_sel)[0])
                if self.shadow_oracle is not None
                else p_hat_sel
            )
            p_hmm_val = (
                float(self.shadow_hmm.predict(inp_sel)[0])
                if self.shadow_hmm is not None
                else p_hat_sel
            )

            # 5. Take send position s
            s_next = st_sel.next_pos(self.cfg.retransmit)
            st_sel.take_position(s_next, self.cfg.retransmit)

            # 6. Issue grant g
            g_id = self.next_gid
            self.next_gid += 1

            tau_g = grant_timeout(t_ns, st_sel, self.D_ns)

            st_sel.outstanding[g_id] = OutstandingGrant(
                pos=s_next,
                t_issue_ns=t_ns,
                p_issue=p_pacing,
                timeout_ns=tau_g,
            )
            self.inflight.add(g_id, p_pacing)
            self.grants_issued += 1
            if isinstance(self.selector, RoundRobinSelector):
                self.rr_pointer = (sel_cid + 1) % self.N

            # Update rate window tracker
            self.rate_windows[sel_cid].set_outstanding(t_ns, len(st_sel.outstanding))

            # Push back if still pending
            if st_sel.has_pending(self.cfg.retransmit):
                v_cur = value_key(st_sel, self.cfg.retransmit)
                heapq.heappush(self.heap, (heap_key(v_cur, sel_cid), sel_cid))

            grant_action = SendGrant(client_id=sel_cid, grant_id=g_id, pos=s_next)
            grants.append(grant_action)

            # Record ML training sample hook if enabled
            if self.sample_hook is not None:
                self.sample_hook({
                    "event": "grant_issued",
                    "client_id": sel_cid,
                    "grant_id": g_id,
                    "pos": s_next,
                    "t_ns": t_ns,
                    "abs_t_ns": self.round_start_abs_ns + t_ns,
                    "tau_ns": tau_g,
                    "p_hat": p_hat_sel,
                    "p_ewma": p_ewma_val,
                    "p_hmm": p_hmm_val,
                    "p_oracle": p_oracle_val,
                    "x48": x48,
                    "ctx": context16(x48),
                })

        return grants

    def on_data(
        self,
        client_id: int,
        grant_id: int,
        pos: int,
        block_idx: int,
        payload: np.ndarray,
        t_ns: int,
    ) -> list[SendGrant]:
        """Procedure OnData(i, g, s, j, payload, t): record arrival, resolve grant, update state."""
        if t_ns > self.D_ns:
            self.late_after_deadline += 1
            return []
        self.arrivals.record(t_ns, 8 * S)

        st = self.client_states[client_id]

        # 1. Store delivered block
        self.rate_windows[client_id].advance(st, t_ns)
        self.rate_windows[client_id].add_delivery(t_ns, 8 * S)
        first_delivery = st.mark_delivered(pos)
        if first_delivery:
            self.delivered_payloads[client_id][block_idx] = payload
        else:
            self.duplicate_packets += 1

        # 2. Resolve grant
        if grant_id in st.outstanding:
            self._resolve(client_id, grant_id, y=1, t_ns=t_ns)
            self.grants_delivered += 1
        else:
            self.late_after_timeout += 1

        return self.pump(t_ns)

    def on_tick(self, t_ns: int) -> list[SendGrant]:
        """Procedure OnTick(t): check timeouts, periodic p_hat & rate updates, pump."""
        for cid, st in enumerate(self.client_states):
            self.rate_windows[cid].advance(st, t_ns)

            # Check timeouts for outstanding grants
            expired_gids = [
                gid for gid, og in st.outstanding.items() if og.timeout_ns <= t_ns
            ]
            for gid in expired_gids:
                og = st.outstanding.get(gid)
                if og is None:
                    continue
                pos = og.pos
                self._resolve(cid, gid, y=0, t_ns=t_ns)
                st.mark_timeout(pos)
                self.grants_timed_out += 1

                if st.has_pending(self.cfg.retransmit):
                    v_cur = value_key(st, self.cfg.retransmit)
                    heapq.heappush(self.heap, (heap_key(v_cur, cid), cid))

        if t_ns % 5_000_000 == 0:
            self._rebuild_heap(t_ns)

        if t_ns % 100_000_000 == 0:
            for st in self.client_states:
                st.cap = compute_cap(st)

        return self.pump(t_ns)

    def on_deadline(self, t_ns: int) -> tuple[list[SendRoundEnd], EmitAggregate]:
        """Procedure OnDeadline(D): resolve outstanding grants with y=0, aggregate updates."""
        for cid, st in enumerate(self.client_states):
            for gid in list(st.outstanding.keys()):
                og = st.outstanding.get(gid)
                if og is None:
                    continue
                pos = og.pos
                self._resolve(cid, gid, y=0, t_ns=t_ns)
                st.mark_timeout(pos)
                self.grants_timed_out += 1

        round_ends: list[SendRoundEnd] = []
        decoded_updates: list[np.ndarray] = []

        for cid, st in enumerate(self.client_states):
            st.update_starve()
            deliv_map = self.delivered_payloads[cid]
            deliv_indices = sorted(deliv_map.keys())

            n_bytes = int(math.ceil(self.K / 8.0))
            bitmap = np.zeros(n_bytes, dtype=np.uint8)
            for j in deliv_indices:
                bitmap[j // 8] |= (1 << (j % 8))

            round_ends.append(SendRoundEnd(client_id=cid, bitmap=bitmap, delivered_indices=deliv_indices))

            u_hat = np.zeros(self.d_model, dtype=np.float32)
            for j, block_payload in deliv_map.items():
                start = j * self.L
                end = min(start + self.L, self.d_model)
                n_coords = end - start
                u_hat[start:end] = block_payload[:n_coords].astype(np.float32)

            decoded_updates.append(u_hat)

        G_hat = np.zeros(self.d_model, dtype=np.float32)
        for w, u_hat in zip(self.weights, decoded_updates, strict=True):
            G_hat += float(w) * u_hat

        return round_ends, EmitAggregate(G=G_hat)

    def _resolve(self, client_id: int, grant_id: int, y: int, t_ns: int) -> None:
        """Procedure Resolve(i, g, y, t): update RFC 6298 estimators, EWMA, and observe."""
        st = self.client_states[client_id]
        if grant_id not in st.outstanding or grant_id not in self.inflight:
            return

        og = st.outstanding[grant_id]
        rtt_sample = (t_ns - og.t_issue_ns) if y == 1 else None

        view = self.get_view(t_ns)
        obs = resolve_event(st, view, grant_id, y, t_ns, rtt_sample)

        self.rate_windows[client_id].set_outstanding(t_ns, len(st.outstanding))

        self.predictor.observe(client_id, y, t_ns, obs)
        self.pewma.observe(client_id, y, t_ns, obs)
        if self.shadow_hmm is not None:
            self.shadow_hmm.observe(client_id, y, t_ns, obs)
        if self.shadow_oracle is not None:
            self.shadow_oracle.observe(client_id, y, t_ns, obs)

        if self.sample_hook is not None:
            self.sample_hook({
                "event": "grant_resolved",
                "client_id": client_id,
                "grant_id": grant_id,
                "y": y,
                "t_ns": t_ns,
                "abs_t_ns": self.round_start_abs_ns + t_ns,
                "obs": obs,
                "rtt_ns": rtt_sample,
            })

    def _predict_one(self, client_id: int, t_ns: int) -> float:
        view = self.get_view(t_ns)
        inp = build_pred_input([self.client_states[client_id]], view)
        return float(self.predictor.predict(inp)[0])

    def _get_eligible_clients(self, t_ns: int) -> list[int]:
        eligible: list[int] = []
        for cid, st in enumerate(self.client_states):
            p = self._predict_one(cid, t_ns)
            if is_eligible(st, t_ns, self.D_ns, p, self.cfg)[0]:
                eligible.append(cid)
        return eligible

    def _build_candidate_view(self, t_ns: int) -> CandidateView:
        cids: list[int] = []
        vals: list[float] = []
        undeliv: list[int] = []

        for cid, st in enumerate(self.client_states):
            p = self._predict_one(cid, t_ns)
            if is_eligible(st, t_ns, self.D_ns, p, self.cfg)[0]:
                cids.append(cid)
                vals.append(value_key(st, self.cfg.retransmit))
                undeliv.append(self.K - st.m - len(st.outstanding))

        return CandidateView(
            client_id=np.array(cids, dtype=np.int32),
            next_value=np.array(vals, dtype=np.float64),
            undelivered=np.array(undeliv, dtype=np.int32),
            rr_pointer=self.rr_pointer,
        )

    def _rebuild_heap(self, t_ns: int) -> None:
        new_heap: list[tuple[tuple[float, int], int]] = []
        for cid, st in enumerate(self.client_states):
            if st.has_pending(self.cfg.retransmit):
                v_cur = value_key(st, self.cfg.retransmit)
                new_heap.append((heap_key(v_cur, cid), cid))
        heapq.heapify(new_heap)
        self.heap = new_heap


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
) -> dict[str, Any]:
    """Execute a single simulated round of FedGrant frame transmission (Section 6)."""
    cfg = policy_config or PolicyConfig(selector="value", predictor="ewma")
    K = K_DEFAULT
    L = L_DEFAULT
    D_ns = compute_deadline_ns(beta, num_clients, K)

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
        ge_model = GilbertElliottModel.for_scenario(scenario, seed=ge_seed)
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
            elif cfg.predictor == "hmm":
                pred = shadow_hmm
            elif cfg.predictor == "ewma":
                pred = PEWMA()
            elif cfg.predictor in {"mlp", "gru"}:
                pred = make_predictor(cfg.predictor, models_dir="models")
            else:
                raise ValueError(f"Unsupported simulator predictor {cfg.predictor!r}")
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
        )
        shadow_oracle.client_states = gateway.client_states
    else:
        gateway.D_ns = D_ns
        gateway.cfg = cfg
        gateway.sample_hook = sample_hook
        if predictor is not None:
            gateway.predictor = predictor
        gateway.shadow_oracle = shadow_oracle
        shadow_oracle.client_states = gateway.client_states
        if gateway.shadow_hmm is None:
            gateway.shadow_hmm = shadow_hmm
        if predictor is None:
            if cfg.predictor == "oracle":
                gateway.predictor = shadow_oracle
            elif cfg.predictor == "hmm":
                gateway.predictor = gateway.shadow_hmm
            elif cfg.predictor == "ewma":
                gateway.predictor = PEWMA()
            elif cfg.predictor in {"mlp", "gru"}:
                gateway.predictor = make_predictor(cfg.predictor, models_dir="models")

    # 4. Cross-traffic (Scenario X2 only)
    crosstraffic: CrossTrafficGenerator | None = None
    if scenario.upper() == "X2":
        crosstraffic = CrossTrafficGenerator(
            engine=engine,
            bottleneck=bottleneck,
            rate_bps=3_000_000.0,
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

            if gid in gateway.client_states[cid].outstanding:
                og = gateway.client_states[cid].outstanding[gid]
                rtt_samples.append(engine.now_ns - og.t_issue_ns)

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
    gateway.on_round_start(round_id=round_id, t_ns=0, round_start_abs_ns=round_start_abs_ns)

    for cid in range(num_clients):
        max_rho_ns = int(0.10 * D_ns)
        rho_ns = int(rng_readiness.integers(0, max_rho_ns + 1)) if max_rho_ns > 0 else 0
        delta_ns = access_links[cid].prop_delay_ns
        ready_arrival_ns = rho_ns + delta_ns + 5_000_000

        engine.schedule(
            delay_ns=ready_arrival_ns,
            event_type=EventType.READY_ARRIVE,
            callback=lambda _evt, c=cid: _on_ready_arrive(c),
        )

    def _on_ready_arrive(cid: int) -> None:
        energies = client_energies[cid]
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
