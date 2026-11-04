"""Discrete-event execution of the B2 push and B3 plan baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fedgrant.baselines.b2_udp import (
    B2ClientCore,
    B2RoundAccumulator,
    B2RoundResult,
    apply_delivery_result,
    encode_data_packet,
)
from fedgrant.baselines.b3_pacing import data_frame_size_bytes, pacing_send_time_ns
from fedgrant.baselines.b3_plan import B3GatewayCore, B3RoundResult
from fedgrant.common.constants import MessageType
from fedgrant.common.wire import encode_message
from fedgrant.core.state import ClientState
from fedgrant.encoder.blocks import block_payload_f16
from fedgrant.encoder.blocks import EncodedUpdate
from fedgrant.mininet.profile_update import profile_bank, synthetic_update
from fedgrant.policy.constants import (
    C,
    D_MODEL,
    K as K_DEFAULT,
    L as L_DEFAULT,
    S,
)
from fedgrant.policy.plan import DeliveryRatioEWMA
from fedgrant.sim.engine import EventType, SimulationEngine
from fedgrant.sim.crosstraffic import CrossTrafficGenerator
from fedgrant.sim.ge import GilbertElliottModel
from fedgrant.sim.links import AccessLink, BottleneckLink, Frame, access_delay_ns, mean_access_rate_bps
from fedgrant.sim.traces import TraceManager
from fedgrant.sim.run_net import compute_deadline_ns

NS_PER_S = 1_000_000_000
BN_PROP_NS = 5_000_000
READY_JITTER_FRACTION = 0.10


@dataclass(slots=True)
class BaselineState:
    """Client error feedback and B3 gateway estimators retained across rounds."""

    mode: str
    clients: dict[int, B2ClientCore]
    b3_states: dict[int, ClientState] = field(default_factory=dict)
    b3_delivery_ewma: DeliveryRatioEWMA = field(default_factory=lambda: DeliveryRatioEWMA(init=0.1))
    last_round_id: int = 0

    @classmethod
    def create(cls, mode: str, num_clients: int, weights: np.ndarray) -> BaselineState:
        return cls(
            mode=mode,
            clients={cid: B2ClientCore(cid, d=D_MODEL, L=L_DEFAULT) for cid in range(num_clients)},
            b3_states=(
                {cid: ClientState(cid, float(weights[cid]), K_DEFAULT) for cid in range(num_clients)}
                if mode == "B3"
                else {}
            ),
        )


def _client_delta(
    cid: int,
    energies: np.ndarray,
    client_payloads: list[dict[int, np.ndarray]] | None,
    send_orders: list[np.ndarray] | None,
    seed: int,
    round_id: int,
) -> np.ndarray:
    """Build the same deterministic synthetic update used by Mininet clients."""
    if client_payloads is not None and cid < len(client_payloads) and client_payloads[cid]:
        delta = np.zeros(D_MODEL, dtype=np.float32)
        for block_idx, payload in client_payloads[cid].items():
            idx = int(block_idx)
            if not 0 <= idx < K_DEFAULT:
                raise ValueError(f"client_payloads[{cid}] has invalid block index {idx}")
            values = np.asarray(payload, dtype=np.float32).reshape(-1)
            count = min(L_DEFAULT, D_MODEL - idx * L_DEFAULT, values.size)
            delta[idx * L_DEFAULT : idx * L_DEFAULT + count] = values[:count]
        return delta

    if send_orders is not None:
        order = np.asarray(send_orders[cid])
        if order.shape != (K_DEFAULT,) or not np.array_equal(np.sort(order), np.arange(K_DEFAULT)):
            raise ValueError(f"send_orders[{cid}] must be a permutation of [0, {K_DEFAULT})")
        # The simulator's energy input is in transmission order; the synthetic
        # update helper expects coordinate block order.
        by_block = np.zeros(K_DEFAULT, dtype=np.float32)
        by_block[order] = np.asarray(energies, dtype=np.float32)
    else:
        by_block = np.asarray(energies, dtype=np.float32)
    if not np.isfinite(by_block).all() or np.any(by_block < 0):
        raise ValueError("client_energies must be finite and non-negative")
    update_seed = int(seed) + int(cid) * 1009 + int(round_id)
    return synthetic_update(by_block, seed=update_seed, d=D_MODEL, block_length=L_DEFAULT)


def _delivery_energy(result: B2RoundResult, client_energy_by_block: dict[int, np.ndarray], weights: np.ndarray) -> tuple[float, float, list[int]]:
    total = delivered = 0.0
    counts: list[int] = []
    for cid, cresult in result.client_results.items():
        block_energy = client_energy_by_block[cid]
        w_sq = float(weights[cid]) ** 2
        total += w_sq * float(np.sum(block_energy, dtype=np.float64))
        delivered += w_sq * sum(float(block_energy[j]) for j in cresult.delivered_indices)
        counts.append(cresult.packets_received)
    return (delivered / total if total > 0 else 0.0), delivered, counts


def run_baseline_round(
    mode: str,
    *,
    round_id: int = 1,
    num_clients: int = 20,
    client_weights: np.ndarray | None = None,
    client_energies: np.ndarray | None = None,
    client_payloads: list[dict[int, np.ndarray]] | None = None,
    send_orders: list[np.ndarray] | None = None,
    encoded_updates: list[EncodedUpdate] | None = None,
    beta: float = 0.10,
    scenario: str = "L2",
    split: str = "test",
    trace_dir: str | Path = "data/traces",
    seed: int = 0,
    state: BaselineState | None = None,
    round_start_abs_ns: int | None = None,
    loss_rate: float | None = None,
    mean_burst_length: float | None = None,
    cross_mbps: float = 0.0,
) -> dict[str, Any]:
    """Simulate B2 or B3 on the standard access links and bottleneck.

    B3's estimator receives only READY and DATA packets that arrive at the
    gateway. The simulator's GE and queue-drop counters are used for measured
    network metrics, never passed to the B3 gateway core.
    """
    mode = mode.upper()
    if mode not in {"B2", "B3"}:
        raise ValueError("mode must be B2 or B3")
    if round_id < 1 or num_clients < 1:
        raise ValueError("round_id and num_clients must be positive")
    if (loss_rate is None) != (mean_burst_length is None):
        raise ValueError("loss_rate and mean_burst_length must be provided together")
    if not np.isfinite(cross_mbps) or cross_mbps < 0:
        raise ValueError("cross_mbps must be finite and non-negative")

    K, L = K_DEFAULT, L_DEFAULT
    duration_ns = compute_deadline_ns(beta, num_clients, K)
    if round_start_abs_ns is None:
        round_start_abs_ns = (round_id - 1) * (1_000_000_000 + duration_ns)
    round_start_abs_ns = int(round_start_abs_ns)
    deadline_abs_ns = round_start_abs_ns + duration_ns
    if client_weights is None:
        client_weights = np.full(num_clients, 1.0 / num_clients, dtype=np.float32)
    weights = np.asarray(client_weights, dtype=np.float32)
    if weights.shape != (num_clients,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError(f"client_weights must be {num_clients} finite non-negative values")
    if not np.isclose(float(weights.sum()), 1.0):
        raise ValueError("client_weights must sum to 1")
    if client_energies is None:
        client_energies = np.stack([
            profile_bank(seed + cid, d=D_MODEL, block_length=L)[(round_id - 1) % 8]
            for cid in range(num_clients)
        ])
    energies = np.asarray(client_energies, dtype=np.float32)
    if energies.shape != (num_clients, K):
        raise ValueError(f"client_energies must have shape ({num_clients}, {K})")
    if send_orders is not None and len(send_orders) != num_clients:
        raise ValueError("send_orders must contain one order per client")
    if encoded_updates is not None:
        if len(encoded_updates) != num_clients:
            raise ValueError("encoded_updates must contain one update per client")
        if any(update.d != D_MODEL or update.L != L for update in encoded_updates):
            raise ValueError(f"encoded_updates must use d={D_MODEL}, L={L}")

    if state is None:
        state = BaselineState.create(mode, num_clients, weights)
    elif state.mode != mode or set(state.clients) != set(range(num_clients)):
        raise ValueError("baseline state mode/client set does not match this run")
    if round_id <= state.last_round_id:
        raise ValueError("round_id must increase monotonically for a baseline state")
    state.last_round_id = round_id

    seed_stream = np.random.SeedSequence(seed).spawn(6)
    rng_trace = np.random.default_rng(seed_stream[0])
    rng_ge = np.random.default_rng(seed_stream[1])
    rng_ready = np.random.default_rng(seed_stream[2])
    engine = SimulationEngine()
    engine.now_ns = round_start_abs_ns
    bottleneck = BottleneckLink(engine, capacity_bps=C, prop_delay_ns=BN_PROP_NS)
    effective_cross_mbps = 3.0 if scenario.upper() == "X2" and cross_mbps == 0 else cross_mbps
    cross_traffic = None
    if effective_cross_mbps > 0:
        cross_traffic = CrossTrafficGenerator(
            engine, bottleneck, rate_bps=effective_cross_mbps * 1_000_000.0,
            frame_size_bytes=S, rng=np.random.default_rng(seed_stream[5]),
        )
        cross_traffic.start()
    trace_mgr = TraceManager(num_clients, scenario, split, trace_dir, rng_trace)
    access_links: list[AccessLink] = []
    for cid in range(num_clients):
        ge_seed = int(rng_ge.integers(0, 2**31))
        ge = (
            GilbertElliottModel.for_scenario(scenario, seed=ge_seed)
            if loss_rate is None
            else GilbertElliottModel.for_loss_burst(loss_rate, mean_burst_length, seed=ge_seed)
        )
        link = AccessLink(
            client_id=cid,
            engine=engine,
            prop_delay_ns=access_delay_ns(cid),
            ge_model=ge,
            initial_rate_bps=trace_mgr.get_client_rate(cid, 0),
        )
        link.next_hop_cb = bottleneck.route_from_access
        access_links.append(link)

    client_encodings = {}
    client_energy_by_block: dict[int, np.ndarray] = {}
    for cid in range(num_clients):
        if encoded_updates is None:
            delta = _client_delta(cid, energies[cid], client_payloads, send_orders, seed, round_id)
            encoded = state.clients[cid].start_round(round_id, deadline_abs_ns, delta)
        else:
            encoded = encoded_updates[cid]
        client_encodings[cid] = encoded
        client_energy_by_block[cid] = np.sum(encoded.blocks * encoded.blocks, axis=1, dtype=np.float32)

    weights_by_id = {cid: float(weights[cid]) for cid in range(num_clients)}
    accumulator = B2RoundAccumulator(
        round_id=round_id,
        client_ids=range(num_clients),
        start_ns=round_start_abs_ns,
        deadline_ns=deadline_abs_ns,
        d=D_MODEL,
        L=L,
        client_weights=weights_by_id,
    )
    b3_core: B3GatewayCore | None = None
    if mode == "B3":
        b3_core = B3GatewayCore(
            round_id=round_id,
            start_ns=round_start_abs_ns,
            deadline_ns=deadline_abs_ns,
            d=D_MODEL,
            L=L,
            client_weights=weights_by_id,
            client_states=state.b3_states,
            delivery_ewma=state.b3_delivery_ewma,
        )

    rho_by_client: dict[int, int] = {}
    packets_sent_by_client = [0] * num_clients
    data_bytes_sent_by_client = [0] * num_clients
    b3_plans: dict[int, Any] = {}
    for cid in range(num_clients):
        rho_rel = int(rng_ready.integers(0, int(READY_JITTER_FRACTION * duration_ns) + 1))
        rho = round_start_abs_ns + rho_rel
        rho_by_client[cid] = rho
        if mode == "B2":
            encoded = client_encodings[cid]
            next_send_ns = rho
            for send_pos, block_idx_value in enumerate(encoded.order, start=1):
                send_ns = next_send_ns
                if send_ns >= deadline_abs_ns:
                    break
                block_idx = int(block_idx_value)
                payload, n_coords = block_payload_f16(encoded, block_idx)
                packet = encode_data_packet(
                    round_id,
                    cid,
                    block_idx,
                    n_coords,
                    payload,
                    send_pos=send_pos,
                    t_send_ns=send_ns,
                )
                frame = Frame(
                    frame_id=f"{mode}_r{round_id}_c{cid}_p{send_pos}",
                    client_id=cid,
                    size_bytes=S,
                    created_at_ns=send_ns,
                    payload={"packet": packet},
                )
                engine.schedule(
                    delay_ns=send_ns - engine.now_ns,
                    event_type=EventType.ACCESS_ENQUEUE,
                    callback=lambda _evt, f=frame, link=access_links[cid]: link.enqueue(f),
                )
                packets_sent_by_client[cid] += 1
                data_bytes_sent_by_client[cid] += len(packet)
                serialization_ns = math.ceil(S * 8 * NS_PER_S / mean_access_rate_bps(cid))
                next_send_ns = send_ns + serialization_ns
        else:
            assert b3_core is not None
            encoded = client_encodings[cid]
            ready_packet = encode_message(
                MessageType.READY,
                round_id=round_id,
                client_id=cid,
                t_send_ns=rho,
                body={"K": K, "energy": tuple(float(x) for x in encoded.energies_sorted)},
                block_length=L,
            )
            ready_arrival = rho + access_links[cid].prop_delay_ns + BN_PROP_NS
            engine.schedule(
                delay_ns=ready_arrival - engine.now_ns,
                event_type=EventType.READY_ARRIVE,
                callback=lambda _evt, packet=ready_packet, t=ready_arrival, core=b3_core: core.accept_ready(packet, t)
                if t <= core.plan_ns
                else None,
            )

    if mode == "B3":
        assert b3_core is not None
        plan_ns = b3_core.plan_ns

        def _send_plans(_evt: Any = None) -> None:
            plans = b3_core.compute_plans(engine.now_ns)
            for cid, _plan in plans.items():
                plan = b3_core.mark_plan_sent(cid, engine.now_ns)
                b3_plans[cid] = plan
                plan_arrival = engine.now_ns + BN_PROP_NS + access_links[cid].prop_delay_ns
                if plan_arrival >= deadline_abs_ns or plan.blocks == 0:
                    continue
                encoded = client_encodings[cid]
                for send_pos, block_idx_value in enumerate(encoded.order[: plan.blocks], start=1):
                    send_ns = pacing_send_time_ns(
                        send_pos,
                        t_start_ns=plan_arrival,
                        k=plan.blocks,
                        rate_bps=plan.rate_bps,
                        block_size_bytes=data_frame_size_bytes(L),
                    )
                    if send_ns >= deadline_abs_ns:
                        break
                    block_idx = int(block_idx_value)
                    payload, n_coords = block_payload_f16(encoded, block_idx)
                    packet = encode_data_packet(
                        round_id,
                        cid,
                        block_idx,
                        n_coords,
                        payload,
                        send_pos=send_pos,
                        t_send_ns=send_ns,
                    )
                    frame = Frame(
                        frame_id=f"B3_r{round_id}_c{cid}_p{send_pos}",
                        client_id=cid,
                        size_bytes=S,
                        created_at_ns=send_ns,
                        payload={"packet": packet},
                    )
                    engine.schedule(
                        delay_ns=send_ns - engine.now_ns,
                        event_type=EventType.ACCESS_ENQUEUE,
                        callback=lambda _evt, f=frame, link=access_links[cid]: link.enqueue(f),
                    )
                    packets_sent_by_client[cid] += 1
                    data_bytes_sent_by_client[cid] += len(packet)

        engine.schedule(delay_ns=plan_ns - engine.now_ns, event_type=EventType.TICK_1MS, callback=_send_plans)

    late_after_deadline = 0

    def _gateway_arrival(frame: Frame) -> None:
        nonlocal late_after_deadline
        if frame.client_id < 0:  # Cross traffic has no FedGrant DATA outcome.
            return
        packet = frame.payload["packet"]
        if engine.now_ns > deadline_abs_ns:
            late_after_deadline += 1
            return
        if mode == "B2":
            accumulator.accept_packet(packet, engine.now_ns)
        else:
            assert b3_core is not None
            b3_core.accept_data(packet, engine.now_ns)

    bottleneck.dest_cb = _gateway_arrival

    for t_rel_ns in range(100_000_000, duration_ns, 100_000_000):
        engine.schedule(
            delay_ns=t_rel_ns,
            event_type=EventType.TICK_1MS,
            callback=lambda _evt, t=t_rel_ns: [link.set_rate(trace_mgr.get_client_rate(cid, t)) for cid, link in enumerate(access_links)],
        )
    backlog_samples: list[float] = []
    for t_rel_ns in range(10_000_000, duration_ns, 10_000_000):
        engine.schedule(
            delay_ns=t_rel_ns,
            event_type=EventType.TICK_1MS,
            callback=lambda _evt: backlog_samples.append(bottleneck.backlog_bits),
        )

    engine.run_until(deadline_abs_ns)
    if cross_traffic is not None:
        cross_traffic.stop()
    if mode == "B2":
        accumulator.stop(deadline_abs_ns, "deadline")
        round_result: B2RoundResult = accumulator.result()
    else:
        assert b3_core is not None
        b3_result: B3RoundResult = b3_core.finish(deadline_abs_ns)
        round_result = b3_result.round_result

    if encoded_updates is None:
        for cid in range(num_clients):
            apply_delivery_result(state.clients[cid], round_result)

    delivered_mask = [[False] * K for _ in range(num_clients)]
    for cid, cresult in round_result.client_results.items():
        for idx in cresult.delivered_indices:
            delivered_mask[cid][idx] = True
    delivered_counts = [sum(delivered_mask[cid]) for cid in range(num_clients)]
    eta, delivered_energy, _ = _delivery_energy(round_result, client_energy_by_block, weights)
    total_delivered = sum(delivered_counts)
    seconds = duration_ns / NS_PER_S
    goodput = total_delivered * 8 * 1400 / seconds if seconds > 0 else 0.0
    utilization = bottleneck.frames_sent * S * 8 / (C * seconds) if seconds > 0 else 0.0
    queueing_ms = [float(bits / C * 1000) for bits in backlog_samples]
    total_ge_drops = sum(link.drops_ge for link in access_links)
    access_q_drops = sum(link.drops_queue for link in access_links)

    metrics: dict[str, Any] = {
        "goodput_bps": goodput,
        "utilization": utilization,
        "frames_sent": bottleneck.frames_sent,
        "grants_issued": None,
        "grants_delivered": None,
        "grants_timed_out": None,
        "clients_represented": sum(count > 0 for count in delivered_counts),
        "deadline_completion": None,
        "queueing_delay_mean_ms": float(np.mean(queueing_ms)) if queueing_ms else 0.0,
        "queueing_delay_p99_ms": float(np.percentile(queueing_ms, 99)) if queueing_ms else 0.0,
        "drops_ge": total_ge_drops,
        "drops_queue_access": access_q_drops,
        "drops_queue_bn_client": bottleneck.drops_queue_client,
        "drops_queue_bn_cross": bottleneck.drops_queue_cross,
        "late_after_timeout": None,
        "late_after_deadline": None,
        "wasted_grants_ratio": None,
        "rtt_median_ms": None,
        "rtt_p99_ms": None,
        "delivered_blocks": total_delivered,
        "data_packets_sent": int(sum(packets_sent_by_client)),
        "data_bytes_sent": int(sum(data_bytes_sent_by_client)),
        "data_packets_delivered": int(round_result.delivered_blocks),
    }
    if mode == "B3":
        metrics["plans_issued"] = len(b3_plans)
        metrics["blocks_planned"] = int(sum(plan.blocks for plan in b3_plans.values()))
    b3_rtt = {} if mode == "B2" else dict(b3_core.rtt_samples_ns)
    engine.run_all()
    metrics["late_after_deadline"] = late_after_deadline
    return {
        "round_id": round_id,
        "deadline_ms": duration_ns / 1_000_000.0,
        "delivered_mask": delivered_mask,
        "delivered_counts": delivered_counts,
        "delivered_energy_vr": delivered_energy,
        "eta": eta,
        "metrics": metrics,
        "late_metrics": {"late_after_deadline": late_after_deadline},
        "baseline_state": state,
        "baseline": mode,
        "b3_rtt_samples_ns": b3_rtt,
        "b3_plans": b3_plans,
        "client_packets_sent": packets_sent_by_client,
        "client_data_bytes_sent": data_bytes_sent_by_client,
        "round_result": round_result,
        "client_energy_by_block": client_energy_by_block,
    }


__all__ = ["BaselineState", "run_baseline_round"]
