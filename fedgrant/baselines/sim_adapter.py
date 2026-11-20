"""Discrete-event adapters for the B2 and B3 network-only baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fedgrant.baselines.b2_udp import (
    B2ClientTransfer,
    B2RoundAccumulator,
    B2RoundResult,
    encode_data_packet,
)
from fedgrant.baselines.b3_pacing import (
    data_frame_size_bytes,
    pacing_send_time_ns,
)
from fedgrant.baselines.b3_plan import (
    B3ClientPlan,
    B3GatewayCore,
    B3PlanError,
    B3RoundResult,
    B3_DELIVERY_RATIO_INIT,
)
from fedgrant.common.constants import MessageType, TransferMode
from fedgrant.common.wire import decode_message, encode_message
from fedgrant.core.state import ClientState
from fedgrant.policy.constants import C, D_MODEL, K as K_DEFAULT, L as L_DEFAULT, S
from fedgrant.policy.plan import DeliveryRatioEWMA
from fedgrant.sim.crosstraffic import CrossTrafficGenerator
from fedgrant.sim.engine import EventType, SimulationEngine
from fedgrant.sim.ge import GilbertElliottModel
from fedgrant.sim.links import AccessLink, BottleneckLink, Frame, access_delay_ns, mean_access_rate_bps
from fedgrant.sim.run_net import compute_deadline_ns
from fedgrant.sim.traces import TraceManager


@dataclass(slots=True)
class BaselineSimulationState:
    protocol: str
    client_states: dict[int, ClientState] = field(default_factory=dict)
    delivery_ewma: DeliveryRatioEWMA = field(
        default_factory=lambda: DeliveryRatioEWMA(init=B3_DELIVERY_RATIO_INIT)
    )


@dataclass(slots=True)
class _Transfer:
    client_id: int
    round_id: int
    expected_packets: int
    sent_indices: list[int] = field(default_factory=list)
    send_times_ns: list[int] = field(default_factory=list)
    application_bytes_sent: int = 0
    completed_ns: int = 0
    stop_reason: str = "deadline"

    def result(self) -> B2ClientTransfer:
        return B2ClientTransfer(
            client_id=self.client_id,
            round_id=self.round_id,
            sent_indices=tuple(self.sent_indices),
            send_times_ns=tuple(self.send_times_ns),
            application_bytes_sent=self.application_bytes_sent,
            completed_ns=self.completed_ns,
            stop_reason=self.stop_reason,
        )


BOTTLENECK_PROP_NS = 5_000_000


def control_leg_ns(access_prop_delay_ns: int) -> int:
    """One-way gateway<->client control delay: access delta_i + bottleneck 5 ms (sim/run_net.py)."""
    return int(access_prop_delay_ns) + BOTTLENECK_PROP_NS


def run_baseline_network_round(
    *,
    round_id: int,
    num_clients: int,
    beta: float,
    scenario: str,
    policy: str,
    seed: int,
    gateway: BaselineSimulationState | None = None,
    trace_dir: str | Path = "data/traces",
    d: int = D_MODEL,
    L: int = L_DEFAULT,
    deadline_ns: int | None = None,
    client_weights: np.ndarray | None = None,
    client_energies: np.ndarray | None = None,
    cross_mbps: float = 0.0,
) -> dict[str, Any]:
    """Run one B2 or B3 round over the shared discrete-event network model.

    ``client_weights`` and ``client_energies`` take the same frozen value-bank
    inputs as the grant simulator: one weight per client and one descending
    energy profile per client in send-position order. Omitted inputs keep the
    legacy uniform weights and synthetic profile. ``cross_mbps`` configures
    bottleneck cross traffic; the legacy ``scenario="X2"`` alias still means
    3 Mbit/s when no explicit rate is given.
    """
    if policy not in {"B2", "B3"}:
        raise ValueError("policy must be B2 or B3")
    if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 1:
        raise ValueError("round_id must be a positive integer")
    if isinstance(num_clients, bool) or not isinstance(num_clients, int) or num_clients < 1:
        raise ValueError("num_clients must be a positive integer")
    if isinstance(d, bool) or not isinstance(d, int) or d < 1:
        raise ValueError("d must be a positive integer")
    if isinstance(L, bool) or not isinstance(L, int) or L < 1:
        raise ValueError("L must be a positive integer")

    K = (d + L - 1) // L
    duration_ns = compute_deadline_ns(beta, num_clients, K) if deadline_ns is None else deadline_ns
    if isinstance(duration_ns, bool) or not isinstance(duration_ns, int) or duration_ns < 1:
        raise ValueError("deadline_ns must be a positive integer")
    round_start_abs_ns = (round_id - 1) * (1_000_000_000 + duration_ns)
    deadline_abs_ns = round_start_abs_ns + duration_ns
    if gateway is None:
        gateway = BaselineSimulationState(protocol=policy)
    elif gateway.protocol != policy:
        raise ValueError("persistent baseline state does not match policy")

    if isinstance(cross_mbps, bool) or not isinstance(cross_mbps, (int, float, np.floating)) or not math.isfinite(float(cross_mbps)) or cross_mbps < 0:
        raise ValueError("cross_mbps must be finite and non-negative")
    if client_weights is None:
        weights = np.full(num_clients, 1.0 / num_clients, dtype=np.float64)
    else:
        weights = np.asarray(client_weights, dtype=np.float64)
        if weights.shape != (num_clients,) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError(f"client_weights must be {num_clients} finite non-negative values")
        if not np.isclose(float(weights.sum()), 1.0):
            raise ValueError("client_weights must sum to 1")
    if client_energies is None:
        energy_rows = np.tile(np.linspace(1.0, 0.01, K, dtype=np.float32), (num_clients, 1))
    else:
        energy_rows = np.asarray(client_energies, dtype=np.float32)
        if energy_rows.shape != (num_clients, K):
            raise ValueError(f"client_energies must have shape ({num_clients}, {K})")
        if not np.isfinite(energy_rows).all() or np.any(energy_rows < 0):
            raise ValueError("client_energies must be finite and non-negative")
        if np.any(np.diff(energy_rows, axis=1) > 0):
            raise ValueError("client_energies must be sorted descending in send-position order")
    seed_streams = np.random.SeedSequence(seed).spawn(6)
    rng_trace = np.random.default_rng(seed_streams[0])
    rng_ge = np.random.default_rng(seed_streams[1])
    rng_readiness = np.random.default_rng(seed_streams[2])
    rng_cross = np.random.default_rng(seed_streams[5])

    engine = SimulationEngine()
    engine.now_ns = round_start_abs_ns
    bottleneck = BottleneckLink(engine=engine, capacity_bps=C, prop_delay_ns=BOTTLENECK_PROP_NS, max_queue_frames=64)
    trace_scenario = "X2" if scenario.upper() == "X2" else scenario
    traces = TraceManager(
        num_clients=num_clients,
        scenario=trace_scenario,
        split="test",
        trace_dir=trace_dir,
        rng=rng_trace,
    )
    access_links: list[AccessLink] = []
    for client_id in range(num_clients):
        ge = GilbertElliottModel.for_scenario(
            trace_scenario,
            seed=int(rng_ge.integers(0, 2**31)),
        )
        link = AccessLink(
            client_id=client_id,
            engine=engine,
            prop_delay_ns=access_delay_ns(client_id),
            ge_model=ge,
            max_queue_frames=32,
            initial_rate_bps=traces.get_client_rate(client_id, 0),
        )
        link.next_hop_cb = bottleneck.route_from_access
        access_links.append(link)

    cross_traffic: CrossTrafficGenerator | None = None
    effective_cross_mbps = 3.0 if trace_scenario == "X2" and cross_mbps == 0 else float(cross_mbps)
    if effective_cross_mbps > 0:
        cross_traffic = CrossTrafficGenerator(
            engine=engine,
            bottleneck=bottleneck,
            rate_bps=effective_cross_mbps * 1_000_000.0,
            frame_size_bytes=S,
            rng=rng_cross,
        )
        cross_traffic.start()

    def update_rates(_event, time_ns: int) -> None:
        for cid, link in enumerate(access_links):
            link.set_rate(traces.get_client_rate(cid, time_ns))

    for t_ns in range(100_000_000, duration_ns, 100_000_000):
        engine.schedule(
            delay_ns=t_ns,
            event_type=EventType.TICK_1MS,
            callback=lambda event, time_ns=t_ns: update_rates(event, time_ns),
        )

    queue_samples: list[dict[str, float | int | str]] = []
    for t_ns in range(10_000_000, duration_ns, 10_000_000):
        engine.schedule(
            delay_ns=t_ns,
            event_type=EventType.TICK_1MS,
            callback=lambda _event, time_ns=t_ns: queue_samples.append({
                "t_ns": round_start_abs_ns + time_ns,
                "backlog_bytes": bottleneck.backlog_bytes,
                "backlog_bits": bottleneck.backlog_bits,
                "drops": bottleneck.drops_queue,
            }),
        )

    coords = np.full(K, L, dtype=np.int64)
    coords[-1] = d - (K - 1) * L
    data_size_bytes = data_frame_size_bytes(L)
    transfers = {
        cid: _Transfer(
            cid,
            round_id,
            K if policy == "B2" else 0,
            stop_reason="deadline" if policy == "B2" else "plan_not_received",
        )
        for cid in range(num_clients)
    }
    packet_events: list[dict[str, Any]] = []
    round_events: list[dict[str, Any]] = []
    late_after_deadline = 0

    if policy == "B2":
        accumulator = B2RoundAccumulator(
            round_id=round_id,
            client_ids=tuple(range(num_clients)),
            start_ns=round_start_abs_ns,
            deadline_ns=deadline_abs_ns,
            d=d,
            L=L,
            client_weights={cid: float(weights[cid]) for cid in range(num_clients)},
        )
        b3_core = None
        plans: dict[int, B3ClientPlan] = {}
    else:
        if not gateway.client_states:
            gateway.client_states = {
                cid: ClientState(cid, float(weights[cid]), K)
                for cid in range(num_clients)
            }
        elif set(gateway.client_states) != set(range(num_clients)):
            raise ValueError("persistent B3 client states do not match num_clients")
        b3_core = B3GatewayCore(
            round_id=round_id,
            start_ns=round_start_abs_ns,
            deadline_ns=deadline_abs_ns,
            d=d,
            L=L,
            client_weights={cid: float(weights[cid]) for cid in range(num_clients)},
            client_states=gateway.client_states,
            delivery_ewma=gateway.delivery_ewma,
        )
        accumulator = None
        plans = {}

    def enqueue_data(client_id: int, send_pos: int, packet: bytes) -> None:
        block_idx = send_pos - 1
        frame = Frame(
            frame_id=f"{policy}-r{round_id}-c{client_id}-p{send_pos}",
            client_id=client_id,
            size_bytes=data_size_bytes,
            created_at_ns=engine.now_ns,
            payload={"packet": packet, "send_pos": send_pos, "block_idx": block_idx},
        )
        access_links[client_id].enqueue(frame)

    def on_gateway_arrival(frame: Frame) -> None:
        nonlocal late_after_deadline
        if frame.client_id < 0:
            return
        packet = frame.payload["packet"]
        if engine.now_ns > deadline_abs_ns:
            late_after_deadline += 1
            return
        if accumulator is not None:
            accumulator.accept_packet(packet, engine.now_ns)
        elif b3_core is not None:
            b3_core.accept_data(packet, engine.now_ns)

    bottleneck.dest_cb = on_gateway_arrival

    def send_data(client_id: int, send_pos: int) -> None:
        tracker = transfers[client_id]
        if engine.now_ns >= deadline_abs_ns:
            return
        block_idx = send_pos - 1
        payload = np.zeros(L, dtype=np.float16)
        n_coords = int(coords[block_idx])
        packet = encode_data_packet(
            round_id,
            client_id,
            block_idx,
            n_coords,
            payload,
            send_pos=send_pos,
            t_send_ns=engine.now_ns,
        )
        tracker.sent_indices.append(block_idx)
        tracker.send_times_ns.append(engine.now_ns)
        tracker.application_bytes_sent += len(packet)
        packet_events.append({
            "t_ns": engine.now_ns,
            "round_id": round_id,
            "client_id": client_id,
            "packet_seq": send_pos,
            "packet_type": "DATA",
            "bytes": len(packet),
            "result": "sent",
        })
        enqueue_data(client_id, send_pos, packet)

    rho_by_client: dict[int, int] = {}
    for cid in range(num_clients):
        max_rho_ns = int(0.10 * duration_ns)
        rho_by_client[cid] = (
            int(rng_readiness.integers(0, max_rho_ns + 1))
            if max_rho_ns > 0 else 0
        )

    if policy == "B2":
        for cid in range(num_clients):
            tracker = transfers[cid]
            pace_bps = mean_access_rate_bps(cid)
            # rho_i starts when the client receives ROUND_START (CHANGELOG 2026-11-13),
            # i.e. one control leg after the round start, as in baseline_proc.
            first_send_ns = (round_start_abs_ns + control_leg_ns(access_links[cid].prop_delay_ns)
                             + rho_by_client[cid])
            for position in range(1, K + 1):
                send_ns = first_send_ns + (position - 1) * math.ceil(
                    (data_size_bytes * 8 * 1_000_000_000) / pace_bps
                )
                if send_ns >= deadline_abs_ns:
                    break
                engine.schedule(
                    delay_ns=send_ns - engine.now_ns,
                    event_type=EventType.ACCESS_ENQUEUE,
                    callback=lambda _event, client=cid, pos=position: send_data(client, pos),
                )
    else:
        assert b3_core is not None
        for cid in range(num_clients):
            rho_us = rho_by_client[cid] // 1_000
            start_packet = encode_message(
                MessageType.ROUND_START,
                round_id=round_id,
                client_id=cid,
                t_send_ns=round_start_abs_ns,
                body={
                    "deadline_ns": deadline_abs_ns,
                    "K": K,
                    "L": L,
                    "rho_us": rho_us,
                    "mode": int(TransferMode.PLAN),
                },
                block_length=L,
            )
            if not start_packet:
                raise RuntimeError("ROUND_START encoding produced an empty datagram")
            start_message = decode_message(start_packet, block_length=L)
            rho_us = start_message.body["rho_us"]
            # Same READY timing as sim/run_net.py: ROUND_START downlink, then rho_i
            # from its receipt, then the READY uplink (CHANGELOG 2026-11-13).
            leg_ns = control_leg_ns(access_links[cid].prop_delay_ns)
            ready_send_ns = round_start_abs_ns + leg_ns + rho_us * 1_000
            ready_packet = encode_message(
                MessageType.READY,
                round_id=round_id,
                client_id=cid,
                t_send_ns=ready_send_ns,
                body={"K": K, "energy": tuple(float(value) for value in energy_rows[cid])},
                block_length=L,
            )
            ready_ns = ready_send_ns + leg_ns

            def on_ready(_event, client=cid, packet=ready_packet) -> None:
                try:
                    b3_core.accept_ready(packet, engine.now_ns)
                    round_events.append({
                        "t_ns": engine.now_ns,
                        "round_id": round_id,
                        "event": "ready_received",
                        "client_id": client,
                        "detail": "",
                    })
                    if b3_core.ready_complete:
                        # Early close: the last expected READY is in (CHANGELOG 2026-11-21).
                        send_plans(None)
                except B3PlanError as exc:
                    if "outside the round start-to-plan window" not in str(exc):
                        raise
                    round_events.append({
                        "t_ns": engine.now_ns,
                        "round_id": round_id,
                        "event": "ready_rejected",
                        "client_id": client,
                        "detail": "outside_plan_window",
                    })

            engine.schedule(
                delay_ns=ready_ns - engine.now_ns,
                event_type=EventType.READY_ARRIVE,
                callback=on_ready,
            )

        def send_plans(_event) -> None:
            # At the last READY (early close) or at the cap, whichever comes first.
            nonlocal plans
            if b3_core.plan_computed:
                return
            plans = b3_core.compute_plans(engine.now_ns)
            for cid, pending_plan in plans.items():
                plan = b3_core.mark_plan_sent(cid, engine.now_ns)
                plan_packet = encode_message(
                    MessageType.PLAN,
                    round_id=round_id,
                    client_id=cid,
                    t_send_ns=engine.now_ns,
                    body={"k": plan.blocks, "rate_bps": plan.rate_bps},
                    block_length=L,
                )
                packet_events.append({
                    "t_ns": engine.now_ns,
                    "round_id": round_id,
                    "client_id": cid,
                    "packet_seq": 0,
                    "packet_type": "PLAN",
                    "bytes": len(plan_packet),
                    "result": "sent",
                })
                round_events.append({
                    "t_ns": engine.now_ns,
                    "round_id": round_id,
                    "event": "plan_sent",
                    "client_id": cid,
                    "detail": f"blocks={pending_plan.blocks};rate_bps={pending_plan.rate_bps}",
                })
                return_delay_ns = control_leg_ns(access_links[cid].prop_delay_ns)
                transfer = transfers[cid]
                transfer.expected_packets = plan.blocks

                def on_plan_arrival(_plan_event, client=cid, sent_plan=plan) -> None:
                    tracker = transfers[client]
                    if engine.now_ns >= deadline_abs_ns:
                        tracker.stop_reason = "plan_not_received"
                        return
                    if sent_plan.blocks == 0:
                        tracker.stop_reason = "all_blocks_sent"
                        tracker.completed_ns = engine.now_ns
                        return
                    tracker.stop_reason = "deadline"
                    frame_size = data_frame_size_bytes(L)
                    for position in range(1, sent_plan.blocks + 1):
                        target_ns = pacing_send_time_ns(
                            position,
                            t_start_ns=engine.now_ns,
                            k=sent_plan.blocks,
                            rate_bps=sent_plan.rate_bps,
                            block_size_bytes=frame_size,
                        )
                        if target_ns >= deadline_abs_ns:
                            break
                        engine.schedule(
                            delay_ns=target_ns - engine.now_ns,
                            event_type=EventType.ACCESS_ENQUEUE,
                            callback=lambda _data_event, client_id=client, pos=position: send_data(client_id, pos),
                        )

                engine.schedule(
                    delay_ns=return_delay_ns,
                    event_type=EventType.ACCESS_ENQUEUE,
                    callback=on_plan_arrival,
                )

        engine.schedule(
            delay_ns=b3_core.plan_cap_ns - engine.now_ns,
            event_type=EventType.TICK_1MS,
            callback=send_plans,
        )

    engine.run_until(deadline_abs_ns)
    frames_sent_at_deadline = bottleneck.frames_sent
    access_queue_drops_at_deadline = sum(link.drops_queue for link in access_links)
    # GE channel losses of DATA on the access links (READY/PLAN are not sent through them).
    ge_data_drops_at_deadline = sum(link.drops_ge for link in access_links)
    bottleneck_data_drops_at_deadline = bottleneck.drops_queue_client
    data_queue_drops_at_deadline = access_queue_drops_at_deadline + bottleneck_data_drops_at_deadline
    cross_queue_drops_at_deadline = bottleneck.drops_queue_cross
    if cross_traffic is not None:
        cross_traffic.stop()
    if accumulator is not None:
        accumulator.stop(deadline_abs_ns, "deadline")
        round_result: B2RoundResult = accumulator.result()
        delivered_by_client = {
            cid: result.delivered_indices
            for cid, result in round_result.client_results.items()
        }
        plans_sent = None
        planned_blocks = None
        rtt_samples_ns: dict[int, int] = {}
        rtt_sample_times_ns: dict[int, int] = {}
        protocol_result: B2RoundResult | B3RoundResult = round_result
    else:
        assert b3_core is not None
        b3_result: B3RoundResult = b3_core.finish(deadline_abs_ns)
        round_result = b3_result.round_result
        delivered_by_client = {
            cid: result.delivered_indices
            for cid, result in round_result.client_results.items()
        }
        plans_sent = len(b3_result.plans)
        planned_blocks = sum(plan.blocks for plan in b3_result.plans.values())
        rtt_samples_ns = dict(b3_result.rtt_samples_ns)
        rtt_sample_times_ns = {
            cid: b3_result.plans[cid].sent_ns + sample_ns
            for cid, sample_ns in rtt_samples_ns.items()
        }
        protocol_result = b3_result

    for cid, tracker in transfers.items():
        if policy == "B2":
            tracker.stop_reason = (
                "all_blocks_sent"
                if len(tracker.sent_indices) == K
                else "deadline"
            )
        elif tracker.stop_reason != "plan_not_received":
            tracker.stop_reason = (
                "all_blocks_sent"
                if len(tracker.sent_indices) == tracker.expected_packets
                else "deadline"
            )
        if tracker.stop_reason == "all_blocks_sent":
            tracker.completed_ns = (
                tracker.send_times_ns[-1]
                if tracker.send_times_ns
                else tracker.completed_ns
            )
        else:
            tracker.completed_ns = deadline_abs_ns
        round_events.append({
            "t_ns": tracker.completed_ns,
            "round_id": round_id,
            "event": "client_transfer_end",
            "client_id": cid,
            "detail": (
                f"stop_reason={tracker.stop_reason};packets_sent={len(tracker.sent_indices)};"
                f"application_bytes_sent={tracker.application_bytes_sent}"
            ),
        })

    engine.run_all()

    total_energy = float(np.sum((weights[:, None] ** 2) * energy_rows.astype(np.float64)))
    delivered_energy = sum(
        float(weights[cid] ** 2) * float(energy_rows[cid, block_idx])
        for cid, indices in delivered_by_client.items()
        for block_idx in indices
    )
    eta = delivered_energy / total_energy if total_energy > 0 else 0.0
    # Same accounting boundary as the grant simulator (run_net) and the FL
    # baseline simulator: every DATA carries a fixed L-coordinate fp16 block
    # (the final block is zero-padded on the wire), so delivered payload bytes
    # are unique delivered blocks x 2L. n_coords is decoder metadata only.
    delivered_payload_bytes = sum(len(indices) for indices in delivered_by_client.values()) * 2 * L
    duration_s = duration_ns / 1_000_000_000.0
    queueing_delays_ms = [
        (float(sample["backlog_bits"]) / float(C)) * 1000.0
        for sample in queue_samples
    ]
    delivered_counts = [len(delivered_by_client[cid]) for cid in range(num_clients)]
    data_packets_sent = sum(len(transfer.sent_indices) for transfer in transfers.values())
    data_bytes_sent = sum(transfer.application_bytes_sent for transfer in transfers.values())
    metrics: dict[str, float | int | None] = {
        "goodput_bps": (delivered_payload_bytes * 8 / duration_s) if duration_s > 0 else 0.0,
        "utilization": (frames_sent_at_deadline * S * 8 / (C * duration_s)) if duration_s > 0 else 0.0,
        "frames_sent": frames_sent_at_deadline,
        "clients_represented": sum(count > 0 for count in delivered_counts),
        "deadline_completion": None,
        "queueing_delay_mean_ms": float(np.mean(queueing_delays_ms)) if queueing_delays_ms else 0.0,
        "queueing_delay_p99_ms": float(np.percentile(queueing_delays_ms, 99)) if queueing_delays_ms else 0.0,
        "drops_queue_data": data_queue_drops_at_deadline,
        "drops_queue_access": access_queue_drops_at_deadline,
        # GE (channel) drops of DATA, separate from the two queues (attribution of missing blocks).
        "drops_ge_data": ge_data_drops_at_deadline,
        # Spec queue drops: bottleneck drop-tail drops of FedGrant frames.
        "drops_queue_bn_client": bottleneck_data_drops_at_deadline,
        "drops_queue_cross": cross_queue_drops_at_deadline,
        "wasted_grants_ratio": None,
        "late_after_timeout": None,
        "late_after_deadline": late_after_deadline,
        "grants_issued": None,
        "grants_delivered": None,
        "grants_timed_out": None,
        "data_packets_sent": data_packets_sent,
        "data_bytes_sent": data_bytes_sent,
        "planned_blocks": planned_blocks,
        "plan_messages_sent": plans_sent,
        "protocol_overhead": None,
    }
    return {
        "round_id": round_id,
        "deadline_ms": duration_ns / 1_000_000.0,
        "gateway": gateway,
        "eta": eta,
        "delivered_counts": delivered_counts,
        "metrics": metrics,
        "late_metrics": {"late_after_deadline": late_after_deadline},
        "round_result": round_result,
        "protocol_result": protocol_result,
        "plans": protocol_result.plans if policy == "B3" else None,
        "transfers": {cid: transfer.result() for cid, transfer in transfers.items()},
        "rtt_samples_ns": rtt_samples_ns,
        "rtt_sample_times_ns": rtt_sample_times_ns,
        "round_events": round_events,
        "packet_events": packet_events,
        "queue_samples": queue_samples,
    }


__all__ = ["BaselineSimulationState", "run_baseline_network_round"]
