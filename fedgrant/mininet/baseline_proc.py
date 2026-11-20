"""Mininet gateway and client processes for baselines B1 (TCP), B2 (UDP push) and B3 (PLAN).

Real sockets only; simulator adapters are never used here (Section 7: Mininet
runs only M1/M2). Protocol logic is reused, not rewritten:

* B1: ``B1TcpGateway`` / ``B1RoundAccumulator`` / ``B1TcpClient`` (TCP 9100).
* B2: ``B2RoundAccumulator`` / ``B2UdpClient`` (UDP 9000 -> 9001).
* B3: ``B3UdpGateway`` / ``B3UdpClient`` / ``B3GatewayCore`` (UDP 9000 <-> 9001).

This module adds the multi-round lifecycle (``round_loop.run_round_loop``:
Tc, ROUND_START with absolute D and per-client rho_us, D, late drain), the
ROUND_START/ROUND_END control path for B1/B2 (design decision D-08),
bank-backed network-only updates (D-10) and raw logs in the ``logschema``
formats (D-12). B1/B2 clients do not send READY (D-07); eta is joined after
the run from client energy logs (D-09, ``baseline_metrics``).

CLI (inside Mininet hosts)::

    python -m fedgrant.mininet.baseline_proc gateway --policy B2 --clients 20 --rounds 100 \
        --seed 0 --beta 0.10 --log-dir <dir> --round-state <file>
    python -m fedgrant.mininet.baseline_proc client --policy B2 --client-id 3 --clients 20 \
        --rounds 100 --seed 0 --log-file <dir>/client_3.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from fedgrant.baselines.b1_tcp import (
    B1ClientCore,
    B1RoundAccumulator,
    B1TcpClient,
    B1TcpGateway,
    decode_block_frame,
)
from fedgrant.baselines.b2_udp import (
    B2ClientCore,
    B2ProtocolError,
    B2RoundAccumulator,
    B2TransferError,
    B2UdpClient,
)
from fedgrant.baselines.b3_plan import B3PlanError, B3UdpClient, B3UdpGateway
from fedgrant.common.constants import MessageType, TransferMode
from fedgrant.common.logschema import PACKET_COLUMNS, ROUND_COLUMNS, ROUND_METRIC_COLUMNS, RTT_COLUMNS
from fedgrant.common.wire import WireProtocolError, decode_message, encode_message
from fedgrant.encoder.blocks import encode
from fedgrant.encoder.ef import ErrorFeedback, bitmap_from_indices, indices_from_bitmap
from fedgrant.mininet.round_loop import RoundSchedule, run_round_loop
from fedgrant.mininet.round_state import RoundStateFile
from fedgrant.mininet.value_profiles import BankUpdateProvider
from fedgrant.policy.constants import D_MODEL, L as L_DEFAULT
from fedgrant.sim.links import mean_access_rate_bps
from fedgrant.sim.run_net import compute_deadline_ns
from fedgrant.sim.value_bank import ValueBankInputs, load_value_bank

BASELINES = ("B1", "B2", "B3")
MODES = {"B1": TransferMode.TCP, "B2": TransferMode.PUSH, "B3": TransferMode.PLAN}
UDP_GATEWAY_PORT = 9000  # Section 7
UDP_CLIENT_PORT = 9001
TCP_GATEWAY_PORT = 9100
ROUND_END_GRACE_NS = 200_000_000  # Section 9
GATEWAY_HOST = "10.0.0.254"  # same addressing as gateway_proc / client_proc
VALUE_BANK_PATH = Path("data/bank/femnist_full_s0.npz")


def mininet_client_endpoints(num_clients: int, port: int = UDP_CLIENT_PORT) -> dict[int, tuple[str, int]]:
    return {cid: (f"10.0.0.{cid + 1}", port) for cid in range(num_clients)}


@dataclass(frozen=True)
class BaselineConfig:
    policy: str
    num_clients: int
    rounds: int
    master_seed: int
    schedule: RoundSchedule
    client_endpoints: Mapping[int, tuple[str, int]]
    gateway_host: str = GATEWAY_HOST
    gateway_bind_host: str = "0.0.0.0"
    udp_port: int = UDP_GATEWAY_PORT
    tcp_port: int = TCP_GATEWAY_PORT
    d: int = D_MODEL
    L: int = L_DEFAULT
    # M1 is network-only: like the simulator, each round's values are the bank
    # profile itself, so the client carries no error-feedback residual (D-19).
    # ROUND_END is still sent and processed; only the residual is discarded.
    network_only: bool = True
    # M2 (end-to-end FL): the gateway sends each round's aggregate to the FL
    # server at this (host, port); None for network-only runs.
    aggregate_endpoint: tuple[str, int] | None = None

    def __post_init__(self) -> None:
        if self.policy not in BASELINES:
            raise ValueError(f"baseline_proc runs only {BASELINES}; FedGrant policies use gateway_proc")
        if self.num_clients < 1 or self.rounds < 1:
            raise ValueError("num_clients and rounds must be positive")
        if set(self.client_endpoints) != set(range(self.num_clients)):
            raise ValueError("client_endpoints must cover exactly clients 0..N-1")

    @property
    def K(self) -> int:
        return (self.d + self.L - 1) // self.L


class _CsvLog:
    def __init__(self, path: Path, columns: tuple[str, ...]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=columns, extrasaction="raise", restval="")
        self._writer.writeheader()

    def write(self, **row: Any) -> None:
        self._writer.writerow(row)
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


class _GatewayLogs:
    """``logschema`` CSVs plus the per-round delivery record used for eta (D-09)."""

    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.dir = log_dir
        self.packets = _CsvLog(log_dir / "packets.csv", PACKET_COLUMNS)
        self.rounds = _CsvLog(log_dir / "rounds.csv", ROUND_COLUMNS)
        self.rtt = _CsvLog(log_dir / "rtt_samples.csv", RTT_COLUMNS)
        self.metrics = _CsvLog(log_dir / "round_metrics.csv", ROUND_METRIC_COLUMNS)
        self._deliveries = (log_dir / "gateway_rounds.jsonl").open("w", encoding="utf-8")
        self.summary: list[dict[str, Any]] = []

    def packet(self, *, t_ns: int, round_id: int, client_id: int, direction: str, packet_type: str,
               size: int, result: str, send_pos: Any = "", block_idx: Any = "", payload_bytes: int = 0) -> None:
        self.packets.write(t_mono_ns=t_ns, round_id=round_id, direction=direction, client_id=client_id,
                           packet_seq="", grant_id="", send_pos=send_pos, block_idx=block_idx,
                           packet_type=packet_type, bytes=size, payload_bytes=payload_bytes, result=result)

    def event(self, t_ns: int, round_id: int, event: str, client_id: Any = "", detail: str = "") -> None:
        self.rounds.write(t_mono_ns=t_ns, round_id=round_id, event=event, client_id=client_id, detail=detail)

    def deliveries(self, record: Mapping[str, Any]) -> None:
        self._deliveries.write(json.dumps(record, sort_keys=True) + "\n")
        self._deliveries.flush()

    def close(self) -> None:
        for log in (self.packets, self.rounds, self.rtt, self.metrics):
            log.close()
        self._deliveries.close()
        (self.dir / "gateway_summary.json").write_text(json.dumps(self.summary, indent=2) + "\n")


def _encode_round_start(cfg: BaselineConfig, round_id: int, cid: int, deadline_ns: int, rho_us: int) -> bytes:
    return encode_message(
        MessageType.ROUND_START, round_id=round_id, client_id=cid, t_send_ns=time.monotonic_ns(),
        body={"deadline_ns": deadline_ns, "K": cfg.K, "L": cfg.L, "rho_us": int(rho_us),
              "mode": int(MODES[cfg.policy])},
        block_length=cfg.L,
    )


def _encode_round_end(cfg: BaselineConfig, round_id: int, cid: int, delivered) -> bytes:
    delivered = sorted(int(i) for i in delivered)
    return encode_message(
        MessageType.ROUND_END, round_id=round_id, client_id=cid, t_send_ns=time.monotonic_ns(),
        body={"K": cfg.K, "n_delivered": len(delivered), "bitmap": bitmap_from_indices(delivered, cfg.K)},
        block_length=cfg.L,
    )


@dataclass
class _RoundBook:
    round_id: int
    start_ns: int
    deadline_ns: int
    rho_us: dict[int, int]
    delivered: dict[int, list[int]] = field(default_factory=dict)
    late_after_deadline: int = 0
    accepted: int = 0
    rejected: int = 0
    invalid: int = 0  # undecodable or unexpected-type datagrams
    extra: dict[str, Any] = field(default_factory=dict)


class _ControlEndpoint(asyncio.DatagramProtocol):
    """Gateway UDP socket for B1/B2: control messages out, B2 DATA in."""

    def __init__(self, gateway: "BaselineGateway") -> None:
        self.gateway = gateway
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport) -> None:
        self.transport = transport

    def datagram_received(self, data: bytes, addr) -> None:
        self.gateway._on_datagram(data, time.monotonic_ns())

    def error_received(self, exc: Exception) -> None:
        self.gateway.socket_errors += 1


class _LoggingB3Gateway(B3UdpGateway):
    """B3UdpGateway with raw packet logging; protocol behaviour unchanged."""

    def __init__(self, *, owner: "BaselineGateway", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.owner = owner

    def datagram_received(self, data: bytes, addr) -> None:
        received_ns = self.clock_ns()  # one timestamp for both the core and the log
        before = self.rejected_datagrams
        self.receive_datagram(data, addr, received_ns)
        self.owner._log_b3_datagram(data, received_ns, accepted=self.rejected_datagrams == before)

    def _send_message(self, message_type, *, client_id: int, t_send_ns: int, body) -> None:
        super()._send_message(message_type, client_id=client_id, t_send_ns=t_send_ns, body=body)
        core = self.core
        size = len(encode_message(message_type, round_id=core.round_id, client_id=client_id,
                                  t_send_ns=t_send_ns, body=body, block_length=self.L))
        self.owner.logs.packet(t_ns=self.clock_ns(), round_id=core.round_id, client_id=client_id,
                               direction="sent", packet_type=MessageType(message_type).name,
                               size=size, result="sent")


class BaselineGateway:
    """Round hooks for ``run_round_loop`` driving one baseline gateway process."""

    def __init__(self, cfg: BaselineConfig, client_weights: np.ndarray, log_dir: Path) -> None:
        weights = np.asarray(client_weights, dtype=np.float64)
        if weights.shape != (cfg.num_clients,) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("client_weights must have one value per client and sum to 1")
        self.cfg = cfg
        self.weights = {cid: float(weights[cid]) for cid in range(cfg.num_clients)}
        self.logs = _GatewayLogs(log_dir)
        self.books: dict[int, _RoundBook] = {}
        self.current: _RoundBook | None = None
        self.socket_errors = 0
        self._control: _ControlEndpoint | None = None
        self._b3: _LoggingB3Gateway | None = None
        self._b2: B2RoundAccumulator | None = None
        self._b1: B1TcpGateway | None = None
        self._aggregate_tasks: set[asyncio.Task] = set()
        self.aggregate_transfer_errors = 0

    async def open(self) -> None:
        loop = asyncio.get_running_loop()
        local = (self.cfg.gateway_bind_host, self.cfg.udp_port)
        if self.cfg.policy == "B3":
            self._b3 = _LoggingB3Gateway(
                owner=self, client_weights=self.weights, endpoints=dict(self.cfg.client_endpoints),
                d=self.cfg.d, duration_ns=self.cfg.schedule.duration_ns, L=self.cfg.L,
                host=self.cfg.gateway_bind_host, port=self.cfg.udp_port,
            )
            await self._b3.bind()
        else:
            _, self._control = await loop.create_datagram_endpoint(lambda: _ControlEndpoint(self), local_addr=local)

    async def close(self) -> None:
        # M2: let the last round's aggregate reach the FL server before exit.
        if self._aggregate_tasks:
            await asyncio.wait(tuple(self._aggregate_tasks), timeout=20.0)
            for task in tuple(self._aggregate_tasks):
                task.cancel()
            await asyncio.gather(*self._aggregate_tasks, return_exceptions=True)
        if self._b1 is not None:
            await self._b1.close()
        if self._b3 is not None:
            await self._b3.close()
        if self._control is not None and self._control.transport is not None:
            self._control.transport.close()
        self.logs.close()

    # ---- round hooks ------------------------------------------------------
    async def start_round(self, round_id: int, start_ns: int, deadline_ns: int, rho_us: dict[int, int]) -> None:
        cfg = self.cfg
        if cfg.policy == "B3":
            core = self._b3.start_round(round_id, rho_us=rho_us)
            start_ns, deadline_ns = core.start_ns, core.deadline_ns  # B3 core owns its window
        book = _RoundBook(round_id, start_ns, deadline_ns, dict(rho_us))
        self.books[round_id] = book
        self.current = book
        self.logs.event(start_ns, round_id, "round_start", detail=f"deadline_ns={deadline_ns};mode={cfg.policy}")
        if cfg.policy == "B3":
            return
        clients = tuple(range(cfg.num_clients))
        if cfg.policy == "B2":
            self._b2 = B2RoundAccumulator(round_id=round_id, client_ids=clients, start_ns=start_ns,
                                          deadline_ns=deadline_ns, d=cfg.d, L=cfg.L, client_weights=self.weights)
        else:
            accumulator = _LoggingB1Accumulator(self, round_id=round_id, client_ids=clients, start_ns=start_ns,
                                                deadline_ns=deadline_ns, d=cfg.d, L=cfg.L,
                                                client_weights=self.weights)
            self._b1 = B1TcpGateway(accumulator, host=cfg.gateway_bind_host, port=cfg.tcp_port)
            await self._b1.start()  # listening before any client can reach readiness
        for cid in clients:
            self._send_control(_encode_round_start(cfg, round_id, cid, deadline_ns, rho_us[cid]), cid,
                               round_id, "ROUND_START")

    async def on_deadline(self, round_id: int, deadline_ns: int) -> None:
        book = self.books[round_id]
        cfg = self.cfg
        if cfg.policy == "B3":
            result = await self._b3.wait()
            self._b3.send_round_end(result)
            core = self._b3.core
            for cid, rtt_ns in sorted(result.rtt_samples_ns.items()):
                send_pos, arrival_ns = core.rtt_first_data[cid]
                plan_sent_ns = result.plans[cid].sent_ns
                self.logs.rtt.write(t_mono_ns=arrival_ns, round_id=round_id, rtt_ms=rtt_ns / 1e6,
                                    rtt_type="plan_data", client_id=cid)
                self.logs.event(arrival_ns, round_id, "plan_data_rtt", cid,
                                f"send_pos={send_pos};plan_sent_ns={plan_sent_ns};arrival_ns={arrival_ns};"
                                f"rtt_ns={rtt_ns}")
            book.extra.update(planned_blocks=sum(p.blocks for p in result.plans.values()),
                              plan_messages_sent=self._b3.plan_local_handoffs,
                              plan_send_attempts=self._b3.plan_send_attempts,
                              plan_data_rtt_samples=len(result.rtt_samples_ns))
            round_result = result.round_result
        elif cfg.policy == "B2":
            self._b2.stop(deadline_ns, "deadline")
            round_result = self._b2.result()
        else:
            round_result = await self._b1.wait()
            self._b1 = None
        book.delivered = {cid: sorted(int(i) for i in r.delivered_indices)
                          for cid, r in round_result.client_results.items()}
        if cfg.policy != "B3":
            for cid in range(cfg.num_clients):
                self._send_control(_encode_round_end(cfg, round_id, cid, book.delivered[cid]), cid,
                                   round_id, "ROUND_END")
        self.logs.event(time.monotonic_ns(), round_id, "round_end_sent")
        if cfg.aggregate_endpoint is not None:
            self._send_aggregate(round_id, round_result)

    def _send_aggregate(self, round_id: int, round_result) -> None:
        """M2: G_r = sum_i w_i u_hat_i (zero-filled, as fl/server.py) to the FL server."""
        from fedgrant.net.fl_tcp import TcpAggregateSender

        cfg = self.cfg
        aggregate = round_result.aggregate(weighted_sum)
        n_coords = {cid: {int(j): block_coords(cfg.d, cfg.L, int(j)) for j in result.delivered_indices}
                    for cid, result in round_result.client_results.items()}
        sender = TcpAggregateSender(*cfg.aggregate_endpoint)
        task = asyncio.create_task(sender.send_aggregate(round_id, aggregate, n_coords))
        self._aggregate_tasks.add(task)
        task.add_done_callback(self._aggregate_done)

    def _aggregate_done(self, task: asyncio.Task) -> None:
        self._aggregate_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self.aggregate_transfer_errors += 1
            self.logs.event(time.monotonic_ns(), self.current.round_id if self.current else 0,
                            "aggregate_transfer_failed", detail=repr(task.exception())[:200])

    def finalize_round(self, round_id: int) -> None:
        book = self.books[round_id]
        cfg = self.cfg
        delivered = sum(len(v) for v in book.delivered.values())
        duration_s = (book.deadline_ns - book.start_ns) / 1e9
        self.logs.metrics.write(
            round_id=round_id, round_start_mono_ns=book.start_ns, deadline_mono_ns=book.deadline_ns,
            eta="",  # joined after the run from client energies (D-09)
            delivered_blocks=delivered,
            clients_represented=sum(1 for v in book.delivered.values() if v),
            goodput_bps=delivered * 2 * cfg.L * 8 / duration_s,
            late_after_deadline=book.late_after_deadline,
            planned_blocks=book.extra.get("planned_blocks", ""),
            plan_messages_sent=book.extra.get("plan_messages_sent", ""),
            plan_data_rtt_samples=book.extra.get("plan_data_rtt_samples", ""),
        )
        self.logs.deliveries({"round_id": round_id, "start_ns": book.start_ns, "deadline_ns": book.deadline_ns,
                              "policy": cfg.policy, "rho_us": {str(k): v for k, v in book.rho_us.items()},
                              "delivered": {str(k): v for k, v in book.delivered.items()}})
        self.logs.summary.append({"round_id": round_id, "start_mono_ns": book.start_ns,
                                  "deadline_mono_ns": book.deadline_ns, "accepted_data": book.accepted,
                                  "rejected_data": book.rejected, "invalid_events": book.invalid,
                                  "late_after_deadline": book.late_after_deadline, **book.extra})

    # ---- packet accounting -------------------------------------------------
    def _send_control(self, packet: bytes, cid: int, round_id: int, kind: str) -> None:
        self._control.transport.sendto(packet, tuple(self.cfg.client_endpoints[cid]))
        self.logs.packet(t_ns=time.monotonic_ns(), round_id=round_id, client_id=cid, direction="sent",
                         packet_type=kind, size=len(packet), result="sent")

    def _classify(self, book: _RoundBook | None, round_id: int, received_ns: int, accepted: bool) -> str:
        if accepted:
            if book is not None:
                book.accepted += 1
            return "accepted"
        if book is not None and book.round_id == round_id and received_ns > book.deadline_ns:
            book.late_after_deadline += 1
            return "late_after_deadline"
        if book is not None:
            book.rejected += 1
        return "rejected"

    def _count_invalid(self) -> None:
        if self.current is not None:
            self.current.invalid += 1

    def _on_datagram(self, data: bytes, received_ns: int) -> None:
        if self.cfg.policy != "B2":
            return  # B1 data uses TCP; its UDP socket only sends control messages
        try:
            message = decode_message(data, block_length=self.cfg.L)
        except WireProtocolError:
            self._count_invalid()
            self.logs.packet(t_ns=received_ns, round_id=0, client_id=-1, direction="received",
                             packet_type="malformed", size=len(data), result="rejected")
            return
        accepted = False
        if message.message_type is not MessageType.DATA:
            self._count_invalid()  # B2 clients send only DATA
        elif self._b2 is not None:
            try:
                self._b2.accept_packet(data, received_ns)
                accepted = True
            except B2ProtocolError:
                accepted = False
        result = self._classify(self.current, message.round_id, received_ns, accepted)
        body = message.body
        self.logs.packet(t_ns=received_ns, round_id=message.round_id, client_id=message.client_id,
                         direction="received", packet_type=MessageType(message.message_type).name, size=len(data),
                         result=result, send_pos=body.get("send_pos", ""), block_idx=body.get("block_idx", ""),
                         payload_bytes=2 * self.cfg.L if message.message_type is MessageType.DATA else 0)

    def _log_b3_datagram(self, data: bytes, received_ns: int, *, accepted: bool) -> None:
        try:
            message = decode_message(data, block_length=self.cfg.L)
        except WireProtocolError:
            self._count_invalid()
            self.logs.packet(t_ns=received_ns, round_id=0, client_id=-1, direction="received",
                             packet_type="malformed", size=len(data), result="rejected")
            return
        if message.message_type is MessageType.READY:
            # accepted_data / rejected_data / late_after_deadline count DATA only
            # (same meaning as GatewayCore's counters); READY is logged, not counted.
            result = "accepted" if accepted else "rejected"
        else:
            if message.message_type is not MessageType.DATA:
                self._count_invalid()
            result = self._classify(self.current, message.round_id, received_ns, accepted)
        body = message.body
        self.logs.packet(t_ns=received_ns, round_id=message.round_id, client_id=message.client_id,
                         direction="received", packet_type=MessageType(message.message_type).name, size=len(data),
                         result=result, send_pos=body.get("send_pos", ""), block_idx=body.get("block_idx", ""),
                         payload_bytes=2 * self.cfg.L if message.message_type is MessageType.DATA else 0)


class _LoggingB1Accumulator(B1RoundAccumulator):
    def __init__(self, owner: BaselineGateway, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.owner = owner

    def accept_frame(self, client_id: int, frame: bytes, received_ns: int) -> None:
        book = self.owner.current
        try:
            super().accept_frame(client_id, frame, received_ns)
        except Exception:
            self.owner._classify(book, self.round_id, received_ns, False)
            self.owner.logs.packet(t_ns=received_ns, round_id=self.round_id, client_id=client_id,
                                   direction="received", packet_type="B1_BLOCK", size=len(frame),
                                   result="rejected")
            raise
        block_idx, _n_coords, _payload = decode_block_frame(frame, self.L)
        self.owner._classify(book, self.round_id, received_ns, True)
        self.owner.logs.packet(t_ns=received_ns, round_id=self.round_id, client_id=client_id,
                               direction="received", packet_type="B1_BLOCK", size=len(frame),
                               result="accepted", block_idx=block_idx, payload_bytes=2 * self.L)


async def run_gateway(cfg: BaselineConfig, client_weights: np.ndarray, log_dir: Path,
                      round_state: RoundStateFile | None = None) -> list:
    gateway = BaselineGateway(cfg, client_weights, Path(log_dir))
    await gateway.open()
    try:
        return await run_round_loop(rounds=cfg.rounds, schedule=cfg.schedule, hooks=gateway,
                                    master_seed=cfg.master_seed, num_clients=cfg.num_clients,
                                    round_state=round_state)
    finally:
        await gateway.close()


# ---------------------------------------------------------------- clients


class _ClientLogs:
    def __init__(self, log_file: Path, client_id: int) -> None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = log_file
        self.packets = _CsvLog(log_file.with_name(f"client_{client_id}_packets.csv"), PACKET_COLUMNS)
        self._rounds = log_file.with_name(f"client_{client_id}_rounds.jsonl").open("w", encoding="utf-8")

    def round(self, record: Mapping[str, Any]) -> None:
        self._rounds.write(json.dumps(record, sort_keys=True) + "\n")
        self._rounds.flush()

    def close(self) -> None:
        self.packets.close()
        self._rounds.close()


class _ClientControl(asyncio.DatagramProtocol):
    def __init__(self, client: "PushOrTcpClient") -> None:
        self.client = client
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport) -> None:
        self.transport = transport

    def datagram_received(self, data: bytes, addr) -> None:
        self.client._on_control(data, tuple(addr[:2]), time.monotonic_ns())

    def error_received(self, exc: Exception) -> None:
        self.client.socket_errors += 1


class _RecordingB1Core(B1ClientCore):
    """B1ClientCore that remembers the last update it encoded.

    ``B1TcpClient.send_round`` encodes the update and, when the TCP connect fails
    before D, finalizes the round itself (empty delivery), which drops the encoded
    update. The energies of that update are still real and are needed for eta
    (D-09); without them every such round's eta is missing.
    """

    prepared: tuple[int, Any] | None = None

    def start_round(self, round_id: int, deadline_ns: int, delta: np.ndarray):
        encoded = super().start_round(round_id, deadline_ns, delta)
        self.prepared = (round_id, encoded)
        return encoded

    def prepared_encoded(self, round_id: int):
        return self.prepared[1] if self.prepared is not None and self.prepared[0] == round_id else None


class PushOrTcpClient:
    """B1/B2 client: ROUND_START -> wait rho_i -> send; ROUND_END -> error feedback."""

    def __init__(self, cfg: BaselineConfig, client_id: int, provider: Callable[[int], np.ndarray],
                 logs: _ClientLogs) -> None:
        self.cfg, self.client_id, self.provider, self.logs = cfg, client_id, provider, logs
        self.gateway_addr = (cfg.gateway_host, cfg.udp_port)
        self.control: _ClientControl | None = None
        self.b2: B2UdpClient | None = None
        self.b1_core = _RecordingB1Core(client_id, d=cfg.d, L=cfg.L) if cfg.policy == "B1" else None
        self.b1_sender = B1TcpClient(port=cfg.tcp_port) if cfg.policy == "B1" else None
        self.rate_bps = mean_access_rate_bps(client_id)  # b_bar_i (Section 3.1, D-15)
        self.finished = asyncio.Queue()
        self.socket_errors = self.rejected = self.round_end_timeouts = 0
        self.sent_packets = 0
        self._tasks: set[asyncio.Task] = set()
        self._timers: dict[int, asyncio.TimerHandle] = {}
        self._rounds: dict[int, dict[str, Any]] = {}

    @property
    def core(self):
        return self.b1_core if self.cfg.policy == "B1" else self.b2.core

    async def open(self) -> None:
        loop = asyncio.get_running_loop()
        local = tuple(self.cfg.client_endpoints[self.client_id])
        _, self.control = await loop.create_datagram_endpoint(lambda: _ClientControl(self), local_addr=local)
        if self.cfg.policy == "B2":
            self.b2 = B2UdpClient(B2ClientCore(self.client_id, d=self.cfg.d, L=self.cfg.L),
                                  gateway_addr=self.gateway_addr)
            self.b2.connection_made(self.control.transport)  # DATA leaves from port 9001
        _configure_error_feedback(self.core, self.cfg)

    async def close(self) -> None:
        for timer in self._timers.values():
            timer.cancel()
        for task in tuple(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self.control is not None and self.control.transport is not None:
            self.control.transport.close()

    def _on_control(self, data: bytes, addr, received_ns: int) -> None:
        if addr != self.gateway_addr:
            self.rejected += 1
            return
        try:
            message = decode_message(data, block_length=self.cfg.L)
            if message.client_id != self.client_id:
                raise ValueError("wrong client id")
            if message.message_type is MessageType.ROUND_START:
                self._on_round_start(message, received_ns)
            elif message.message_type is MessageType.ROUND_END:
                self._on_round_end(message)
            else:
                raise ValueError("unexpected message type")
        except (WireProtocolError, ValueError, TypeError, KeyError):
            self.rejected += 1

    def _on_round_start(self, message, received_ns: int) -> None:
        body = message.body
        if body["mode"] != int(MODES[self.cfg.policy]) or body["K"] != self.cfg.K or body["L"] != self.cfg.L:
            raise ValueError("ROUND_START does not match this baseline client")
        round_id, deadline_ns, rho_us = message.round_id, body["deadline_ns"], body["rho_us"]
        if round_id in self._rounds:
            raise ValueError("duplicate ROUND_START")
        delta = self.provider(round_id)
        self._rounds[round_id] = {"round_id": round_id, "round_start_rx_ns": received_ns,
                                  "deadline_ns": deadline_ns, "rho_us": rho_us, "status": "started"}
        loop = asyncio.get_running_loop()
        self._timers[round_id] = loop.call_later(
            max(0.0, (deadline_ns + ROUND_END_GRACE_NS - time.monotonic_ns()) / 1e9),
            self._round_end_timeout, round_id)
        task = asyncio.create_task(self._send_after_readiness(round_id, deadline_ns, rho_us, delta, received_ns))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _send_after_readiness(self, round_id: int, deadline_ns: int, rho_us: int, delta: np.ndarray,
                                    received_ns: int) -> None:
        record = self._rounds[round_id]
        # Readiness is rho_i after ROUND_START receipt (Section 2.1); local work
        # done since receipt counts toward it.
        await asyncio.sleep(max(0.0, (received_ns + rho_us * 1_000 - time.monotonic_ns()) / 1e9))
        record["readiness_ns"] = time.monotonic_ns()
        try:
            if self.cfg.policy == "B2":
                transfer = await self.b2.send_round(round_id=round_id, deadline_ns=deadline_ns,
                                                    delta=delta, rate_bps=self.rate_bps)
                encoded = self.b2.core.round.encoded if self.b2.core.round is not None else None
                for pos, (block_idx, sent_ns) in enumerate(zip(transfer.sent_indices, transfer.send_times_ns), 1):
                    self.logs.packets.write(t_mono_ns=sent_ns, round_id=round_id, direction="sent",
                                            client_id=self.client_id, packet_seq="", grant_id="", send_pos=pos,
                                            block_idx=block_idx, packet_type="DATA",
                                            bytes=32 + 2 * self.cfg.L, payload_bytes=2 * self.cfg.L,
                                            result="sent")
                record.update(packets_sent=transfer.packets_sent, bytes_sent=transfer.application_bytes_sent,
                              stop_reason=transfer.stop_reason,
                              first_send_ns=transfer.send_times_ns[0] if transfer.send_times_ns else None)
                self.sent_packets += transfer.packets_sent
            else:
                transfer = await self.b1_sender.send_round(self.b1_core, gateway_host=self.cfg.gateway_host,
                                                           round_id=round_id, deadline_ns=deadline_ns, delta=delta)
                encoded = (self.b1_core.round.encoded if self.b1_core.round is not None
                           else self.b1_core.prepared_encoded(round_id))  # connect failed: finalized by sender
                self.logs.packets.write(t_mono_ns=transfer.completed_ns, round_id=round_id, direction="sent",
                                        client_id=self.client_id, packet_seq=transfer.blocks_written, grant_id="",
                                        send_pos="", block_idx="", packet_type="B1_TCP_STREAM",
                                        bytes=transfer.application_bytes_written,
                                        payload_bytes=2 * self.cfg.L * transfer.blocks_written, result="sent")
                record.update(blocks_written=transfer.blocks_written,
                              bytes_sent=transfer.application_bytes_written,
                              connection_error=transfer.connection_error, close_reason=transfer.close_reason)
            if encoded is not None:
                record["block_energies"] = [float(x) for x in _block_energies(encoded)]
        except (B2TransferError, OSError) as exc:
            record["send_error"] = f"{type(exc).__name__}: {exc}"

    def _on_round_end(self, message) -> None:
        round_id = message.round_id
        if message.body["K"] != self.cfg.K:
            raise ValueError("ROUND_END K mismatch")
        delivered = tuple(int(i) for i in indices_from_bitmap(message.body["bitmap"], self.cfg.K))
        if len(delivered) != message.body["n_delivered"]:
            raise ValueError("ROUND_END n_delivered disagrees with its bitmap")
        self._finish(round_id, delivered, "round_end")

    def _round_end_timeout(self, round_id: int) -> None:
        if self._rounds.get(round_id, {}).get("status") == "started":
            self.round_end_timeouts += 1
            self._finish(round_id, (), "round_end_timeout")

    def _finish(self, round_id: int, delivered: tuple[int, ...], how: str) -> None:
        record = self._rounds.get(round_id)
        if record is None or record["status"] != "started":
            return
        timer = self._timers.pop(round_id, None)
        if timer is not None:
            timer.cancel()
        core = self.core
        if core.round is not None and core.round.round_id == round_id:
            if "block_energies" not in record:
                # The record is written here, possibly before the transfer task
                # returns (B1 holds its TCP stream until D), and finish_round
                # drops the encoded update: take the energies of the update this
                # round actually encoded now (D-09). Never inferred otherwise.
                record["block_energies"] = [float(x) for x in _block_energies(core.round.encoded)]
            core.finish_round(round_id, delivered)  # error feedback e <- u - u_hat (Eq. 6)
            record["status"] = how
        else:
            record["status"] = f"{how}_without_prepared_round"
            prepared = core.prepared_encoded(round_id) if hasattr(core, "prepared_encoded") else None
            if "block_energies" not in record and prepared is not None:
                # B1 connect failed: the sender already applied the empty delivery.
                record["block_energies"] = [float(x) for x in _block_energies(prepared)]
        record["delivered"] = list(delivered)
        self.logs.round(record)
        self.finished.put_nowait(round_id)


def weighted_sum(updates, weights) -> np.ndarray:
    """sum_i w_i u_hat_i in float32 (``fl/server.py::aggregate_updates`` without its d check)."""
    out = np.zeros_like(updates[0], dtype=np.float64)
    for w, update in zip(np.asarray(weights, dtype=np.float64), updates, strict=True):
        out += w * np.asarray(update, dtype=np.float64)
    return out.astype(np.float32)


def block_coords(d: int, L: int, block_idx: int) -> int:
    """Real coordinates in block j: L, except the zero-padded last block."""
    return min(L, d - block_idx * L)


def _configure_error_feedback(core, cfg: BaselineConfig) -> None:
    if cfg.network_only:
        core.error_feedback = ErrorFeedback(cfg.d, cfg.L, enabled=False)


def _block_energies(encoded) -> np.ndarray:
    """Per-block energies by block index (READY order inverted), for eta (D-09)."""
    by_block = np.empty(encoded.K, dtype=np.float64)
    by_block[np.asarray(encoded.order)] = np.asarray(encoded.energies_sorted, dtype=np.float64)
    return by_block


class _PrefetchedUpdates:
    """Prepare round r's update during the compute window, before ROUND_START.

    Mirrors Section 7 ("clients finish local training within Tc"): the
    ROUND_START handler only looks the update up, so READY/first DATA follow
    ROUND_START by rho_i rather than by rho_i plus payload construction.
    """

    def __init__(self, provider: Callable[[int], np.ndarray], rounds: int) -> None:
        self.provider, self.rounds = provider, rounds
        self._cache: dict[int, np.ndarray] = {}

    def prefetch(self, round_id: int) -> None:
        if 1 <= round_id <= self.rounds and round_id not in self._cache:
            self._cache[round_id] = self.provider(round_id)

    def __call__(self, round_id: int) -> np.ndarray:
        cached = self._cache.pop(round_id, None)
        return self.provider(round_id) if cached is None else cached


async def run_client(cfg: BaselineConfig, client_id: int, provider: BankUpdateProvider | Callable[[int], np.ndarray],
                     log_file: Path, *, round_timeout_s: float = 10.0) -> dict[str, Any]:
    logs = _ClientLogs(Path(log_file), client_id)
    weight = getattr(provider, "weight", None)
    completed: list[int] = []
    summary: dict[str, Any] = {"client_id": client_id, "policy": cfg.policy}
    if getattr(provider, "trains_in_background", False):
        updates = provider  # M2: trains when the FL server's model arrives (fl_service)
    else:
        updates = _PrefetchedUpdates(provider, cfg.rounds)
        updates.prefetch(1)
    try:
        if cfg.policy == "B3":
            summary.update(await _run_b3_client(cfg, client_id, updates, logs, round_timeout_s, completed, weight))
        else:
            client = PushOrTcpClient(cfg, client_id, updates, logs)
            await client.open()
            try:
                for _ in range(cfg.rounds):
                    try:
                        finished = await asyncio.wait_for(client.finished.get(), timeout=round_timeout_s)
                    except asyncio.TimeoutError:
                        break
                    completed.append(finished)
                    if hasattr(updates, "prefetch"):
                        updates.prefetch(finished + 1)  # during late drain + Tc
            finally:
                await client.close()
            summary.update(sent=client.sent_packets, rejected=client.rejected,
                           round_end_timeouts=client.round_end_timeouts, socket_errors=client.socket_errors)
    finally:
        logs.close()
    summary["completed_rounds"] = completed
    summary["client_weight"] = weight
    if hasattr(provider, "metadata"):
        summary["value_profile"] = provider.metadata()
    Path(log_file).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


async def _run_b3_client(cfg, client_id, provider, logs, round_timeout_s, completed, weight) -> dict[str, Any]:
    core = B2ClientCore(client_id, d=cfg.d, L=cfg.L)
    _configure_error_feedback(core, cfg)
    client = await B3UdpClient.bind(core, gateway_addr=(cfg.gateway_host, cfg.udp_port),
                                    local_addr=tuple(cfg.client_endpoints[client_id]), update_provider=provider)
    sent = 0
    try:
        for _ in range(cfg.rounds):
            try:
                # One immutable record per round, published after the PLAN
                # transfer has settled (completed, cancelled at round end or failed).
                record = await client.wait_for_round_record(timeout=round_timeout_s)
            except asyncio.TimeoutError:
                break
            round_id = record.round_id
            transfer = record.transfer
            if transfer is not None:
                sent += transfer.packets_sent
                for pos, (block_idx, sent_ns) in enumerate(zip(transfer.sent_indices, transfer.send_times_ns), 1):
                    logs.packets.write(t_mono_ns=sent_ns, round_id=round_id, direction="sent", client_id=client_id,
                                       packet_seq="", grant_id="", send_pos=pos, block_idx=block_idx,
                                       packet_type="DATA", bytes=32 + 2 * cfg.L, payload_bytes=2 * cfg.L,
                                       result="sent")
            entry = {"round_id": round_id, "ready_sent_ns": record.ready_sent_ns,
                     "plan_start_ns": record.plan_start_ns, "delivered": list(record.delivered_indices),
                     "packets_sent": record.packets_sent, "status": record.finished_by,
                     "transfer_status": record.transfer_status, "transfer_error": record.transfer_error}
            if cfg.network_only:
                # With network_only, u = provider(round_id) exactly (no residual), so
                # these are the energies the client reported in READY. (M2 trains a
                # fresh update and carries a residual, so it cannot be recomputed.)
                base = getattr(provider, "provider", provider)  # deterministic: same u as sent
                entry["block_energies"] = [float(x) for x in _block_energies(encode(base(round_id), cfg.L))]
            logs.round(entry)
            completed.append(round_id)
            if hasattr(provider, "prefetch"):
                provider.prefetch(round_id + 1)  # during late drain + Tc
    finally:
        await client.close()
    return {"sent": sent, "rejected": client.rejected_datagrams, "round_end_timeouts": client.round_end_timeouts}


# ------------------------------------------------------------------- CLI


MODE_TO_POLICY = {"tcp": "B1", "push": "B2", "plan": "B3"}  # runner contract --mode


def _bank_client_count(path: str) -> int:
    """N from the bank's energy array, so a client needs only its own id."""
    with np.load(path, allow_pickle=False) as bank:
        return int(bank["energies_sorted"].shape[1])


def _config_from_args(args: argparse.Namespace, num_clients: int) -> BaselineConfig:
    duration_ns = args.duration_ns or compute_deadline_ns(args.beta, num_clients)
    schedule = RoundSchedule(duration_ns=int(duration_ns),
                             compute_window_ns=int(round(args.compute_window_s * 1e9)),
                             late_drain_ns=int(round(args.late_drain_s * 1e9)))
    return BaselineConfig(policy=args.policy, num_clients=num_clients, rounds=args.rounds,
                          master_seed=args.seed, schedule=schedule,
                          client_endpoints=mininet_client_endpoints(num_clients, args.client_port),
                          gateway_host=args.gateway_host, udp_port=args.port, tcp_port=args.tcp_port)


def _load_real_bank(path: str, num_clients: int) -> ValueBankInputs:
    # No fallback: real M1 runs require the FULL seed-0 bank (Section 5.2).
    return load_value_bank(path, num_clients)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="role", required=True)
    for role in ("gateway", "client"):
        p = sub.add_parser(role)
        which = p.add_mutually_exclusive_group(required=True)
        which.add_argument("--mode", choices=tuple(MODE_TO_POLICY), help="runner contract: tcp|push|plan")
        which.add_argument("--policy", choices=BASELINES, help="equivalent: B1|B2|B3")
        p.add_argument("--clients", type=int, default=None,
                       help="N (required for the gateway; clients infer it from the bank)")
        p.add_argument("--rounds", type=int, default=1)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--beta", type=float, default=0.10)
        p.add_argument("--duration-ns", type=int, default=None, help="D; default round(beta*T_full)")
        p.add_argument("--compute-window-s", type=float, default=1.0)
        p.add_argument("--late-drain-s", type=float, default=0.1)
        p.add_argument("--value-bank", "--bank", dest="bank", default=str(VALUE_BANK_PATH))
        p.add_argument("--gateway-host", default=GATEWAY_HOST)
        p.add_argument("--port", type=int, default=UDP_GATEWAY_PORT)
        p.add_argument("--client-port", type=int, default=UDP_CLIENT_PORT)
        p.add_argument("--tcp-port", type=int, default=TCP_GATEWAY_PORT)
        if role == "gateway":
            p.add_argument("--aggregate-endpoint", default=None,
                           help="M2: FL server HOST:PORT that receives each round's aggregate")
            p.add_argument("--log-dir", required=True)
            p.add_argument("--round-state", required=True)
            p.add_argument("--profile-gateway", action="store_true",
                           help="accepted for runner compatibility; CPU/RSS are sampled by the runner")
        else:
            p.add_argument("--client-id", type=int, required=True)
            p.add_argument("--log-file", required=True)
            p.add_argument("--round-timeout-s", type=float, default=10.0)
            p.add_argument("--fl-model-port", type=int, default=None,
                           help="M2: receive the FL server's model on this TCP port and train locally")
            p.add_argument("--fl-data-root", default="data/femnist/reproduced/leaf/data")
            p.add_argument("--fl-shard-root", default="data/femnist/reproduced/clients")
            p.add_argument("--fl-trainer", default=None, help="module:function trainer factory (tests)")
            p.add_argument("--fl-wait-s", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.mode is not None:
        args.policy = MODE_TO_POLICY[args.mode]
    if args.role == "gateway" and args.clients is None:
        parser.error("gateway requires --clients")
    num_clients = args.clients if args.clients is not None else _bank_client_count(args.bank)
    cfg = _config_from_args(args, num_clients)
    bank = _load_real_bank(args.bank, num_clients)
    if args.role == "gateway":
        if args.aggregate_endpoint:
            from fedgrant.mininet.gateway_proc import parse_endpoint

            cfg = dataclasses.replace(cfg, aggregate_endpoint=parse_endpoint(args.aggregate_endpoint))
        asyncio.run(run_gateway(cfg, bank.client_weights, Path(args.log_dir), RoundStateFile(args.round_state)))
    elif args.fl_model_port is not None:
        asyncio.run(run_fl_client(cfg, args))
    else:
        provider = BankUpdateProvider(bank, args.seed, args.client_id, source_path=args.bank)
        asyncio.run(run_client(cfg, args.client_id, provider, Path(args.log_file),
                               round_timeout_s=args.round_timeout_s))


async def run_fl_client(cfg: BaselineConfig, args: argparse.Namespace) -> dict[str, Any]:
    """M2 baseline client: local training on the received model, error feedback on (D-19)."""
    from fedgrant.mininet.client_proc import make_fl_updates
    from fedgrant.mininet.fl_service import start_model_receiver

    cfg = dataclasses.replace(cfg, network_only=False)
    updates = make_fl_updates(args)
    receiver = await start_model_receiver(updates, port=args.fl_model_port)
    try:
        summary = await run_client(cfg, args.client_id, updates, Path(args.log_file),
                                   round_timeout_s=args.round_timeout_s)
    finally:
        await receiver.close()
        updates.close()
    summary.update(fl_models_received=updates.models_received, fl_missing_models=updates.missing_models)
    Path(args.log_file).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    main()
