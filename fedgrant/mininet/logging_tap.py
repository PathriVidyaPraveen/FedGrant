"""UDP gateway adapter that records packet/action events to the raw CSV schema."""

from __future__ import annotations

import asyncio
import csv
import time
from pathlib import Path
from typing import Any

from fedgrant.common.constants import MessageType
from fedgrant.common.wire import WireProtocolError, decode_message
from fedgrant.common.logschema import GRANT_COLUMNS, PACKET_COLUMNS, RTT_COLUMNS, SCHEDULER_COLUMNS
from fedgrant.core.gateway import SendGrant
from fedgrant.net.udp_gateway import UdpGateway


class LoggingUdpGateway(UdpGateway):
    """Same UDP gateway API as UdpGateway plus one strict raw packet log."""

    def __init__(self, core, *, log_dir: str | Path, **kwargs: Any):
        super().__init__(core, **kwargs)
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self._packet_file = (directory / "packets.csv").open("w", newline="", encoding="utf-8")
        self._packet_writer = csv.DictWriter(self._packet_file, fieldnames=PACKET_COLUMNS, extrasaction="raise")
        self._packet_writer.writeheader()
        self._rtt_file = (directory / "rtt_samples.csv").open("w", newline="", encoding="utf-8")
        self._rtt_writer = csv.DictWriter(self._rtt_file, fieldnames=RTT_COLUMNS, extrasaction="raise")
        self._rtt_writer.writeheader()
        self._scheduler_file = (directory / "scheduler_samples.csv").open("w", newline="", encoding="utf-8")
        self._scheduler_writer = csv.DictWriter(self._scheduler_file, fieldnames=SCHEDULER_COLUMNS, extrasaction="raise")
        self._scheduler_writer.writeheader()
        self._grant_file = (directory / "grants.csv").open("w", newline="", encoding="utf-8")
        self._grant_writer = csv.DictWriter(self._grant_file, fieldnames=GRANT_COLUMNS, extrasaction="raise")
        self._grant_writer.writeheader()
        self._dispatch_trigger_perf_ns: int | None = None
        self._grant_sent_ns: dict[tuple[int, int], int] = {}
        self._deadline_snapshot: tuple[int, int] | None = None

    @classmethod
    async def bind(cls, core, *, log_dir: str | Path, local_addr=("0.0.0.0", 9000), **kwargs: Any):
        """Bind with the subclass-only log_dir constructor argument forwarded."""
        protocol = cls(core, log_dir=log_dir, **kwargs)
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(lambda: protocol, local_addr=local_addr)
        return protocol

    def datagram_received(self, data: bytes, addr) -> None:
        received_perf_ns = time.perf_counter_ns()
        received_mono_ns = time.monotonic_ns()
        packet_type = "malformed"
        client_id = -1
        round_id = 0
        message = None
        try:
            block_length = self.core.round.L if self.core.round is not None else None
            message = decode_message(data, block_length=block_length)
            packet_type = MessageType(message.message_type).name
            client_id = int(message.client_id)
            round_id = int(message.round_id)
        except (WireProtocolError, ValueError, TypeError):
            pass
        body = message.body if message is not None else {}
        before = self.core.stats.accepted_data
        self._dispatch_trigger_perf_ns = received_perf_ns
        try:
            super().datagram_received(data, addr)
        finally:
            self._dispatch_trigger_perf_ns = None
        # Persist after the core has scheduled any immediate response, so disk
        # I/O does not inflate event-to-GRANT latency.
        self._write_packet(round_id, client_id, packet_type, len(data), "received", now_ns=received_mono_ns,
                           grant_id=body.get("grant_id", ""), send_pos=body.get("send_pos", ""),
                           block_idx=body.get("block_idx", ""),
                           payload_bytes=(2 * self.core.round.L if packet_type == "DATA" and self.core.round else 0))
        if packet_type == "DATA" and self.core.stats.accepted_data > before:
            try:
                message = decode_message(data, block_length=self.core.round.L if self.core.round else None)
                grant_id = int(message.body["grant_id"])
                grant = self.core.resolved_grants.get(client_id, {}).get(grant_id)
                if grant is not None:
                    sent_ns = self._grant_sent_ns.get((client_id, grant_id), grant.t_issue_ns)
                    rtt_ms = (received_mono_ns - sent_ns) / 1e6
                    self._rtt_writer.writerow({"t_mono_ns": received_mono_ns, "round_id": round_id,
                                               "rtt_ms": rtt_ms})
                    self._rtt_file.flush()
                    self._grant_writer.writerow({"t_mono_ns": received_mono_ns, "round_id": round_id,
                                                 "client_id": client_id, "grant_id": grant_id,
                                                 "send_pos": message.body["send_pos"], "event": "delivered",
                                                 "issue_ns": grant.t_issue_ns, "timeout_ns": grant.timeout_ns,
                                                 "latency_us": "", "rtt_ms": rtt_ms})
                    self._grant_file.flush()
            except (WireProtocolError, ValueError, TypeError, KeyError):
                pass

    def _write_packet(self, round_id: int, client_id: int, packet_type: str, size: int, result: str,
                      *, now_ns: int | None = None, grant_id="", send_pos="", block_idx="",
                      payload_bytes=0) -> None:
        self._packet_writer.writerow({
            "t_mono_ns": time.monotonic_ns() if now_ns is None else now_ns, "round_id": round_id, "client_id": client_id,
            "direction": result, "packet_seq": "", "grant_id": grant_id, "send_pos": send_pos,
            "block_idx": block_idx, "packet_type": packet_type, "bytes": size,
            "payload_bytes": payload_bytes, "result": result,
        })
        self._packet_file.flush()

    def dispatch_actions(self, actions) -> None:
        for action in actions:
            round_id = int(getattr(action, "round_id", self.core.round.round_id if self.core.round else 0))
            client_id = int(getattr(action, "client_id", -1))
            before_sent = self.stats.sent
            super().dispatch_actions((action,))
            if self.stats.sent == before_sent:
                continue
            now_ns = time.monotonic_ns()
            sent_perf_ns = time.perf_counter_ns()
            if isinstance(action, SendGrant):
                grant = self.core.clients[client_id].outstanding.get(action.grant_id)
                if grant is not None:
                    self._grant_sent_ns[(client_id, action.grant_id)] = now_ns
                    packet_bytes = 20 + 8
                    self._write_packet(round_id, client_id, "GRANT", packet_bytes, "sent", now_ns=now_ns,
                                       grant_id=action.grant_id, send_pos=action.send_pos)
                    self._grant_writer.writerow({"t_mono_ns": now_ns, "round_id": round_id,
                                                 "client_id": client_id, "grant_id": action.grant_id,
                                                 "send_pos": action.send_pos, "event": "issued",
                                                 "issue_ns": grant.t_issue_ns, "timeout_ns": grant.timeout_ns,
                                                 "latency_us": ((sent_perf_ns - self._dispatch_trigger_perf_ns) / 1e3
                                                                if self._dispatch_trigger_perf_ns is not None else ""),
                                                 "rtt_ms": ""})
                    self._grant_file.flush()
                    if self._dispatch_trigger_perf_ns is not None:
                        self._scheduler_writer.writerow({"t_mono_ns": now_ns, "round_id": round_id,
                                                         "latency_us": (sent_perf_ns - self._dispatch_trigger_perf_ns) / 1e3})
                        self._scheduler_file.flush()
                else:
                    self._write_packet(round_id, client_id, "GRANT", 28, "sent", now_ns=now_ns,
                                       grant_id=action.grant_id, send_pos=action.send_pos)
            else:
                name = type(action).__name__
                packet_type = "ROUND_START" if name == "SendRoundStart" else "ROUND_END"
                size = (20 + 20 if name == "SendRoundStart" else
                        20 + 4 + ((self.core.clients[client_id].K + 7) // 8))
                self._write_packet(round_id, client_id, packet_type, size, "sent", now_ns=now_ns)

    def _run_tick(self) -> None:
        """Include tick-triggered grants in event-to-send latency samples."""
        if (self.core.phase.name == "ACTIVE" and self.core.round is not None
                and self.clock_ns() >= self.core.round.deadline_ns):
            self.capture_deadline_state()
        self._dispatch_trigger_perf_ns = time.perf_counter_ns()
        try:
            super()._run_tick()
        finally:
            self._dispatch_trigger_perf_ns = None

    def capture_deadline_state(self) -> int:
        """Snapshot unresolved grants before the core's deadline cleanup runs."""
        if self.core.round is None:
            return 0
        round_id = self.core.round.round_id
        if self._deadline_snapshot is None or self._deadline_snapshot[0] != round_id:
            unresolved = sum(len(state.outstanding) for state in self.core.clients.values())
            self._deadline_snapshot = (round_id, unresolved)
        return self._deadline_snapshot[1]

    def log_timeouts(self) -> None:
        """Persist timeout outcomes after the core has resolved deadline grants."""
        for client_id, grant_ids in self.core.timed_out_grants.items():
            for grant_id in sorted(grant_ids):
                grant = self.core.resolved_grants.get(client_id, {}).get(grant_id)
                if grant is None:
                    continue
                self._grant_writer.writerow({"t_mono_ns": grant.timeout_ns,
                                             "round_id": self.core.round.round_id if self.core.round else 0,
                                             "client_id": client_id, "grant_id": grant_id,
                                             "send_pos": grant.pos, "event": "timeout",
                                             "issue_ns": grant.t_issue_ns, "timeout_ns": grant.timeout_ns,
                                             "latency_us": "", "rtt_ms": ""})
        self._grant_file.flush()

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            if not self._packet_file.closed:
                self._packet_file.close()
            if not self._rtt_file.closed:
                self._rtt_file.close()
            if not self._scheduler_file.closed:
                self._scheduler_file.close()
            if not self._grant_file.closed:
                self._grant_file.close()


__all__ = ["LoggingUdpGateway"]
