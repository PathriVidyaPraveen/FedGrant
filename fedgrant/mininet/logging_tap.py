"""UDP gateway adapter that records packet/action events to the raw CSV schema."""

from __future__ import annotations

import asyncio
import csv
import time
from pathlib import Path
from typing import Any

from fedgrant.common.constants import MessageType
from fedgrant.common.wire import WireProtocolError, decode_message
from fedgrant.common.logschema import PACKET_COLUMNS, RTT_COLUMNS, SCHEDULER_COLUMNS
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
        self._dispatch_trigger_ns: int | None = None

    @classmethod
    async def bind(cls, core, *, log_dir: str | Path, local_addr=("0.0.0.0", 9000), **kwargs: Any):
        """Bind with the subclass-only log_dir constructor argument forwarded."""
        protocol = cls(core, log_dir=log_dir, **kwargs)
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(lambda: protocol, local_addr=local_addr)
        return protocol

    def datagram_received(self, data: bytes, addr) -> None:
        packet_type = "malformed"
        client_id = -1
        round_id = 0
        try:
            block_length = self.core.round.L if self.core.round is not None else None
            message = decode_message(data, block_length=block_length)
            packet_type = MessageType(message.message_type).name
            client_id = int(message.client_id)
            round_id = int(message.round_id)
        except (WireProtocolError, ValueError, TypeError):
            pass
        now_ns = time.monotonic_ns()
        self._write_packet(round_id, client_id, packet_type, len(data), "received", now_ns=now_ns)
        before = self.core.stats.accepted_data
        self._dispatch_trigger_ns = now_ns
        try:
            super().datagram_received(data, addr)
        finally:
            self._dispatch_trigger_ns = None
        if packet_type == "DATA" and self.core.stats.accepted_data > before:
            try:
                message = decode_message(data, block_length=self.core.round.L if self.core.round else None)
                grant_id = int(message.body["grant_id"])
                grant = self.core.resolved_grants.get(client_id, {}).get(grant_id)
                if grant is not None:
                    self._rtt_writer.writerow({"t_mono_ns": now_ns, "round_id": round_id,
                                               "rtt_ms": (now_ns - grant.t_issue_ns) / 1e6})
                    self._rtt_file.flush()
            except (WireProtocolError, ValueError, TypeError, KeyError):
                pass

    def _write_packet(self, round_id: int, client_id: int, packet_type: str, size: int, result: str,
                      *, now_ns: int | None = None) -> None:
        self._packet_writer.writerow({
            "t_mono_ns": time.monotonic_ns() if now_ns is None else now_ns, "round_id": round_id, "client_id": client_id,
            "packet_seq": "", "packet_type": packet_type, "bytes": size, "result": result,
        })
        self._packet_file.flush()

    def dispatch_actions(self, actions) -> None:
        for action in actions:
            round_id = int(getattr(action, "round_id", self.core.round.round_id if self.core.round else 0))
            client_id = int(getattr(action, "client_id", -1))
            self._write_packet(round_id, client_id, type(action).__name__, 0, "send_action")
            if isinstance(action, SendGrant) and self._dispatch_trigger_ns is not None:
                grant = self.core.clients[client_id].outstanding.get(action.grant_id)
                if grant is not None:
                    self._scheduler_writer.writerow({"t_mono_ns": grant.t_issue_ns, "round_id": round_id,
                                                     "latency_us": (grant.t_issue_ns - self._dispatch_trigger_ns) / 1e3})
                    self._scheduler_file.flush()
        super().dispatch_actions(actions)

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


__all__ = ["LoggingUdpGateway"]
