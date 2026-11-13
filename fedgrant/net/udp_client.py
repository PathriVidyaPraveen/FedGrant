"""Asyncio UDP client agent for the FedGrant GRANT-mode protocol."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np

from fedgrant.common.constants import MessageType
from fedgrant.common.wire import WireProtocolError, decode_message, encode_message
from fedgrant.core.client_core import ClientCore, ClientCoreError

log = logging.getLogger(__name__)

DEFAULT_CLIENT_PORT = 9001
ROUND_END_GRACE_NS = 200_000_000
_NANOSECONDS_PER_SECOND = 1_000_000_000

Endpoint: TypeAlias = tuple[str, int]
UpdateProvider: TypeAlias = Callable[[int], np.ndarray]


@dataclass(slots=True)
class UdpClientStats:
    received: int = 0
    malformed: int = 0
    unexpected_type: int = 0
    wrong_client: int = 0
    wrong_endpoint: int = 0
    stale_or_invalid: int = 0
    sent: int = 0
    skipped_ready: int = 0
    round_end_timeouts: int = 0


class UdpClient(asyncio.DatagramProtocol):
    """Transport adapter; encoding, block selection, and residuals stay in ClientCore."""

    def __init__(
        self,
        core: ClientCore,
        *,
        gateway_addr: Endpoint,
        update_provider: UpdateProvider,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        packet_observer: Callable[[bytes, int], None] | None = None,
    ) -> None:
        if not isinstance(core, ClientCore):
            raise TypeError("core must be a ClientCore")
        if not callable(update_provider):
            raise TypeError("update_provider must be callable")
        self.core = core
        self.gateway_addr = self._normalise_endpoint(gateway_addr)
        self.update_provider = update_provider
        self.clock_ns = clock_ns
        self.packet_observer = packet_observer
        self.transport: asyncio.DatagramTransport | None = None
        self._ready_handle: asyncio.TimerHandle | None = None
        self._round_end_handle: asyncio.TimerHandle | None = None
        self._ready_round_id: int | None = None
        self._closing = False
        self._round_end_event = asyncio.Event()
        self.last_reconstruction: np.ndarray | None = None
        self.stats = UdpClientStats()

    @classmethod
    async def bind(
        cls,
        core: ClientCore,
        *,
        gateway_addr: Endpoint,
        local_addr: Endpoint = ("0.0.0.0", DEFAULT_CLIENT_PORT),
        update_provider: UpdateProvider,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        packet_observer: Callable[[bytes, int], None] | None = None,
    ) -> "UdpClient":
        """Create, bind, and start a client UDP endpoint."""
        protocol = cls(
            core,
            gateway_addr=gateway_addr,
            update_provider=update_provider,
            clock_ns=clock_ns,
            packet_observer=packet_observer,
        )
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(lambda: protocol, local_addr=local_addr)
        return protocol

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = cast(asyncio.DatagramTransport, transport)
        self._closing = False

    def connection_lost(self, exc: Exception | None) -> None:
        self.transport = None
        self._closing = True
        self._cancel_timers()

    def error_received(self, exc: Exception) -> None:
        log.warning("FedGrant UDP client transport error: %r", exc)

    def datagram_received(self, data: bytes, addr: tuple[str, int] | tuple[str, int, int, int]) -> None:
        self.stats.received += 1
        if self._normalise_endpoint(addr) != self.gateway_addr:
            self.stats.wrong_endpoint += 1
            return
        block_length = self.core.round.L if self.core.round is not None else self.core.error_feedback.L
        try:
            message = decode_message(data, block_length=block_length)
        except WireProtocolError:
            self.stats.malformed += 1
            return
        if message.client_id != self.core.client_id:
            self.stats.wrong_client += 1
            return
        try:
            if message.message_type is MessageType.ROUND_START:
                self._handle_round_start(message)
            elif message.message_type is MessageType.GRANT:
                self._handle_grant(message)
            elif message.message_type is MessageType.ROUND_END:
                self._handle_round_end(message)
            else:
                self.stats.unexpected_type += 1
        except (ClientCoreError, WireProtocolError, TypeError, ValueError, OverflowError):
            self.stats.stale_or_invalid += 1
            log.warning("ignoring invalid FedGrant message type %s", message.message_type.name, exc_info=True)

    async def close(self) -> None:
        """Cancel pending round callbacks and close the UDP transport."""
        self._closing = True
        self._cancel_timers()
        transport, self.transport = self.transport, None
        if transport is not None:
            transport.close()

    async def wait_for_round_end(self, timeout: float | None = None) -> int:
        """Wait for ROUND_END or the specified 200 ms post-deadline fallback."""
        if timeout is None:
            await self._round_end_event.wait()
        else:
            await asyncio.wait_for(self._round_end_event.wait(), timeout)
        if self.core.last_completed_round_id is None:
            raise ClientCoreError("no round has completed")
        return self.core.last_completed_round_id

    def _handle_round_start(self, message) -> None:
        # rho_i is measured from ROUND_START receipt (design decision D-20):
        # take the timer origin before the update is prepared and encoded.
        loop = asyncio.get_running_loop()
        received_at = loop.time()
        if self.core.round is not None:
            raise ClientCoreError("ROUND_START arrived while a round is active")
        if (
            self.core.last_completed_round_id is not None
            and message.round_id <= self.core.last_completed_round_id
        ):
            raise ClientCoreError("stale ROUND_START")
        body = message.body
        update = self.update_provider(message.round_id)
        energies = self.core.start_round(
            round_id=message.round_id,
            start_ns=message.t_send_ns,
            deadline_ns=body["deadline_ns"],
            K=body["K"],
            L=body["L"],
            mode=body["mode"],
            delta=update,
        )
        self._cancel_timers()
        self._round_end_event.clear()
        self.last_reconstruction = None
        round_id = message.round_id
        self._ready_round_id = None
        delay_seconds = body["rho_us"] / 1_000_000.0
        # Preparation above is synchronous, so READY can never precede it. If it
        # overran received_at + rho, call_at with a past time sends READY on the
        # next loop iteration, i.e. as soon as preparation has completed.
        self._ready_handle = loop.call_at(
            received_at + delay_seconds, self._send_ready, round_id, energies
        )
        timeout_ns = body["deadline_ns"] + ROUND_END_GRACE_NS
        timeout_seconds = max(0, timeout_ns - self.clock_ns()) / _NANOSECONDS_PER_SECOND
        self._round_end_handle = loop.call_later(
            timeout_seconds, self._finish_round_timeout, round_id
        )

    def _handle_grant(self, message) -> None:
        current = self.core.round
        if current is None or message.round_id != current.round_id:
            raise ClientCoreError("GRANT does not match the active round")
        if self._ready_round_id != current.round_id:
            raise ClientCoreError("GRANT arrived before READY was sent")
        body = self.core.data_for_grant(
            message.round_id,
            message.body["grant_id"],
            message.body["send_pos"],
        )
        packet = encode_message(
            MessageType.DATA,
            round_id=current.round_id,
            client_id=self.core.client_id,
            t_send_ns=self.clock_ns(),
            block_length=current.L,
            body=body,
        )
        self._send(packet)

    def _handle_round_end(self, message) -> None:
        current = self.core.round
        if current is None or message.round_id != current.round_id:
            raise ClientCoreError("ROUND_END does not match the active round")
        if message.body["K"] != current.K:
            raise ClientCoreError("ROUND_END K does not match the active round")
        self.last_reconstruction = self.core.finish_round(message.round_id, message.body["bitmap"])
        self._ready_round_id = None
        self._cancel_timers()
        self._round_end_event.set()

    def _send_ready(self, round_id: int, energies: tuple[float, ...]) -> None:
        self._ready_handle = None
        current = self.core.round
        if self._closing or current is None or current.round_id != round_id:
            return
        if self.clock_ns() >= current.deadline_ns:
            self.stats.skipped_ready += 1
            return
        packet = encode_message(
            MessageType.READY,
            round_id=round_id,
            client_id=self.core.client_id,
            t_send_ns=self.clock_ns(),
            body={"K": current.K, "energy": energies},
        )
        if self._send(packet):
            self._ready_round_id = round_id

    def _finish_round_timeout(self, round_id: int) -> None:
        self._round_end_handle = None
        current = self.core.round
        if self._closing or current is None or current.round_id != round_id:
            return
        self.last_reconstruction = self.core.finish_round_timeout(round_id)
        self._ready_round_id = None
        self._cancel_ready()
        self.stats.round_end_timeouts += 1
        self._round_end_event.set()

    def _send(self, packet: bytes) -> bool:
        if self.transport is None or self._closing:
            return False
        now_ns = self.clock_ns()
        self.transport.sendto(packet, self.gateway_addr)
        if self.packet_observer is not None:
            self.packet_observer(packet, now_ns)
        self.stats.sent += 1
        return True

    def _cancel_ready(self) -> None:
        if self._ready_handle is not None:
            self._ready_handle.cancel()
            self._ready_handle = None

    def _cancel_timers(self) -> None:
        self._cancel_ready()
        if self._round_end_handle is not None:
            self._round_end_handle.cancel()
            self._round_end_handle = None

    @staticmethod
    def _normalise_endpoint(endpoint: tuple[object, ...]) -> Endpoint:
        if not isinstance(endpoint, tuple) or len(endpoint) < 2:
            raise ValueError("UDP endpoint must be a (host, port) tuple")
        host, port = endpoint[0], endpoint[1]
        if (
            not isinstance(host, str)
            or isinstance(port, bool)
            or not isinstance(port, int)
            or not 0 <= port <= 65535
        ):
            raise ValueError("UDP endpoint must contain a host string and port in [0, 65535]")
        return host, port


__all__ = [
    "DEFAULT_CLIENT_PORT",
    "ROUND_END_GRACE_NS",
    "UdpClient",
    "UdpClientStats",
    "UpdateProvider",
]
