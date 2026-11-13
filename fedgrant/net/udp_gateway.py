"""Asyncio UDP adapter and stable-cadence scheduler for :mod:`fedgrant.core.gateway`.

The adapter owns datagram I/O and endpoint routing only.  It deliberately
delegates protocol/state validation, grant matching, retransmission, and
deadline handling to :class:`fedgrant.core.gateway.GatewayCore`.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, cast

from fedgrant.common.constants import L as L_DEFAULT, MessageType
from fedgrant.common.wire import WireProtocolError, decode_message, encode_message
from fedgrant.core.gateway import (
    EmitAggregate,
    GatewayAction,
    GatewayCore,
    GatewayError,
    SendGrant,
    SendRoundEnd,
    SendRoundStart,
)

log = logging.getLogger(__name__)

DEFAULT_TICK_INTERVAL_NS = 1_000_000
DEFAULT_GATEWAY_PORT = 9000
_NANOSECONDS_PER_SECOND = 1_000_000_000

Endpoint: TypeAlias = tuple[str, int]
AggregateHandler: TypeAlias = Callable[[Any], Awaitable[None] | None]


class AggregateSender(Protocol):
    async def send_aggregate(
        self,
        round_id: int,
        aggregate: Any,
        client_n_coords: Mapping[int, Mapping[int, int]],
    ) -> None:
        ...


@dataclass(slots=True)
class UdpGatewayStats:
    """Transport-level counters; core owns protocol decision counters."""

    received: int = 0
    malformed: int = 0
    unexpected_type: int = 0
    unknown_client: int = 0
    endpoint_conflict: int = 0
    sent: int = 0
    dropped_actions: int = 0
    tick_count: int = 0
    tick_errors: int = 0
    aggregate_transfer_errors: int = 0


class UdpGateway(asyncio.DatagramProtocol):
    """Bind a :class:`GatewayCore` to one asyncio UDP socket.

    READY and DATA are the only client-to-gateway protocol messages, so this
    protocol accepts only those frame types from the network.  ROUND_START is
    intentionally a local control-plane operation exposed by :meth:`start_round`:
    it uses the gateway's monotonic clock and emits unicast ROUND_START actions.
    """

    def __init__(
        self,
        core: GatewayCore,
        *,
        endpoints: Mapping[int, Endpoint] | None = None,
        tick_interval_ns: int = DEFAULT_TICK_INTERVAL_NS,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        aggregate_handler: AggregateHandler | None = None,
        aggregate_sender: AggregateSender | None = None,
    ) -> None:
        if not isinstance(core, GatewayCore):
            raise TypeError("core must be a GatewayCore")
        if isinstance(tick_interval_ns, bool) or not isinstance(tick_interval_ns, int) or tick_interval_ns < 1:
            raise ValueError("tick_interval_ns must be a positive integer")
        self.core = core
        self.tick_interval_ns = tick_interval_ns
        self.clock_ns = clock_ns
        self._sleep = sleep
        self.aggregate_handler = aggregate_handler
        if aggregate_handler is not None and aggregate_sender is not None:
            raise ValueError("configure either aggregate_handler or aggregate_sender, not both")
        self.aggregate_sender = aggregate_sender
        self.endpoints: dict[int, Endpoint] = {}
        for client_id, endpoint in (endpoints or {}).items():
            self.set_endpoint(client_id, endpoint)
        self.transport: asyncio.DatagramTransport | None = None
        self.scheduler_task: asyncio.Task[None] | None = None
        self._aggregate_tasks: set[asyncio.Task[None]] = set()
        self._closing = False
        self._tick_active = False
        # Grid time the scheduler loop was waiting for when it last called
        # _run_tick (monotonic ns). Read-only observability for profiling; the
        # loop never reads it back.
        self.scheduled_tick_ns: int | None = None
        self.stats = UdpGatewayStats()

    @classmethod
    async def bind(
        cls,
        core: GatewayCore,
        *,
        local_addr: Endpoint = ("0.0.0.0", DEFAULT_GATEWAY_PORT),
        endpoints: Mapping[int, Endpoint] | None = None,
        tick_interval_ns: int = DEFAULT_TICK_INTERVAL_NS,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        aggregate_handler: AggregateHandler | None = None,
        aggregate_sender: AggregateSender | None = None,
    ) -> "UdpGateway":
        """Create, bind, and start an asyncio UDP gateway daemon."""
        protocol = cls(
            core,
            endpoints=endpoints,
            tick_interval_ns=tick_interval_ns,
            clock_ns=clock_ns,
            aggregate_handler=aggregate_handler,
            aggregate_sender=aggregate_sender,
        )
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(lambda: protocol, local_addr=local_addr)
        return protocol

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = cast(asyncio.DatagramTransport, transport)
        self._closing = False
        self.scheduler_task = asyncio.create_task(self._scheduler_loop(), name="fedgrant-gateway-tick")

    def connection_lost(self, exc: Exception | None) -> None:
        self.transport = None
        self._closing = True
        task = self.scheduler_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        for aggregate_task in tuple(self._aggregate_tasks):
            aggregate_task.cancel()

    def error_received(self, exc: Exception) -> None:
        log.warning("FedGrant UDP transport error: %r", exc)

    def datagram_received(self, data: bytes, addr: tuple[str, int] | tuple[str, int, int, int]) -> None:
        """Decode one datagram, bind/check its endpoint, and delegate to the core."""
        self.stats.received += 1
        now_ns = self.clock_ns()
        block_length = self.core.round.L if self.core.round is not None else L_DEFAULT
        try:
            message = decode_message(data, block_length=block_length)
        except WireProtocolError:
            self.stats.malformed += 1
            return
        if message.message_type not in (MessageType.READY, MessageType.DATA):
            self.stats.unexpected_type += 1
            return
        if message.client_id not in self.core.clients:
            self.stats.unknown_client += 1
            return
        endpoint = self._normalise_endpoint(addr)
        known_endpoint = self.endpoints.get(message.client_id)
        if known_endpoint is None:
            self.endpoints[message.client_id] = endpoint
        elif known_endpoint != endpoint:
            self.stats.endpoint_conflict += 1
            return
        try:
            if message.message_type is MessageType.READY:
                actions = self.core.on_ready_message(data, block_length=block_length, received_at_ns=now_ns)
            else:
                actions = self.core.on_data_message(data, block_length=block_length, received_at_ns=now_ns)
        except (GatewayError, ValueError, TypeError):
            # GatewayCore normally counts and drops invalid messages itself, but
            # transport input must never take down the datagram callback.
            self.stats.malformed += 1
            return
        self.dispatch_actions(actions)

    def start_round(self, **kwargs: Any) -> list[GatewayAction]:
        """Start a local round with a gateway-owned monotonic ``t0``.

        ``deadline_ns`` is in the same absolute monotonic clock domain as this
        daemon; callers may instead provide ``duration_ns`` to the core.
        """
        actions = self.core.on_round_start(self.clock_ns(), **kwargs)
        self.dispatch_actions(actions)
        return actions

    def set_endpoint(self, client_id: int, endpoint: Endpoint) -> None:
        """Configure a registered client's destination before ROUND_START."""
        if client_id not in self.core.clients:
            raise ValueError("endpoint client_id must be registered with the gateway")
        self.endpoints[client_id] = self._normalise_endpoint(endpoint)

    def dispatch_actions(self, actions: Iterable[GatewayAction]) -> None:
        """Encode core actions and send them to their registered endpoints."""
        for action in actions:
            try:
                if isinstance(action, EmitAggregate):
                    self._dispatch_aggregate(action)
                else:
                    self._send_action(action)
            except (GatewayError, WireProtocolError, ValueError, TypeError):
                self.stats.dropped_actions += 1
                log.exception("dropping invalid gateway action")

    async def close(self) -> None:
        """Cancel scheduler/aggregate tasks and close the UDP transport."""
        self._closing = True
        task = self.scheduler_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.scheduler_task = None
        for aggregate_task in tuple(self._aggregate_tasks):
            aggregate_task.cancel()
        if self._aggregate_tasks:
            await asyncio.gather(*self._aggregate_tasks, return_exceptions=True)
        transport, self.transport = self.transport, None
        if transport is not None:
            transport.close()

    async def _scheduler_loop(self) -> None:
        """Drive the core at stable tick deadlines without overlapping work."""
        next_tick_ns = self.clock_ns() + self.tick_interval_ns
        try:
            while not self._closing:
                delay_ns = next_tick_ns - self.clock_ns()
                if delay_ns > 0:
                    await self._sleep(delay_ns / _NANOSECONDS_PER_SECOND)
                if self._closing:
                    break
                self.scheduled_tick_ns = next_tick_ns
                self._run_tick()
                next_tick_ns += self.tick_interval_ns
                now_ns = self.clock_ns()
                if next_tick_ns <= now_ns:
                    skipped = (now_ns - next_tick_ns) // self.tick_interval_ns + 1
                    next_tick_ns += skipped * self.tick_interval_ns
        except asyncio.CancelledError:
            raise

    def _run_tick(self) -> None:
        if self._tick_active or self._closing:
            return
        self._tick_active = True
        try:
            actions = self.core.on_tick(self.clock_ns())
            self.stats.tick_count += 1
            self.dispatch_actions(actions)
        except (GatewayError, ValueError, TypeError):
            self.stats.tick_errors += 1
            log.exception("FedGrant scheduler tick failed")
        finally:
            self._tick_active = False

    def _send_action(self, action: SendRoundStart | SendGrant | SendRoundEnd) -> None:
        if self.transport is None:
            self.stats.dropped_actions += 1
            return
        endpoint = self.endpoints.get(action.client_id)
        if endpoint is None:
            self.stats.dropped_actions += 1
            return
        now_ns = self.clock_ns()
        if isinstance(action, SendRoundStart):
            packet = encode_message(
                MessageType.ROUND_START,
                round_id=action.round_id,
                client_id=action.client_id,
                t_send_ns=now_ns,
                body={
                    "deadline_ns": action.deadline_ns,
                    "K": action.K,
                    "L": action.L,
                    "rho_us": action.rho_us,
                    "mode": int(action.mode),
                },
            )
        elif isinstance(action, SendGrant):
            if self.core.round is None:
                raise GatewayError("cannot send a grant without an active round")
            packet = encode_message(
                MessageType.GRANT,
                round_id=self.core.round.round_id,
                client_id=action.client_id,
                t_send_ns=now_ns,
                body={"grant_id": action.grant_id, "send_pos": action.send_pos},
            )
        else:
            state = self.core.clients.get(action.client_id)
            if state is None:
                raise GatewayError("round-end action references an unknown client")
            packet = encode_message(
                MessageType.ROUND_END,
                round_id=action.round_id,
                client_id=action.client_id,
                t_send_ns=now_ns,
                body={
                    "K": state.K,
                    "n_delivered": sum(byte.bit_count() for byte in action.bitmap),
                    "bitmap": action.bitmap,
                },
            )
        self.transport.sendto(packet, endpoint)
        self.stats.sent += 1

    def _dispatch_aggregate(self, action: EmitAggregate) -> None:
        if self.aggregate_sender is not None:
            result = self.aggregate_sender.send_aggregate(
                action.round_id,
                action.aggregate,
                action.client_n_coords,
            )
        elif self.aggregate_handler is not None:
            result = self.aggregate_handler(action.aggregate)
        else:
            self.stats.aggregate_transfer_errors += 1
            log.error("completed aggregate for round %d has no configured receiver", action.round_id)
            return
        if inspect.isawaitable(result):
            task = asyncio.create_task(cast(Awaitable[None], result))
            self._aggregate_tasks.add(task)
            task.add_done_callback(self._aggregate_task_done)

    def _aggregate_task_done(self, task: asyncio.Task[None]) -> None:
        self._aggregate_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self.stats.aggregate_transfer_errors += 1
            log.error(
                "FedGrant aggregate transfer failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    @staticmethod
    def _normalise_endpoint(endpoint: tuple[Any, ...]) -> Endpoint:
        if not isinstance(endpoint, tuple) or len(endpoint) < 2:
            raise ValueError("UDP endpoint must be a (host, port) tuple")
        host, port = endpoint[0], endpoint[1]
        if not isinstance(host, str) or isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("UDP endpoint must contain a host string and port in [0, 65535]")
        return host, port


__all__ = [
    "AggregateHandler",
    "DEFAULT_GATEWAY_PORT",
    "DEFAULT_TICK_INTERVAL_NS",
    "Endpoint",
    "UdpGateway",
    "UdpGatewayStats",
]
