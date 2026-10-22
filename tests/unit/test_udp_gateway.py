"""Async UDP gateway adapter tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np
import pytest

from fedgrant.common.constants import MessageType
from fedgrant.common.wire import decode_message, encode_message
from fedgrant.core.gateway import EmitAggregate, GatewayCore, GatewayPhase
from fedgrant.core.state import ClientState
from fedgrant.net.udp_gateway import UdpGateway
from fedgrant.policy.types import CandidateView


@dataclass
class Clock:
    now: int = 100

    def __call__(self) -> int:
        return self.now


@dataclass
class FakeTransport:
    sent: list[tuple[bytes, tuple[str, int]]] = field(default_factory=list)
    closed: bool = False

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:
        self.sent.append((data, addr))

    def close(self) -> None:
        self.closed = True


class Eligibility:
    def candidates(self, clients, now_ns, deadline_ns, retransmit):
        eligible = [
            client_id
            for client_id, state in sorted(clients.items())
            if state.ready and not state.outstanding and state.has_pending(retransmit)
        ]
        return CandidateView(
            np.asarray(eligible, dtype=np.int32),
            np.asarray([clients[i].val[clients[i].next_pos(retransmit) - 1] for i in eligible], dtype=np.float64),
            np.asarray([clients[i].K - clients[i].m - len(clients[i].outstanding) for i in eligible], dtype=np.int32),
            0,
        )


class Selector:
    def select(self, candidates):
        return int(candidates.client_id[0])


class Pacing:
    def allow(self, client_id, p_hat, now_ns, deadline_ns):
        return True


class Prediction:
    def probability(self, client_id, now_ns):
        return 0.75


class Timeout:
    def timeout_ns(self, client_id, issue_ns):
        return issue_ns + 100


@dataclass
class TickEstimator:
    clock: Clock | None = None
    daemon: UdpGateway | None = None
    ticks: list[int] = field(default_factory=list)

    def reset_round(self, ctx):
        pass

    def resolve(self, client_id, grant, delivered, t_ns):
        pass

    def on_tick(self, t_ns):
        self.ticks.append(t_ns)
        if len(self.ticks) == 1 and self.clock is not None:
            self.clock.now += 2_500_000  # emulate scheduler work longer than one interval
        elif len(self.ticks) == 3 and self.daemon is not None:
            self.daemon._closing = True

    def on_deadline(self, t_ns):
        pass


def make_core(*, estimator=None) -> GatewayCore:
    return GatewayCore(
        {1: ClientState(1, 0.5, K=3)},
        selector=Selector(),
        eligibility=Eligibility(),
        pacing=Pacing(),
        prediction=Prediction(),
        timeout=Timeout(),
        estimator=estimator,
    )


def test_round_start_is_encoded_and_routed_to_configured_endpoint() -> None:
    async def scenario() -> None:
        clock = Clock()
        core = make_core()
        daemon = UdpGateway(core, endpoints={1: ("127.0.0.1", 9001)}, clock_ns=clock, tick_interval_ns=1_000_000_000)
        transport = FakeTransport()
        daemon.connection_made(transport)  # type: ignore[arg-type]
        try:
            daemon.start_round(round_id=1, deadline_ns=1_000, K=3, L=2)
            assert len(transport.sent) == 1
            packet, endpoint = transport.sent[0]
            message = decode_message(packet)
            assert endpoint == ("127.0.0.1", 9001)
            assert message.message_type is MessageType.ROUND_START
            assert message.round_id == 1 and message.client_id == 1
            assert message.t_send_ns == 100
            assert message.body == {"deadline_ns": 1_000, "K": 3, "L": 2, "rho_us": 0, "mode": 0}
        finally:
            await daemon.close()
        assert transport.closed

    asyncio.run(scenario())


def test_ready_dispatch_uses_gateway_receive_time_and_sends_grant() -> None:
    async def scenario() -> None:
        clock = Clock(200)
        core = make_core()
        core.on_round_start(100, round_id=1, deadline_ns=1_000, K=3, L=2)
        daemon = UdpGateway(core, clock_ns=clock, tick_interval_ns=1_000_000_000)
        transport = FakeTransport()
        daemon.connection_made(transport)  # type: ignore[arg-type]
        packet = encode_message(
            MessageType.READY,
            round_id=1,
            client_id=1,
            t_send_ns=99_999,
            body={"K": 3, "energy": [4.0, 2.0, 0.0]},
        )
        try:
            daemon.datagram_received(packet, ("127.0.0.1", 9001))
            assert daemon.endpoints == {1: ("127.0.0.1", 9001)}
            assert core.clients[1].ready
            assert len(transport.sent) == 1
            grant = decode_message(transport.sent[0][0])
            assert grant.message_type is MessageType.GRANT
            assert grant.t_send_ns == 200
            assert grant.body == {"grant_id": 1, "send_pos": 1}

            clock.now = 210
            data = encode_message(
                MessageType.DATA,
                round_id=1,
                client_id=1,
                t_send_ns=88_888,
                block_length=2,
                body={
                    "grant_id": 1,
                    "send_pos": 1,
                    "block_idx": 0,
                    "n_coords": 2,
                    "payload": [1.0, 0.0],
                },
            )
            daemon.datagram_received(data, ("127.0.0.1", 9001))
            assert core.clients[1].m == 1
            assert len(transport.sent) == 2
            assert decode_message(transport.sent[1][0]).body == {"grant_id": 2, "send_pos": 2}
        finally:
            await daemon.close()

    asyncio.run(scenario())


def test_malformed_unexpected_and_conflicting_endpoint_packets_are_dropped() -> None:
    async def scenario() -> None:
        core = make_core()
        daemon = UdpGateway(core, endpoints={1: ("127.0.0.1", 9001)}, tick_interval_ns=1_000_000_000)
        transport = FakeTransport()
        daemon.connection_made(transport)  # type: ignore[arg-type]
        try:
            daemon.datagram_received(b"bad", ("127.0.0.1", 9001))
            wrong_direction = encode_message(
                MessageType.GRANT,
                round_id=1,
                client_id=1,
                t_send_ns=1,
                body={"grant_id": 1, "send_pos": 1},
            )
            daemon.datagram_received(wrong_direction, ("127.0.0.1", 9001))
            ready = encode_message(
                MessageType.READY,
                round_id=1,
                client_id=1,
                t_send_ns=1,
                body={"K": 3, "energy": [4.0, 2.0, 0.0]},
            )
            daemon.datagram_received(ready, ("127.0.0.1", 9002))
            assert daemon.stats.malformed == 1
            assert daemon.stats.unexpected_type == 1
            assert daemon.stats.endpoint_conflict == 1
            assert transport.sent == []
        finally:
            await daemon.close()

    asyncio.run(scenario())


def test_scheduler_uses_stable_deadlines_without_overlapping_ticks() -> None:
    async def scenario() -> None:
        clock = Clock(0)
        estimator = TickEstimator(clock=clock)
        core = make_core(estimator=estimator)
        core.on_round_start(0, round_id=1, deadline_ns=20_000_000, K=3, L=2)

        async def fake_sleep(seconds: float) -> None:
            clock.now += round(seconds * 1_000_000_000)

        daemon = UdpGateway(core, clock_ns=clock, sleep=fake_sleep)
        estimator.daemon = daemon
        await daemon._scheduler_loop()

        assert estimator.ticks == [1_000_000, 4_000_000, 5_000_000]
        assert daemon.stats.tick_count == 3
        assert not daemon._tick_active

    asyncio.run(scenario())


def test_scheduler_deadline_and_close_cleanup() -> None:
    async def scenario() -> None:
        clock = Clock(100)
        core = make_core()
        core.on_round_start(100, round_id=1, deadline_ns=200, K=3, L=2)
        daemon = UdpGateway(core, endpoints={1: ("127.0.0.1", 9001)}, clock_ns=clock, tick_interval_ns=1_000_000_000)
        transport = FakeTransport()
        daemon.connection_made(transport)  # type: ignore[arg-type]
        clock.now = 200
        daemon._run_tick()
        assert core.phase is GatewayPhase.DEADLINE
        assert decode_message(transport.sent[0][0]).message_type is MessageType.ROUND_END
        await daemon.close()
        assert transport.closed and daemon.scheduler_task is None

    asyncio.run(scenario())


def test_missing_aggregate_receiver_is_reported() -> None:
    daemon = UdpGateway(make_core())

    daemon.dispatch_actions([EmitAggregate(object(), round_id=4)])

    assert daemon.stats.aggregate_transfer_errors == 1


def test_async_aggregate_sender_failure_is_counted() -> None:
    class BrokenSender:
        async def send_aggregate(self, round_id, aggregate, client_n_coords):
            raise OSError("unavailable")

    async def scenario() -> None:
        daemon = UdpGateway(make_core(), aggregate_sender=BrokenSender())
        daemon.dispatch_actions([EmitAggregate(object(), round_id=4)])
        await asyncio.gather(*tuple(daemon._aggregate_tasks), return_exceptions=True)
        await asyncio.sleep(0)

        assert daemon.stats.aggregate_transfer_errors == 1
        assert daemon._aggregate_tasks == set()

    asyncio.run(scenario())


def test_connection_lost_cancels_and_awaits_aggregate_tasks() -> None:
    async def scenario() -> None:
        started = asyncio.Event()

        async def aggregate_handler(aggregate) -> None:
            started.set()
            await asyncio.Event().wait()

        daemon = UdpGateway(make_core(), aggregate_handler=aggregate_handler)
        daemon.connection_made(FakeTransport())  # type: ignore[arg-type]
        daemon.dispatch_actions([EmitAggregate(object())])
        await started.wait()
        aggregate_task = next(iter(daemon._aggregate_tasks))

        daemon.connection_lost(None)
        await daemon.close()

        assert aggregate_task.done() and aggregate_task.cancelled()
        assert daemon._aggregate_tasks == set()
        assert daemon.scheduler_task is None
        assert daemon.transport is None

    asyncio.run(scenario())


def test_loopback_udp_ready_smoke() -> None:
    class Client(asyncio.DatagramProtocol):
        def __init__(self, received: asyncio.Future[bytes]) -> None:
            self.received = received
            self.transport: asyncio.DatagramTransport | None = None

        def connection_made(self, transport) -> None:
            self.transport = transport

        def datagram_received(self, data, addr) -> None:
            if not self.received.done():
                self.received.set_result(data)

    async def scenario() -> None:
        core = make_core()
        start_ns = __import__("time").monotonic_ns()
        core.on_round_start(start_ns, round_id=1, deadline_ns=start_ns + 1_000_000_000, K=3, L=2)
        try:
            daemon = await UdpGateway.bind(core, local_addr=("127.0.0.1", 0), tick_interval_ns=50_000_000)
        except PermissionError:
            pytest.skip("the current environment forbids opening UDP sockets")
        assert daemon.transport is not None
        server_addr = daemon.transport.get_extra_info("sockname")
        loop = asyncio.get_running_loop()
        received: asyncio.Future[bytes] = loop.create_future()
        client_transport, _ = await loop.create_datagram_endpoint(lambda: Client(received), local_addr=("127.0.0.1", 0))
        try:
            ready = encode_message(
                MessageType.READY,
                round_id=1,
                client_id=1,
                t_send_ns=1,
                body={"K": 3, "energy": [4.0, 2.0, 0.0]},
            )
            client_transport.sendto(ready, server_addr)
            response = await asyncio.wait_for(received, timeout=1.0)
            assert decode_message(response).message_type is MessageType.GRANT
        finally:
            client_transport.close()
            await daemon.close()

    asyncio.run(scenario())
