"""Tests for the asyncio UDP client adapter."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np
import pytest

from fedgrant.common.constants import MessageType, TransferMode
from fedgrant.common.wire import decode_message, encode_message
from fedgrant.core.client_core import ClientCore
from fedgrant.encoder.ef import bitmap_from_indices
from fedgrant.net import udp_client
from fedgrant.net.udp_client import UdpClient


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


def packet(message_type, *, round_id=1, client_id=3, t_send_ns=100, body, block_length=700):
    return encode_message(
        message_type,
        round_id=round_id,
        client_id=client_id,
        t_send_ns=t_send_ns,
        body=body,
        block_length=block_length,
    )


def test_round_start_waits_readiness_delay_and_grants_send_requested_data() -> None:
    async def scenario() -> None:
        clock = Clock(100)
        core = ClientCore(3, d=5, L=2)
        transport = FakeTransport()
        client = UdpClient(
            core,
            gateway_addr=("127.0.0.1", 9000),
            update_provider=lambda round_id: np.asarray([1, 2, 3, 4, 5], dtype=np.float32),
            clock_ns=clock,
        )
        client.connection_made(transport)  # type: ignore[arg-type]
        round_start = packet(
            MessageType.ROUND_START,
            body={
                "deadline_ns": 1_000_000_000,
                "K": 3,
                "L": 2,
                "rho_us": 10_000,
                "mode": TransferMode.GRANT,
            },
        )
        client.datagram_received(round_start, ("127.0.0.1", 9000))
        assert transport.sent == []
        await asyncio.sleep(0.02)
        assert len(transport.sent) == 1
        ready = decode_message(transport.sent[0][0])
        assert ready.message_type is MessageType.READY
        assert ready.body == {"K": 3, "energy": (25.0, 25.0, 5.0)}

        grant = packet(
            MessageType.GRANT,
            body={"grant_id": 12, "send_pos": 2},
        )
        client.datagram_received(grant, ("127.0.0.1", 9000))
        assert len(transport.sent) == 2
        data = decode_message(transport.sent[1][0], block_length=2)
        assert data.message_type is MessageType.DATA
        assert data.body["grant_id"] == 12
        assert data.body["send_pos"] == 2
        assert data.body["block_idx"] == 2
        assert data.body["n_coords"] == 1
        assert data.body["payload"] == (5.0, 0.0)

        client.datagram_received(grant, ("127.0.0.1", 9000))
        assert len(transport.sent) == 2
        assert client.stats.stale_or_invalid == 1
        await client.close()
        assert transport.closed

    asyncio.run(scenario())


def test_round_end_updates_residual_once_and_rejects_stale_or_wrong_source() -> None:
    async def scenario() -> None:
        core = ClientCore(3, d=5, L=2)
        transport = FakeTransport()
        client = UdpClient(
            core,
            gateway_addr=("127.0.0.1", 9000),
            update_provider=lambda round_id: np.asarray([1, 2, 3, 4, 5], dtype=np.float32),
        )
        client.connection_made(transport)  # type: ignore[arg-type]
        start = packet(
            MessageType.ROUND_START,
            body={
                "deadline_ns": 1_000_000_000,
                "K": 3,
                "L": 2,
                "rho_us": 0,
                "mode": TransferMode.GRANT,
            },
        )
        client.datagram_received(start, ("127.0.0.1", 9000))
        await asyncio.sleep(0)
        u = core.round.u.copy()

        bitmap = bitmap_from_indices([2], 3)
        end = packet(
            MessageType.ROUND_END,
            body={"K": 3, "n_delivered": 1, "bitmap": bitmap},
        )
        client.datagram_received(end, ("127.0.0.1", 9001))
        assert client.stats.wrong_endpoint == 1
        client.datagram_received(end, ("127.0.0.1", 9000))
        assert await client.wait_for_round_end(0.1) == 1
        expected = np.zeros(5, dtype=np.float32)
        expected[4:5] = u[4:5].astype(np.float16).astype(np.float32)
        np.testing.assert_array_equal(client.last_reconstruction, expected)
        np.testing.assert_array_equal(core.error_feedback.e, u - expected)

        client.datagram_received(end, ("127.0.0.1", 9000))
        assert client.stats.stale_or_invalid == 1
        np.testing.assert_array_equal(core.error_feedback.e, u - expected)
        await client.close()

    asyncio.run(scenario())


def test_malformed_wrong_client_and_unexpected_messages_are_ignored() -> None:
    core = ClientCore(3, d=5, L=2)
    client = UdpClient(
        core,
        gateway_addr=("127.0.0.1", 9000),
        update_provider=lambda round_id: np.ones(5, dtype=np.float32),
    )
    client.datagram_received(b"bad", ("127.0.0.1", 9000))
    wrong_client = packet(
        MessageType.ROUND_START,
        client_id=4,
        body={
            "deadline_ns": 1000,
            "K": 3,
            "L": 2,
            "rho_us": 0,
            "mode": TransferMode.GRANT,
        },
    )
    client.datagram_received(wrong_client, ("127.0.0.1", 9000))
    unexpected = packet(
        MessageType.PLAN,
        body={"k": 1, "rate_bps": 100},
    )
    client.datagram_received(unexpected, ("127.0.0.1", 9000))

    assert client.stats.malformed == 1
    assert client.stats.wrong_client == 1
    assert client.stats.unexpected_type == 1
    assert core.round is None


def test_round_end_timeout_uses_all_blocks_undelivered(monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(udp_client, "ROUND_END_GRACE_NS", 5_000_000)
        clock = Clock(100)
        core = ClientCore(3, d=5, L=2)
        client = UdpClient(
            core,
            gateway_addr=("127.0.0.1", 9000),
            update_provider=lambda round_id: np.asarray([1, 2, 3, 4, 5], dtype=np.float32),
            clock_ns=clock,
        )
        client.connection_made(FakeTransport())  # type: ignore[arg-type]
        start = packet(
            MessageType.ROUND_START,
            t_send_ns=clock.now,
            body={
                "deadline_ns": clock.now + 5_000_000,
                "K": 3,
                "L": 2,
                "rho_us": 0,
                "mode": TransferMode.GRANT,
            },
        )
        client.datagram_received(start, ("127.0.0.1", 9000))
        u = core.round.u.copy()

        assert await client.wait_for_round_end(0.2) == 1
        assert client.stats.round_end_timeouts == 1
        assert client.last_reconstruction is not None
        assert not client.last_reconstruction.any()
        np.testing.assert_array_equal(core.error_feedback.e, u)
        await client.close()

    asyncio.run(scenario())


# -- READY timer origin: rho_i runs from ROUND_START receipt (design decision D-20) --

ENCODE_S = 0.050
UPDATE = np.asarray([1, 2, 3, 4, 5], dtype=np.float32)
EXPECTED_READY = {"K": 3, "energy": (25.0, 25.0, 5.0)}


class _VirtualHandle:
    def __init__(self, when: float, callback, args) -> None:
        self._when, self._callback, self._args = when, callback, args
        self._cancelled = False

    def when(self) -> float:
        return self._when

    def cancel(self) -> None:
        self._cancelled = True

    def cancelled(self) -> bool:
        return self._cancelled


class VirtualLoop:
    """Deterministic stand-in for the running loop's timer API.

    Due handles fire in ``when`` order. As in asyncio, a handle whose time has
    already passed runs on the next iteration, at the current time.
    """

    def __init__(self, now: float = 100.0) -> None:
        self.now = now
        self.handles: list[_VirtualHandle] = []

    def time(self) -> float:
        return self.now

    def call_at(self, when: float, callback, *args) -> _VirtualHandle:
        handle = _VirtualHandle(when, callback, args)
        self.handles.append(handle)
        return handle

    def call_later(self, delay: float, callback, *args) -> _VirtualHandle:
        return self.call_at(self.now + delay, callback, *args)

    def run_until(self, t: float) -> None:
        while True:
            due = [h for h in self.handles if not h.cancelled() and h.when() <= t]
            if not due:
                break
            handle = min(due, key=_VirtualHandle.when)
            self.handles.remove(handle)
            self.now = max(self.now, handle.when())
            handle._callback(*handle._args)
        self.now = max(self.now, t)

    def run_once(self) -> None:
        """One loop iteration: run everything already due, without advancing time."""
        self.run_until(self.now)


def _virtual_client(monkeypatch, rho_s: float):
    loop = VirtualLoop()
    monkeypatch.setattr(udp_client.asyncio, "get_running_loop", lambda: loop)
    trace: dict[str, float] = {}
    sends: list[tuple[MessageType, float]] = []

    def slow_provider(round_id: int) -> np.ndarray:
        trace["encode_start"] = loop.time()
        loop.now += ENCODE_S  # deliberately slow update preparation
        trace["encode_end"] = loop.time()
        return UPDATE

    def observe(data: bytes, now_ns: int) -> None:
        sends.append((decode_message(data).message_type, now_ns / 1e9))

    client = UdpClient(
        ClientCore(3, d=5, L=2),
        gateway_addr=("127.0.0.1", 9000),
        update_provider=slow_provider,
        clock_ns=lambda: round(loop.time() * 1e9),
        packet_observer=observe,
    )
    transport = FakeTransport()
    client.connection_made(transport)  # type: ignore[arg-type]
    received_at = loop.time()
    client.datagram_received(
        packet(
            MessageType.ROUND_START,
            body={
                "deadline_ns": round((received_at + 5.0) * 1e9),
                "K": 3,
                "L": 2,
                "rho_us": round(rho_s * 1e6),
                "mode": TransferMode.GRANT,
            },
        ),
        ("127.0.0.1", 9000),
    )
    return loop, client, transport, trace, sends, received_at


def test_ready_deadline_is_receipt_plus_rho_when_encoding_finishes_first(monkeypatch) -> None:
    rho_s = 0.080
    loop, client, transport, trace, sends, received_at = _virtual_client(monkeypatch, rho_s)

    # The timer origin is taken before preparation starts, and encoding time
    # does not move the READY deadline.
    assert trace["encode_start"] == received_at
    assert trace["encode_end"] == pytest.approx(received_at + ENCODE_S)
    assert client._ready_handle.when() == pytest.approx(received_at + rho_s)
    assert transport.sent == []

    loop.run_until(received_at + rho_s - 1e-6)
    assert transport.sent == []
    loop.run_until(received_at + rho_s)
    assert len(transport.sent) == 1
    ready = decode_message(transport.sent[0][0])
    assert ready.message_type is MessageType.READY and ready.body == EXPECTED_READY
    kind, sent_at = sends[0]
    assert kind is MessageType.READY
    assert sent_at - received_at == pytest.approx(rho_s, abs=1e-9)  # not rho + encoding
    assert sent_at >= trace["encode_end"]


def test_ready_is_sent_as_soon_as_overrunning_encoding_completes(monkeypatch) -> None:
    rho_s = 0.020  # shorter than the 50 ms preparation
    loop, client, transport, trace, sends, received_at = _virtual_client(monkeypatch, rho_s)

    assert trace["encode_start"] == received_at
    assert client._ready_handle.when() == pytest.approx(received_at + rho_s)
    # Never sent from inside preparation or before it completes.
    assert transport.sent == []
    assert loop.time() == pytest.approx(trace["encode_end"])

    loop.run_once()  # the next loop iteration, with no time elapsed
    assert len(transport.sent) == 1
    assert decode_message(transport.sent[0][0]).body == EXPECTED_READY
    _, sent_at = sends[0]
    assert sent_at == pytest.approx(trace["encode_end"], abs=1e-9)
    assert sent_at - received_at == pytest.approx(ENCODE_S, abs=1e-9)  # not rho + encoding


@pytest.mark.parametrize("rho_s", [0.100, 0.020])
def test_real_loop_ready_gap_excludes_encoding_time(rho_s: float) -> None:
    async def scenario() -> tuple[int, int, int]:
        ready_sent = asyncio.Event()
        marks: dict[str, int] = {}

        def slow_provider(round_id: int) -> np.ndarray:
            time.sleep(ENCODE_S)  # blocking, like real encoding work
            marks["encode_end"] = time.monotonic_ns()
            return UPDATE

        def observe(data: bytes, now_ns: int) -> None:
            if decode_message(data).message_type is MessageType.READY:
                marks["ready"] = now_ns
                ready_sent.set()

        client = UdpClient(
            ClientCore(3, d=5, L=2),
            gateway_addr=("127.0.0.1", 9000),
            update_provider=slow_provider,
            packet_observer=observe,
        )
        client.connection_made(FakeTransport())  # type: ignore[arg-type]
        received_ns = time.monotonic_ns()
        client.datagram_received(
            packet(
                MessageType.ROUND_START,
                body={
                    "deadline_ns": received_ns + 5_000_000_000,
                    "K": 3,
                    "L": 2,
                    "rho_us": round(rho_s * 1e6),
                    "mode": TransferMode.GRANT,
                },
            ),
            ("127.0.0.1", 9000),
        )
        await asyncio.wait_for(ready_sent.wait(), 2.0)
        await client.close()
        return received_ns, marks["encode_end"], marks["ready"]

    received_ns, encode_end_ns, ready_ns = asyncio.run(scenario())
    gap_s = (ready_ns - received_ns) / 1e9
    assert ready_ns >= encode_end_ns  # never before the update is ready
    # Expected gap max(rho, encoding); the old origin gave at least rho + encoding.
    assert gap_s >= max(rho_s, ENCODE_S) - 0.001
    assert gap_s < rho_s + ENCODE_S
