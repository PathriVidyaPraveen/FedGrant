"""Tests for the asyncio UDP client adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np

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
