from __future__ import annotations

import asyncio

import numpy as np
import pytest

from fedgrant.net import fl_tcp
from fedgrant.net.fl_tcp import (
    AggregateEnvelope,
    ModelBackedUpdateProvider,
    ModelMailbox,
    TcpAggregateReceiver,
    TcpAggregateSender,
    TcpModelReceiver,
    TransferError,
    broadcast_model,
    send_model,
)


class ModelSource:
    def __init__(self, model: np.ndarray) -> None:
        self.model = model

    def current_model(self) -> tuple[int, np.ndarray]:
        return 4, self.model


def test_model_broadcast_installs_model_for_client_update_provider() -> None:
    async def scenario() -> None:
        model = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        mailbox = ModelMailbox(expected_shape=(3,))
        receiver = await TcpModelReceiver(mailbox).start()
        try:
            version = await broadcast_model(
                ModelSource(model),
                [receiver.address],
                round_id=7,
                timeout=1,
            )
            observed: list[tuple[int, np.ndarray]] = []
            provider = ModelBackedUpdateProvider(
                mailbox,
                lambda round_id, received: observed.append((round_id, received.copy()))
                or received * np.float32(2),
            )
            update = provider(7)
            assert version == 4
            np.testing.assert_array_equal(update, np.asarray([2.0, 4.0, 6.0], np.float32))
            assert observed[0][0] == 7
            np.testing.assert_array_equal(observed[0][1], model)
            with pytest.raises(TransferError, match="has not been received"):
                provider(8)
        finally:
            await receiver.close()

    asyncio.run(scenario())


def test_model_broadcast_failure_does_not_unlock_update_provider() -> None:
    async def scenario() -> None:
        mailbox = ModelMailbox()
        receiver = await TcpModelReceiver(mailbox, max_transfer_bytes=128).start()
        try:
            with pytest.raises(TransferError):
                await broadcast_model(
                    ModelSource(np.ones(100, dtype=np.float32)),
                    [receiver.address],
                    round_id=1,
                    timeout=1,
                )
            with pytest.raises(TransferError, match="has not been received"):
                mailbox.model_for_round(1)
        finally:
            await receiver.close()

    asyncio.run(scenario())


def test_model_broadcast_waits_for_every_client_acknowledgement() -> None:
    async def scenario() -> None:
        async def no_ack(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                await fl_tcp._read_frame(reader, fl_tcp.MAX_TRANSFER_BYTES)
            finally:
                writer.close()
                await writer.wait_closed()

        mailbox = ModelMailbox()
        receiver = await TcpModelReceiver(mailbox).start()
        silent_server = await asyncio.start_server(no_ack, "127.0.0.1", 0)
        silent_host, silent_port = silent_server.sockets[0].getsockname()[:2]
        try:
            with pytest.raises(TransferError, match="truncated TCP frame"):
                await broadcast_model(
                    ModelSource(np.asarray([1.0], dtype=np.float32)),
                    [receiver.address, (silent_host, silent_port)],
                    round_id=1,
                    timeout=1,
                )
            assert mailbox.model_for_round(1).tolist() == [1.0]
        finally:
            silent_server.close()
            await silent_server.wait_closed()
            await receiver.close()

    asyncio.run(scenario())


def test_incompatible_model_is_rejected_without_replacing_mailbox_model() -> None:
    async def scenario() -> None:
        mailbox = ModelMailbox(expected_shape=(3,))
        receiver = await TcpModelReceiver(mailbox).start()
        try:
            with pytest.raises(TransferError, match="shape"):
                await broadcast_model(
                    ModelSource(np.asarray([1.0, 2.0], dtype=np.float32)),
                    [receiver.address],
                    round_id=1,
                    timeout=1,
                )
            with pytest.raises(TransferError, match="has not been received"):
                mailbox.model_for_round(1)
        finally:
            await receiver.close()

    asyncio.run(scenario())


def test_model_ack_must_match_model_version_and_payload() -> None:
    async def scenario() -> None:
        async def stale_ack(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            header, payload = await fl_tcp._read_frame(reader, fl_tcp.MAX_TRANSFER_BYTES)
            await fl_tcp._write_frame(
                writer,
                {
                    "version": fl_tcp.PROTOCOL_VERSION,
                    "kind": "ACK",
                    "round_id": header["round_id"],
                    "model_version": header["model_version"] + 1,
                    "payload_sha256": header["payload_sha256"],
                    "status": "ok",
                    "payload_bytes": 0,
                },
                b"",
                fl_tcp.MAX_TRANSFER_BYTES,
            )

        server = await asyncio.start_server(stale_ack, "127.0.0.1", 0)
        host, port = server.sockets[0].getsockname()[:2]
        try:
            with pytest.raises(TransferError, match="different model version"):
                await send_model(
                    host,
                    port,
                    round_id=1,
                    model_version=4,
                    model=np.asarray([1.0], dtype=np.float32),
                    timeout=1,
                )
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_aggregate_transfer_preserves_round_clients_and_counts_and_deduplicates() -> None:
    async def scenario() -> None:
        received: list[AggregateEnvelope] = []
        accepted = asyncio.Event()

        async def accept(envelope: AggregateEnvelope) -> None:
            received.append(envelope)
            accepted.set()

        receiver = await TcpAggregateReceiver(accept).start()
        sender = TcpAggregateSender(*receiver.address, timeout=1)
        aggregate = np.asarray([1.0, -2.0, 3.0], dtype=np.float32)
        counts = {2: {0: 700, 4: 11}, 5: {}}
        try:
            await sender.send_aggregate(12, aggregate, counts)
            await accepted.wait()
            await sender.send_aggregate(12, aggregate, counts)
            assert len(received) == 1
            assert received[0].round_id == 12
            assert received[0].client_ids == (2, 5)
            assert received[0].client_n_coords == counts
            np.testing.assert_array_equal(received[0].aggregate, aggregate)
            with pytest.raises(TransferError, match="conflicting duplicate"):
                await sender.send_aggregate(12, np.asarray([9.0], np.float32), counts)
        finally:
            await receiver.close()

    asyncio.run(scenario())


def test_lost_aggregate_ack_retries_without_reapplying(monkeypatch) -> None:
    async def scenario() -> None:
        received: list[AggregateEnvelope] = []
        successful_acks = 0
        original_write_frame = fl_tcp._write_frame

        async def drop_first_success_ack(writer, header, payload, max_transfer_bytes) -> None:
            nonlocal successful_acks
            if header.get("kind") == "ACK" and header.get("status") == "ok":
                successful_acks += 1
                if successful_acks == 1:
                    return
            await original_write_frame(writer, header, payload, max_transfer_bytes)

        monkeypatch.setattr(fl_tcp, "_write_frame", drop_first_success_ack)

        async def accept(envelope: AggregateEnvelope) -> None:
            received.append(envelope)

        receiver = await TcpAggregateReceiver(accept).start()
        sender = TcpAggregateSender(*receiver.address, timeout=1)
        try:
            await sender.send_aggregate(3, np.asarray([2.0], np.float32), {1: {0: 1}})
            assert len(received) == 1
            assert successful_acks == 2
        finally:
            await receiver.close()

    asyncio.run(scenario())


def test_aggregate_deduplication_is_lost_on_receiver_restart() -> None:
    async def scenario() -> None:
        applied: list[int] = []

        async def accept(envelope: AggregateEnvelope) -> None:
            applied.append(envelope.round_id)

        first_receiver = await TcpAggregateReceiver(accept).start()
        sender = TcpAggregateSender(*first_receiver.address, timeout=1)
        aggregate = np.asarray([1.0], np.float32)
        try:
            await sender.send_aggregate(9, aggregate, {1: {0: 1}})
        finally:
            await first_receiver.close()

        restarted_receiver = await TcpAggregateReceiver(accept).start()
        sender = TcpAggregateSender(*restarted_receiver.address, timeout=1)
        try:
            await sender.send_aggregate(9, aggregate, {1: {0: 1}})
            assert applied == [9, 9]
        finally:
            await restarted_receiver.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (b"\x00\x01\x00\x65", "outside the permitted bound"),
        (b"\x00\x00\x00\x10\x00\x00\x00\x04{}", "truncated TCP frame"),
        (b"\x00\x00\x00\x07\x00\x00\x00\x03{x}", "invalid JSON transfer header"),
    ],
)
def test_tcp_reader_rejects_oversized_truncated_and_malformed_frames(
    frame: bytes,
    message: str,
) -> None:
    async def scenario() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(frame)
        reader.feed_eof()
        with pytest.raises(TransferError, match=message):
            await fl_tcp._read_frame(reader, 64)

    asyncio.run(scenario())


def test_aggregate_transfer_failure_can_retry_without_double_application() -> None:
    async def scenario() -> None:
        attempts = 0
        received: list[AggregateEnvelope] = []

        async def accept(envelope: AggregateEnvelope) -> None:
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                raise RuntimeError("temporary server failure")
            received.append(envelope)

        receiver = await TcpAggregateReceiver(accept).start()
        sender = TcpAggregateSender(*receiver.address, timeout=1)
        aggregate = np.asarray([5.0], dtype=np.float32)
        try:
            with pytest.raises(TransferError, match="receiver failed"):
                await sender.send_aggregate(2, aggregate, {1: {0: 1}})
            await sender.send_aggregate(2, aggregate, {1: {0: 1}})
            await sender.send_aggregate(2, aggregate, {1: {0: 1}})
            assert attempts == 3
            assert len(received) == 1
        finally:
            await receiver.close()

    asyncio.run(scenario())
