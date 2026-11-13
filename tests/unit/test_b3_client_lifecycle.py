"""B3 client round lifecycle: ROUND_END vs. the PLAN transfer task.

All interleavings are forced deterministically: a fake clock, a controlled
``sleep`` whose futures the test resolves, and ``asyncio.sleep(0)`` only to let
already-runnable callbacks execute. No test waits on wall-clock time.
"""

from __future__ import annotations

import asyncio
import gc

import numpy as np
import pytest

from fedgrant.baselines.b2_udp import B2ClientCore
from fedgrant.baselines.b3_plan import B3PlanError, B3UdpClient
from fedgrant.common.constants import MessageType, TransferMode
from fedgrant.common.wire import decode_message, encode_message
from fedgrant.encoder.ef import bitmap_from_indices

GW = ("127.0.0.1", 9000)
D_TEST, L_TEST = 6, 2  # K = 3 blocks
K_TEST = 3
RATE = 624_000  # one 78-byte DATA frame per millisecond


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000_000

    def __call__(self) -> int:
        return self.now


class _ControlledSleep:
    """``sleep`` that blocks until the test releases it, then advances the clock."""

    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.pending: list[tuple[float, asyncio.Future]] = []

    async def sleep(self, seconds: float) -> None:
        future = asyncio.get_running_loop().create_future()
        self.pending.append((seconds, future))
        await future
        self.clock.now += int(round(seconds * 1e9))

    def release(self) -> int:
        released = 0
        for _, future in self.pending:
            if not future.done():
                future.set_result(None)
                released += 1
        self.pending.clear()
        return released


class _Transport:
    def __init__(self, fail_on_data: int | None = None) -> None:
        self.sent: list = []
        self.fail_on_data = fail_on_data
        self.data_attempts = 0

    def sendto(self, packet: bytes, addr) -> None:
        message = decode_message(packet, block_length=L_TEST)
        if message.message_type is MessageType.DATA:
            self.data_attempts += 1
            if self.data_attempts == self.fail_on_data:
                raise OSError("simulated send failure")
        self.sent.append(message)

    def data(self, round_id: int | None = None) -> list:
        return [m for m in self.sent if m.message_type is MessageType.DATA
                and (round_id is None or m.round_id == round_id)]

    def close(self) -> None:
        pass


def _update(round_id: int) -> np.ndarray:
    return np.asarray([3.0, 0.5, 2.0, 0.1, 1.0, 0.2], dtype=np.float32) * round_id


async def _client(fail_on_data: int | None = None):
    clock = _Clock()
    sleeper = _ControlledSleep(clock)
    client = B3UdpClient(B2ClientCore(5, d=D_TEST, L=L_TEST), gateway_addr=GW, update_provider=_update,
                         clock_ns=clock, sleep=sleeper.sleep)
    transport = _Transport(fail_on_data)
    client.connection_made(transport)
    return client, clock, sleeper, transport


def _msg(message_type, round_id, body):
    return encode_message(message_type, round_id=round_id, client_id=5, t_send_ns=0, body=body,
                          block_length=L_TEST)


def _round_start(clock, round_id, *, rho_us=0, deadline_in_ns=10_000_000):
    return _msg(MessageType.ROUND_START, round_id, {
        "deadline_ns": clock.now + deadline_in_ns, "K": K_TEST, "L": L_TEST,
        "rho_us": rho_us, "mode": int(TransferMode.PLAN)})


def _plan(round_id, k=K_TEST, rate=RATE):
    return _msg(MessageType.PLAN, round_id, {"k": k, "rate_bps": rate})


def _round_end(round_id, delivered=()):
    delivered = sorted(delivered)
    return _msg(MessageType.ROUND_END, round_id, {
        "K": K_TEST, "n_delivered": len(delivered), "bitmap": bitmap_from_indices(delivered, K_TEST)})


async def _yield(times: int = 5) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


async def _assert_clean(client: B3UdpClient) -> None:
    await client.close()
    assert not client._settling
    assert client.sender._sending is False
    others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert others == [], others


def _run(scenario):
    """Run a scenario and fail on any unretrieved task exception."""
    errors: list[dict] = []

    async def main():
        asyncio.get_running_loop().set_exception_handler(lambda _loop, ctx: errors.append(ctx))
        await scenario()

    asyncio.run(main())
    gc.collect()
    assert errors == [], errors


def test_round_end_during_running_transfer_records_exact_partial_counts() -> None:
    async def scenario():
        client, clock, sleeper, transport = await _client()
        client.datagram_received(_round_start(clock, 1), GW)
        client.datagram_received(_plan(1), GW)
        await _yield()
        task = client._plan_task
        assert not task.done() and len(transport.data(1)) == 1  # sent pos 1, now pacing
        sent_idx = transport.data(1)[0].body["block_idx"]

        client.datagram_received(_round_end(1, [sent_idx]), GW)  # task still running
        assert client.core.round is None  # error feedback applied at once, not deferred
        record = await client.wait_for_round_record(timeout=1.0)
        assert task.cancelled()
        assert (record.round_id, record.finished_by, record.transfer_status) == (1, "round_end", "cancelled")
        assert record.transfer.sent_indices == (sent_idx,) and record.packets_sent == 1
        assert record.transfer.application_bytes_sent == 32 + 2 * L_TEST
        assert record.delivered_indices == (sent_idx,)
        assert client.last_transfer is record.transfer
        # A released stale sleep cannot produce further DATA for the round.
        sleeper.release()
        await _yield()
        assert len(transport.data()) == 1 and len(client.round_records) == 1
        assert client._finished_rounds.empty()
        await _assert_clean(client)

    _run(scenario)


def test_round_end_right_after_transfer_completes_uses_the_task_result_once() -> None:
    async def scenario():
        client, clock, sleeper, transport = await _client()
        client.datagram_received(_round_start(clock, 1), GW)
        client.datagram_received(_plan(1), GW)
        while not client._plan_task.done():
            await _yield()
            sleeper.release()
        result = client._plan_task.result()
        assert result.packets_sent == K_TEST and result.stop_reason == "all_blocks_sent"

        client.datagram_received(_round_end(1, result.sent_indices), GW)
        record = await client.wait_for_round_record(timeout=1.0)
        assert record.transfer is result and record.transfer_status == "completed"
        assert record.transfer.sent_indices == tuple(m.body["block_idx"] for m in transport.data(1))
        assert [m.body["send_pos"] for m in transport.data(1)] == [1, 2, 3]
        assert client._finished_rounds.empty() and len(client.round_records) == 1
        await _assert_clean(client)

    _run(scenario)


def test_fallback_timeout_finalizes_once_and_a_later_round_end_is_rejected() -> None:
    async def scenario():
        client, clock, sleeper, transport = await _client()
        client.datagram_received(_round_start(clock, 1), GW)
        client.datagram_received(_plan(1), GW)
        await _yield()
        client._round_end_timeout(1)  # the 200 ms fallback timer firing
        record = await client.wait_for_round_record(timeout=1.0)
        assert (record.finished_by, record.delivered_indices) == ("round_end_timeout", ())
        assert record.transfer_status == "cancelled" and record.packets_sent == len(transport.data(1)) == 1
        assert client.round_end_timeouts == 1

        rejected = client.rejected_datagrams
        client.datagram_received(_round_end(1, [0]), GW)  # too late
        await _yield()
        assert client.rejected_datagrams == rejected + 1
        assert client._finished_rounds.empty() and list(client.round_records) == [1]
        client._round_end_timeout(1)  # a second firing is a no-op
        assert client.round_end_timeouts == 1
        await _assert_clean(client)

    _run(scenario)


def test_duplicate_round_end_is_rejected_and_the_round_finalized_once() -> None:
    async def scenario():
        client, clock, sleeper, transport = await _client()
        client.datagram_received(_round_start(clock, 1), GW)  # no PLAN this round
        client.datagram_received(_round_end(1), GW)
        client.datagram_received(_round_end(1), GW)
        record = await client.wait_for_round_record(timeout=1.0)
        await _yield()
        assert (record.transfer, record.transfer_status, record.packets_sent) == (None, "no_plan", 0)
        assert client.rejected_datagrams == 1
        assert client._finished_rounds.empty() and list(client.round_records) == [1]
        await _assert_clean(client)

    _run(scenario)


def test_previous_round_end_after_next_round_start_does_not_touch_the_new_round() -> None:
    async def scenario():
        client, clock, sleeper, transport = await _client()
        client.datagram_received(_round_start(clock, 1), GW)
        client.datagram_received(_round_end(1), GW)
        assert (await client.wait_for_round_record(timeout=1.0)).round_id == 1
        clock.now += 50_000_000
        client.datagram_received(_round_start(clock, 2), GW)
        client.datagram_received(_plan(2), GW)
        await _yield()
        before = (client.core.round.round_id, len(transport.data(2)), client.rejected_datagrams)

        client.datagram_received(_round_end(1, [0, 1]), GW)  # delayed duplicate of round 1
        await _yield()
        assert client.rejected_datagrams == before[2] + 1
        assert (client.core.round.round_id, len(transport.data(2))) == before[:2]
        assert not client._plan_task.done() and 2 not in client.round_records
        client.datagram_received(_round_end(2), GW)
        record = await client.wait_for_round_record(timeout=1.0)
        assert record.round_id == 2 and record.transfer.round_id == 2
        await _assert_clean(client)

    _run(scenario)


def test_stale_ready_callback_never_sends_for_a_finished_or_newer_round() -> None:
    async def scenario():
        client, clock, sleeper, transport = await _client()
        client.datagram_received(_round_start(clock, 1, rho_us=5_000), GW)
        handle = client._ready_handle
        assert handle is not None and not [m for m in transport.sent if m.message_type is MessageType.READY]
        client.datagram_received(_round_end(1), GW)
        assert handle.cancelled() and client._ready_handle is None
        energies = (1.0, 0.5, 0.25)
        client._send_ready(1, energies)  # a stale callback firing after finish
        client.datagram_received(_round_start(clock, 2, rho_us=5_000), GW)
        client._send_ready(1, energies)  # ... or after the next round started
        assert not [m for m in transport.sent if m.message_type is MessageType.READY]
        assert client.ready_sent_ns is None  # round 2 READY time untouched
        record = await client.wait_for_round_record(timeout=1.0)
        assert record.round_id == 1 and record.ready_sent_ns is None
        await _assert_clean(client)

    _run(scenario)


def test_round_end_after_failed_send_records_failure_and_exact_partial_counts() -> None:
    async def scenario():
        client, clock, sleeper, transport = await _client(fail_on_data=2)
        client.datagram_received(_round_start(clock, 1), GW)
        client.datagram_received(_plan(1), GW)
        while not client._plan_task.done():
            await _yield()
            sleeper.release()
        assert isinstance(client._plan_task.exception(), Exception)
        client.datagram_received(_round_end(1), GW)
        record = await client.wait_for_round_record(timeout=1.0)
        assert record.transfer_status == "failed" and "simulated send failure" in record.transfer_error
        assert record.packets_sent == len(transport.data(1)) == 1  # the failed send is not counted
        assert record.transfer.stop_reason == "send_failed"
        await _assert_clean(client)

    _run(scenario)  # also asserts the task exception was retrieved


def test_round_end_while_a_cancellation_is_pending_records_once() -> None:
    async def scenario():
        client, clock, sleeper, transport = await _client()
        client.datagram_received(_round_start(clock, 1), GW)
        client.datagram_received(_plan(1), GW)
        await _yield()
        client._plan_task.cancel()  # e.g. shutdown racing with ROUND_END
        client.datagram_received(_round_end(1), GW)
        record = await client.wait_for_round_record(timeout=1.0)
        assert record.transfer_status == "cancelled" and record.packets_sent == 1
        await _yield()
        assert client._finished_rounds.empty() and list(client.round_records) == [1]
        await _assert_clean(client)

    _run(scenario)


def test_old_transfer_settling_after_next_round_start_stays_in_its_own_round() -> None:
    async def scenario():
        client, clock, sleeper, transport = await _client()
        client.datagram_received(_round_start(clock, 1), GW)
        client.datagram_received(_plan(1), GW)
        await _yield()
        client.datagram_received(_round_end(1), GW)  # cancellation requested, not yet settled
        client.datagram_received(_round_start(clock, 2), GW)  # same loop step
        assert client.last_transfer is None and client.plan_start_ns is None
        record = await client.wait_for_round_record(timeout=1.0)
        assert record.round_id == 1 and record.transfer.round_id == 1
        assert record.plan_start_ns is not None
        assert client.last_transfer is None and client.plan_start_ns is None  # round 2 untouched
        assert client.sender._sending is False  # round 2 may send
        client.datagram_received(_plan(2), GW)
        await _yield()
        assert len(transport.data(2)) == 1
        client.datagram_received(_round_end(2), GW)
        assert (await client.wait_for_round_record(timeout=1.0)).transfer.round_id == 2
        await _assert_clean(client)

    _run(scenario)


def test_wait_for_plan_reports_a_round_end_cancellation_without_cancelling_the_caller() -> None:
    async def scenario():
        client, clock, sleeper, transport = await _client()
        client.datagram_received(_round_start(clock, 1), GW)
        client.datagram_received(_plan(1), GW)
        waiter = asyncio.create_task(client.wait_for_plan())
        await _yield()
        client.datagram_received(_round_end(1), GW)
        with pytest.raises(B3PlanError, match="cancelled at round end"):
            await waiter
        assert (await client.wait_for_round_record(timeout=1.0)).transfer_status == "cancelled"
        await _assert_clean(client)

    _run(scenario)


def test_three_rounds_each_finalized_once_with_their_own_transfer() -> None:
    async def scenario():
        client, clock, sleeper, transport = await _client()
        outcomes = {1: "completed", 2: "cancelled", 3: "no_plan"}
        for round_id in (1, 2, 3):
            client.datagram_received(_round_start(clock, round_id), GW)
            if round_id != 3:
                client.datagram_received(_plan(round_id), GW)
            await _yield()
            if round_id == 1:
                while not client._plan_task.done():
                    await _yield()
                    sleeper.release()
            client.datagram_received(_round_end(round_id), GW)
            record = await client.wait_for_round_record(timeout=1.0)
            assert (record.round_id, record.transfer_status) == (round_id, outcomes[round_id])
            assert record.packets_sent == len(transport.data(round_id))
            if record.transfer is not None:
                assert record.transfer.round_id == round_id
                assert set(record.transfer.sent_indices) == {m.body["block_idx"] for m in transport.data(round_id)}
            clock.now += 50_000_000
        assert sorted(client.round_records) == [1, 2, 3] and client._finished_rounds.empty()
        assert sum(r.packets_sent for r in client.round_records.values()) == len(transport.data())
        await _assert_clean(client)

    _run(scenario)
