"""B1 client round records carry block_energies like B2's (design decision D-09).

B1's TCP transfer deliberately holds its connection until D, so ROUND_END can be
handled before ``send_round`` returns. The client round record is written at
ROUND_END and the encoded update is dropped by ``finish_round``; the energies
must therefore be captured from the prepared round at that point.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from fedgrant.baselines.b1_tcp import B1ClientTransfer
from fedgrant.common.constants import MessageType, TransferMode
from fedgrant.common.wire import decode_message, encode_message
from fedgrant.encoder.blocks import encode
from fedgrant.encoder.ef import bitmap_from_indices
from fedgrant.mininet import baseline_proc
from fedgrant.mininet.baseline_metrics import eta_from_run_dir, round_eta
from fedgrant.mininet.round_loop import RoundSchedule

D_SMALL, L_SMALL = 40, 8  # K = 5 blocks
K_SMALL = 5


def _cfg(policy: str = "B1") -> baseline_proc.BaselineConfig:
    return baseline_proc.BaselineConfig(
        policy=policy, num_clients=1, rounds=1, master_seed=0, schedule=RoundSchedule(duration_ns=10**9),
        client_endpoints={0: ("127.0.0.1", 9001)}, gateway_host="127.0.0.1", d=D_SMALL, L=L_SMALL)


def _update(round_id: int) -> np.ndarray:
    rng = np.random.default_rng(round_id)
    return rng.normal(size=D_SMALL).astype(np.float32)


def _expected_energies(round_id: int) -> list[float]:
    return [float(x) for x in baseline_proc._block_energies(encode(_update(round_id), L_SMALL))]


def _round_start(round_id: int, deadline_ns: int) -> object:
    return decode_message(encode_message(
        MessageType.ROUND_START, round_id=round_id, client_id=0, t_send_ns=0,
        body={"deadline_ns": deadline_ns, "K": K_SMALL, "L": L_SMALL, "rho_us": 0, "mode": int(TransferMode.TCP)},
        block_length=L_SMALL), block_length=L_SMALL)


def _round_end(round_id: int, delivered: tuple[int, ...]) -> object:
    return decode_message(encode_message(
        MessageType.ROUND_END, round_id=round_id, client_id=0, t_send_ns=0,
        body={"K": K_SMALL, "n_delivered": len(delivered), "bitmap": bitmap_from_indices(delivered, K_SMALL)},
        block_length=L_SMALL), block_length=L_SMALL)


class _HeldTransfer:
    """Stands in for B1TcpClient.send_round: prepares the round, then holds until released."""

    def __init__(self, *, prepare: bool = True) -> None:
        self.release = asyncio.Event()
        self.prepare = prepare

    async def send_round(self, core, *, gateway_host, round_id, deadline_ns, delta):
        if self.prepare:
            core.start_round(round_id, deadline_ns, delta)  # the real encoding path
        await self.release.wait()
        return B1ClientTransfer(client_id=core.client_id, round_id=round_id, blocks_written=K_SMALL,
                                application_bytes_written=K_SMALL * (4 + 2 * L_SMALL), connected=True,
                                connection_error=None, close_reason="gateway_closed", completed_ns=1,
                                residual_finalized=False)


def _run(tmp_path, *, round_end_first: bool, delivered: tuple[int, ...], prepare: bool = True) -> dict:
    async def scenario():
        logs = baseline_proc._ClientLogs(tmp_path / "client_0.json", 0)
        client = baseline_proc.PushOrTcpClient(_cfg(), 0, _update, logs)
        baseline_proc._configure_error_feedback(client.core, client.cfg)
        sender = _HeldTransfer(prepare=prepare)
        client.b1_sender = sender
        client._on_round_start(_round_start(1, deadline_ns=2**62), received_ns=0)
        await asyncio.sleep(0)  # readiness task starts and (optionally) prepares the round
        await asyncio.sleep(0)
        if round_end_first:
            client._on_round_end(_round_end(1, delivered))  # ROUND_END while the TCP stream is held open
            sender.release.set()
        else:
            sender.release.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            client._on_round_end(_round_end(1, delivered))
        await asyncio.gather(*client._tasks, return_exceptions=True)
        for timer in client._timers.values():
            timer.cancel()
        logs.close()

    asyncio.run(scenario())
    (record,) = [json.loads(line) for line in (tmp_path / "client_0_rounds.jsonl").read_text().splitlines()]
    return record


@pytest.mark.parametrize("round_end_first", [True, False])
def test_b1_record_has_block_energies_whichever_finishes_first(tmp_path, round_end_first: bool) -> None:
    record = _run(tmp_path, round_end_first=round_end_first, delivered=(0, 2))
    assert record["status"] == "round_end" and record["delivered"] == [0, 2]
    assert record["block_energies"] == _expected_energies(1)
    assert len(record["block_energies"]) == K_SMALL and all(np.isfinite(record["block_energies"]))


def test_b1_multiple_and_empty_deliveries_feed_eta(tmp_path) -> None:
    many = _run(tmp_path / "many", round_end_first=True, delivered=(0, 1, 3))
    empty = _run(tmp_path / "empty", round_end_first=True, delivered=())
    energies = many["block_energies"]
    assert empty["block_energies"] == energies and empty["delivered"] == []
    eta = round_eta({1: {0: many["delivered"]}}, {1: {0: energies}}, {0: 1.0})[1]
    assert eta == pytest.approx(sum(energies[j] for j in (0, 1, 3)) / sum(energies))
    assert round_eta({1: {0: []}}, {1: {0: energies}}, {0: 1.0})[1] == 0.0


def test_unprepared_round_logs_no_invented_energies_and_eta_is_missing(tmp_path) -> None:
    # The transfer failed before encoding (no prepared round): nothing to log.
    record = _run(tmp_path, round_end_first=True, delivered=(), prepare=False)
    assert "block_energies" not in record
    assert record["status"] == "round_end_without_prepared_round"
    (tmp_path / "gateway_rounds.jsonl").write_text(json.dumps({"round_id": 1, "delivered": {"0": []}}) + "\n")
    (tmp_path / "client_0.json").write_text(json.dumps({"client_weight": 1.0}))
    assert eta_from_run_dir(tmp_path, 1) == {1: None}  # explicit missing value, never zero


def test_b1_and_b2_records_use_the_same_energy_schema(tmp_path) -> None:
    b1 = _run(tmp_path, round_end_first=True, delivered=(1,))
    # B2 stores _block_energies(core.round.encoded) under the same key: per block
    # index (not send order), K floats, from the client's own encoded update.
    encoded = encode(_update(1), L_SMALL)
    by_block = baseline_proc._block_energies(encoded)
    assert np.all(by_block[np.asarray(encoded.order)] == np.asarray(encoded.energies_sorted))
    assert b1["block_energies"] == [float(x) for x in by_block]
    assert isinstance(b1["block_energies"], list) and all(isinstance(x, float) for x in b1["block_energies"])


class _FailedConnect:
    """Stands in for B1TcpClient.send_round when the TCP connect times out before D:
    the real sender encodes the update, then finalizes the round with no delivery."""

    async def send_round(self, core, *, gateway_host, round_id, deadline_ns, delta):
        core.start_round(round_id, deadline_ns, delta)
        core.finish_round(round_id, ())
        return B1ClientTransfer(client_id=core.client_id, round_id=round_id, blocks_written=0,
                                application_bytes_written=0, connected=False, connection_error="",
                                close_reason="connection_failed", completed_ns=1, residual_finalized=True)


@pytest.mark.parametrize("round_end_first", [True, False])
def test_b1_failed_connect_still_logs_the_encoded_energies(tmp_path, round_end_first: bool) -> None:
    async def scenario():
        logs = baseline_proc._ClientLogs(tmp_path / "client_0.json", 0)
        client = baseline_proc.PushOrTcpClient(_cfg(), 0, _update, logs)
        baseline_proc._configure_error_feedback(client.core, client.cfg)
        client.b1_sender = _FailedConnect()
        if round_end_first:  # ROUND_END handled after encoding+finalize but before the task records energies

            async def late(*args):
                client.b1_core.start_round(1, 2**62, _update(1))
                client.b1_core.finish_round(1, ())
                client._on_round_end(_round_end(1, ()))
            client._send_after_readiness = late
        client._on_round_start(_round_start(1, deadline_ns=2**62), received_ns=0)
        for _ in range(4):
            await asyncio.sleep(0)
        if not round_end_first:
            client._on_round_end(_round_end(1, ()))
        await asyncio.gather(*client._tasks, return_exceptions=True)
        for timer in client._timers.values():
            timer.cancel()
        logs.close()

    asyncio.run(scenario())
    (record,) = [json.loads(line) for line in (tmp_path / "client_0_rounds.jsonl").read_text().splitlines()]
    assert record["delivered"] == []
    assert record["block_energies"] == _expected_energies(1)
    (tmp_path / "gateway_rounds.jsonl").write_text(json.dumps({"round_id": 1, "delivered": {"0": []}}) + "\n")
    (tmp_path / "client_0.json").write_text(json.dumps({"client_weight": 1.0}))
    assert eta_from_run_dir(tmp_path, 1) == {1: 0.0}  # a real zero, not missing
