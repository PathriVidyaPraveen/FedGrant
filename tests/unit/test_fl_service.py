"""M2 FL service pieces: server update rule, missing/stale aggregates, client trainer."""

from __future__ import annotations

import asyncio
import json
import threading

import numpy as np
import pytest

from fedgrant.mininet.baseline_proc import block_coords, weighted_sum
from fedgrant.mininet.fl_service import FLServer, ModelUnavailable, TrainedUpdates
from fedgrant.mininet.gateway_proc import aggregate_sender_from_args, parse_endpoint
from fedgrant.net.fl_tcp import AggregateEnvelope, TcpAggregateSender, TransferError

D = 8


def _server(tmp_path, rounds: int = 3, **kwargs) -> FLServer:
    return FLServer(np.arange(D, dtype=np.float32), rounds=rounds, client_endpoints={0: ("127.0.0.1", 1)},
                    history_path=tmp_path / "fl_history.jsonl", **kwargs)


def _env(round_id: int, value: float, clients=None) -> AggregateEnvelope:
    return AggregateEnvelope(round_id, np.full(D, value, dtype=np.float32),
                             clients if clients is not None else {0: {0: 700, 1: 700}, 1: {}})


def _no_broadcast(server: FLServer) -> list[int]:
    sent: list[int] = []

    async def broadcast(round_id):
        sent.append(round_id)
        return {}

    server.broadcast = broadcast
    return sent


def test_server_applies_theta_plus_aggregate_and_sends_the_next_model(tmp_path) -> None:
    async def scenario():
        server = _server(tmp_path)
        sent = _no_broadcast(server)
        await server.receive_aggregate(_env(1, 0.5))
        await server.drain()
        # theta <- theta + G_r, exactly fl/server.py step 5.
        np.testing.assert_array_equal(server.theta, np.arange(D, dtype=np.float32) + 0.5)
        assert server.theta.dtype == np.float32 and server.model_version == 1 and server.next_round == 2
        assert sent == [2]
        record = server.history[-1]
        assert record["status"] == "applied" and record["round_id"] == 1
        assert (record["clients"], record["delivered_blocks"], record["delivered_coords"]) == (2, 2, 1400)
        assert json.loads((tmp_path / "fl_history.jsonl").read_text().splitlines()[0])["round_id"] == 1
        await server.receive_aggregate(_env(2, 1.0))
        await server.receive_aggregate(_env(3, 1.0))
        await server.drain()
        assert server.done.is_set() and sent == [2, 3]  # no MODEL 4 after the last round
        np.testing.assert_array_equal(server.theta, np.arange(D, dtype=np.float32) + 2.5)

    asyncio.run(scenario())


def test_a_missing_aggregate_is_recorded_and_the_run_continues(tmp_path) -> None:
    async def scenario():
        server = _server(tmp_path, rounds=4)
        _no_broadcast(server)
        await server.receive_aggregate(_env(1, 1.0))
        await server.receive_aggregate(_env(3, 2.0))       # round 2's aggregate never arrived
        await server.drain()
        assert [(r["round_id"], r["status"]) for r in server.history] == [
            (1, "applied"), (2, "aggregate_missing"), (3, "applied")]
        assert server.history[1]["model_version"] == 1   # model unchanged for the missing round
        assert server.model_version == 2 and server.next_round == 4

    asyncio.run(scenario())


@pytest.mark.parametrize(("round_id", "aggregate", "message"), [
    (1, np.full(D, 1.0, np.float32), "stale"),
    (9, np.full(D, 1.0, np.float32), "past the last round"),
    (2, np.full(D + 1, 1.0, np.float32), "shape"),
    (2, np.full(D, np.nan, np.float32), "non-finite"),
    (2, np.full(D, 1.0, np.float64), "float32"),
])
def test_invalid_aggregates_are_rejected_without_changing_the_model(tmp_path, round_id, aggregate, message) -> None:
    async def scenario():
        server = _server(tmp_path)
        _no_broadcast(server)
        await server.receive_aggregate(_env(1, 1.0))
        before = server.theta.copy()
        with pytest.raises(TransferError, match=message):
            await server.receive_aggregate(AggregateEnvelope(round_id, aggregate, {}))
        np.testing.assert_array_equal(server.theta, before)
        assert server.model_version == 1 and server.next_round == 2

    asyncio.run(scenario())


def test_evaluation_runs_after_the_ack_on_its_schedule(tmp_path) -> None:
    async def scenario():
        seen: list[float] = []

        def evaluator(theta):
            seen.append(float(theta[0]))
            return 1.5, 0.25

        server = _server(tmp_path, rounds=4, evaluator=evaluator, eval_every=2)
        _no_broadcast(server)
        for r in range(1, 5):
            await server.receive_aggregate(_env(r, 1.0))
        await server.drain()
        by_round = {rec["round_id"]: rec for rec in server.history}
        assert "test_accuracy" not in by_round[1] and "test_accuracy" not in by_round[3]
        assert by_round[2]["test_accuracy"] == 0.25 and by_round[4]["test_loss"] == 1.5
        assert seen == [2.0, 4.0]   # evaluated on theta after rounds 2 and 4
        assert server.done.is_set()

    asyncio.run(scenario())


def test_trained_updates_train_on_the_received_model_once() -> None:
    calls: list[tuple[int, float, str]] = []

    def trainer(round_id, theta):
        calls.append((round_id, float(theta[0]), threading.current_thread().name))
        return (theta * 0 + round_id).astype(np.float32)

    updates = TrainedUpdates(trainer, d=D)
    try:
        with pytest.raises(ModelUnavailable):
            updates(1)                               # no model yet: sit the round out
        assert isinstance(ModelUnavailable("x"), ValueError)  # both clients ignore that ROUND_START
        theta = np.full(D, 7.0, np.float32)
        updates.mailbox.install(1, 0, theta)
        updates.mailbox.install(1, 0, theta)        # a resent identical model is not retrained
        np.testing.assert_array_equal(updates(1), np.full(D, 1.0, np.float32))
        assert [c[:2] for c in calls] == [(1, 7.0)] and calls[0][2].startswith("fl-train")
        assert updates.models_received == [(1, 0)] and updates.missing_models == [1]
        with pytest.raises(ModelUnavailable):
            updates(1)                               # each round's update is used once
    finally:
        updates.close()


def test_trained_updates_reject_a_bad_update_shape() -> None:
    updates = TrainedUpdates(lambda r, theta: np.zeros(D + 1, np.float32), d=D)
    try:
        updates.mailbox.install(1, 0, np.zeros(D, np.float32))
        with pytest.raises(TransferError, match="shape"):
            updates(1)
    finally:
        updates.close()


def test_b3_aggregate_helpers_match_the_fl_path() -> None:
    a, b = np.array([1.0, 2.0], np.float32), np.array([3.0, -1.0], np.float32)
    np.testing.assert_allclose(weighted_sum([a, b], np.array([0.25, 0.75], np.float32)), [2.5, -0.25])
    assert weighted_sum([a, b], np.array([0.25, 0.75])).dtype == np.float32
    # d = 117,694 = 168 * 700 + 94: the last block carries 94 real coordinates.
    assert [block_coords(117_694, 700, j) for j in (0, 167, 168)] == [700, 700, 94]


def test_gateway_aggregate_endpoint_option(tmp_path) -> None:
    from argparse import Namespace

    assert parse_endpoint("10.0.0.253:9200") == ("10.0.0.253", 9200)
    for bad in ("10.0.0.253", ":9200", "h:0", "h:x"):
        with pytest.raises(ValueError):
            parse_endpoint(bad)
    sender = aggregate_sender_from_args(Namespace(aggregate_endpoint="127.0.0.1:9200", aggregate_timeout_s=3.0,
                                                  log_dir=str(tmp_path)))
    assert isinstance(sender, TcpAggregateSender) and (sender.host, sender.port, sender.timeout) == (
        "127.0.0.1", 9200, 3.0)
    sink = aggregate_sender_from_args(Namespace(aggregate_endpoint=None, log_dir=str(tmp_path)))
    assert type(sink).__name__ == "NetworkOnlyAggregateSink"   # M1/smoke default unchanged
