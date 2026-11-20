"""End-to-end FL over the network (experiment M2): FL server service and client trainer.

One M2 round r, for the grant gateway (FG-GRU) and for B3 alike:

1. The FL server sends MODEL r (TCP, ``fedgrant.net.fl_tcp``) to every client
   during the compute window Tc before ROUND_START r.
2. Each client trains its local update Delta_r on its own data in a background
   thread as soon as MODEL r arrives (:class:`TrainedUpdates`), so ROUND_START r
   only looks the update up. Error feedback is on (an FL run, design decision D-19).
3. ROUND_START r -> READY/GRANT/DATA (or READY/PLAN/DATA for B3), as in M1.
4. At D the gateway forms G_r = sum_i w_i u_hat_i and sends it with its per-client
   coordinate counts to the server (``TcpAggregateSender``).
5. The server applies theta <- theta + G_r, exactly step 5 of
   ``fedgrant.fl.server.run_fl_simulated``, then sends MODEL r+1.

Failure handling. If the aggregate of round r never reaches the server, no
MODEL r+1 is sent, every client skips round r+1 (no update, so no READY), and
the gateway's round r+1 aggregate (zeros) arrives while round r is expected. The
server then records rounds r..r'-1 as ``aggregate_missing`` (model unchanged),
applies round r' and continues, so one lost transfer does not stall the run.

The trainer and the evaluator are injected. :func:`make_femnist_trainer` and
:func:`make_femnist_evaluator` give the real FEMNIST CNN path; they need the
FEMNIST shards on the host and PyTorch. Tests use a small synthetic trainer;
such runs are transport tests, never M2 results (Section 11; D-16).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from fedgrant.net.fl_tcp import (
    AggregateEnvelope,
    ModelMailbox,
    TcpAggregateReceiver,
    TcpModelReceiver,
    TransferError,
    send_model,
)

log = logging.getLogger(__name__)

AGGREGATE_PORT = 9200    # FL server (srv): aggregates from the gateway
MODEL_PORT = 9201        # every client: models from the FL server

Trainer = Callable[[int, np.ndarray], np.ndarray]           # (round_id, theta_r) -> Delta_r
Evaluator = Callable[[np.ndarray], tuple[float, float]]      # theta -> (loss, accuracy)


class ModelUnavailable(ValueError):
    """No model was received for the round, so this client has no update to send.

    A ValueError, so both the grant client (``UdpClient``) and the B3 client
    ignore that ROUND_START and the client simply sits the round out.
    """


def _as_vector(name: str, value: np.ndarray, d: int | None = None) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.ndim != 1:
        raise TransferError(f"{name} must be a 1-D float32 vector")
    if d is not None and value.shape != (d,):
        raise TransferError(f"{name} has shape {value.shape}, expected ({d},)")
    if not np.isfinite(value).all():
        raise TransferError(f"{name} contains non-finite values")
    return value


# --------------------------------------------------------------------- server


class FLServer:
    """Holds the global model; applies each round's aggregate and sends the next model.

    ``theta`` starts at ``theta0`` (``create_model(seed)`` in a real run, as in the
    simulated FL path). ``model_version`` counts the aggregates applied.
    """

    def __init__(
        self,
        theta0: np.ndarray,
        *,
        rounds: int,
        client_endpoints: Mapping[int, tuple[str, int]],
        evaluator: Evaluator | None = None,
        eval_every: int = 5,
        history_path: str | Path | None = None,
        first_broadcast_timeout_s: float = 30.0,
        broadcast_timeout_s: float = 10.0,
        retry_interval_s: float = 0.05,
        transfer_timeout_s: float = 10.0,
    ) -> None:
        self.theta = _as_vector("theta0", np.array(theta0, dtype=np.float32, copy=True))
        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        if not client_endpoints:
            raise ValueError("at least one client endpoint is required")
        if eval_every < 1:
            raise ValueError("eval_every must be >= 1")
        self.rounds = rounds
        self.client_endpoints = {int(c): (str(h), int(p)) for c, (h, p) in client_endpoints.items()}
        self.evaluator = evaluator
        self.eval_every = int(eval_every)
        self.history_path = Path(history_path) if history_path is not None else None
        self.first_broadcast_timeout_s = float(first_broadcast_timeout_s)
        self.broadcast_timeout_s = float(broadcast_timeout_s)
        self.retry_interval_s = float(retry_interval_s)
        self.transfer_timeout_s = float(transfer_timeout_s)
        self.model_version = 0
        self.next_round = 1                 # the round whose aggregate is expected next
        self.history: list[dict[str, Any]] = []
        self.broadcasts: dict[int, dict[int, bool]] = {}   # round -> client -> model delivered
        self.done = asyncio.Event()
        self.address: tuple[str, int] | None = None   # aggregate endpoint once listening
        self._tasks: set[asyncio.Task] = set()
        if self.history_path is not None:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_path.write_text("", encoding="utf-8")

    # -- model broadcast ------------------------------------------------------
    async def broadcast(self, round_id: int) -> dict[int, bool]:
        """Send MODEL round_id to every client, retrying each until its timeout."""
        version, theta = self.model_version, self.theta.copy()
        timeout_s = self.first_broadcast_timeout_s if round_id == 1 else self.broadcast_timeout_s
        results = await asyncio.gather(*(
            self._send_with_retry(cid, endpoint, round_id, version, theta, timeout_s)
            for cid, endpoint in sorted(self.client_endpoints.items())))
        delivered = dict(zip(sorted(self.client_endpoints), results, strict=True))
        self.broadcasts[round_id] = delivered
        missing = [cid for cid, ok in delivered.items() if not ok]
        if missing:
            log.error("MODEL %d not delivered to clients %s", round_id, missing)
        return delivered

    async def _send_with_retry(self, client_id: int, endpoint: tuple[str, int], round_id: int,
                               version: int, theta: np.ndarray, timeout_s: float) -> bool:
        give_up = time.monotonic() + timeout_s
        while True:
            try:
                await send_model(endpoint[0], endpoint[1], round_id=round_id, model_version=version,
                                 model=theta, timeout=self.transfer_timeout_s)
                return True
            except (TransferError, OSError) as exc:
                if time.monotonic() + self.retry_interval_s >= give_up:
                    log.warning("MODEL %d to client %d failed: %s", round_id, client_id, exc)
                    return False
                await asyncio.sleep(self.retry_interval_s)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("FL server task failed", exc_info=task.exception())

    # -- aggregates (TcpAggregateReceiver callback) ---------------------------
    async def receive_aggregate(self, envelope: AggregateEnvelope) -> None:
        """theta <- theta + G_r (fl/server.py step 5), then send MODEL r+1.

        Kept fast: the ACK to the gateway is not held up by evaluation or by the
        next broadcast, which run as background tasks.
        """
        round_id = int(envelope.round_id)
        aggregate = _as_vector("aggregate", envelope.aggregate, self.theta.size)
        if round_id < self.next_round:
            raise TransferError(f"stale aggregate for round {round_id}; expecting {self.next_round}")
        if round_id > self.rounds:
            raise TransferError(f"aggregate for round {round_id} is past the last round {self.rounds}")
        for skipped in range(self.next_round, round_id):
            self._record({"round_id": skipped, "status": "aggregate_missing",
                          "model_version": self.model_version})
        self.theta = (self.theta + aggregate).astype(np.float32)
        self.model_version += 1
        record: dict[str, Any] = {
            "round_id": round_id, "status": "applied", "model_version": self.model_version,
            "clients": len(envelope.client_n_coords),
            "delivered_blocks": int(sum(len(blocks) for blocks in envelope.client_n_coords.values())),
            "delivered_coords": int(sum(sum(blocks.values()) for blocks in envelope.client_n_coords.values())),
            "aggregate_l2": float(np.linalg.norm(aggregate.astype(np.float64))),
            "models_sent": (None if round_id not in self.broadcasts
                            else sum(self.broadcasts[round_id].values())),
        }
        self.next_round = round_id + 1
        if self.next_round <= self.rounds:
            self._spawn(self.broadcast(self.next_round))
        evaluate = self.evaluator is not None and (round_id % self.eval_every == 0 or round_id == self.rounds)
        if evaluate:
            self._spawn(self._evaluate_and_record(record, self.theta.copy()))
        else:
            self._record(record)
            if self.next_round > self.rounds:
                self.done.set()

    async def _evaluate_and_record(self, record: dict[str, Any], theta: np.ndarray) -> None:
        try:
            loss, accuracy = await asyncio.to_thread(self.evaluator, theta)
            record.update(test_loss=float(loss), test_accuracy=float(accuracy))
        except Exception as exc:  # noqa: BLE001 - an evaluation failure must not stop training
            log.error("evaluation after round %d failed: %s", record["round_id"], exc)
            record["eval_error"] = repr(exc)[:200]
        self._record(record)
        if record["round_id"] >= self.rounds:
            self.done.set()

    def _record(self, record: dict[str, Any]) -> None:
        self.history.append(record)
        if self.history_path is not None:
            with self.history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")

    async def drain(self) -> None:
        """Wait for pending broadcasts and evaluations."""
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)


async def serve_fl_server(server: FLServer, *, host: str = "0.0.0.0", port: int = AGGREGATE_PORT,
                          idle_timeout_s: float = 120.0) -> FLServer:
    """Accept aggregates, send MODEL 1, and run until the last round or an idle timeout.

    The idle timeout (no aggregate for that long) ends a run whose gateway has
    stopped; the history then shows how far it got.
    """
    receiver = await TcpAggregateReceiver(server).start(host, port)
    server.address = receiver.address
    try:
        server._spawn(server.broadcast(1))
        last_version = -1
        while not server.done.is_set():
            try:
                await asyncio.wait_for(server.done.wait(), idle_timeout_s)
            except asyncio.TimeoutError:
                if server.model_version == last_version:
                    log.error("no aggregate for %.0f s after round %d; stopping", idle_timeout_s,
                              server.next_round - 1)
                    break
            last_version = server.model_version
        await server.drain()
    finally:
        await receiver.close()
    return server


# --------------------------------------------------------------------- client


class _NotifyingMailbox(ModelMailbox):
    def __init__(self, expected_shape, on_install: Callable[[int, int, np.ndarray], None]) -> None:
        super().__init__(expected_shape)
        self._on_install = on_install

    def install(self, round_id: int, model_version: int, model: np.ndarray) -> None:
        current = self._current
        super().install(round_id, model_version, model)
        if current is None or current[0] != round_id:   # a repeat of the same model is ignored
            self._on_install(round_id, model_version, model)


class TrainedUpdates:
    """Update provider for M2 clients: Delta_r trained on MODEL r in the background.

    Training starts when MODEL r arrives (during Tc), so ROUND_START r normally
    finds the update ready. If training is still running, the call waits for it
    (at most ``wait_s``); READY then leaves when it finishes, never before
    (D-20). With no MODEL r the client sits the round out (:class:`ModelUnavailable`).
    """

    trains_in_background = True  # callers must not prefetch: training starts when the model arrives

    def __init__(self, trainer: Trainer, *, d: int, wait_s: float = 30.0) -> None:
        if not callable(trainer):
            raise TypeError("trainer must be callable")
        self.trainer = trainer
        self.d = int(d)
        self.wait_s = float(wait_s)
        self.mailbox = _NotifyingMailbox((self.d,), self._on_model)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="fl-train")
        self._futures: dict[int, concurrent.futures.Future] = {}
        self.models_received: list[tuple[int, int]] = []    # (round_id, model_version)
        self.missing_models: list[int] = []

    def _on_model(self, round_id: int, model_version: int, model: np.ndarray) -> None:
        self.models_received.append((int(round_id), int(model_version)))
        for old in [r for r in self._futures if r < round_id]:
            self._futures.pop(old).cancel()
        self._futures[round_id] = self._executor.submit(self._train, round_id, model.copy())

    def _train(self, round_id: int, theta: np.ndarray) -> np.ndarray:
        delta = self.trainer(round_id, theta)
        return _as_vector("trained update", np.asarray(delta), self.d)

    def __call__(self, round_id: int) -> np.ndarray:
        future = self._futures.pop(round_id, None)
        if future is None:
            self.missing_models.append(int(round_id))
            raise ModelUnavailable(f"no model received for round {round_id}")
        try:
            return future.result(timeout=self.wait_s).copy()
        except concurrent.futures.TimeoutError as exc:
            self.missing_models.append(int(round_id))
            raise ModelUnavailable(f"local training for round {round_id} did not finish") from exc

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


async def start_model_receiver(updates: TrainedUpdates, *, host: str = "0.0.0.0",
                               port: int = MODEL_PORT) -> TcpModelReceiver:
    return await TcpModelReceiver(updates.mailbox).start(host, port)


# --------------------------------------------------- FEMNIST trainer / evaluator


def femnist_initial_model(seed: int) -> np.ndarray:
    """theta_0 as in the simulated FL path (``create_model(seed)``)."""
    from fedgrant.fl.model import create_model, get_flat_params

    return get_flat_params(create_model(seed)).astype(np.float32)


def make_femnist_trainer(client_id: int, *, seed: int, data_root: str, shard_root: str,
                         device: str = "cpu", torch_threads: int = 1) -> Trainer:
    """Client i's local training exactly as ``fl/server.py`` does it (20 SGD steps, lr 0.05).

    Uses the client's own FEMNIST shard and its persistent, once-seeded batch
    loader (``client_loader_seed(seed, i)``), so client i draws the same batches
    as in the simulated FL run with the same seed.
    """
    import torch

    from fedgrant.fl.femnist import load_femnist
    from fedgrant.fl.local import CyclicBatchLoader, client_loader_seed, train_client
    from fedgrant.fl.model import create_model, set_flat_params

    torch.set_num_threads(int(torch_threads))
    dataset = load_femnist(data_root, shard_root, 200, 10)
    images, labels = dataset.clients[client_id].load_train()
    loader = CyclicBatchLoader(images, labels, batch_size=32, seed=client_loader_seed(seed, client_id))
    model = create_model(seed)

    def train(round_id: int, theta: np.ndarray) -> np.ndarray:
        set_flat_params(model, theta)
        return train_client(client_id, images, labels, model, round_idx=round_id - 1,
                            master_seed=seed, loader=loader, device=device)

    return train


def make_femnist_evaluator(*, data_root: str, shard_root: str, device: str = "cpu") -> Evaluator:
    """Test loss and accuracy over every client's test shard (``fl/server.py::evaluate``)."""
    from fedgrant.fl.femnist import load_femnist
    from fedgrant.fl.model import create_model, set_flat_params
    from fedgrant.fl.server import evaluate

    dataset = load_femnist(data_root, shard_root, 200, 10)
    test_clients = [client.load_test() for client in dataset.clients]
    model = create_model(0)

    def evaluator(theta: np.ndarray) -> tuple[float, float]:
        set_flat_params(model, theta)
        return evaluate(model, test_clients, device=device)

    return evaluator


def femnist_client_weights(*, data_root: str, shard_root: str) -> np.ndarray:
    from fedgrant.fl.femnist import load_femnist

    return np.asarray(load_femnist(data_root, shard_root, 200, 10).client_weights, dtype=np.float32)


def main(argv: list[str] | None = None) -> None:
    """FL server process for the ``srv`` host (M2)."""
    import argparse

    parser = argparse.ArgumentParser(description="FedGrant M2 FL server (model broadcast, aggregate apply)")
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=AGGREGATE_PORT, help="aggregates from the gateway")
    parser.add_argument("--client-host-template", default="10.0.0.{i}",
                        help="client i's address; {i} is client_id + 1 (Mininet hosts 10.0.0.1..N)")
    parser.add_argument("--model-port", type=int, default=MODEL_PORT, help="each client's model port")
    parser.add_argument("--history", required=True, help="per-round JSONL history")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--no-eval", action="store_true", help="skip test-set evaluation")
    parser.add_argument("--data-root", default="data/femnist/reproduced/leaf/data")
    parser.add_argument("--shard-root", default="data/femnist/reproduced/clients")
    parser.add_argument("--initial-model", default=None,
                        help="module:function(seed) -> theta_0 (tests); default: the FEMNIST CNN")
    parser.add_argument("--first-broadcast-timeout-s", type=float, default=30.0)
    parser.add_argument("--idle-timeout-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    if args.clients < 1 or args.rounds < 1:
        parser.error("clients and rounds must be positive")

    if args.initial_model:
        from fedgrant.mininet.client_proc import load_factory

        theta0 = np.asarray(load_factory(args.initial_model)(args.seed), dtype=np.float32)
    else:
        theta0 = femnist_initial_model(args.seed)
    evaluator = None if args.no_eval else make_femnist_evaluator(data_root=args.data_root,
                                                                 shard_root=args.shard_root)
    endpoints = {cid: (args.client_host_template.format(i=cid + 1), args.model_port)
                 for cid in range(args.clients)}
    server = FLServer(theta0, rounds=args.rounds, client_endpoints=endpoints, evaluator=evaluator,
                      eval_every=args.eval_every, history_path=args.history,
                      first_broadcast_timeout_s=args.first_broadcast_timeout_s)
    asyncio.run(serve_fl_server(server, host=args.host, port=args.port, idle_timeout_s=args.idle_timeout_s))


__all__ = [
    "AGGREGATE_PORT",
    "FLServer",
    "MODEL_PORT",
    "ModelUnavailable",
    "TrainedUpdates",
    "femnist_client_weights",
    "femnist_initial_model",
    "make_femnist_evaluator",
    "make_femnist_trainer",
    "serve_fl_server",
    "start_model_receiver",
]


if __name__ == "__main__":
    main()
