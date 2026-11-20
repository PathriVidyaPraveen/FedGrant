"""M2 FL loop over localhost sockets: model -> local training -> grants -> aggregate -> model.

Real pieces: FLServer behind TcpAggregateReceiver, model broadcast over TCP,
TcpModelReceiver + TrainedUpdates on every client, the production gateway
(create_gateway) on UdpGateway with TcpAggregateSender, UdpClient with error
feedback on, and the B3 baseline processes. The trainer is a small synthetic
function, so these are transport/orchestration tests, not M2 results (D-16).
"""

from __future__ import annotations

import asyncio
import json
import socket

import numpy as np
import pytest

from fedgrant.common.constants import TransferMode
from fedgrant.core.client_core import ClientCore
from fedgrant.mininet.fl_service import FLServer, TrainedUpdates, serve_fl_server, start_model_receiver
from fedgrant.mininet.production import GatewayContext, create_gateway
from fedgrant.net.fl_tcp import TcpAggregateSender
from fedgrant.net.udp_client import UdpClient
from fedgrant.net.udp_gateway import UdpGateway
from fedgrant.policy.constants import D_MODEL, K, L

N = 3
ROUNDS = 3
SEED = 0
MS = 1_000_000


def theta0(_seed: int = SEED) -> np.ndarray:
    return np.linspace(-1.0, 1.0, D_MODEL, dtype=np.float32)


class SyntheticTrainer:
    """Delta_r = 0.1 (target_i - theta_r): a deterministic stand-in for local SGD."""

    def __init__(self, client_id: int) -> None:
        self.client_id = client_id
        self.target = np.full(D_MODEL, float(client_id + 1), dtype=np.float32)
        self.seen: list[tuple[int, np.ndarray]] = []

    def __call__(self, round_id: int, theta: np.ndarray) -> np.ndarray:
        self.seen.append((round_id, theta.copy()))
        return (0.1 * (self.target - theta)).astype(np.float32)


class RecordingSender(TcpAggregateSender):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sent: dict[int, tuple[np.ndarray, dict]] = {}

    async def send_aggregate(self, round_id, aggregate, client_n_coords) -> None:
        self.sent[round_id] = (aggregate.copy(), {c: dict(b) for c, b in client_n_coords.items()})
        await super().send_aggregate(round_id, aggregate, client_n_coords)


async def _until(predicate, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not predicate():
        if loop.time() > end:
            raise TimeoutError("condition not reached")
        await asyncio.sleep(0.005)


def test_grant_gateway_fl_loop_applies_each_aggregate_and_trains_on_each_model(tmp_path) -> None:
    async def scenario():
        trainers = [SyntheticTrainer(cid) for cid in range(N)]
        updates = [TrainedUpdates(trainers[cid], d=D_MODEL) for cid in range(N)]
        receivers = [await start_model_receiver(u, host="127.0.0.1", port=0) for u in updates]
        server = FLServer(theta0(), rounds=ROUNDS, client_endpoints={c: r.address for c, r in enumerate(receivers)},
                          history_path=tmp_path / "fl_history.jsonl")
        server_task = asyncio.create_task(serve_fl_server(server, host="127.0.0.1", port=0))
        await _until(lambda: server.address is not None)
        weights = np.full(N, 1.0 / N, dtype=np.float32)
        core = create_gateway(GatewayContext(num_clients=N, policy="B6", client_weights=weights))
        sender = RecordingSender(*server.address, timeout=5.0)
        gateway = await UdpGateway.bind(core, local_addr=("127.0.0.1", 0), aggregate_sender=sender)
        gw_addr = gateway.transport.get_extra_info("sockname")[:2]
        clients = []
        for cid in range(N):
            client = await UdpClient.bind(ClientCore(cid, error_feedback=True), gateway_addr=gw_addr,
                                          local_addr=("127.0.0.1", 0), update_provider=updates[cid])
            gateway.set_endpoint(cid, client.transport.get_extra_info("sockname")[:2])
            clients.append(client)
        try:
            for round_id in range(1, ROUNDS + 1):
                # Tc: every client has MODEL r and its update trained before ROUND_START r.
                await _until(lambda: all((round_id, round_id - 1) in u.models_received for u in updates))
                gateway.start_round(round_id=round_id, duration_ns=400 * MS, K=K, L=L,
                                    rho_us={cid: 1_000 * (cid + 1) for cid in range(N)}, mode=TransferMode.GRANT)
                await asyncio.gather(*(c.wait_for_round_end(timeout=5.0) for c in clients))
                for c in clients:
                    c._round_end_event.clear()
                await _until(lambda: server.next_round == round_id + 1)
            await asyncio.wait_for(server_task, 10.0)
        finally:
            for c in clients:
                await c.close()
            await gateway.close()
            for r in receivers:
                await r.close()
            for u in updates:
                u.close()
            if not server_task.done():
                server_task.cancel()
        return server, sender, trainers, updates, clients

    server, sender, trainers, updates, clients = asyncio.run(scenario())

    # Every round was applied in order; the server sent MODEL 2.. after each aggregate.
    assert [(r["round_id"], r["status"]) for r in server.history] == [(r, "applied") for r in range(1, ROUNDS + 1)]
    assert server.model_version == ROUNDS and server.done.is_set()
    assert all(server.broadcasts[r] == {0: True, 1: True, 2: True} for r in range(1, ROUNDS + 1))
    # theta_R = theta_0 + sum_r G_r with exactly the aggregates the gateway sent.
    expected = theta0().copy()
    for r in range(1, ROUNDS + 1):
        aggregate, n_coords = sender.sent[r]
        assert aggregate.dtype == np.float32 and aggregate.shape == (D_MODEL,)
        assert server.history[r - 1]["delivered_blocks"] == sum(len(b) for b in n_coords.values())
        expected = (expected + aggregate).astype(np.float32)
    np.testing.assert_array_equal(server.theta, expected)
    # Something was delivered over the run (a single round can be empty when the shared loop is starved).
    assert sum(r["delivered_blocks"] for r in server.history) > 0
    # Each client trained round r on exactly theta_{r-1} (the model after r-1 aggregates).
    thetas = [theta0()]
    for r in range(1, ROUNDS + 1):
        thetas.append((thetas[-1] + sender.sent[r][0]).astype(np.float32))
    for trainer in trainers:
        assert [r for r, _ in trainer.seen] == list(range(1, ROUNDS + 1))
        for r, theta in trainer.seen:
            np.testing.assert_array_equal(theta, thetas[r - 1])
    for u in updates:
        assert u.models_received == [(r, r - 1) for r in range(1, ROUNDS + 1)] and u.missing_models == []
    # Error feedback is on (FL): undelivered blocks leave a residual for the next round.
    assert any(np.any(c.core.error_feedback.e != 0) for c in clients)



def _free_port(kind: int) -> int:
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_b3_processes_fl_loop_send_each_round_aggregate_to_the_server(tmp_path) -> None:
    from fedgrant.mininet import baseline_proc
    from fedgrant.mininet.baseline_proc import BaselineConfig, run_client, run_gateway
    from fedgrant.mininet.round_loop import RoundSchedule
    from fedgrant.mininet.round_state import RoundStateFile

    schedule = RoundSchedule(duration_ns=500 * MS, compute_window_ns=200 * MS, late_drain_ns=80 * MS)
    weights = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)

    async def scenario():
        trainers = [SyntheticTrainer(cid) for cid in range(N)]
        updates = [TrainedUpdates(trainers[cid], d=D_MODEL) for cid in range(N)]
        receivers = [await start_model_receiver(u, host="127.0.0.1", port=0) for u in updates]
        server = FLServer(theta0(), rounds=ROUNDS, client_endpoints={c: r.address for c, r in enumerate(receivers)})
        applied: dict[int, np.ndarray] = {}
        receive = server.receive_aggregate

        async def spy(envelope):
            applied[envelope.round_id] = envelope.aggregate.copy()
            await receive(envelope)

        server.receive_aggregate = spy
        server_task = asyncio.create_task(serve_fl_server(server, host="127.0.0.1", port=0))
        await _until(lambda: server.address is not None)
        cfg = BaselineConfig(
            policy="B3", num_clients=N, rounds=ROUNDS, master_seed=SEED, schedule=schedule,
            client_endpoints={cid: ("127.0.0.1", _free_port(socket.SOCK_DGRAM)) for cid in range(N)},
            gateway_host="127.0.0.1", gateway_bind_host="127.0.0.1",
            udp_port=_free_port(socket.SOCK_DGRAM), tcp_port=_free_port(socket.SOCK_STREAM),
            network_only=False, aggregate_endpoint=server.address)
        run_dir = tmp_path / "b3"
        opened = asyncio.Event()
        original_open = baseline_proc.BaselineGateway.open

        async def open_and_signal(self) -> None:
            await original_open(self)
            opened.set()

        baseline_proc.BaselineGateway.open = open_and_signal
        try:
            gateway = asyncio.create_task(run_gateway(cfg, weights, run_dir, RoundStateFile(run_dir / "rs.json")))
            await asyncio.wait_for(opened.wait(), 10.0)
        finally:
            baseline_proc.BaselineGateway.open = original_open
        clients = [asyncio.create_task(run_client(cfg, cid, updates[cid], run_dir / f"client_{cid}.json",
                                                  round_timeout_s=10.0)) for cid in range(N)]
        try:
            await asyncio.wait_for(gateway, 30.0)
            summaries = await asyncio.wait_for(asyncio.gather(*clients), 30.0)
            await asyncio.wait_for(server_task, 10.0)
        finally:
            for r in receivers:
                await r.close()
            for u in updates:
                u.close()
        return server, applied, trainers, updates, summaries, run_dir

    server, applied, trainers, updates, summaries, run_dir = asyncio.run(scenario())

    assert [(r["round_id"], r["status"]) for r in server.history] == [(r, "applied") for r in range(1, ROUNDS + 1)]
    assert all(s["completed_rounds"] == list(range(1, ROUNDS + 1)) for s in summaries)
    thetas = [theta0()]
    for r in range(1, ROUNDS + 1):
        thetas.append((thetas[-1] + applied[r]).astype(np.float32))
    np.testing.assert_array_equal(server.theta, thetas[-1])
    for trainer in trainers:
        assert [r for r, _ in trainer.seen] == list(range(1, ROUNDS + 1))
        for r, theta in trainer.seen:
            np.testing.assert_array_equal(theta, thetas[r - 1])   # trained on the current global model
    # The aggregate is sum_i w_i u_hat_i over exactly the blocks B3 delivered.
    gateway_rounds = [json.loads(line) for line in (run_dir / "gateway_rounds.jsonl").read_text().splitlines()]
    for record, history in zip(gateway_rounds, server.history, strict=True):
        delivered = sum(len(v) for v in record["delivered"].values())
        assert history["delivered_blocks"] == delivered
    assert sum(h["delivered_blocks"] for h in server.history) > 0 and any(np.any(a != 0) for a in applied.values())
