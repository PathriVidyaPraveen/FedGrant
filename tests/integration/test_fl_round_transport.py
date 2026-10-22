from __future__ import annotations

import asyncio

import numpy as np
import pytest

from fedgrant.common.constants import TransferMode
from fedgrant.core.client_core import ClientCore
from fedgrant.core.gateway import GatewayCore
from fedgrant.core.state import ClientState
from fedgrant.net.fl_tcp import (
    AggregateEnvelope,
    ModelBackedUpdateProvider,
    ModelMailbox,
    TcpAggregateReceiver,
    TcpAggregateSender,
    TcpModelReceiver,
    broadcast_model,
)
from fedgrant.net.udp_client import UdpClient
from fedgrant.net.udp_gateway import UdpGateway
from fedgrant.policy.types import CandidateView


class Eligibility:
    def candidates(self, clients, now_ns, deadline_ns, retransmit):
        ids = [
            client_id
            for client_id, state in sorted(clients.items())
            if state.ready and not state.outstanding and state.has_pending(retransmit)
        ]
        return CandidateView(
            np.asarray(ids, dtype=np.int32),
            np.asarray(
                [clients[cid].val[clients[cid].next_pos(retransmit) - 1] for cid in ids],
                dtype=np.float64,
            ),
            np.asarray([clients[cid].K - clients[cid].m for cid in ids], dtype=np.int32),
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
        return 1.0


class Timeout:
    def timeout_ns(self, client_id, issue_ns):
        return issue_ns + 100_000_000


class ModelSource:
    def current_model(self) -> tuple[int, np.ndarray]:
        return 3, np.asarray([1.0, 2.0, 3.0], dtype=np.float32)


class TestAggregateProvider:
    def aggregate(self, clients) -> np.ndarray:
        state = clients[1]
        assert state.delivered_n_coords == {0: 2, 1: 1}
        assert set(state.delivered_payloads) == {0, 1}
        return np.asarray([1.0, 2.0, 3.0], dtype=np.float32)


def test_full_round_broadcast_udp_and_aggregate_transfer_loopback() -> None:
    async def scenario() -> None:
        mailbox = ModelMailbox(expected_shape=(3,))
        model_receiver = await TcpModelReceiver(mailbox).start()
        accepted_aggregate: list[AggregateEnvelope] = []
        aggregate_received = asyncio.Event()

        async def accept_aggregate(envelope: AggregateEnvelope) -> None:
            accepted_aggregate.append(envelope)
            aggregate_received.set()

        aggregate_receiver = await TcpAggregateReceiver(accept_aggregate).start()
        gateway_core = GatewayCore(
            {1: ClientState(1, 1.0, K=2)},
            selector=Selector(),
            eligibility=Eligibility(),
            pacing=Pacing(),
            prediction=Prediction(),
            timeout=Timeout(),
            aggregate=TestAggregateProvider(),
        )
        gateway: UdpGateway | None = None
        client: UdpClient | None = None
        try:
            version = await broadcast_model(
                ModelSource(),
                [model_receiver.address],
                round_id=1,
                timeout=1.0,
            )
            assert version == 3
            observed_model: list[np.ndarray] = []
            update_provider = ModelBackedUpdateProvider(
                mailbox,
                lambda round_id, model: observed_model.append(model.copy()) or model.copy(),
            )

            try:
                gateway = await UdpGateway.bind(
                    gateway_core,
                    local_addr=("127.0.0.1", 0),
                    tick_interval_ns=500_000,
                    aggregate_sender=TcpAggregateSender(*aggregate_receiver.address, timeout=1.0),
                )
            except PermissionError:
                pytest.skip("the environment does not permit localhost UDP sockets")
            assert gateway.transport is not None
            gateway_addr = gateway.transport.get_extra_info("sockname")[:2]
            try:
                client = await UdpClient.bind(
                    ClientCore(1, d=3, L=2),
                    gateway_addr=(gateway_addr[0], gateway_addr[1]),
                    local_addr=("127.0.0.1", 0),
                    update_provider=update_provider,
                )
            except PermissionError:
                pytest.skip("the environment does not permit localhost UDP sockets")

            assert client.transport is not None
            client_addr = client.transport.get_extra_info("sockname")[:2]
            gateway.set_endpoint(1, (client_addr[0], client_addr[1]))
            gateway.start_round(
                round_id=1,
                duration_ns=60_000_000,
                K=2,
                L=2,
                rho_us=1_000,
                mode=TransferMode.GRANT,
            )
            assert await client.wait_for_round_end(timeout=1.0) == 1
            await asyncio.wait_for(aggregate_received.wait(), timeout=1.0)

            assert len(observed_model) == 1
            np.testing.assert_array_equal(observed_model[0], [1.0, 2.0, 3.0])
            assert gateway_core.clients[1].m == 2
            assert client.core.error_feedback.e.tolist() == [0.0, 0.0, 0.0]
            assert len(accepted_aggregate) == 1
            received = accepted_aggregate[0]
            assert received.round_id == 1
            assert received.client_ids == (1,)
            assert received.client_n_coords == {1: {0: 2, 1: 1}}
            np.testing.assert_array_equal(received.aggregate, [1.0, 2.0, 3.0])
        finally:
            if client is not None:
                await client.close()
            if gateway is not None:
                await gateway.close()
            await model_receiver.close()
            await aggregate_receiver.close()

    asyncio.run(scenario())
