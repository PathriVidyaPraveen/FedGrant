"""Bounded localhost integration of the UDP client and gateway."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from fedgrant.common.constants import TransferMode
from fedgrant.core.client_core import ClientCore
from fedgrant.core.gateway import GatewayCore
from fedgrant.core.state import ClientState
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


def test_localhost_udp_round_from_round_start_through_error_feedback() -> None:
    async def scenario() -> None:
        delta = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        gateway_core = GatewayCore(
            {1: ClientState(1, 1.0, K=2)},
            selector=Selector(),
            eligibility=Eligibility(),
            pacing=Pacing(),
            prediction=Prediction(),
            timeout=Timeout(),
        )
        try:
            gateway = await UdpGateway.bind(
                gateway_core,
                local_addr=("127.0.0.1", 0),
                tick_interval_ns=500_000,
            )
        except PermissionError:
            pytest.skip("the environment does not permit localhost UDP sockets")

        client: UdpClient | None = None
        try:
            assert gateway.transport is not None
            gateway_addr = gateway.transport.get_extra_info("sockname")[:2]
            try:
                client = await UdpClient.bind(
                    ClientCore(1, d=3, L=2),
                    gateway_addr=(gateway_addr[0], gateway_addr[1]),
                    local_addr=("127.0.0.1", 0),
                    update_provider=lambda round_id: delta,
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
            assert gateway_core.phase.name == "DEADLINE"
            assert gateway_core.clients[1].m == 2
            assert gateway_core.delivered_blocks[1] == {0, 1}
            assert client.last_reconstruction is not None
            expected = delta.astype(np.float16).astype(np.float32)
            np.testing.assert_array_equal(client.last_reconstruction, expected)
            np.testing.assert_array_equal(
                client.core.error_feedback.e,
                delta - expected,
            )
            assert client.stats.sent == 3  # READY and two grant-directed DATA packets.
            assert client.stats.round_end_timeouts == 0
        finally:
            if client is not None:
                await client.close()
            await gateway.close()

    asyncio.run(scenario())
