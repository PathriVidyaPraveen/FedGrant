from __future__ import annotations

import asyncio
import time

import numpy as np

from fedgrant.baselines.b2_udp import (
    B2ClientCore,
    B2RoundAccumulator,
    B2UdpClient,
    B2UdpGateway,
    apply_delivery_result,
)


def test_real_udp_b2_loopback_pushes_multiple_clients_and_zero_fills() -> None:
    async def scenario() -> None:
        round_id, d, L = 1, 5, 2
        start_ns = time.monotonic_ns()
        deadline_ns = start_ns + 250_000_000
        cores = {
            client_id: B2ClientCore(client_id, d=d, L=L)
            for client_id in (2, 5)
        }
        updates = {
            2: np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            5: np.asarray([0.0, 0.0, 8.0, 0.0, 0.0], dtype=np.float32),
        }
        accumulator = B2RoundAccumulator(
            round_id=round_id,
            client_ids=(2, 5),
            start_ns=start_ns,
            deadline_ns=deadline_ns,
            d=d,
            L=L,
            client_weights={2: 0.4, 5: 0.6},
        )
        gateway = await B2UdpGateway(
            accumulator,
            host="127.0.0.1",
            port=0,
        ).start()
        clients = [
            await B2UdpClient.bind(
                cores[client_id],
                gateway_addr=gateway.address,
                local_addr=("127.0.0.1", 0),
            )
            for client_id in (2, 5)
        ]
        try:
            transfers = await asyncio.wait_for(
                asyncio.gather(
                    *(
                        client.send_round(
                            round_id=round_id,
                            deadline_ns=deadline_ns,
                            delta=updates[client_id],
                            rate_bps=20_000_000.0,
                        )
                        for client_id, client in zip((2, 5), clients, strict=True)
                    )
                ),
                timeout=1.0,
            )
            result = await asyncio.wait_for(gateway.wait(), timeout=1.0)

            assert result.completed_at_deadline
            assert result.close_reason == "deadline"
            for cid, transfer in zip((2, 5), transfers, strict=True):
                assert transfer.client_id == cid
                assert transfer.sent_indices == tuple(
                    int(idx) for idx in cores[cid].round.encoded.order
                )
                assert transfer.packets_sent == cores[cid].round.encoded.K
                assert result.client_results[cid].receive_order == transfer.sent_indices
                assert result.client_results[cid].delivered_indices == tuple(sorted(transfer.sent_indices))
                assert result.client_results[cid].n_coords == {
                    idx: int(cores[cid].round.encoded.n_coords[idx])
                    for idx in transfer.sent_indices
                }
                apply_delivery_result(cores[cid], result)
                assert cores[cid].round is None

            assert result.delivered_blocks == sum(item.packets_sent for item in transfers)
            assert result.rejected_duplicates == 0
            aggregate = result.aggregate(
                lambda client_updates, weights: sum(
                    (
                        weight * update
                        for weight, update in zip(weights, client_updates, strict=True)
                    ),
                    start=np.zeros(d, dtype=np.float32),
                ).astype(np.float32)
            )
            expected = (
                0.4 * result.client_results[2].decoded_update
                + 0.6 * result.client_results[5].decoded_update
            ).astype(np.float32)
            np.testing.assert_array_equal(aggregate, expected)
        finally:
            for client in clients:
                await client.close()
            await gateway.close()

    asyncio.run(scenario())
