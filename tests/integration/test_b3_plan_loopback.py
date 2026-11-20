from __future__ import annotations

import asyncio
import time

import numpy as np

from fedgrant.baselines.b2_udp import B2ClientCore, apply_delivery_result
from fedgrant.baselines.b3_plan import B3UdpClient, B3UdpGateway
def test_real_udp_b3_round_start_ready_plan_and_energy_order_data() -> None:
    async def scenario() -> None:
        round_id, d, L = 1, 5, 2
        client_cores = {
            cid: B2ClientCore(cid, d=d, L=L)
            for cid in (3, 8)
        }
        updates = {
            3: np.asarray([1.0, 3.0, 0.0, 0.0, 4.0], dtype=np.float32),
            8: np.asarray([0.0, 0.0, 9.0, 0.0, 0.0], dtype=np.float32),
        }
        gateway = await B3UdpGateway(
            client_weights={3: 0.4, 8: 0.6},
            endpoints={3: ("127.0.0.1", 0), 8: ("127.0.0.1", 0)},
            d=d,
            L=L,
            duration_ns=500_000_000,
            host="127.0.0.1",
            port=0,
        ).bind()
        clients = [
            await B3UdpClient.bind(
                client_cores[cid],
                gateway_addr=gateway.address,
                local_addr=("127.0.0.1", 0),
                update_provider=lambda _received_round, cid=cid: updates[cid].copy(),
            )
            for cid in (3, 8)
        ]
        try:
            for cid, client in zip((3, 8), clients, strict=True):
                gateway.set_endpoint(cid, client.transport.get_extra_info("sockname"))
            core = gateway.start_round(round_id)
            assert await asyncio.gather(
                *(client.wait_for_round_start() for client in clients)
            ) == [round_id, round_id]
            transfers = await asyncio.wait_for(
                asyncio.gather(*(client.wait_for_plan() for client in clients)),
                timeout=2.0,
            )
            result = await asyncio.wait_for(gateway.wait(), timeout=2.0)

            for cid, transfer in zip((3, 8), transfers, strict=True):
                encoded = client_cores[cid].round.encoded
                expected = tuple(int(block) for block in encoded.order[: result.plans[cid].blocks])
                assert transfer.sent_indices == expected
                assert transfer.packets_sent == result.plans[cid].blocks
                assert transfer.round_id == round_id
                assert result.plans[cid].rate_bps == 4_000_000
                interval_ns = (36 + 42) * 8 * 1_000_000_000 // result.plans[cid].rate_bps
                assert all(
                    later - earlier >= interval_ns
                    for earlier, later in zip(
                        transfer.send_times_ns,
                        transfer.send_times_ns[1:],
                    )
                ), transfer.send_times_ns
                assert len(set(transfer.sent_indices)) == transfer.packets_sent
                apply_delivery_result(client_cores[cid], result.round_result)
                assert client_cores[cid].round is None

            assert result.completed_at_deadline
            assert result.round_result.delivered_blocks == sum(
                plan.blocks for plan in result.plans.values()
            )
            assert result.rtt_samples_ns.keys() == result.plans.keys()
            assert all(sample >= 0 for sample in result.rtt_samples_ns.values())
            # Both READYs arrived (rho = 0), so the plan was made at the last one,
            # well before the 0.1 D + 105 ms = 155 ms cap (CHANGELOG 2026-11-21).
            # (the plan is stamped when the gateway processes that READY).
            assert max(core.ready_times_ns.values()) <= core.plan_ns < core.plan_cap_ns
            assert core.plan_cap_ns == core.start_ns + 155_000_000
            assert result.planned_early and result.ready_clients == 2
            assert core.accumulator.rejected_duplicates == 0
        finally:
            for client in clients:
                await client.close()
            await gateway.close()

    asyncio.run(scenario())
