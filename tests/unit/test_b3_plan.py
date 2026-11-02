from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest

from fedgrant.baselines.b2_udp import (
    B2ClientCore,
    apply_delivery_result,
    encode_data_packet,
)
from fedgrant.baselines.b3_plan import (
    B3ClientPlan,
    B3GatewayCore,
    B3PlanError,
    B3UdpClient,
    B3UdpGateway,
)
from fedgrant.baselines.b3_pacing import (
    data_frame_size_bytes,
    paced_blocks_sent,
    pacing_send_time_ns,
)
from fedgrant.common.constants import MessageType
from fedgrant.common.wire import decode_message, encode_message
from fedgrant.core.estimators import rtt_update
from fedgrant.policy.plan import plan_budget, plan_caps, plan_time_ns


def _ready(
    client_id: int,
    energies: tuple[float, ...] = (9.0, 4.0, 1.0),
    *,
    round_id: int = 1,
    t_send_ns: int = 0,
) -> bytes:
    return encode_message(
        MessageType.READY,
        round_id=round_id,
        client_id=client_id,
        t_send_ns=t_send_ns,
        body={"K": len(energies), "energy": energies},
    )


def _core(
    *,
    client_ids: tuple[int, ...] = (1, 2, 3),
    duration_ns: int = 398_000_000,
    bottleneck_bps: int = 100_000,
    delivery_ratio_init: float = 0.1,
) -> B3GatewayCore:
    weights = {cid: 1.0 / len(client_ids) for cid in client_ids}
    return B3GatewayCore(
        round_id=1,
        start_ns=1_000_000_000,
        deadline_ns=1_000_000_000 + duration_ns,
        d=6,
        L=2,
        client_weights=weights,
        bottleneck_bps=bottleneck_bps,
        delivery_ratio_init=delivery_ratio_init,
    )


class _FakeTransport:
    def __init__(self) -> None:
        self.packets: list[tuple[bytes, tuple[str, int]]] = []

    def sendto(self, packet: bytes, endpoint: tuple[str, int]) -> None:
        self.packets.append((packet, endpoint))

    def close(self) -> None:
        pass


class _FakeClock:
    def __init__(self, now_ns: int) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns

    async def sleep(self, seconds: float) -> None:
        self.now_ns += round(seconds * 1_000_000_000)


def test_plan_wire_round_trip_uses_existing_message_definition() -> None:
    packet = encode_message(
        MessageType.PLAN,
        round_id=8,
        client_id=13,
        t_send_ns=1234,
        body={"k": 7, "rate_bps": 4_000_000},
    )
    message = decode_message(packet)
    assert message.message_type is MessageType.PLAN
    assert (message.round_id, message.client_id, message.t_send_ns) == (8, 13, 1234)
    assert message.body == {"k": 7, "rate_bps": 4_000_000}


def test_t_plan_ready_cutoff_median_rtt_budget_caps_and_greedy_allocation() -> None:
    core = _core()
    assert core.plan_ns == core.start_ns + plan_time_ns(core.duration_ns)
    core.states[1].srtt_ns = 100_000_000
    core.states[2].srtt_ns = 200_000_000
    core.states[3].srtt_ns = 300_000_000

    core.accept_ready(_ready(1, (9.0, 2.0, 1.0)), core.plan_ns - 1)
    core.accept_ready(_ready(2, (8.0, 7.0, 1.0)), core.plan_ns)
    with pytest.raises(B3PlanError, match="outside"):
        core.accept_ready(_ready(3), core.plan_ns + 1)

    plans = core.compute_plans(core.plan_ns)
    median_rtt = 150_000_000
    budget = plan_budget(
        100_000,
        core.duration_ns,
        core.plan_ns - core.start_ns,
        median_rtt,
    )
    caps = {
        cid: plan_caps(
            0.1,
            4_000_000.0,
            core.duration_ns,
            core.plan_ns - core.start_ns,
            core.states[cid].srtt_ns,
        )
        for cid in (1, 2)
    }
    assert core.median_rtt_ns == median_rtt
    assert core.budget == budget == 1
    assert core.caps == caps
    assert plans[1].blocks == 1
    assert plans[2].blocks == 0
    assert plans[3].blocks == 0
    assert sum(plan.blocks for plan in plans.values()) <= budget
    assert all(plans[cid].blocks <= caps[cid] for cid in caps)
    assert all(plan.rate_bps == 4_000_000 for plan in plans.values())


def test_no_ready_clients_get_zero_plan_and_missing_ready_is_counted() -> None:
    core = _core(client_ids=(1, 2))
    plans = core.compute_plans(core.plan_ns)
    assert core.budget == 0
    assert core.median_rtt_ns is None
    assert all(plan.blocks == 0 for plan in plans.values())
    with pytest.raises(B3PlanError, match="already been computed"):
        core.compute_plans(core.plan_ns)


def test_rtt_sample_uses_first_data_after_plan_and_existing_rfc_update() -> None:
    core = _core(client_ids=(1,), duration_ns=398_000_000, bottleneck_bps=1_000_000)
    core.accept_ready(_ready(1), core.plan_ns)
    plan = core.compute_plans(core.plan_ns)[1]
    assert plan.blocks == 3
    sent_ns = core.plan_ns + 1_000
    core.mark_plan_sent(1, sent_ns)
    old_state = core.states[1]
    expected_srtt = old_state.srtt_ns
    expected_rttvar = old_state.rttvar_ns
    received_ns = sent_ns + 5_000_000
    packet = encode_data_packet(
        1,
        1,
        0,
        2,
        np.asarray([3.0, 0.0], dtype=np.float16),
        send_pos=1,
        t_send_ns=sent_ns + 10,
    )
    core.accept_data(packet, received_ns)
    assert core.rtt_samples_ns == {1: 5_000_000}
    expected_state = _core(client_ids=(1,), duration_ns=398_000_000).states[1]
    expected_state.srtt_ns = expected_srtt
    expected_state.rttvar_ns = expected_rttvar
    rtt_update(expected_state, 5_000_000)
    assert old_state.srtt_ns == expected_state.srtt_ns
    assert old_state.rttvar_ns == expected_state.rttvar_ns
    second = encode_data_packet(
        1,
        1,
        1,
        2,
        np.asarray([2.0, 0.0], dtype=np.float16),
        send_pos=2,
        t_send_ns=sent_ns + 20,
    )
    core.accept_data(second, received_ns + 100)
    assert core.rtt_samples_ns == {1: 5_000_000}
    core.finish(core.deadline_ns)
    assert old_state.bhat_bps == 4_000_000.0
    with pytest.raises(B3PlanError, match="already sent"):
        core.mark_plan_sent(1, sent_ns + 1)


def test_persistent_rtt_rate_and_delivery_ewma_carry_into_next_round() -> None:
    first = _core(client_ids=(1,), duration_ns=398_000_000, bottleneck_bps=1_000_000)
    first.accept_ready(_ready(1), first.plan_ns)
    plan = first.compute_plans(first.plan_ns)[1]
    sent_ns = first.plan_ns + 1_000
    first.mark_plan_sent(1, sent_ns)
    packet = encode_data_packet(
        1,
        1,
        0,
        2,
        np.asarray([3.0, 0.0], dtype=np.float16),
        send_pos=1,
        t_send_ns=sent_ns,
    )
    first.accept_data(packet, sent_ns + 2_000_000)
    first_result = first.finish(first.deadline_ns)
    expected_srtt = first.states[1].srtt_ns
    expected_bhat = first.states[1].bhat_bps
    expected_ebar = first.delivery_ewma.get(1)

    next_start = first.deadline_ns + 1
    second = B3GatewayCore(
        round_id=2,
        start_ns=next_start,
        deadline_ns=next_start + first.duration_ns,
        d=6,
        L=2,
        client_weights={1: 1.0},
        bottleneck_bps=1_000_000,
        client_states=first.states,
        delivery_ewma=first.delivery_ewma,
    )
    assert second.states[1] is first.states[1]
    assert second.states[1].srtt_ns == expected_srtt
    assert second.states[1].bhat_bps == expected_bhat
    assert second.delivery_ewma.get(1) == expected_ebar
    assert first_result.rtt_samples_ns[1] == 2_000_000


def test_data_before_plan_wrong_send_position_and_after_deadline_are_rejected() -> None:
    core = _core(client_ids=(1,))
    packet = encode_data_packet(
        1, 1, 0, 2, np.asarray([1.0, 0.0], dtype=np.float16), send_pos=1
    )
    with pytest.raises(B3PlanError, match="before PLAN"):
        core.accept_data(packet, core.plan_ns)
    core.accept_ready(_ready(1), core.plan_ns)
    plan = core.compute_plans(core.plan_ns)[1]
    core.mark_plan_sent(1, core.plan_ns)
    too_far = encode_data_packet(
        1, 1, 0, 2, np.asarray([1.0, 0.0], dtype=np.float16), send_pos=plan.blocks + 1
    )
    with pytest.raises(B3PlanError, match="exceeds"):
        core.accept_data(too_far, core.plan_ns + 1)
    late = encode_data_packet(
        1, 1, 0, 2, np.asarray([1.0, 0.0], dtype=np.float16), send_pos=1
    )
    with pytest.raises(B3PlanError, match="deadline"):
        core.accept_data(late, core.deadline_ns + 1)


def test_client_missing_plan_wait_is_bounded() -> None:
    async def scenario() -> None:
        core = B2ClientCore(1, d=6, L=2)
        client = B3UdpClient(
            core,
            gateway_addr=("127.0.0.1", 9102),
            update_provider=lambda _round: np.zeros(6, dtype=np.float32),
        )
        transport = _FakeTransport()
        client.connection_made(transport)  # type: ignore[arg-type]
        start_ns = client.clock_ns()
        deadline_ns = start_ns + 10_000_000
        start = encode_message(
            MessageType.ROUND_START,
            round_id=1,
            client_id=1,
            t_send_ns=start_ns,
            body={
                "deadline_ns": deadline_ns,
                "K": 3,
                "L": 2,
                "rho_us": 0,
                "mode": 2,
            },
        )
        client.datagram_received(start, ("127.0.0.1", 9102))
        with pytest.raises(B3PlanError, match="PLAN was not received"):
            await client.wait_for_plan()
        assert decode_message(transport.packets[0][0]).message_type is MessageType.READY
        await client.close()

    asyncio.run(scenario())


def test_new_client_uses_conservative_delivery_prior_and_updates_from_deliveries() -> None:
    successful = _core(client_ids=(1,), bottleneck_bps=1_000_000)
    assert successful.delivery_ewma.get(1) == 0.1
    successful.accept_ready(_ready(1), successful.plan_ns)
    plan = successful.compute_plans(successful.plan_ns)[1]
    assert successful.caps[1] == plan_caps(
        0.1,
        successful.states[1].bhat_bps,
        successful.duration_ns,
        successful.plan_offset_ns,
        successful.states[1].srtt_ns,
    )
    assert successful.caps[1] < plan_caps(
        1.0,
        successful.states[1].bhat_bps,
        successful.duration_ns,
        successful.plan_offset_ns,
        successful.states[1].srtt_ns,
    )
    assert plan.blocks == 3
    plan_sent_ns = successful.plan_ns + 1
    successful.mark_plan_sent(1, plan_sent_ns)
    for send_pos in range(1, plan.blocks + 1):
        packet = encode_data_packet(
            1,
            1,
            send_pos - 1,
            2,
            np.asarray([float(send_pos), 0.0], dtype=np.float16),
            send_pos=send_pos,
            t_send_ns=plan_sent_ns + send_pos,
        )
        successful.accept_data(packet, plan_sent_ns + send_pos)
    successful.finish(successful.deadline_ns)
    assert successful.delivery_ewma.get(1) == pytest.approx(0.19)

    poor = _core(client_ids=(1,), bottleneck_bps=1_000_000)
    poor.accept_ready(_ready(1), poor.plan_ns)
    assert poor.compute_plans(poor.plan_ns)[1].blocks == 3
    poor.finish(poor.deadline_ns)
    assert poor.delivery_ewma.get(1) == pytest.approx(0.09)

    configured_core = _core(client_ids=(1,), delivery_ratio_init=0.25)
    assert configured_core.delivery_ewma.get(1) == 0.25
    gateway = B3UdpGateway(
        client_weights={1: 1.0},
        endpoints={1: ("127.0.0.1", 0)},
        d=6,
        duration_ns=398_000_000,
        delivery_ratio_init=0.25,
    )
    assert gateway.delivery_ewma.get(1) == 0.25


@pytest.mark.parametrize(
    ("estimate", "expected"),
    [
        (4_000_000.4, 4_000_000),
        (4_000_000.5, 4_000_000),
        (4_000_001.5, 4_000_002),
        (0.0, 0),
        (0xFFFFFFFF, 0xFFFFFFFF),
    ],
)
def test_plan_rate_rounds_to_nearest_integer_without_early_precision_loss(
    estimate: float, expected: int
) -> None:
    core = _core(client_ids=(1,))
    core.states[1].bhat_bps = estimate
    plan = core.compute_plans(core.plan_ns)[1]
    assert plan.rate_bps == expected


def test_plan_rate_rejects_value_that_rounds_outside_u32() -> None:
    core = _core(client_ids=(1,))
    core.states[1].bhat_bps = 0xFFFFFFFF + 0.6
    with pytest.raises(B3PlanError, match="wire range"):
        core.compute_plans(core.plan_ns)


def test_client_paces_at_the_integer_rate_received_in_plan() -> None:
    async def scenario() -> None:
        clock = _FakeClock(1_000_000_000)
        core = B2ClientCore(1, d=4, L=2)
        client = B3UdpClient(
            core,
            gateway_addr=("127.0.0.1", 9102),
            update_provider=lambda _round: np.asarray(
                [1.0, 2.0, 3.0, 4.0], dtype=np.float32
            ),
            clock_ns=clock,
            sleep=clock.sleep,
        )
        transport = _FakeTransport()
        client.connection_made(transport)  # type: ignore[arg-type]
        deadline_ns = clock() + 1_000_000_000
        start = encode_message(
            MessageType.ROUND_START,
            round_id=1,
            client_id=1,
            t_send_ns=clock(),
            body={
                "deadline_ns": deadline_ns,
                "K": 2,
                "L": 2,
                "rho_us": 0,
                "mode": 2,
            },
        )
        client.datagram_received(start, ("127.0.0.1", 9102))
        wire_rate = 1_234_568
        plan = encode_message(
            MessageType.PLAN,
            round_id=1,
            client_id=1,
            t_send_ns=clock(),
            body={"k": 2, "rate_bps": wire_rate},
        )
        client.datagram_received(plan, ("127.0.0.1", 9102))
        transfer = await client.wait_for_plan()
        expected_interval_ns = math.ceil((36 + 42) * 8 * 1_000_000_000 / wire_rate)
        assert transfer.send_times_ns[1] - transfer.send_times_ns[0] == expected_interval_ns
        assert transfer.packets_sent == 2
        await client.close()

    asyncio.run(scenario())


def _planned_single_client() -> tuple[B3GatewayCore, int]:
    core = _core(client_ids=(1,), bottleneck_bps=1_000_000)
    core.accept_ready(_ready(1), core.plan_ns)
    plan = core.compute_plans(core.plan_ns)[1]
    assert plan.blocks == 3
    core.mark_plan_sent(1, core.plan_ns + 1)
    return core, plan.blocks


def test_missing_reordered_and_duplicate_data_do_not_forge_delivery_or_backlog() -> None:
    core, blocks = _planned_single_client()
    initial_bhat = core.states[1].bhat_bps
    sent_ns = core.plans[1].sent_ns
    later_position = encode_data_packet(
        1, 1, 1, 2, np.asarray([2.0, 0.0], dtype=np.float16),
        send_pos=2, t_send_ns=sent_ns + 10,
    )
    earlier_position = encode_data_packet(
        1, 1, 0, 2, np.asarray([1.0, 0.0], dtype=np.float16),
        send_pos=1, t_send_ns=sent_ns + 20,
    )
    core.accept_data(later_position, sent_ns + 10)
    core.accept_data(earlier_position, sent_ns + 20)
    with pytest.raises(B3PlanError, match="duplicate"):
        core.accept_data(later_position, sent_ns + 30)

    assert tuple(core.accumulator.clients[1].delivered) == (1, 0)
    assert len(core.accumulator.clients[1].delivered) == 2 < blocks
    core._advance_rate_window(1, core.start_ns + 100_000_000, backlogged=None)
    assert core.states[1].bhat_bps == initial_bhat

    client = B2ClientCore(1, d=6, L=2)
    delta = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32)
    client.start_round(1, core.deadline_ns, delta)
    prepared = client.round.u.copy()
    result = core.finish(core.deadline_ns)
    apply_delivery_result(client, result.round_result)
    expected_residual = prepared.copy()
    expected_residual[:4] -= prepared[:4].astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(client.error_feedback.e, expected_residual)

    missing_packet = encode_data_packet(
        1, 1, 2, 2, np.asarray([3.0, 0.0], dtype=np.float16),
        send_pos=3, t_send_ns=core.deadline_ns + 1,
    )
    with pytest.raises(B3PlanError, match="closed"):
        core.accept_data(missing_packet, core.deadline_ns + 1)
    assert result.round_result.client_results[1].delivered_indices == (0, 1)


def test_delayed_final_data_at_deadline_counts_once_and_completes_delivery_ratio() -> None:
    core, blocks = _planned_single_client()
    sent_ns = core.plans[1].sent_ns
    for arrival_offset, send_pos in enumerate((2, 1), start=10):
        packet = encode_data_packet(
            1,
            1,
            send_pos - 1,
            2,
            np.asarray([float(send_pos), 0.0], dtype=np.float16),
            send_pos=send_pos,
            t_send_ns=sent_ns + send_pos,
        )
        core.accept_data(packet, sent_ns + arrival_offset)
    final_packet = encode_data_packet(
        1,
        1,
        2,
        2,
        np.asarray([3.0, 0.0], dtype=np.float16),
        send_pos=3,
        t_send_ns=core.deadline_ns,
    )
    core.accept_data(final_packet, core.deadline_ns)
    with pytest.raises(B3PlanError, match="duplicate"):
        core.accept_data(final_packet, core.deadline_ns)
    result = core.finish(core.deadline_ns)
    assert result.round_result.client_results[1].packets_received == blocks
    assert core.delivery_ewma.get(1) == pytest.approx(0.19)


def test_shared_pacing_helper_boundaries_and_validation() -> None:
    start_ns = 1_000_000_000
    rate_bps = 800
    size_bytes = 10
    interval_ns = 100_000_000
    assert paced_blocks_sent(
        start_ns - 1,
        t_start_ns=start_ns,
        k=3,
        rate_bps=rate_bps,
        block_size_bytes=size_bytes,
    ) == 0
    assert paced_blocks_sent(
        start_ns,
        t_start_ns=start_ns,
        k=3,
        rate_bps=rate_bps,
        block_size_bytes=size_bytes,
    ) == 1
    assert paced_blocks_sent(
        start_ns + interval_ns - 1,
        t_start_ns=start_ns,
        k=3,
        rate_bps=rate_bps,
        block_size_bytes=size_bytes,
    ) == 1
    assert paced_blocks_sent(
        start_ns + interval_ns,
        t_start_ns=start_ns,
        k=3,
        rate_bps=rate_bps,
        block_size_bytes=size_bytes,
    ) == 2
    assert paced_blocks_sent(
        start_ns + 3 * interval_ns,
        t_start_ns=start_ns,
        k=3,
        rate_bps=rate_bps,
        block_size_bytes=size_bytes,
    ) == 3
    assert paced_blocks_sent(
        start_ns,
        t_start_ns=start_ns,
        k=0,
        rate_bps=0,
        block_size_bytes=size_bytes,
    ) == 0
    assert pacing_send_time_ns(
        3,
        t_start_ns=start_ns,
        k=3,
        rate_bps=rate_bps,
        block_size_bytes=size_bytes,
    ) == start_ns + 2 * interval_ns
    with pytest.raises(ValueError, match="positive"):
        paced_blocks_sent(
            start_ns,
            t_start_ns=start_ns,
            k=1,
            rate_bps=0,
            block_size_bytes=size_bytes,
        )
    with pytest.raises(ValueError, match="block_size_bytes"):
        paced_blocks_sent(
            start_ns,
            t_start_ns=start_ns,
            k=1,
            rate_bps=rate_bps,
            block_size_bytes=0,
        )
    with pytest.raises(ValueError, match="u64"):
        paced_blocks_sent(
            1 << 64,
            t_start_ns=0,
            k=1,
            rate_bps=rate_bps,
            block_size_bytes=size_bytes,
        )
    with pytest.raises(ValueError, match="u64 range"):
        pacing_send_time_ns(
            2,
            t_start_ns=(1 << 64) - 1,
            k=2,
            rate_bps=1,
            block_size_bytes=size_bytes,
        )


def test_first_data_position_one_refines_start_from_plan_and_rtt() -> None:
    core, blocks = _planned_single_client()
    plan = core.plans[1]
    srtt_at_plan = core.plan_srtt_ns[1]
    frame_size = data_frame_size_bytes(core.L)
    candidate = plan.sent_ns + 5_000_000
    arrival_ns = candidate + srtt_at_plan // 2
    packet = encode_data_packet(
        1, 1, 0, 2, np.asarray([1.0, 0.0], dtype=np.float16),
        send_pos=1, t_send_ns=plan.sent_ns,
    )
    core.accept_data(packet, arrival_ns)
    assert blocks == 3
    assert core.transmission_start_ns[1] == candidate
    assert frame_size == 78


def test_first_data_position_after_one_accounts_for_pacing_offset() -> None:
    core, blocks = _planned_single_client()
    plan = core.plans[1]
    offset = pacing_send_time_ns(
        2,
        t_start_ns=0,
        k=blocks,
        rate_bps=plan.rate_bps,
        block_size_bytes=data_frame_size_bytes(core.L),
    )
    candidate = plan.sent_ns + 7_000_000
    packet = encode_data_packet(
        1, 1, 1, 2, np.asarray([2.0, 0.0], dtype=np.float16),
        send_pos=2, t_send_ns=plan.sent_ns,
    )
    core.accept_data(packet, candidate + core.plan_srtt_ns[1] // 2 + offset)
    assert core.transmission_start_ns[1] == candidate


def test_multiple_data_refines_start_to_minimum_and_never_before_plan() -> None:
    core, blocks = _planned_single_client()
    plan = core.plans[1]
    frame_size = data_frame_size_bytes(core.L)
    half_rtt = core.plan_srtt_ns[1] // 2
    offset_1 = pacing_send_time_ns(
        1, t_start_ns=0, k=blocks, rate_bps=plan.rate_bps, block_size_bytes=frame_size
    )
    offset_2 = pacing_send_time_ns(
        2, t_start_ns=0, k=blocks, rate_bps=plan.rate_bps, block_size_bytes=frame_size
    )
    first_candidate = plan.sent_ns + 10_000_000
    first = encode_data_packet(
        1, 1, 0, 2, np.asarray([1.0, 0.0], dtype=np.float16),
        send_pos=1, t_send_ns=plan.sent_ns,
    )
    core.accept_data(first, first_candidate + half_rtt + offset_1)
    second_candidate = plan.sent_ns + 9_900_000
    second = encode_data_packet(
        1, 1, 1, 2, np.asarray([2.0, 0.0], dtype=np.float16),
        send_pos=2, t_send_ns=plan.sent_ns,
    )
    core.accept_data(second, second_candidate + half_rtt + offset_2)
    assert core.transmission_start_ns[1] == second_candidate

    earlier_candidate = plan.sent_ns + 9_800_000
    third = encode_data_packet(
        1, 1, 2, 2, np.asarray([3.0, 0.0], dtype=np.float16),
        send_pos=3, t_send_ns=plan.sent_ns,
    )
    core.accept_data(third, earlier_candidate + half_rtt + pacing_send_time_ns(
        3, t_start_ns=0, k=blocks, rate_bps=plan.rate_bps, block_size_bytes=frame_size
    ))
    assert core.transmission_start_ns[1] == earlier_candidate

    clamped, clamped_blocks = _planned_single_client()
    clamped_plan = clamped.plans[1]
    before_plan = encode_data_packet(
        1, 1, 0, 2, np.asarray([1.0, 0.0], dtype=np.float16),
        send_pos=1, t_send_ns=clamped_plan.sent_ns,
    )
    clamped.accept_data(
        before_plan,
        clamped_plan.sent_ns - 10_000_000 + clamped.plan_srtt_ns[1] // 2,
    )
    assert clamped.transmission_start_ns[1] == clamped_plan.sent_ns
    assert clamped_blocks == 3


def test_no_data_uses_plan_plus_half_rtt_fallback_for_windows_and_loss_rate() -> None:
    core = B3GatewayCore(
        round_id=1,
        start_ns=0,
        deadline_ns=1_000_000_000,
        d=200,
        L=2,
        client_weights={1: 1.0},
    )
    core.states[1].bhat_bps = 100_000.0
    core.compute_plans(core.plan_ns)
    core.plans[1] = B3ClientPlan(1, 100, 100_000, core.plan_ns)
    core._plan_sent.add(1)
    core.plan_srtt_ns[1] = 100_000_000
    core.transmission_start_ns[1] = core.plan_ns + 50_000_000
    assert core.transmission_start_ns[1] == 150_000_000

    result = core.finish(core.deadline_ns)
    assert result.round_result.delivered_blocks == 0
    # Windows [200, 300) through [600, 700) began after t_start and still
    # had paced blocks due after their end. Each empty backlogged window applies
    # the existing 0.7 old-rate multiplier.
    assert core.states[1].bhat_bps == pytest.approx(100_000.0 * 0.7**5)
    assert core.delivery_ewma.get(1) == pytest.approx(0.09)


@pytest.mark.parametrize(
    ("window_start", "t_start", "k", "rate", "expected_backlogged"),
    [
        (1_000_000_000, 1_200_000_000, 100, 100_000, False),
        (1_000_000_000, 1_050_000_000, 100, 100_000, False),
        (1_100_000_000, 1_100_000_000, 100, 100_000, True),
        (1_100_000_000, 1_100_000_000, 3, 1_000_000, False),
    ],
)
def test_rate_window_backlog_uses_shared_pacing_progress(
    window_start: int,
    t_start: int,
    k: int,
    rate: int,
    expected_backlogged: bool,
) -> None:
    core = _core(client_ids=(1,))
    core.compute_plans(core.plan_ns)
    core.plans[1] = B3ClientPlan(1, k, rate, core.plan_ns)
    core._plan_sent.add(1)
    core.plan_srtt_ns[1] = 0
    core.transmission_start_ns[1] = t_start
    core.windows[1].start(window_start)
    old_rate = core.states[1].bhat_bps
    core._advance_rate_window(1, window_start + 100_000_000, backlogged=None)
    expected = old_rate * 0.7 if expected_backlogged else old_rate
    assert core.states[1].bhat_bps == pytest.approx(expected)


def test_client_and_gateway_use_same_pacing_schedule_and_send_positions() -> None:
    async def scenario() -> None:
        gateway_core = _core(client_ids=(1,), bottleneck_bps=1_000_000)
        gateway_core.accept_ready(_ready(1), gateway_core.plan_ns)
        gateway_core.states[1].bhat_bps = 1_500_000.0
        gateway_plan = gateway_core.compute_plans(gateway_core.plan_ns)[1]
        assert gateway_plan.blocks == 3
        start_ns = gateway_core.plan_ns + 1
        gateway_core.mark_plan_sent(1, start_ns)
        clock = _FakeClock(start_ns)
        client_core = B2ClientCore(1, d=6, L=2)
        client = B3UdpClient(
            client_core,
            gateway_addr=("127.0.0.1", 9102),
            update_provider=lambda _round: np.asarray(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32
            ),
            clock_ns=clock,
            sleep=clock.sleep,
        )
        transport = _FakeTransport()
        client.connection_made(transport)  # type: ignore[arg-type]
        deadline_ns = gateway_core.deadline_ns
        round_start = encode_message(
            MessageType.ROUND_START,
            round_id=1,
            client_id=1,
            t_send_ns=start_ns,
            body={
                "deadline_ns": deadline_ns,
                "K": 3,
                "L": 2,
                "rho_us": 0,
                "mode": 2,
            },
        )
        client.datagram_received(round_start, ("127.0.0.1", 9102))
        rate = gateway_plan.rate_bps
        plan = encode_message(
            MessageType.PLAN,
            round_id=1,
            client_id=1,
            t_send_ns=start_ns,
            body={"k": gateway_plan.blocks, "rate_bps": rate},
        )
        client.datagram_received(plan, ("127.0.0.1", 9102))
        transfer = await client.wait_for_plan()

        for (packet, _endpoint), send_time in zip(
            transport.packets[1:],
            transfer.send_times_ns,
            strict=True,
        ):
            message = decode_message(packet, block_length=2)
            assert message.body["send_pos"] == len(
                [t for t in transfer.send_times_ns if t <= send_time]
            )
            gateway_core.accept_data(
                packet,
                send_time + gateway_core.plan_srtt_ns[1] // 2,
            )
        assert transfer.send_times_ns == tuple(
            pacing_send_time_ns(
                pos,
                t_start_ns=start_ns,
                k=gateway_plan.blocks,
                rate_bps=rate,
                block_size_bytes=data_frame_size_bytes(2),
            )
            for pos in range(1, 4)
        )
        assert gateway_core.transmission_start_ns[1] == start_ns
        await client.close()

    asyncio.run(scenario())


def test_duplicate_data_does_not_change_start_estimate_or_delivery_ratio() -> None:
    core, blocks = _planned_single_client()
    plan = core.plans[1]
    candidate = plan.sent_ns + 5_000_000
    offset = pacing_send_time_ns(
        1, t_start_ns=0, k=blocks, rate_bps=plan.rate_bps,
        block_size_bytes=data_frame_size_bytes(core.L),
    )
    packet = encode_data_packet(
        1, 1, 0, 2, np.asarray([1.0, 0.0], dtype=np.float16),
        send_pos=1, t_send_ns=plan.sent_ns,
    )
    arrival = candidate + core.plan_srtt_ns[1] // 2 + offset
    core.accept_data(packet, arrival)
    estimated = core.transmission_start_ns[1]
    delivered = len(core.accumulator.clients[1].delivered)
    with pytest.raises(B3PlanError, match="duplicate"):
        core.accept_data(packet, arrival + 1)
    assert core.transmission_start_ns[1] == estimated
    assert len(core.accumulator.clients[1].delivered) == delivered == 1
    core.finish(core.deadline_ns)
    assert core.delivery_ewma.get(1) == pytest.approx(0.1 + 0.1 * (1 / blocks - 0.1))


def test_zero_plan_does_not_update_delivery_ratio_and_finish_is_once() -> None:
    core = _core(client_ids=(1,), delivery_ratio_init=0.25)
    core.compute_plans(core.plan_ns)
    assert core.plans[1].blocks == 0
    core.mark_plan_sent(1, core.plan_ns)
    core.finish(core.deadline_ns)
    assert core.delivery_ewma.get(1) == 0.25
    with pytest.raises(B3PlanError, match="already finished"):
        core.finish(core.deadline_ns)
