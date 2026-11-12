"""B3 PLAN->first-DATA RTT producer (specification Section 10, B3 mode 2).

Definition under test (unchanged): one sample per client per round = arrival
time of the client's first *accepted* DATA of the round minus the send time of
its PLAN, where "send time" is the time the PLAN was handed to the socket
(``mark_plan_sent`` runs only after ``sendto`` succeeds). The same value feeds
the RFC 6298 update.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from fedgrant.baselines.b2_udp import encode_data_packet
from fedgrant.baselines.b3_plan import B3GatewayCore, B3PlanError, B3UdpGateway
from fedgrant.common.constants import MessageType
from fedgrant.common.wire import decode_message, encode_message
from fedgrant.core.estimators import rtt_update
from fedgrant.core.state import ClientState

START, DURATION = 1_000_000_000, 398_000_000
L_TEST = 2


def _ready(cid: int, round_id: int = 1) -> bytes:
    return encode_message(MessageType.READY, round_id=round_id, client_id=cid, t_send_ns=0,
                          body={"K": 3, "energy": (9.0, 4.0, 1.0)})


def _data(cid: int, send_pos: int, block_idx: int, round_id: int = 1) -> bytes:
    n_coords = 2  # d = 6, L = 2: every block is full
    return encode_data_packet(round_id, cid, block_idx, n_coords, np.asarray([1.0, 0.0], np.float16),
                              send_pos=send_pos)


def _core(round_id: int = 1, start: int = START, states=None, ewma=None, clients=(1, 2, 3)) -> B3GatewayCore:
    core = B3GatewayCore(round_id=round_id, start_ns=start, deadline_ns=start + DURATION, d=6, L=L_TEST,
                         client_weights={cid: 1.0 / len(clients) for cid in clients},
                         bottleneck_bps=10_000_000, client_states=states, delivery_ewma=ewma)
    for cid in clients:
        core.accept_ready(_ready(cid, round_id), core.plan_ns)
    plans = core.compute_plans(core.plan_ns)
    assert all(plan.blocks == 3 for plan in plans.values())
    return core


def test_sample_is_plan_handoff_to_first_accepted_data_and_records_its_send_pos() -> None:
    core = _core()
    sent_1 = core.plan_ns + 1_000
    sent_2 = core.plan_ns + 2_000
    core.mark_plan_sent(1, sent_1)
    core.mark_plan_sent(2, sent_2)
    # client 3's PLAN is never handed off to the socket.

    srtt_before = core.states[1].srtt_ns
    # Client 1: send position 1 was lost; position 2 is the first accepted DATA.
    core.accept_data(_data(1, 2, 1), sent_1 + 7_000_000)
    assert core.rtt_samples_ns == {1: 7_000_000}
    assert core.rtt_first_data == {1: (2, sent_1 + 7_000_000)}
    expected = ClientState(1, 1 / 3, 3)
    expected.srtt_ns, expected.rttvar_ns = srtt_before, ClientState(1, 1 / 3, 3).rttvar_ns
    rtt_update(expected, 7_000_000)
    assert core.states[1].srtt_ns == expected.srtt_ns  # the sample also drove RFC 6298

    # A later-arriving lower position, and a duplicate, add no sample.
    core.accept_data(_data(1, 1, 0), sent_1 + 9_000_000)
    with pytest.raises(B3PlanError, match="duplicate"):
        core.accept_data(_data(1, 2, 1), sent_1 + 10_000_000)
    # Client 2: DATA timed before its PLAN send is rejected, as is DATA after D.
    with pytest.raises(B3PlanError, match="outside the PLAN-to-deadline window"):
        core.accept_data(_data(2, 1, 0), sent_2 - 1)
    with pytest.raises(B3PlanError, match="outside the PLAN-to-deadline window"):
        core.accept_data(_data(2, 1, 0), core.deadline_ns + 1)
    # Client 3: no successful PLAN handoff, so its DATA is never accepted.
    with pytest.raises(B3PlanError, match="before this client's PLAN"):
        core.accept_data(_data(3, 1, 0), sent_1 + 5_000_000)

    result = core.finish(core.deadline_ns)
    assert result.rtt_samples_ns == {1: 7_000_000}  # at most one per client; none for 2 and 3
    assert core.rtt_first_data == {1: (2, sent_1 + 7_000_000)}
    assert result.round_result.client_results[1].delivered_indices == (0, 1)


def test_samples_belong_to_their_round_while_estimator_state_persists() -> None:
    first = _core(round_id=1)
    first.mark_plan_sent(1, first.plan_ns)
    first.accept_data(_data(1, 1, 0), first.plan_ns + 4_000_000)
    r1 = first.finish(first.deadline_ns)
    srtt_after_r1 = first.states[1].srtt_ns

    start2 = START + DURATION + 1_000_000_000
    second = _core(round_id=2, start=start2, states=first.states, ewma=first.delivery_ewma)
    assert second.rtt_samples_ns == {} and second.rtt_first_data == {}
    second.mark_plan_sent(1, second.plan_ns)
    second.mark_plan_sent(2, second.plan_ns)
    second.accept_data(_data(2, 1, 0, round_id=2), second.plan_ns + 6_000_000)
    second.accept_data(_data(1, 1, 0, round_id=2), second.plan_ns + 9_000_000)
    with pytest.raises(B3PlanError):
        second.accept_data(_data(1, 2, 1, round_id=1), second.plan_ns + 9_500_000)  # stale round id
    r2 = second.finish(second.deadline_ns)

    assert r1.rtt_samples_ns == {1: 4_000_000}
    assert r2.rtt_samples_ns == {1: 9_000_000, 2: 6_000_000}
    assert second.rtt_first_data == {1: (1, second.plan_ns + 9_000_000), 2: (1, second.plan_ns + 6_000_000)}
    assert second.states[1] is first.states[1] and second.states[1].srtt_ns != srtt_after_r1


class _FakeClock:
    def __init__(self, now: int) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


class _Transport:
    def __init__(self, fail_plan_for: set[int]) -> None:
        self.fail_plan_for, self.sent = fail_plan_for, []

    def sendto(self, packet: bytes, endpoint) -> None:
        message = decode_message(packet, block_length=L_TEST)
        if message.message_type is MessageType.PLAN and message.client_id in self.fail_plan_for:
            raise OSError("simulated PLAN send failure")
        self.sent.append(message)

    def close(self) -> None:
        pass


@pytest.mark.parametrize("fail", [False, True])
def test_udp_gateway_samples_from_successful_plan_handoff_only(fail: bool) -> None:
    async def scenario():
        endpoint = ("127.0.0.1", 9001)
        clock = _FakeClock(START)
        gateway = B3UdpGateway(client_weights={1: 1.0}, endpoints={1: endpoint}, d=6, L=L_TEST,
                               duration_ns=DURATION, clock_ns=clock)
        gateway.transport = _Transport({1} if fail else set())
        core = gateway.start_round(1)
        core.accept_ready(_ready(1), core.plan_ns)
        clock.now = core.plan_ns + 3_000
        if fail:
            with pytest.raises(B3PlanError, match="failed to send PLAN"):
                await gateway._plan_at_deadline()
            assert gateway.plan_local_handoffs == 0 and core._plan_sent == set()
        else:
            await gateway._plan_at_deadline()
            assert gateway.plan_local_handoffs == 1
            assert core.plans[1].sent_ns == clock.now  # timestamp of the successful handoff
        rejected = gateway.rejected_datagrams
        clock.now += 8_000_000
        gateway.datagram_received(_data(1, 1, 0), endpoint)
        if fail:
            assert gateway.rejected_datagrams == rejected + 1
            assert core.rtt_samples_ns == {}
        else:
            assert core.rtt_samples_ns == {1: 8_000_000} and core.rtt_first_data == {1: (1, clock.now)}
        await gateway.close()

    asyncio.run(scenario())
