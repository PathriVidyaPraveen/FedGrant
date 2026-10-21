"""Gateway-core boundary and round-transition tests."""

from dataclasses import dataclass, field

import numpy as np
import pytest

from fedgrant.common.constants import DATA_GRANT_ID, MessageType, TransferMode
from fedgrant.common.wire import encode_message
from fedgrant.core.gateway import (
    EmitAggregate,
    GatewayCore,
    GatewayError,
    GatewayPhase,
    SendGrant,
    SendRoundStart,
    SendRoundEnd,
)
from fedgrant.core.state import ClientState
from fedgrant.policy.types import CandidateView


@dataclass
class FakeEligibility:
    def candidates(self, clients, now_ns, deadline_ns, retransmit):
        eligible = [
            client_id
            for client_id, state in sorted(clients.items())
            if state.ready and not state.outstanding and state.has_pending(retransmit)
        ]
        return CandidateView(
            np.asarray(eligible, dtype=np.int32),
            np.asarray([clients[i].val[clients[i].next_pos(retransmit) - 1] for i in eligible], dtype=np.float64),
            np.asarray([clients[i].K - clients[i].m - len(clients[i].outstanding) for i in eligible], dtype=np.int32),
            0,
        )


@dataclass
class FakeSelector:
    def select(self, cand):
        return int(cand.client_id[0])


@dataclass
class FakePacing:
    allowed: bool = True

    def allow(self, client_id, p_hat, now_ns, deadline_ns):
        return self.allowed


@dataclass
class FakePrediction:
    p: float = 0.75

    def probability(self, client_id, now_ns):
        return self.p


@dataclass
class FakeTimeout:
    delta_ns: int = 10

    def timeout_ns(self, client_id, issue_ns):
        return issue_ns + self.delta_ns


@dataclass
class FakeEstimator:
    reset: list = field(default_factory=list)
    resolutions: list = field(default_factory=list)
    ticks: list = field(default_factory=list)
    deadlines: list = field(default_factory=list)

    def reset_round(self, ctx):
        self.reset.append(ctx)

    def resolve(self, client_id, grant, delivered, t_ns):
        self.resolutions.append((client_id, grant, delivered, t_ns))

    def on_tick(self, t_ns):
        self.ticks.append(t_ns)

    def on_deadline(self, t_ns):
        self.deadlines.append(t_ns)


@dataclass
class FakeAggregate:
    def aggregate(self, clients):
        return tuple(sorted((client_id, state.m) for client_id, state in clients.items()))


@dataclass
class FakeIntegrity:
    valid_result: bool = True
    seen_n_coords: list[int] = field(default_factory=list)

    def valid(self, client_id, block_idx, payload, reported_energies, n_coords):
        self.seen_n_coords.append(n_coords)
        return self.valid_result


def make_gateway(*, mode=TransferMode.GRANT, integrity=None, dependencies=True):
    state = ClientState(1, 0.5, K=3)
    kwargs = {}
    if dependencies:
        kwargs.update(
            selector=FakeSelector(),
            eligibility=FakeEligibility(),
            pacing=FakePacing(),
            prediction=FakePrediction(),
            timeout=FakeTimeout(),
        )
    gateway = GatewayCore(
        {1: state},
        estimator=FakeEstimator(),
        aggregate=FakeAggregate(),
        value_integrity=integrity,
        **kwargs,
    )
    gateway.on_round_start(100, round_id=7, deadline_ns=1000, K=3, L=2, mode=mode)
    return gateway, state


def ready(gateway):
    return gateway.on_ready(1, [4.0, 2.0, 0.0], 200)


def test_initial_and_round_start_state() -> None:
    gateway, state = make_gateway()
    assert gateway.phase is GatewayPhase.ACTIVE
    assert gateway.round is not None
    assert gateway.round.round_id == 7
    assert gateway.next_grant_id == 1
    assert state.ready is False and state.K == 3 and state.outstanding == {}
    assert gateway.estimator.reset[0].D_ns == 900


def test_ready_rejects_unknown_or_invalid_events_without_mutating_state() -> None:
    gateway, state = make_gateway()
    assert gateway.on_ready(99, [4.0, 2.0, 0.0], 200) == []
    assert gateway.on_ready(1, [1.0, 2.0, 0.0], 200) == []
    assert gateway.on_ready(1, [4.0, 2.0, 0.0], 1001) == []
    assert not state.ready
    assert gateway.stats.rejected_ready == 3


def test_ready_installs_weighted_values_and_issues_first_grant() -> None:
    gateway, state = make_gateway()
    actions = ready(gateway)
    assert actions == [SendGrant(1, 1, 1)]
    np.testing.assert_array_equal(state.val, np.asarray([1.0, 0.5, 0.0], np.float32))
    assert state.outstanding[1].pos == 1
    assert state.outstanding[1].p_issue == 0.75
    assert gateway.on_ready(1, [4.0, 2.0, 0.0], 201) == []
    assert gateway.stats.duplicate_ready == 1


def test_grant_data_requires_matching_id_and_position_before_state_mutation() -> None:
    gateway, state = make_gateway()
    ready(gateway)
    assert gateway.on_data(1, 999, 1, 0, [1.0, 0.0], 210) == []
    assert state.m == 0 and state.status[0] == 1 and len(state.outstanding) == 1
    assert gateway.on_data(1, 1, 2, 0, [1.0, 0.0], 210) == []
    assert state.m == 0 and len(state.outstanding) == 1

    actions = gateway.on_data(1, 1, 1, 0, [1.0, 0.0], 210)
    assert actions == [SendGrant(1, 2, 2)]
    assert state.m == 1 and state.status[0] == 2
    assert gateway.stats.accepted_data == 1
    assert gateway.estimator.resolutions[0][2] is True


def test_timeout_marks_lost_and_retransmits_through_state_transition() -> None:
    gateway, state = make_gateway()
    ready(gateway)
    assert gateway.on_tick(211) == [SendGrant(1, 2, 1)]
    assert state.status[0] == 1
    assert gateway.on_tick(222) == [SendGrant(1, 3, 1)]
    assert state.status[0] == 1
    assert [resolution[2] for resolution in gateway.estimator.resolutions] == [False, False]


def test_push_and_plan_accept_sentinel_without_outstanding_grant() -> None:
    gateway, state = make_gateway(mode=TransferMode.PUSH)
    assert gateway.on_data(1, DATA_GRANT_ID, 2, 1, [0.0, 0.0], 200) == []
    assert state.m == 1 and state.status[1] == 2
    assert gateway.on_data(1, DATA_GRANT_ID, 2, 1, [0.0, 0.0], 201) == []
    assert gateway.stats.accepted_data == 1 and gateway.stats.duplicate_data == 1
    assert gateway.on_data(1, 1, 3, 2, [0.0, 0.0], 200) == []
    assert state.m == 1


def test_grant_mode_rejects_push_sentinel_and_accepts_data_at_deadline() -> None:
    gateway, state = make_gateway()
    ready(gateway)
    assert gateway.on_data(1, DATA_GRANT_ID, 1, 0, [0.0, 0.0], 210) == []
    assert state.m == 0
    assert gateway.on_data(1, 1, 1, 0, [0.0, 0.0], 1000) == []
    assert state.m == 1


def test_deadline_resolves_grants_sends_bitmaps_and_emits_external_aggregate() -> None:
    gateway, state = make_gateway()
    ready(gateway)
    actions = gateway.on_deadline(1000)
    assert actions[0] == SendRoundEnd(1, b"\x00")
    assert actions[1] == EmitAggregate(((1, 0),))
    assert gateway.phase is GatewayPhase.DEADLINE
    assert state.outstanding == {}
    assert gateway.estimator.deadlines == [1000]
    assert gateway.on_deadline(1001) == []


def test_value_integrity_rule_overwrites_remaining_values_directly() -> None:
    gateway, state = make_gateway(integrity=FakeIntegrity(False))
    ready(gateway)
    gateway.on_data(1, 1, 1, 0, [1.0, 0.0], 210)
    # New grant for position 2 is issued by the test eligibility double.
    gateway.on_data(1, 2, 2, 1, [1.0, 0.0], 220)
    gateway.on_data(1, 3, 3, 2, [1.0, 0.0], 230)
    # The rule is activated on the third violation without calling set_ready.
    np.testing.assert_array_equal(state.val, np.asarray([1.0, 0.5, 0.0], np.float32))
    assert gateway.stats.energy_violations == {1: 3}


def test_missing_scheduling_dependencies_are_explicit_and_non_mutating() -> None:
    gateway, state = make_gateway(dependencies=False)
    assert ready(gateway) == []
    assert gateway.stats.missing_dependencies == 1
    assert state.ready and state.outstanding == {}


def test_invalid_round_configuration_is_rejected() -> None:
    gateway = GatewayCore()
    with pytest.raises(GatewayError):
        gateway.on_round_start(100, round_id=1, deadline_ns=100, K=3, L=2)


def test_round_start_message_initializes_and_propagates_protocol_parameters() -> None:
    state = ClientState(1, 0.5, K=3)
    gateway = GatewayCore({1: state})
    packet = encode_message(
        MessageType.ROUND_START,
        round_id=7,
        client_id=1,
        t_send_ns=100,
        body={"deadline_ns": 1000, "K": 3, "L": 2, "rho_us": 125, "mode": TransferMode.PLAN},
    )

    assert gateway.on_round_start_message(packet) == [
        SendRoundStart(1, 7, 100, 1000, 3, 2, 125, TransferMode.PLAN)
    ]
    assert gateway.round is not None
    assert gateway.round.round_id == 7
    assert gateway.round.deadline_ns == 1000
    assert gateway.round.K == 3
    assert gateway.round.L == 2
    assert gateway.round.rho_us == 125
    assert gateway.round.mode is TransferMode.PLAN
    assert state.ready is False
    assert state.outstanding == {}


def test_round_start_message_rejects_wrong_client_and_duplicate_or_stale_rounds() -> None:
    gateway, state = make_gateway(dependencies=False)
    before = (gateway.round, state.K, state.ready, gateway.next_grant_id)

    wrong_client = encode_message(
        MessageType.ROUND_START,
        round_id=8,
        client_id=99,
        t_send_ns=300,
        body={"deadline_ns": 2000, "K": 3, "L": 2, "rho_us": 0, "mode": TransferMode.GRANT},
    )
    assert gateway.on_round_start_message(wrong_client) == []
    assert (gateway.round, state.K, state.ready, gateway.next_grant_id) == before

    duplicate = encode_message(
        MessageType.ROUND_START,
        round_id=7,
        client_id=1,
        t_send_ns=300,
        body={"deadline_ns": 2000, "K": 3, "L": 2, "rho_us": 0, "mode": TransferMode.GRANT},
    )
    assert gateway.on_round_start_message(duplicate) == []
    assert gateway.round.round_id == 7

    gateway.on_deadline(1000)
    stale = encode_message(
        MessageType.ROUND_START,
        round_id=6,
        client_id=1,
        t_send_ns=1100,
        body={"deadline_ns": 2000, "K": 3, "L": 2, "rho_us": 0, "mode": TransferMode.GRANT},
    )
    assert gateway.on_round_start_message(stale) == []
    assert gateway.round.round_id == 7


def test_round_start_message_rejects_malformed_input_without_initializing() -> None:
    gateway = GatewayCore({1: ClientState(1, 0.5, K=3)})
    packet = bytearray(
        encode_message(
            MessageType.ROUND_START,
            round_id=1,
            client_id=1,
            t_send_ns=100,
            body={"deadline_ns": 1000, "K": 3, "L": 2, "rho_us": 0, "mode": TransferMode.GRANT},
        )
    )
    packet[0] ^= 0x01

    assert gateway.on_round_start_message(packet) == []
    assert gateway.phase is GatewayPhase.IDLE
    assert gateway.round is None


def test_ready_message_validates_round_client_and_energy_length_before_mutation() -> None:
    gateway, state = make_gateway(dependencies=False)
    ready_packet = encode_message(
        MessageType.READY,
        round_id=7,
        client_id=1,
        t_send_ns=200,
        body={"K": 3, "energy": [4.0, 2.0, 0.0]},
    )
    assert gateway.on_ready_message(ready_packet) == []
    assert state.ready is True
    np.testing.assert_array_equal(gateway.energies[1], np.asarray([4.0, 2.0, 0.0], dtype=np.float32))

    duplicate = gateway.on_ready_message(ready_packet)
    assert duplicate == []
    assert gateway.stats.duplicate_ready == 1

    other_round = encode_message(
        MessageType.READY,
        round_id=8,
        client_id=1,
        t_send_ns=200,
        body={"K": 3, "energy": [3.0, 2.0, 1.0]},
    )
    unknown_client = encode_message(
        MessageType.READY,
        round_id=7,
        client_id=99,
        t_send_ns=200,
        body={"K": 3, "energy": [3.0, 2.0, 1.0]},
    )
    assert gateway.on_ready_message(other_round) == []
    assert gateway.on_ready_message(unknown_client) == []


def test_ready_message_rejects_inconsistent_count_without_partial_state_change() -> None:
    gateway, state = make_gateway(dependencies=False)
    packet = encode_message(
        MessageType.READY,
        round_id=7,
        client_id=1,
        t_send_ns=200,
        body={"K": 2, "energy": [4.0, 2.0]},
    )

    assert gateway.on_ready_message(packet) == []
    assert state.ready is False
    assert gateway.energies == {}
    assert state.m == 0


def test_round_start_is_unicast_per_registered_client_and_floors_rho() -> None:
    class View:
        arrivals = np.asarray([4, 5, 6], dtype=np.int64)

    clients = {1: ClientState(1, 0.5, K=2), 2: ClientState(2, 0.5, K=2)}
    gateway = GatewayCore(clients, view=View())
    actions = gateway.on_round_start(
        1_000,
        round_id=4,
        duration_ns=9_000,
        K=2,
        L=2,
        rho_us={1: 12.9, 2: 27.1},
        mode=TransferMode.PUSH,
    )

    assert actions == [
        SendRoundStart(1, 4, 1_000, 10_000, 2, 2, 12, TransferMode.PUSH),
        SendRoundStart(2, 4, 1_000, 10_000, 2, 2, 27, TransferMode.PUSH),
    ]
    assert gateway.stats.bad_client_id == 0
    np.testing.assert_array_equal(gateway.view.arrivals, np.zeros(3, dtype=np.int64))
    assert clients[1].round_start_ns == clients[2].round_start_ns == 1_000


def test_invalid_client_id_is_counted_and_does_not_mutate_state() -> None:
    gateway, state = make_gateway(dependencies=False)
    packet = encode_message(
        MessageType.READY,
        round_id=7,
        client_id=99,
        t_send_ns=200,
        body={"K": 3, "energy": [4.0, 2.0, 0.0]},
    )

    assert gateway.on_ready_message(packet) == []
    assert gateway.stats.bad_client_id == 1
    assert not state.ready


def test_late_data_for_a_lost_position_is_delivered_and_classified() -> None:
    gateway, state = make_gateway()
    ready(gateway)
    gateway.pacing.allowed = False
    assert gateway.on_tick(211) == []
    assert state.status[0] == 3  # LOST

    assert gateway.on_data(1, 1, 1, 0, [1.0, 0.0], 220) == []
    assert state.m == 1
    assert state.status[0] == 2  # DELIVERED
    assert gateway.stats.late_after_timeout == 1
    assert gateway.delivered_data[1][0][1] == 2
    np.testing.assert_array_equal(state.delivered_payloads[0], np.asarray([1.0, 0.0], np.float32))


def test_duplicate_data_resolves_the_active_retransmission_grant() -> None:
    gateway, state = make_gateway()
    ready(gateway)
    assert gateway.on_tick(211) == [SendGrant(1, 2, 1)]

    # A delayed response to grant 1 delivers the block while grant 2 is outstanding.
    assert gateway.on_data(1, 1, 1, 0, [1.0, 0.0], 220) == []
    assert set(state.outstanding) == {2}

    # The retransmission response is a duplicate block, but it still resolves grant 2.
    assert gateway.on_data(1, 2, 1, 0, [1.0, 0.0], 230) == [SendGrant(1, 3, 2)]
    assert set(state.outstanding) == {3}
    assert gateway.stats.duplicate_data == 1
    assert gateway.stats.accepted_data == 1
    assert gateway.estimator.resolutions[-1][1].pos == 1
    assert gateway.estimator.resolutions[-1][2:] == (True, 230)


def test_timeout_of_active_retransmission_does_not_lose_delivered_position() -> None:
    gateway, state = make_gateway()
    ready(gateway)
    assert gateway.on_tick(211) == [SendGrant(1, 2, 1)]

    # Grant 1's delayed response delivers position 1 while its retry is outstanding.
    assert gateway.on_data(1, 1, 1, 0, [1.0, 0.0], 220) == []
    assert state.status[0] == 2
    assert set(state.outstanding) == {2}

    # Grant 2 times out as a failed grant, but the delivered position stays delivered.
    assert gateway.on_tick(222) == [SendGrant(1, 3, 2)]
    assert state.status[0] == 2
    assert state.status[1] == 1
    assert gateway.estimator.resolutions[-1][1].pos == 1
    assert gateway.estimator.resolutions[-1][2:] == (False, 222)


def test_data_after_deadline_is_counted_without_mutating_state() -> None:
    gateway, state = make_gateway()
    ready(gateway)
    gateway.on_deadline(1000)
    before_status = state.status.copy()

    assert gateway.on_data(1, 1, 1, 0, [1.0, 0.0], 1001) == []

    assert gateway.stats.late_after_deadline == 1
    assert gateway.stats.rejected_data == 0
    assert state.m == 0
    np.testing.assert_array_equal(state.status, before_status)


def test_data_message_preserves_true_n_coords_for_integrity_and_storage() -> None:
    integrity = FakeIntegrity()
    gateway, state = make_gateway(integrity=integrity, mode=TransferMode.PUSH, dependencies=False)
    packet = encode_message(
        MessageType.DATA,
        round_id=7,
        client_id=1,
        t_send_ns=1000,
        block_length=2,
        body={
            "grant_id": DATA_GRANT_ID,
            "send_pos": 1,
            "block_idx": 2,
            "n_coords": 1,
            "payload": [1.0, 0.0],
        },
    )

    assert gateway.on_data_message(packet, block_length=2) == []
    assert integrity.seen_n_coords == [1]
    assert gateway.delivered_data[1][2][1] == 1
    np.testing.assert_array_equal(state.delivered_payloads[2], np.asarray([1.0, 0.0], np.float32))
