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
from fedgrant.policy.features import GatewayView
from fedgrant.predictors.fallback import FallbackPredictor
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
class RaisingPredictor:
    calls: int = 0

    def reset_round(self, ctx):
        pass

    def observe(self, client, y, t_ns, obs):
        pass

    def predict(self, inp):
        self.calls += 1
        raise RuntimeError("predictor unavailable")


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
    calls: int = 0

    def aggregate(self, clients):
        self.calls += 1
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


def test_predictor_is_used_by_pump_and_falls_back_for_the_round() -> None:
    primary = RaisingPredictor()
    state = ClientState(1, 0.5, K=3)
    gateway = GatewayCore(
        {1: state},
        selector=FakeSelector(),
        eligibility=FakeEligibility(),
        pacing=FakePacing(),
        timeout=FakeTimeout(),
        predictor=primary,
        view=GatewayView(t_ns=0, D_ns=900, P_target=1.0),
    )
    gateway.on_round_start(100, round_id=7, deadline_ns=1000, K=3, L=2)

    actions = gateway.on_ready(1, [4.0, 2.0, 0.0], 200)

    assert actions == [SendGrant(1, 1, 1)]
    assert primary.calls == 1
    assert isinstance(gateway.predictor, FallbackPredictor)
    assert gateway.predictor.in_fallback
    assert state.outstanding[1].p_issue == pytest.approx(0.9)


def test_predictor_fallback_resets_when_gateway_starts_a_new_round() -> None:
    primary = RaisingPredictor()
    state = ClientState(1, 0.5, K=3)
    gateway = GatewayCore(
        {1: state},
        selector=FakeSelector(),
        eligibility=FakeEligibility(),
        pacing=FakePacing(),
        timeout=FakeTimeout(),
        predictor=primary,
        view=GatewayView(t_ns=0, D_ns=900, P_target=1.0),
    )
    gateway.on_round_start(100, round_id=7, deadline_ns=1000, K=3, L=2)
    gateway.on_ready(1, [4.0, 2.0, 0.0], 200)
    assert gateway.predictor.in_fallback

    gateway.on_deadline(1000)
    gateway.on_round_start(2000, round_id=8, deadline_ns=3000, K=3, L=2)

    assert not gateway.predictor.in_fallback


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
    assert gateway.on_tick(211) == []
    assert state.status[0] == 1
    assert gateway.on_tick(222) == [SendGrant(1, 3, 1)]
    assert state.status[0] == 1
    assert [resolution[2] for resolution in gateway.estimator.resolutions] == [False, False]


@pytest.mark.parametrize(
    ("arrival_ns", "delivered_on_time"),
    [(209, True), (210, True), (211, False)],
)
def test_data_at_grant_timeout_boundary_is_classified_by_arrival_time(
    arrival_ns: int,
    delivered_on_time: bool,
) -> None:
    gateway, state = make_gateway()
    ready(gateway)

    actions = gateway.on_data(1, 1, 1, 0, [1.0, 0.0], arrival_ns)

    assert actions == [SendGrant(1, 2, 2)]
    assert state.m == 1 and state.status[0] == 2
    assert gateway.stats.accepted_data == 1
    assert gateway.stats.late_after_timeout == (0 if delivered_on_time else 1)
    assert gateway.estimator.resolutions[0][2] is delivered_on_time
    assert gateway.estimator.resolutions[0][3] == arrival_ns
    assert set(state.outstanding) == {2}
    np.testing.assert_array_equal(state.delivered_payloads[0], [1.0, 0.0])

    # A duplicate reply to the expired/successfully resolved first grant cannot
    # redeliver the block or add another aggregation contribution.
    assert gateway.on_data(1, 1, 1, 0, [2.0, 0.0], arrival_ns + 1) == []
    assert state.m == 1 and gateway.stats.accepted_data == 1
    assert gateway.stats.late_after_timeout == (0 if delivered_on_time else 2)
    assert len(gateway.estimator.resolutions) == 1
    np.testing.assert_array_equal(state.delivered_payloads[0], [1.0, 0.0])
    actions = gateway.on_deadline(1000)
    assert SendRoundEnd(1, 7, b"\x01") in actions
    assert EmitAggregate(((1, 1),)) in actions


def test_grant_timeout_is_capped_at_round_deadline() -> None:
    gateway, state = make_gateway()
    gateway.timeout.delta_ns = 2_000

    assert ready(gateway) == [SendGrant(1, 1, 1)]
    assert state.outstanding[1].timeout_ns == gateway.round.deadline_ns == 1000
    assert gateway.on_tick(999) == []
    assert gateway.phase is GatewayPhase.ACTIVE
    assert gateway.on_tick(1000) == [
        SendRoundEnd(1, 7, b"\x00"),
        EmitAggregate(((1, 0),)),
    ]
    assert gateway.estimator.resolutions[0][2:] == (False, 1000)
    assert gateway.estimator.deadlines == [1000]


def test_data_at_deadline_is_on_time_when_grant_timeout_is_deadline() -> None:
    gateway, state = make_gateway()
    gateway.timeout.delta_ns = 2_000
    assert ready(gateway) == [SendGrant(1, 1, 1)]
    assert state.outstanding[1].timeout_ns == 1000

    assert gateway.on_data(1, 1, 1, 0, [1.0, 0.0], 1000) == []

    assert state.m == 1 and state.status[0] == 2
    assert gateway.stats.accepted_data == 1
    assert gateway.estimator.resolutions[0][2:] == (True, 1000)
    actions = gateway.on_deadline(1000)
    assert SendRoundEnd(1, 7, b"\x01") in actions
    assert EmitAggregate(((1, 1),)) in actions


def test_early_deadline_callback_is_rejected_without_ending_round() -> None:
    gateway, state = make_gateway()
    ready(gateway)

    assert gateway.on_deadline(999) == []

    assert gateway.phase is GatewayPhase.ACTIVE
    assert set(state.outstanding) == {1}
    assert gateway.estimator.deadlines == []
    assert gateway.aggregate.aggregate(gateway.clients) == ((1, 0),)


def test_round_reset_clears_timed_out_grants_and_rejects_old_round_data() -> None:
    gateway, state = make_gateway()
    ready(gateway)
    assert gateway.on_tick(211) == [SendGrant(1, 2, 1)]
    assert 1 in gateway.resolved_grants[1]
    assert 1 in gateway.timed_out_grants[1]
    gateway.on_deadline(1000)

    gateway.on_round_start(1100, round_id=8, deadline_ns=2000, K=3, L=2)
    assert gateway.resolved_grants == {1: {}}
    assert gateway.timed_out_grants == {1: set()}
    assert state.outstanding == {}
    assert gateway.on_ready(1, [4.0, 2.0, 0.0], 1200) == [SendGrant(1, 1, 1)]

    old_round_data = encode_message(
        MessageType.DATA,
        round_id=7,
        client_id=1,
        t_send_ns=1200,
        block_length=2,
        body={
            "grant_id": 1,
            "send_pos": 1,
            "block_idx": 0,
            "n_coords": 2,
            "payload": [1.0, 0.0],
        },
    )
    assert gateway.on_data_message(old_round_data, block_length=2) == []
    assert state.m == 0 and set(state.outstanding) == {1}
    assert gateway.stats.rejected_data == 1


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
    assert actions[0] == SendRoundEnd(1, 7, b"\x00")
    assert actions[1] == EmitAggregate(((1, 0),))
    assert gateway.phase is GatewayPhase.DEADLINE
    assert state.outstanding == {}
    assert gateway.estimator.deadlines == [1000]
    assert gateway.aggregate.calls == 1
    assert gateway.on_deadline(1001) == []
    assert gateway.on_tick(1000) == []
    assert gateway.aggregate.calls == 1


def test_round_transition_clears_round_data_and_preserves_lifetime_state() -> None:
    gateway, state = make_gateway()
    ready(gateway)
    assert gateway.on_data(1, 1, 1, 0, [1.0, 0.0], 210) == [SendGrant(1, 2, 2)]
    assert gateway.on_deadline(1000)[0] == SendRoundEnd(1, 7, b"\x01")
    previous_resolved_count = state.n_resolved
    accepted_before = gateway.stats.accepted_data

    actions = gateway.on_round_start(1100, round_id=8, deadline_ns=2000, K=2, L=1)

    assert actions == [SendRoundStart(1, 8, 1100, 2000, 2, 1, 0, TransferMode.GRANT)]
    assert gateway.round.round_id == 8 and gateway.phase is GatewayPhase.ACTIVE
    assert gateway.next_grant_id == 1
    assert gateway.energies == {}
    assert gateway.delivered_blocks == {1: set()}
    assert state.delivered_payloads == {}
    assert state.delivered_n_coords == {}
    assert gateway.resolved_grants == {1: {}}
    assert gateway.timed_out_grants == {1: set()}
    assert state.K == 2 and state.m == 0 and state.ready is False
    assert state.outstanding == {} and state.delivered_payloads == {}
    assert state.delivered_n_coords == {}
    assert np.all(state.status == 0)
    assert state.n_resolved == previous_resolved_count
    assert gateway.stats.accepted_data == accepted_before == 1


def test_round_start_cannot_precede_completed_round_deadline() -> None:
    gateway, state = make_gateway(dependencies=False)
    gateway.on_deadline(1000)
    previous_round = gateway.round
    previous_start = state.round_start_ns

    with pytest.raises(GatewayError):
        gateway.on_round_start(999, round_id=8, deadline_ns=2000, K=3, L=2)

    assert gateway.round is previous_round
    assert gateway.phase is GatewayPhase.DEADLINE
    assert state.round_start_ns == previous_start and state.m == 0
    assert gateway.last_round_id == 7


def test_round_start_prevalidates_all_client_timestamps_before_mutation() -> None:
    first = ClientState(1, 0.5, K=3)
    second = ClientState(2, 0.5, K=3)
    second.reset_round(3, round_start_ns=500)
    estimator = FakeEstimator()
    gateway = GatewayCore({1: first, 2: second}, estimator=estimator)

    with pytest.raises(GatewayError):
        gateway.on_round_start(100, round_id=1, deadline_ns=1000, K=3, L=2)

    assert first.round_start_ns is None and first.K == 3 and first.m == 0
    assert second.round_start_ns == 500
    assert gateway.phase is GatewayPhase.IDLE and gateway.round is None
    assert gateway.last_round_id is None and estimator.reset == []


def test_aggregate_failure_is_surfaced_and_deadline_cannot_be_replayed() -> None:
    class FailingAggregate:
        calls = 0

        def aggregate(self, clients):
            self.calls += 1
            raise RuntimeError("aggregation failed")

    aggregate = FailingAggregate()
    gateway, state = make_gateway()
    gateway.aggregate = aggregate
    ready(gateway)

    with pytest.raises(RuntimeError, match="aggregation failed"):
        gateway.on_deadline(1000)

    assert gateway.phase is GatewayPhase.DEADLINE
    assert state.outstanding == {} and state.status[0] == 3
    assert gateway.estimator.resolutions[0][2:] == (False, 1000)
    assert gateway.on_deadline(1001) == []
    assert gateway.on_tick(1000) == []
    assert aggregate.calls == 1


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


def test_non_finite_prediction_is_rejected_before_grant_state_changes() -> None:
    gateway, state = make_gateway(dependencies=True)
    gateway.prediction.p = float("nan")
    state.set_ready(np.asarray([4.0, 2.0, 0.0], dtype=np.float32))
    before_status = state.status.copy()

    with pytest.raises(GatewayError, match="prediction probability"):
        gateway._pump(200)

    np.testing.assert_array_equal(state.status, before_status)
    assert state.outstanding == {}
    assert gateway.next_grant_id == 1


@pytest.mark.parametrize("delivered_blocks", [(), (1,), (0, 1, 2)])
def test_aggregate_interface_receives_authoritative_payloads_and_counts(delivered_blocks) -> None:
    class AggregateSpy:
        def __init__(self):
            self.snapshot = None

        def aggregate(self, clients):
            self.snapshot = {
                client_id: (
                    {idx: payload.copy() for idx, payload in state.delivered_payloads.items()},
                    state.delivered_n_coords.copy(),
                )
                for client_id, state in clients.items()
            }
            return self.snapshot

    gateway, state = make_gateway(mode=TransferMode.PLAN, dependencies=False)
    spy = AggregateSpy()
    gateway.aggregate = spy
    for block_idx in delivered_blocks:
        count = 1 if block_idx == 2 else 2
        payload = [float(block_idx + 1), 0.0]
        gateway.on_data(1, DATA_GRANT_ID, block_idx + 1, block_idx, payload, 200, n_coords=count)

    actions = gateway.on_deadline(1000)
    emitted = next(action.aggregate for action in actions if isinstance(action, EmitAggregate))
    payloads, counts = emitted[1]
    assert set(payloads) == set(delivered_blocks)
    assert set(counts) == set(delivered_blocks)
    for block_idx in delivered_blocks:
        assert counts[block_idx] == (1 if block_idx == 2 else 2)
        np.testing.assert_array_equal(payloads[block_idx], [float(block_idx + 1), 0.0])


def test_invalid_round_configuration_is_rejected() -> None:
    gateway = GatewayCore()
    with pytest.raises(GatewayError):
        gateway.on_round_start(100, round_id=1, deadline_ns=100, K=3, L=2)

    with pytest.raises(GatewayError):
        gateway.on_round_start(
            100, round_id=0x1_0000_0000, deadline_ns=1000, K=3, L=2
        )


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


def test_wire_ready_uses_gateway_receive_timestamp_when_provided() -> None:
    gateway, state = make_gateway()
    packet = encode_message(
        MessageType.READY,
        round_id=7,
        client_id=1,
        t_send_ns=200,
        body={"K": 3, "energy": [4.0, 2.0, 0.0]},
    )

    assert gateway.on_ready_message(packet, received_at_ns=1001) == []
    assert state.ready is False and gateway.stats.rejected_ready == 1
    assert gateway.on_ready_message(packet, received_at_ns=200) == [SendGrant(1, 1, 1)]
    assert state.ready is True


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
    assert state.delivered_n_coords[0] == 2
    np.testing.assert_array_equal(state.delivered_payloads[0], np.asarray([1.0, 0.0], np.float32))


def test_duplicate_data_resolves_the_active_retransmission_grant() -> None:
    gateway, state = make_gateway()
    ready(gateway)
    assert gateway.on_tick(211) == [SendGrant(1, 2, 1)]

    # A delayed response to grant 1 delivers the block while grant 2 is outstanding.
    assert gateway.on_data(1, 1, 1, 0, [1.0, 0.0], 220) == []
    assert set(state.outstanding) == {2}

    # The retransmission response is a duplicate block, but it still resolves grant 2.
    assert gateway.on_data(1, 2, 1, 0, [1.0, 0.0], 220) == [SendGrant(1, 3, 2)]
    assert set(state.outstanding) == {3}
    assert gateway.stats.duplicate_data == 1
    assert gateway.stats.accepted_data == 1
    assert gateway.estimator.resolutions[-1][1].pos == 1
    assert gateway.estimator.resolutions[-1][2:] == (True, 220)


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
    assert state.delivered_n_coords[2] == 1
    np.testing.assert_array_equal(state.delivered_payloads[2], np.asarray([1.0, 0.0], np.float32))


def test_wire_data_uses_gateway_receive_timestamp_when_provided() -> None:
    gateway, state = make_gateway(mode=TransferMode.PUSH, dependencies=False)
    packet = encode_message(
        MessageType.DATA,
        round_id=7,
        client_id=1,
        t_send_ns=200,
        block_length=2,
        body={
            "grant_id": DATA_GRANT_ID,
            "send_pos": 1,
            "block_idx": 0,
            "n_coords": 2,
            "payload": [1.0, 0.0],
        },
    )

    assert gateway.on_data_message(packet, block_length=2, received_at_ns=1001) == []
    assert state.m == 0 and gateway.stats.late_after_deadline == 1
    assert gateway.on_data_message(packet, block_length=2, received_at_ns=1000) == []
    assert state.m == 1 and gateway.stats.accepted_data == 1


def test_direct_data_rejects_numpy_coercible_non_protocol_values() -> None:
    gateway, state = make_gateway(mode=TransferMode.PUSH, dependencies=False)

    assert gateway.on_data(1, DATA_GRANT_ID, 1, 0, ["1.0", "0.0"], 200) == []
    assert gateway.on_data(1, DATA_GRANT_ID, 1, 0, [True, False], 200) == []
    assert gateway.on_data(1, DATA_GRANT_ID, 1, 0, [1.0, 0.0], 200, n_coords=0) == []
    assert state.m == 0
    assert gateway.stats.rejected_data == 3


def test_data_message_does_not_treat_explicit_invalid_block_length_as_default() -> None:
    gateway, state = make_gateway(mode=TransferMode.PUSH, dependencies=False)
    packet = encode_message(
        MessageType.DATA,
        round_id=7,
        client_id=1,
        t_send_ns=200,
        block_length=2,
        body={
            "grant_id": DATA_GRANT_ID,
            "send_pos": 1,
            "block_idx": 0,
            "n_coords": 2,
            "payload": [1.0, 0.0],
        },
    )

    assert gateway.on_data_message(packet, block_length=0) == []
    assert state.m == 0
    assert gateway.stats.rejected_data == 1


# -- stale-round DATA accounting -------------------------------------------------


def wire_data(round_id: int, *, grant_id: int = 1, send_pos: int = 1, block_idx: int = 0) -> bytes:
    return encode_message(
        MessageType.DATA,
        round_id=round_id,
        client_id=1,
        t_send_ns=0,
        block_length=2,
        body={
            "grant_id": grant_id,
            "send_pos": send_pos,
            "block_idx": block_idx,
            "n_coords": 2,
            "payload": [1.0, 0.0],
        },
    )


def round_snapshot(gateway, state) -> tuple:
    return (
        gateway.round,
        gateway.phase,
        gateway.next_grant_id,
        dict(state.outstanding),
        state.status.copy().tolist(),
        state.m,
        dict(state.delivered_payloads),
        {cid: set(blocks) for cid, blocks in gateway.delivered_blocks.items()},
        {cid: dict(grants) for cid, grants in gateway.resolved_grants.items()},
        len(gateway.estimator.resolutions),
        gateway.stats.accepted_data,
        gateway.stats.late_after_deadline,
    )


def start_round_8_with_grant_outstanding():
    gateway, state = make_gateway()
    ready(gateway)
    gateway.on_deadline(1000)
    gateway.on_round_start(1100, round_id=8, deadline_ns=2000, K=3, L=2)
    assert gateway.on_ready(1, [4.0, 2.0, 0.0], 1200) == [SendGrant(1, 1, 1)]
    return gateway, state


def test_previous_round_data_is_stale_and_mutates_nothing() -> None:
    gateway, state = start_round_8_with_grant_outstanding()
    before = round_snapshot(gateway, state)
    stats_before = (gateway.stats.rejected_data, gateway.stats.invalid_events)

    # Same grant id and position as the live round-8 grant: only round_id differs.
    assert gateway.on_data_message(wire_data(7), block_length=2, received_at_ns=1205) == []

    assert gateway.stats.stale_round_data == 1
    assert (gateway.stats.rejected_data, gateway.stats.invalid_events) == (
        stats_before[0] + 1,
        stats_before[1] + 1,
    )
    assert gateway.stats.late_after_deadline == 0
    assert round_snapshot(gateway, state) == before

    # The live grant was not consumed: the real round-8 DATA is still accepted.
    gateway.on_data_message(wire_data(8), block_length=2, received_at_ns=1205)
    assert gateway.stats.accepted_data == 1 and state.m == 1
    assert gateway.stats.stale_round_data == 1
    actions = gateway.on_deadline(2000)
    assert EmitAggregate(((1, 1),)) in actions  # only the valid round-8 block


def test_future_and_unrelated_round_data_count_once_per_packet() -> None:
    gateway, state = start_round_8_with_grant_outstanding()
    before = round_snapshot(gateway, state)
    aggregate_calls = gateway.aggregate.calls  # round 7's deadline already aggregated once

    for round_id in (9, 1234, 1):  # future, unrelated, much older
        assert gateway.on_data_message(wire_data(round_id), block_length=2, received_at_ns=1205) == []
    stale = wire_data(7)
    for _ in range(3):  # repeated copies are separate packets
        gateway.on_data_message(stale, block_length=2, received_at_ns=1205)

    assert gateway.stats.stale_round_data == 6
    assert gateway.stats.rejected_data == 6 and gateway.stats.invalid_events == 6
    assert gateway.stats.late_after_deadline == 0
    assert round_snapshot(gateway, state) == before
    assert gateway.aggregate.calls == aggregate_calls


def test_stale_data_from_unknown_client_is_counted_as_bad_client_only() -> None:
    gateway, _ = start_round_8_with_grant_outstanding()
    packet = encode_message(
        MessageType.DATA,
        round_id=7,
        client_id=99,
        t_send_ns=0,
        block_length=2,
        body={"grant_id": 1, "send_pos": 1, "block_idx": 0, "n_coords": 2, "payload": [1.0, 0.0]},
    )
    assert gateway.on_data_message(packet, block_length=2, received_at_ns=1205) == []
    assert gateway.stats.bad_client_id == 1 and gateway.stats.stale_round_data == 0
    assert gateway.stats.rejected_data == 1


def test_data_before_any_round_started_is_stale() -> None:
    # No round has started, so no round_id is current (the existing contract
    # rejects it in the same branch as a mismatched round).
    state = ClientState(1, 0.5, K=3)
    gateway = GatewayCore({1: state}, estimator=FakeEstimator(), aggregate=FakeAggregate())
    assert gateway.round is None and gateway.phase is GatewayPhase.IDLE

    assert gateway.on_data_message(wire_data(1), block_length=2, received_at_ns=50) == []

    assert gateway.stats.stale_round_data == 1 and gateway.stats.rejected_data == 1
    assert gateway.stats.late_after_deadline == 0
    assert gateway.round is None and gateway.phase is GatewayPhase.IDLE
    assert state.m == 0 and state.outstanding == {} and gateway.estimator.resolutions == []


def test_same_round_data_in_drain_is_late_not_stale() -> None:
    gateway, state = make_gateway()
    ready(gateway)
    gateway.on_deadline(1000)
    assert gateway.phase is GatewayPhase.DEADLINE
    before = round_snapshot(gateway, state)[:10]

    assert gateway.on_data_message(wire_data(7), block_length=2, received_at_ns=1050) == []

    assert gateway.stats.late_after_deadline == 1
    assert gateway.stats.stale_round_data == 0
    assert gateway.stats.rejected_data == 0
    assert round_snapshot(gateway, state)[:10] == before


def test_stale_and_late_counters_are_cumulative_across_round_transitions() -> None:
    gateway, state = make_gateway()  # round 7
    ready(gateway)
    gateway.on_deadline(1000)
    gateway.on_data_message(wire_data(7), block_length=2, received_at_ns=1050)  # late, drain of 7

    gateway.on_round_start(1100, round_id=8, deadline_ns=2000, K=3, L=2)
    gateway.on_data_message(wire_data(7), block_length=2, received_at_ns=1150)  # stale in 8
    gateway.on_deadline(2000)
    gateway.on_data_message(wire_data(8), block_length=2, received_at_ns=2050)  # late, drain of 8
    gateway.on_data_message(wire_data(9), block_length=2, received_at_ns=2060)  # future, stale

    gateway.on_round_start(2100, round_id=9, deadline_ns=3000, K=3, L=2)
    assert (gateway.stats.stale_round_data, gateway.stats.late_after_deadline) == (2, 2)  # not reset
    gateway.on_data_message(wire_data(7), block_length=2, received_at_ns=2150)
    gateway.on_data_message(wire_data(8), block_length=2, received_at_ns=2160)

    assert gateway.stats.stale_round_data == 4
    assert gateway.stats.late_after_deadline == 2
    assert gateway.stats.rejected_data == 4
    assert gateway.stats.accepted_data == 0 and state.m == 0


def test_grant_timed_out_before_deadline_is_not_resolved_again_at_deadline() -> None:
    gateway, state = make_gateway()
    ready(gateway)
    assert gateway.on_tick(211) == [SendGrant(1, 2, 1)]  # grant 1 timed out; retransmit as grant 2
    timed_out = [r for r in gateway.estimator.resolutions if r[1].pos == 1 and r[2] is False]
    assert len(timed_out) == 1 and 1 not in state.outstanding

    gateway.on_deadline(1000)

    grant_1 = gateway.resolved_grants[1][1]
    assert [r for r in gateway.estimator.resolutions if r[1] is grant_1] == [timed_out[0]]
    assert timed_out[0][3] == 211  # resolved at its timeout, not at D
    assert state.outstanding == {}
