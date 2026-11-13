"""Cross-round regressions for the GatewayCore-backed network simulator."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from fedgrant.ml.schema import SplitRecorder
from fedgrant.policy.types import PolicyConfig
from fedgrant.sim.run_net import (
    GatewaySim,
    _normalize_decision_event,
    compute_deadline_ns,
    run_network_round,
)
from fedgrant.sim.seeding import network_round_seed


def _run_round_with_recorder(recorder, run_id, round_id, seed, start_ns, gateway):
    cfg = PolicyConfig(selector="value", predictor="ewma")
    handles: dict[int, int] = {}
    events: list[dict] = []

    def sample_hook(entry: dict) -> None:
        events.append(entry)
        assert entry["abs_t_ns"] == start_ns + entry["t_ns"]
        grant_id = int(entry["grant_id"])
        if entry["event"] == "grant_issued":
            handles[grant_id] = recorder.add_sample(
                run_id, int(entry["client_id"]), int(entry["abs_t_ns"]), entry["x48"],
                float(entry["p_ewma"]), float(entry["p_hmm"]), float(entry["p_oracle"]),
            )
        else:
            handle = handles.pop(grant_id)
            recorder.add_event(
                handle, int(entry["y"]), int(entry["abs_t_ns"]), entry["obs"]
            )

    result = run_network_round(
        round_id=round_id,
        num_clients=20,
        beta=0.10,
        policy_config=cfg,
        seed=seed,
        sample_hook=sample_hook,
        gateway=gateway,
        round_start_abs_ns=start_ns,
    )
    assert not handles
    return result, events


def test_round_two_rtt_and_split_recorder_timestamps_are_run_absolute():
    run_id = 1001
    master_seed = 73
    deadline_ns = compute_deadline_ns(beta=0.10, num_clients=20)
    second_start_ns = 1_000_000_000 + deadline_ns
    recorder = SplitRecorder("train", generator="cross_round_test")
    recorder.begin_run(
        run_id, run_id, burst=8, pi=0.05, beta=0.10,
        D_ns=deadline_ns, capacity="constant",
    )

    first, first_events = _run_round_with_recorder(
        recorder, run_id, 1, network_round_seed(master_seed, 1), 0, None
    )
    second, second_events = _run_round_with_recorder(
        recorder, run_id, 2, network_round_seed(master_seed, 2),
        second_start_ns, first["gateway"],
    )

    second_success_rtts = tuple(
        int(event["rtt_ns"])
        for event in second_events
        if event["event"] == "grant_resolved" and event["rtt_ns"] is not None
    )
    issue_times = {
        (int(event["client_id"]), int(event["grant_id"])): int(event["abs_t_ns"])
        for event in second_events if event["event"] == "grant_issued"
    }
    for event in second_events:
        if event["event"] == "grant_resolved" and event["rtt_ns"] is not None:
            assert event["rtt_ns"] == event["abs_t_ns"] - issue_times[
                (int(event["client_id"]), int(event["grant_id"]))
            ]
    assert second["rtt_samples_ns"]
    assert all(sample >= 0 for sample in second["rtt_samples_ns"])
    assert second["rtt_samples_ns"] == second_success_rtts
    samples_ms = np.asarray(second_success_rtts, dtype=np.float64) / 1e6
    assert second["metrics"]["rtt_median_ms"] == pytest.approx(float(np.median(samples_ms)))
    assert second["metrics"]["rtt_p99_ms"] == pytest.approx(float(np.percentile(samples_ms, 99)))

    for events, start_ns in ((first_events, 0), (second_events, second_start_ns)):
        for event in events:
            assert event["abs_t_ns"] == start_ns + event["t_ns"]
    recorder.end_run(run_id)
    arrays = recorder.arrays()
    assert np.any(arrays["sample_t_ns"] >= second_start_ns)
    assert np.any(arrays["event_t_ns"] >= second_start_ns)
    assert np.all(np.diff(arrays["sample_t_ns"]) >= 0)


def test_decision_hooks_normalize_timestamps_across_rounds_without_changing_core_time(monkeypatch):
    config = PolicyConfig(selector="value", predictor="ewma")
    master_seed = 287
    deadline_ns = compute_deadline_ns(beta=0.10, num_clients=20)
    round_starts = {1: 0, 2: 1_000_000_000 + deadline_ns}
    core_events: list[dict] = []
    original_core_hook = GatewaySim._decision_hook

    def capture_core_event(self, event):
        core_events.append(dict(event))
        original_core_hook(self, event)

    monkeypatch.setattr(GatewaySim, "_decision_hook", capture_core_event)

    def execute_round(round_id, gateway=None):
        decisions: list[dict] = []
        events: list[dict] = []
        result = run_network_round(
            round_id=round_id,
            num_clients=20,
            policy_config=config,
            seed=network_round_seed(master_seed, round_id),
            gateway=gateway,
            round_start_abs_ns=round_starts[round_id],
            decision_hook=decisions.append,
            event_hook=events.append,
        )
        return result, decisions, events

    first, first_decisions, first_events = execute_round(1)
    second, second_decisions, second_events = execute_round(2, first["gateway"])

    all_decisions = first_decisions + second_decisions
    all_events = first_events + second_events
    forwarded_decisions = [event for event in all_events if event["event"] == "grant_decision"]
    assert forwarded_decisions == all_decisions
    assert len(core_events) == len(all_decisions)
    for internal, external in zip(core_events, all_decisions, strict=True):
        start_ns = round_starts[external["round_id"]]
        assert internal["t_ns"] == external["abs_t_ns"]
        assert external["t_ns"] == internal["t_ns"] - start_ns
        assert external["abs_t_ns"] == start_ns + external["t_ns"]

    for round_id, round_events in ((1, first_events), (2, second_events)):
        start_ns = round_starts[round_id]
        for event in round_events:
            assert event["round_id"] == round_id
            assert event["abs_t_ns"] == start_ns + event["t_ns"]
        kinds = {event["event"] for event in round_events}
        assert {"ready_received", "data_received", "tick", "grant_decision"} <= kinds

    # A fresh replay of round 1 remains byte-for-byte deterministic at both
    # external hook boundaries after adding the timestamp normalization.
    _, replay_decisions, replay_events = execute_round(1)
    assert replay_decisions == first_decisions
    assert replay_events == first_events


def test_decision_event_normalization_preserves_existing_absolute_timestamp():
    start_ns = 1_398_000_000
    raw = {"event": "grant_decision", "t_ns": start_ns + 17, "grant_id": 9}
    normalized = _normalize_decision_event(raw, start_ns)
    assert normalized == {
        "event": "grant_decision", "t_ns": 17,
        "abs_t_ns": start_ns + 17, "grant_id": 9,
    }
    assert raw == {"event": "grant_decision", "t_ns": start_ns + 17, "grant_id": 9}

    already_normalized = {
        "event": "grant_decision", "t_ns": 17,
        "abs_t_ns": start_ns + 17, "grant_id": 9,
    }
    copied = _normalize_decision_event(already_normalized, start_ns)
    assert copied == already_normalized
    assert copied is not already_normalized


@pytest.mark.parametrize(
    ("changed", "field"),
    [
        (PolicyConfig(selector="srpt", predictor="ewma"), "selector"),
        (PolicyConfig(selector="value", predictor="ewma", retransmit=False), "retransmit"),
    ],
)
def test_gateway_reuse_rejects_any_policy_config_change_before_round_mutation(changed, field):
    config = PolicyConfig(selector="value", predictor="ewma")
    first = run_network_round(num_clients=20, policy_config=config, seed=91)
    gateway = first["gateway"]
    old_round = gateway.core.round
    old_state = gateway.client_states[0].status.copy()
    old_selector = gateway.selector
    old_pointer = gateway.rr_pointer

    with pytest.raises(ValueError, match=field):
        run_network_round(
            round_id=2,
            num_clients=20,
            policy_config=changed,
            seed=92,
            gateway=gateway,
            round_start_abs_ns=1_398_000_000,
        )

    assert gateway.core.round is old_round
    np.testing.assert_array_equal(gateway.client_states[0].status, old_state)
    assert gateway.selector is old_selector
    assert gateway.rr_pointer == old_pointer


def test_gateway_reuse_preserves_rr_pointer_predictor_and_client_history(monkeypatch):
    config = PolicyConfig(selector="rr", predictor="ewma")
    first = run_network_round(num_clients=20, policy_config=config, seed=101)
    gateway = first["gateway"]
    original_selector = gateway.selector
    original_predictor = gateway.predictor
    gateway.rr_pointer = 7
    gateway.predictor.observe(0, 1, 0, np.zeros(8, dtype=np.float32))
    persisted_prediction = gateway.predictor.value(0)
    state = gateway.client_states[0]
    state.srtt_ns = 987_654
    state.hist.clear()
    state.hist.extend((1, 0))

    reset_snapshots: list[dict] = []
    original_reset = GatewaySim.reset_round

    def capture_reset(self, ctx):
        original_reset(self, ctx)
        reset_snapshots.append({
            "rr_pointer": self.rr_pointer,
            "predictor_value": self.predictor.value(0),
            "srtt_ns": self.client_states[0].srtt_ns,
            "hist": tuple(self.client_states[0].hist),
            "round_counters": (self.grants_issued, self.grants_delivered, self.grants_timed_out),
        })

    monkeypatch.setattr(GatewaySim, "reset_round", capture_reset)
    decisions: list[dict] = []
    second = run_network_round(
        round_id=2,
        num_clients=20,
        policy_config=config,
        seed=102,
        gateway=gateway,
        round_start_abs_ns=1_398_000_000,
        decision_hook=decisions.append,
    )

    assert reset_snapshots
    snapshot = reset_snapshots[0]
    assert snapshot["rr_pointer"] == 7
    assert snapshot["predictor_value"] == pytest.approx(persisted_prediction)
    assert snapshot["srtt_ns"] == 987_654
    assert snapshot["hist"] == (1, 0)
    assert snapshot["round_counters"] == (0, 0, 0)
    assert gateway.selector is original_selector
    assert gateway.predictor is original_predictor
    assert second["metrics"]["grants_issued"] == gateway.grants_issued
    assert GatewaySim(1, np.ones(1, dtype=np.float32), gateway.predictor,
                      selector="rr", policy_config=config).rr_pointer == 0


def test_exploration_rng_is_derived_per_round_and_replaced_on_gateway_reuse(monkeypatch):
    master_seed = 876
    config = PolicyConfig(selector="rr", predictor="ewma", explore_eps=1.0)
    captured_states: list[dict] = []
    original_start = GatewaySim.on_round_start

    def capture_rng_state(self, *args, **kwargs):
        captured_states.append(copy.deepcopy(self.rng.bit_generator.state))
        return original_start(self, *args, **kwargs)

    monkeypatch.setattr(GatewaySim, "on_round_start", capture_rng_state)

    def choices(seed: int) -> list[tuple]:
        records: list[dict] = []
        run_network_round(
            num_clients=20, policy_config=config, seed=seed, decision_hook=records.append
        )
        return [
            (event["t_ns"], event["candidate_client_ids"], event["client_id"], event["outcome"])
            for event in records if event["event"] == "grant_decision"
        ]

    round1_seed = network_round_seed(master_seed, 1)
    round2_seed = network_round_seed(master_seed, 2)
    assert choices(round1_seed) == choices(round1_seed)

    round2_start_ns = 1_000_000_000 + compute_deadline_ns(beta=0.10, num_clients=20)
    fresh_round2 = run_network_round(
        round_id=2, num_clients=20, policy_config=config, seed=round2_seed,
        round_start_abs_ns=round2_start_ns,
    )
    fresh_round2_state = captured_states[-1]
    expected_round2_state = np.random.default_rng(
        np.random.SeedSequence(round2_seed).spawn(6)[3]
    ).bit_generator.state
    assert fresh_round2_state == expected_round2_state
    fresh_round1 = run_network_round(
        num_clients=20, policy_config=config, seed=round1_seed
    )
    fresh_round1_state = captured_states[-1]
    assert fresh_round1_state != fresh_round2_state

    # Simulate arbitrary extra draws consumed during the preceding round. The
    # next run must replace this generator with round 2's child stream.
    fresh_round1["gateway"].rng.random(10_000)
    run_network_round(
        round_id=2, num_clients=20, policy_config=config, seed=round2_seed,
        gateway=fresh_round1["gateway"], round_start_abs_ns=round2_start_ns,
    )
    assert captured_states[-1] == fresh_round2_state
