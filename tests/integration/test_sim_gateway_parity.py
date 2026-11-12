"""Deterministic, observable simulator replay hooks for gateway parity work."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from fedgrant.policy.types import PolicyConfig
from fedgrant.core.estimators import RateWindow, grant_timeout
from fedgrant.core.gateway import GatewayCore, SendGrant
from fedgrant.core.state import ClientState, values_from_energies
from fedgrant.policy.eligibility import is_eligible
from fedgrant.policy.features import (
    ArrivalLog,
    GatewayView,
    build_pred_input,
    features48,
    resolve_event,
)
from fedgrant.policy.pacing import InflightTracker, p_target, pacing_p
from fedgrant.policy.selectors import make_selector, value_key
from fedgrant.policy.types import CandidateView
from fedgrant.predictors.ewma import PEWMA
from fedgrant.policy.constants import K, L, S
from fedgrant.sim.run_net import compute_deadline_ns, run_network_round


def _run(
    seed: int,
    *,
    num_clients: int = 20,
    selector: str = "value",
    scenario: str = "L2",
):
    sample_records: list[dict] = []
    decision_records: list[dict] = []
    observed_events: list[dict] = []
    # Match fedgrant.fl.server._network_round_seed(master_seed, round_id).
    round_seed = int(np.random.SeedSequence([seed, 0]).generate_state(1)[0])
    weights = np.full(num_clients, 1.0 / num_clients, dtype=np.float32)
    energies = np.tile(np.linspace(1.0, 0.01, K, dtype=np.float32), (num_clients, 1))
    result = run_network_round(
        round_id=1,
        num_clients=num_clients,
        client_weights=weights,
        client_energies=energies,
        beta=0.10,
        scenario=scenario,
        policy_config=PolicyConfig(selector=selector, predictor="ewma"),
        seed=round_seed,
        sample_hook=sample_records.append,
        decision_hook=decision_records.append,
        event_hook=observed_events.append,
    )
    return result, sample_records, decision_records, observed_events


class _ReplayAssembly:
    """Test-only provider assembly around the real GatewayCore and shared policies."""

    def __init__(self, num_clients: int, weights: np.ndarray, deadline_ns: int):
        self.cfg = PolicyConfig(selector="srpt", predictor="ewma")
        self.states = {
            cid: ClientState(cid, float(weights[cid]), K=K) for cid in range(num_clients)
        }
        self.inflight = InflightTracker()
        self.arrivals = ArrivalLog()
        self.view = GatewayView(0, deadline_ns, 1.0, self.inflight, self.arrivals)
        self.predictor = _RecordingEWMA()
        self.prediction = _ReplayPrediction(self)
        self.rate_windows = {cid: RateWindow(0, 0) for cid in range(num_clients)}
        self.decisions: list[dict] = []
        self.pacing_inputs: list[tuple[int, float, float, float, bool]] = []
        self.resolutions: list[dict] = []
        self.selected_inputs: list[np.ndarray] = []
        self.gateway: GatewayCore
        self.gateway = GatewayCore(
            self.states,
            selector=make_selector("srpt"),
            eligibility=self,
            pacing=self,
            timeout=self,
            estimator=self,
            prediction=self.prediction,
            view=self.view,
            decision_hook=self._decision,
        )

    def _target(self) -> float:
        ready = [state.srtt_ns for state in self.states.values() if state.ready]
        return p_target(ready) if ready else p_target([state.srtt_ns for state in self.states.values()])

    def _decision(self, record: dict) -> None:
        self.decisions.append(record)
        self.selected_inputs.append(self.predictor.inputs[-1][2].copy())

    def predict_state(self, state: ClientState, now_ns: int) -> float:
        self.view.P_target = self._target()
        feature_view = copy.copy(self.view)
        feature_view.t_ns = now_ns - self.gateway.round.start_ns
        feature_view.D_ns = self.gateway.round.deadline_ns - self.gateway.round.start_ns
        prediction_input = build_pred_input([state], feature_view)
        return float(self.predictor.predict(prediction_input)[0])

    def candidates(self, clients, now_ns, deadline_ns, retransmit):
        self.view.P_target = self._target()
        ids, scores, remaining = [], [], []
        for cid, state in sorted(clients.items()):
            p_hat = self.predict_state(state, now_ns)
            eligible, _ = is_eligible(state, now_ns, deadline_ns, p_hat, self.cfg)
            if eligible:
                ids.append(cid)
                scores.append(value_key(state, retransmit))
                remaining.append(state.K - state.m - len(state.outstanding))
        return CandidateView(
            np.asarray(ids, dtype=np.int32),
            np.asarray(scores, dtype=np.float64),
            np.asarray(remaining, dtype=np.int32),
            0,
        )

    def allow(self, client_id, p_hat, now_ns, deadline_ns):
        self.view.P_target = self._target()
        p_issue = pacing_p(p_hat, self.cfg)
        allowed = self.inflight.can_issue(p_issue, self.view.P_target)
        self.pacing_inputs.append((client_id, p_hat, p_issue, self.view.P_target, allowed))
        return allowed

    def issue_probability(self, client_id, p_hat):
        # GatewayCore records this value in view.inflight when it issues the grant.
        return pacing_p(p_hat, self.cfg)

    def data_arrival(self, client_id, t_ns):
        window = self.rate_windows[client_id]
        window.advance(self.states[client_id], t_ns)
        window.add_delivery(t_ns, 8 * S)

    def grant_issued(self, client_id, grant, t_ns):
        state = self.states[client_id]
        self.rate_windows[client_id].set_outstanding(t_ns, len(state.outstanding))

    def timeout_ns(self, client_id, issue_ns):
        return grant_timeout(issue_ns, self.states[client_id], self.gateway.round.deadline_ns)

    def reset_round(self, ctx):
        self.inflight.clear()
        self.arrivals.reset()
        self.predictor.reset_round(ctx)
        for window in self.rate_windows.values():
            window.start(0, 0)

    def resolve(self, client_id, grant, delivered, t_ns):
        state = self.states[client_id]
        window = self.rate_windows[client_id]
        window.advance(state, t_ns)
        # GatewayCore calls resolve() before removing the grant (same contract
        # as sim/run_net.py), so recover its id from the client's outstanding map;
        # resolve_event then performs the authoritative Phi/state transition.
        grant_id = next(
            (gid for gid, value in state.outstanding.items() if value is grant), None
        )
        if grant_id is None:
            raise KeyError("GatewayCore estimator callback ran after grant removal")
        self.view.t_ns = t_ns
        self.view.P_target = self._target()
        rtt_ns = t_ns - grant.t_issue_ns if delivered else None
        obs = resolve_event(
            state, self.view, grant_id, int(delivered), t_ns, rtt_ns
        )
        self.rate_windows[client_id].set_outstanding(t_ns, len(state.outstanding))
        self.predictor.observe(client_id, int(delivered), t_ns, obs)
        self.resolutions.append({
            "client_id": client_id,
            "grant_id": grant_id,
            "y": int(delivered),
            "t_ns": t_ns,
            "obs": obs.copy(),
            "phi": self.view.Phi,
            "p_target": self.view.P_target,
        })

    def on_tick(self, t_ns):
        for cid, state in self.states.items():
            self.rate_windows[cid].advance(state, t_ns)

    def on_deadline(self, t_ns):
        self.view.t_ns = t_ns

    def replay(self, events: list[dict]) -> list[dict]:
        """Replay only gateway-visible inputs, in the captured event order."""
        inputs: list[dict] = []
        for event in events:
            kind = event["event"]
            if kind == "ready_received":
                inputs.append(event)
                self.gateway.on_ready(event["client_id"], event["energies"], event["t_ns"])
            elif kind == "data_received":
                inputs.append(event)
                t_ns = event["t_ns"]
                if t_ns <= self.gateway.round.deadline_ns:
                    self.view.t_ns = t_ns
                # GatewayCore records the arrival in view.arrivals and calls
                # data_arrival() for the rate window (same as sim/run_net.py).
                self.gateway.on_data(
                    event["client_id"], event["grant_id"], event["send_pos"],
                    event["block_idx"], event["payload"], t_ns,
                    n_coords=event["n_coords"],
                )
            elif kind == "tick":
                inputs.append(event)
                self.gateway.on_tick(event["t_ns"])
            elif kind == "deadline":
                inputs.append(event)
                self.gateway.on_deadline(event["t_ns"])
        return inputs


class _RecordingEWMA:
    def __init__(self):
        self.base = PEWMA()
        self.inputs: list[tuple[tuple[int, ...], int, np.ndarray]] = []

    def reset_round(self, ctx):
        self.base.reset_round(ctx)

    def observe(self, client, y, t_ns, obs):
        self.base.observe(client, y, t_ns, obs)

    def predict(self, inp):
        self.inputs.extend((tuple(int(cid) for cid in inp.client_ids), inp.t_ns, x.copy())
                           for x in inp.x48)
        return self.base.predict(inp)


class _ReplayPrediction:
    """Deterministic test-only prediction adapter for GatewayCore's provider API."""

    def __init__(self, assembly: _ReplayAssembly):
        self.assembly = assembly

    def probability(self, client_id, now_ns):
        return self.assembly.predict_state(
            self.assembly.states[client_id], now_ns
        )


def test_simulator_replay_is_deterministic_and_reports_only_gateway_observables():
    first = _run(481)
    second = _run(481)
    a_result, a_samples, a_decisions, a_events = first
    b_result, b_samples, b_decisions, b_events = second

    assert a_result["delivered_mask"] == b_result["delivered_mask"]
    assert a_decisions == b_decisions
    assert a_events == b_events
    assert len(a_samples) == len(b_samples)
    for left, right in zip(a_samples, b_samples, strict=True):
        assert left.keys() == right.keys()
        for key in left:
            if isinstance(left[key], np.ndarray):
                np.testing.assert_array_equal(left[key], right[key])
            else:
                assert left[key] == right[key]

    issued = [event for event in a_decisions if event["outcome"] == "grant_issued"]
    arrived = [event for event in a_events if event["event"] == "data_received"]
    assert issued and arrived
    assert all("loss" not in key.lower() and "ge_" not in key.lower()
               for event in a_events for key in event)
    event_fields = {
        "event", "round_id", "client_id", "grant_id", "send_pos", "block_idx",
        "n_coords", "payload", "t_ns", "abs_t_ns", "energies", "deadline_ns",
        "selection_method", "candidate_client_ids", "selected_client_id", "p_hat",
        "p_pacing", "p_target", "timeout_ns", "outcome",
        # GatewayCore decision_hook fields forwarded into the event stream.
        "candidate_next_values", "candidate_undelivered", "rr_pointer", "pacing_allowed",
    }
    assert all(set(event).issubset(event_fields) for event in a_events)
    assert all("y" not in event for event in a_events)
    assert all("x48" in event and event["x48"].shape == (48,)
               for event in a_samples if event["event"] == "grant_issued")


def test_event_hook_data_means_frame_reached_gateway():
    # Input to the hook is at the simulator's gateway-arrival callback, not at
    # enqueue/transmit time. Thus simulator-private link drops never become a
    # GatewayCore input or an observed DATA event.
    result, _, _, events = _run(702, scenario="L1")
    data = [event for event in events if event["event"] == "data_received"]
    assert data
    assert all(event["grant_id"] > 0 and event["send_pos"] > 0 for event in data)
    assert all(0 < event["n_coords"] <= 700 for event in data)
    assert result["metrics"]["drops_ge"] > 0
    assert len(data) < result["metrics"]["grants_issued"]


def test_gatewaycore_replays_observable_events_with_scheduling_and_resolution_parity():
    # Keep this paired parity claim inside the common, stable pre-deadline
    # interval. A separate later-round mismatch is recorded in PROJECT_MAP.
    replay_through_ns = 300_000_000
    num_clients = 20
    weights = np.full(num_clients, 1.0 / num_clients, dtype=np.float32)
    result, samples, sim_decisions, events = _run(
        1934, num_clients=num_clients, selector="srpt"
    )
    deadline_ns = int(result["deadline_ms"] * 1_000_000)
    replay = _ReplayAssembly(num_clients, weights, deadline_ns)
    replay.gateway.on_round_start(
        0, round_id=1, deadline_ns=deadline_ns, K=K, L=L
    )

    replay_events = [event for event in events
                     if event.get("t_ns", 0) <= replay_through_ns]
    replay_inputs = replay.replay(replay_events)
    sim_inputs = [event for event in replay_events if event["event"] in {
        "ready_received", "data_received", "tick",
    }]
    assert replay_inputs == sim_inputs
    assert all(set(event).isdisjoint({"ge_state", "loss_state", "future_outcome"})
               for event in replay_inputs)

    sim_resolutions = [event for event in samples
                       if event["event"] == "grant_resolved"
                       and event["t_ns"] <= replay_through_ns]
    for index, (sim, core) in enumerate(zip(sim_resolutions, replay.resolutions)):
        assert (sim["client_id"], sim["grant_id"], sim["y"], sim["t_ns"]) == (
            core["client_id"], core["grant_id"], core["y"], core["t_ns"]
        ), (index, sim, core)
        assert np.array_equal(sim["obs"], core["obs"]), (
            index, (sim["client_id"], sim["grant_id"], sim["t_ns"]),
            sim["obs"], core["obs"], core["phi"], core["p_target"],
        )

    sim_decisions = [event for event in sim_decisions if event["t_ns"] <= replay_through_ns]
    assert len(replay.decisions) == len(sim_decisions)
    for sim, core in zip(sim_decisions, replay.decisions, strict=True):
        assert sim["t_ns"] == core["t_ns"]
        assert sim["candidate_client_ids"] == core["candidate_client_ids"]
        assert sim["selected_client_id"] == core["client_id"]
        assert sim["outcome"] == core["outcome"]
        assert sim["p_hat"] == pytest.approx(core["p_hat"], abs=1e-7)
        if sim["grant_id"] is not None:
            assert sim["grant_id"] == core["grant_id"]
            assert sim["send_pos"] == core["send_pos"]
            assert sim["timeout_ns"] == core["timeout_ns"]

    assert len(replay.pacing_inputs) == len(sim_decisions)
    for sim, (client, p_hat, p_issue, target, allowed) in zip(
        sim_decisions, replay.pacing_inputs, strict=True
    ):
        assert sim["selected_client_id"] == client
        assert sim["p_hat"] == pytest.approx(p_hat, abs=1e-7)
        assert sim["p_pacing"] == pytest.approx(p_issue, abs=1e-12)
        assert sim["p_target"] == pytest.approx(target, abs=1e-12)
        assert allowed is (sim["outcome"] == "grant_issued")

    sim_issues = {
        (event["client_id"], event["grant_id"]): event
        for event in samples
        if event["event"] == "grant_issued" and event["t_ns"] <= replay_through_ns
    }
    core_issues = {
        (event["client_id"], event["grant_id"]): features
        for event, features in zip(replay.decisions, replay.selected_inputs, strict=True)
        if event.get("grant_id") is not None
    }
    assert sim_issues.keys() == core_issues.keys()
    for key in sim_issues:
        np.testing.assert_array_equal(sim_issues[key]["x48"], core_issues[key])

    # Both successful deliveries and losses should be exercised by this
    # deterministic network scenario.
    assert sim_resolutions
    assert {event["y"] for event in sim_resolutions} == {0, 1}


def test_gateway_predictor_uses_shared_features_before_position_mutation():
    class CapturePredictor:
        def __init__(self, state):
            self.inputs = []
            self.state = state
            self.statuses_during_predict = []

        def reset_round(self, ctx):
            pass

        def observe(self, client, y, t_ns, obs):
            pass

        def predict(self, inp):
            self.inputs.append(inp)
            self.statuses_during_predict.append(int(self.state.status[0]))
            return np.asarray([0.75], dtype=np.float32)

    class OneCandidate:
        def candidates(self, clients, now_ns, deadline_ns, retransmit):
            return CandidateView(
                np.asarray([1], dtype=np.int32),
                np.asarray([1.0], dtype=np.float64),
                np.asarray([3], dtype=np.int32),
                0,
            )

    class FirstSelector:
        def select(self, candidates):
            return int(candidates.client_id[0])

    class Permit:
        calls = 0

        def allow(self, client_id, p_hat, now_ns, deadline_ns):
            self.calls += 1
            return self.calls == 1

    class FixedTimeout:
        def timeout_ns(self, client_id, issue_ns):
            return issue_ns + 10

    state = ClientState(1, 0.5, K=3)
    capture = CapturePredictor(state)
    records: list[dict] = []
    view = GatewayView(t_ns=0, D_ns=900, P_target=1.0)
    gateway = GatewayCore(
        {1: state}, selector=FirstSelector(), eligibility=OneCandidate(),
        pacing=Permit(), timeout=FixedTimeout(), predictor=capture, view=view,
        decision_hook=records.append,
    )
    gateway.on_round_start(100, round_id=1, deadline_ns=1000, K=3, L=2)
    actions = gateway.on_ready(1, [4.0, 2.0, 0.0], 200)

    assert actions == [SendGrant(1, 1, 1)]
    assert capture.inputs and capture.inputs[0].t_ns == 100
    expected_state = ClientState(1, 0.5, K=3)
    expected_state.set_ready(values_from_energies(np.asarray([4, 2, 0], np.float32), 0.5))
    expected_view = GatewayView(t_ns=100, D_ns=900, P_target=1.0)
    np.testing.assert_array_equal(capture.inputs[0].x48[0], features48(expected_state, expected_view))
    assert capture.statuses_during_predict[0] == 0  # still UNSENT during prediction work
    assert records[0]["event"] == "grant_decision"
    assert records[0]["candidate_client_ids"] == (1,)
    assert records[0]["outcome"] == "grant_issued"
    assert records[0]["grant_id"] == 1 and records[0]["send_pos"] == 1
    assert records[1]["outcome"] == "pacing_blocked"
    assert state.status[0] == 1  # OUTSTANDING after the selected grant is committed
