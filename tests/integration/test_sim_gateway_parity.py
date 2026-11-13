"""Deterministic, observable simulator replay hooks for gateway parity work.

Ownership contract exercised here (``fedgrant/core/gateway.py``):

* ``GatewayCore._pump`` alone allocates grant IDs, takes the send position,
  inserts ``state.outstanding[gid]`` and inserts ``view.inflight`` (Phi), then
  calls ``EstimatorHooks.grant_issued``. Providers never insert grants.
* ``GatewayCore`` records every valid DATA arrival in ``view.arrivals`` and
  calls ``EstimatorHooks.data_arrival``; providers do not record arrivals.
* On delivery/timeout/deadline ``GatewayCore`` calls
  ``EstimatorHooks.resolve`` *before* it pops the grant; the shared
  ``resolve_event`` removes it from ``outstanding`` and Phi exactly once.

Observable ``event_hook`` contract (``fedgrant/sim/run_net.py``): gateway
inputs (``round_start``, ``ready_received``, ``data_received`` at gateway
arrival, ``tick``, ``deadline``) interleaved with GatewayCore's own
``grant_decision`` records. Deliberately excluded: Gilbert-Elliott/loss state,
link enqueue/drop and queue internals, future delivery outcomes, and the
learning records (``y``, ``obs``, ``x48``) that go to ``sample_hook`` only.
"""

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

_INPUT_FIELDS = {
    "round_start": {"event", "round_id", "t_ns", "abs_t_ns", "deadline_ns"},
    "ready_received": {"event", "round_id", "client_id", "t_ns", "abs_t_ns", "energies"},
    "data_received": {
        "event", "round_id", "client_id", "grant_id", "send_pos", "block_idx",
        "n_coords", "payload", "t_ns", "abs_t_ns",
    },
    "tick": {"event", "round_id", "t_ns", "abs_t_ns"},
    "deadline": {"event", "round_id", "t_ns", "abs_t_ns"},
}
# GatewayCore._pump decision record, plus the adapter's absolute timestamp.
_DECISION_FIELDS = {
    "event", "round_id", "t_ns", "abs_t_ns", "client_id", "candidate_client_ids",
    "candidate_next_values", "candidate_undelivered", "rr_pointer", "p_hat",
    "pacing_allowed", "outcome",
}
_GRANT_FIELDS = ("grant_id", "send_pos", "timeout_ns")
# GatewaySim._decision_hook keeps the simulator's public decision schema on top
# of GatewayCore's record; pacing-blocked decisions carry None grant fields.
_SIM_COMPAT_FIELDS = {"selection_method", "selected_client_id", "p_pacing", "p_target"}
_DECISION_FIELDS = _DECISION_FIELDS | _SIM_COMPAT_FIELDS | set(_GRANT_FIELDS)


def _round_seed(seed: int, round_id: int) -> int:
    # Match fedgrant.fl.server._network_round_seed(master_seed, round_id).
    return int(np.random.SeedSequence([seed, round_id - 1]).generate_state(1)[0])


def _run_rounds(seed: int, rounds: int, *, num_clients: int = 20, selector: str = "value",
                scenario: str = "L2"):
    """Run consecutive rounds on one reused GatewaySim, as the FL server does."""
    weights = np.full(num_clients, 1.0 / num_clients, dtype=np.float32)
    energies = np.tile(np.linspace(1.0, 0.01, K, dtype=np.float32), (num_clients, 1))
    gateway = None
    out = []
    for round_id in range(1, rounds + 1):
        samples: list[dict] = []
        decisions: list[dict] = []
        events: list[dict] = []
        result = run_network_round(
            round_id=round_id,
            num_clients=num_clients,
            client_weights=weights,
            client_energies=energies,
            beta=0.10,
            scenario=scenario,
            policy_config=PolicyConfig(selector=selector, predictor="ewma"),
            seed=_round_seed(seed, round_id),
            gateway=gateway,
            sample_hook=samples.append,
            decision_hook=decisions.append,
            event_hook=events.append,
        )
        gateway = result["gateway"]
        out.append((result, samples, decisions, events))
    return out


def _run(seed: int, *, num_clients: int = 20, selector: str = "value", scenario: str = "L2"):
    return _run_rounds(seed, 1, num_clients=num_clients, selector=selector, scenario=scenario)[0]


class _ReplayAssembly:
    """Independent test assembly of GatewayCore providers and the shared policies.

    It mirrors the production provider contract (see module docstring) but is
    fed only gateway-observable inputs, never simulator-private state.
    """

    def __init__(self, num_clients: int, weights: np.ndarray, deadline_ns: int):
        self.cfg = PolicyConfig(selector="srpt", predictor="ewma")
        assert self.cfg.explore_eps == 0.0  # selection needs no exploration RNG
        self.D_ns = deadline_ns
        self.states = {
            cid: ClientState(cid, float(weights[cid]), K=K) for cid in range(num_clients)
        }
        self.view = GatewayView(0, deadline_ns, 1.0, InflightTracker(), ArrivalLog())
        self.rate_windows = {cid: RateWindow(0, 0) for cid in range(num_clients)}
        self.decisions: list[dict] = []
        self.pacing_inputs: list[tuple[int, float, float, float, bool]] = []
        self.resolutions: list[dict] = []
        self.issued_x48: dict[tuple[int, int, int], np.ndarray] = {}
        self.gateway = GatewayCore(
            self.states,
            selector=make_selector("srpt"),
            eligibility=self,
            pacing=self,
            timeout=self,
            estimator=self,
            predictor=PEWMA(),
            view=self.view,
            decision_hook=self._decision,
            retransmit=self.cfg.retransmit,
        )

    # -- time/view helpers -------------------------------------------------
    @property
    def start_ns(self) -> int:
        return self.gateway.round.start_ns

    def _target(self) -> float:
        ready = [state.srtt_ns for state in self.states.values() if state.ready]
        return float(p_target(ready or [state.srtt_ns for state in self.states.values()]))

    def _set_time(self, absolute_ns: int) -> None:
        self.view.t_ns = absolute_ns - self.start_ns
        self.view.D_ns = self.D_ns
        self.view.P_target = self._target()

    # -- EligibilityProvider ------------------------------------------------
    def candidates(self, clients, now_ns, deadline_ns, retransmit):
        self._set_time(now_ns)
        ids, values, remaining = [], [], []
        for cid, state in clients.items():
            if not state.ready:
                continue
            p_hat = self.gateway._predict_probability(cid, now_ns)
            if not is_eligible(state, now_ns - self.start_ns, self.D_ns, p_hat, self.cfg)[0]:
                continue
            ids.append(cid)
            values.append(value_key(state, retransmit))
            remaining.append(state.K - state.m - len(state.outstanding))
        return CandidateView(
            np.asarray(ids, dtype=np.int32),
            np.asarray(values, dtype=np.float64),
            np.asarray(remaining, dtype=np.int32),
            0,
        )

    # -- PacingProvider: a pure check; GatewayCore inserts into Phi --------
    def allow(self, client_id, p_hat, now_ns, deadline_ns):
        self._set_time(now_ns)
        p_issue = pacing_p(p_hat, self.cfg)
        allowed = self.view.inflight.can_issue(p_issue, self.view.P_target)
        self.pacing_inputs.append((client_id, p_hat, p_issue, self.view.P_target, allowed))
        return allowed

    def issue_probability(self, client_id, p_hat):
        return pacing_p(p_hat, self.cfg)

    # -- TimeoutProvider ------------------------------------------------------
    def timeout_ns(self, client_id, issue_ns):
        return self.start_ns + grant_timeout(issue_ns - self.start_ns, self.states[client_id], self.D_ns)

    # -- EstimatorHooks -------------------------------------------------------
    def reset_round(self, ctx):
        for window in self.rate_windows.values():
            window.start(0, 0)

    def grant_issued(self, client_id, grant, t_ns):
        relative = t_ns - self.start_ns
        state = self.states[client_id]
        assert grant in state.outstanding.values()  # inserted by GatewayCore, once
        self.rate_windows[client_id].advance(state, relative)
        self.rate_windows[client_id].set_outstanding(relative, len(state.outstanding))

    def data_arrival(self, client_id, t_ns):
        relative = t_ns - self.start_ns
        window = self.rate_windows[client_id]
        window.advance(self.states[client_id], relative)
        window.add_delivery(relative, 8 * S)

    def resolve(self, client_id, grant, delivered, t_ns):
        state = self.states[client_id]
        # Contract: resolve runs before GatewayCore pops the grant, so it is
        # still outstanding and in Phi; nothing is re-inserted here.
        matches = [gid for gid, value in state.outstanding.items() if value is grant]
        assert len(matches) == 1, "resolve must run while the grant is still outstanding"
        grant_id = matches[0]
        assert grant_id in self.view.inflight
        relative = t_ns - self.start_ns
        self._set_time(t_ns)
        window = self.rate_windows[client_id]
        window.advance(state, relative)
        rtt_ns = t_ns - grant.t_issue_ns if delivered else None
        obs = resolve_event(state, self.view, grant_id, int(delivered), relative, rtt_ns)
        assert grant_id not in state.outstanding and grant_id not in self.view.inflight
        window.set_outstanding(relative, len(state.outstanding))
        self.gateway.predictor.observe(client_id, int(delivered), relative, obs)
        self.resolutions.append({
            "round_id": self.gateway.round.round_id,
            "client_id": client_id,
            "grant_id": grant_id,
            "y": int(delivered),
            "t_ns": relative,
            "obs": obs.copy(),
        })

    def on_tick(self, t_ns):
        relative = t_ns - self.start_ns
        for cid, state in self.states.items():
            self.rate_windows[cid].advance(state, relative)
        self._set_time(t_ns)

    def on_deadline(self, t_ns):
        relative = t_ns - self.start_ns
        for cid, state in self.states.items():
            self.rate_windows[cid].advance(state, relative)

    # -- observation ------------------------------------------------------------
    def _decision(self, record: dict) -> None:
        self.decisions.append(dict(record))
        if record["outcome"] == "grant_issued":
            self._set_time(record["t_ns"])
            key = (record["round_id"], record["client_id"], record["grant_id"])
            self.issued_x48[key] = features48(self.states[record["client_id"]], self.view)

    def replay(self, events: list[dict]) -> list[dict]:
        """Replay only gateway-visible inputs, in captured order, on absolute time."""
        inputs: list[dict] = []
        for event in events:
            kind = event["event"]
            t_abs = event["abs_t_ns"]
            if kind == "round_start":
                inputs.append(event)
                self.gateway.on_round_start(
                    t_abs, round_id=event["round_id"], deadline_ns=t_abs + event["deadline_ns"], K=K, L=L,
                )
            elif kind == "ready_received":
                inputs.append(event)
                self.gateway.on_ready(event["client_id"], np.asarray(event["energies"], np.float32), t_abs)
            elif kind == "data_received":
                inputs.append(event)
                self.gateway.on_data(
                    event["client_id"], event["grant_id"], event["send_pos"],
                    event["block_idx"], event["payload"], t_abs, n_coords=event["n_coords"],
                )
            elif kind == "tick":
                inputs.append(event)
                for cid, state in self.states.items():
                    self.rate_windows[cid].advance(state, t_abs - self.start_ns)
                self._set_time(t_abs)
                self.gateway.on_tick(t_abs)
            elif kind == "deadline":
                inputs.append(event)
                self.gateway.on_deadline(t_abs)
            else:
                assert kind == "grant_decision"  # an output, never replayed as input
        return inputs


def _assert_decisions_match(sim_decisions: list[dict], core_decisions: list[dict],
                            pacing_inputs: list[tuple], selector: str) -> None:
    assert len(sim_decisions) == len(core_decisions) == len(pacing_inputs)
    for index, (sim, core, pacing) in enumerate(zip(sim_decisions, core_decisions, pacing_inputs, strict=True)):
        expected = {k: v for k, v in sim.items()
                    if k not in _SIM_COMPAT_FIELDS and not (k in _GRANT_FIELDS and v is None)}
        absolute = expected.pop("abs_t_ns")
        relative = expected.pop("t_ns")
        assert core["t_ns"] == absolute, index
        assert {k: v for k, v in core.items() if k != "t_ns"} == expected, (index, sim, core)
        assert relative >= 0
        # The compat fields are derived exactly from the same decision.
        client_id, _, p_issue, target, _ = pacing
        assert sim["selected_client_id"] == core["client_id"] == client_id, index
        assert sim["selection_method"] == selector, index
        assert sim["p_pacing"] == pytest.approx(p_issue), index
        assert sim["p_target"] == pytest.approx(target), index


def _assert_resolutions_match(sim_samples: list[dict], core_resolutions: list[dict], round_id: int) -> None:
    sim = [event for event in sim_samples if event["event"] == "grant_resolved"]
    core = [event for event in core_resolutions if event["round_id"] == round_id]
    assert len(sim) == len(core)
    for index, (left, right) in enumerate(zip(sim, core, strict=True)):
        assert (left["client_id"], left["grant_id"], left["y"], left["t_ns"]) == (
            right["client_id"], right["grant_id"], right["y"], right["t_ns"]
        ), (index, left, right)
        np.testing.assert_array_equal(left["obs"], right["obs"], err_msg=str(index))


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
    # Exact per-type contract: no simulator-private or learning-label field.
    for event in a_events:
        kind = event["event"]
        if kind == "grant_decision":
            expected = _DECISION_FIELDS
            assert event["outcome"] in {"grant_issued", "pacing_blocked"}
            issued = event["outcome"] == "grant_issued"
            assert all((event[k] is not None) == issued for k in _GRANT_FIELDS), event
        else:
            expected = _INPUT_FIELDS[kind]
        assert set(event) == expected, (kind, sorted(set(event) ^ expected))
        assert event["abs_t_ns"] == event["t_ns"]  # round 1 starts at absolute 0
    assert all("y" not in event and "obs" not in event and "x48" not in event for event in a_events)
    # The event stream carries exactly the decision_hook records, in order.
    assert [event for event in a_events if event["event"] == "grant_decision"] == a_decisions
    times = [event["t_ns"] for event in a_events]
    assert times == sorted(times)
    assert a_events[0]["event"] == "round_start"
    # After the deadline the gateway only observes late DATA drained from the
    # network (counted as late_after_deadline); it makes no further decisions.
    deadline_index = next(i for i, event in enumerate(a_events) if event["event"] == "deadline")
    late = a_events[deadline_index + 1:]
    assert all(event["event"] == "data_received" and event["t_ns"] > a_events[deadline_index]["t_ns"]
               for event in late)
    assert a_result["gateway"].core.stats.late_after_deadline == len(late)
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


def _assert_replay_parity(runs, num_clients: int) -> _ReplayAssembly:
    """Replay each round's observable inputs into one persistent assembly."""
    weights = np.full(num_clients, 1.0 / num_clients, dtype=np.float32)
    deadline_ns = int(runs[0][0]["deadline_ms"] * 1_000_000)
    replay = _ReplayAssembly(num_clients, weights, deadline_ns)
    for round_index, (result, samples, sim_decisions, events) in enumerate(runs, start=1):
        decisions_before = len(replay.decisions)
        replay_inputs = replay.replay(events)
        assert replay_inputs == [event for event in events if event["event"] != "grant_decision"]

        # Identical scheduling: every decision, field for field, same order.
        _assert_decisions_match(sim_decisions, replay.decisions[decisions_before:],
                                replay.pacing_inputs[decisions_before:], replay.cfg.selector)
        assert [inputs[4] for inputs in replay.pacing_inputs[decisions_before:]] == [
            event["pacing_allowed"] for event in sim_decisions
        ]
        # Identical resolution: same grant, outcome, time and observation vector.
        _assert_resolutions_match(samples, replay.resolutions, round_index)
        # Identical feature vectors for every issued grant.
        sim_issues = {
            (round_index, event["client_id"], event["grant_id"]): event["x48"]
            for event in samples if event["event"] == "grant_issued"
        }
        core_issues = {key: x for key, x in replay.issued_x48.items() if key[0] == round_index}
        assert sim_issues.keys() == core_issues.keys()
        for key, x48 in sim_issues.items():
            np.testing.assert_array_equal(x48, core_issues[key], err_msg=str(key))

        # Grant lifecycle closes at the deadline in both state owners.
        sim_gateway = result["gateway"]
        for state in list(replay.states.values()) + list(sim_gateway.client_states):
            assert not state.outstanding
        assert replay.view.inflight.phi == 0.0 and sim_gateway.inflight.phi == 0.0
        assert not replay.gateway.view.inflight._p and not sim_gateway.inflight._p
        # The GatewaySim object is reused across rounds, so compare with this
        # round's own snapshot rather than the gateway's latest state.
        assert {cid: sorted(blocks) for cid, blocks in replay.gateway.delivered_blocks.items()} == {
            cid: [idx for idx, hit in enumerate(mask) if hit]
            for cid, mask in enumerate(result["delivered_mask"])
        }
        assert {event["y"] for event in samples if event["event"] == "grant_resolved"} == {0, 1}
    return replay


def test_gatewaycore_replays_observable_events_with_scheduling_and_resolution_parity():
    # Whole round, through the deadline: replaying only gateway-observable
    # inputs reproduces every decision, resolution and feature vector.
    runs = _run_rounds(1934, 1, selector="srpt")
    replay = _assert_replay_parity(runs, 20)
    # Sanity: timeouts and deadline resolutions were exercised.
    assert any(event["y"] == 0 for event in replay.resolutions)
    sim_core = runs[0][0]["gateway"].core
    assert replay.gateway.stats.accepted_data == sim_core.stats.accepted_data == sum(
        len(blocks) for blocks in sim_core.delivered_blocks.values()
    )
    for name in ("duplicate_data", "late_after_timeout", "late_after_deadline", "rejected_data"):
        assert getattr(replay.gateway.stats, name) == getattr(sim_core.stats, name), name


def test_multi_round_replay_parity_with_persistent_state_and_absolute_clock():
    # Three rounds on one reused GatewaySim (as in FL): grant IDs restart each
    # round, estimator/predictor state carries over, round starts are nonzero
    # absolute times and every hook exposes a consistent relative/absolute pair.
    runs = _run_rounds(2207, 3, selector="srpt")
    deadline_ns = compute_deadline_ns(0.10, 20)
    for round_id, (result, samples, decisions, events) in enumerate(runs, start=1):
        start = (round_id - 1) * (1_000_000_000 + deadline_ns)
        assert events[0] == {"event": "round_start", "round_id": round_id, "t_ns": 0,
                             "abs_t_ns": start, "deadline_ns": deadline_ns}
        for record in (*events, *decisions, *samples):
            assert record["abs_t_ns"] == start + record["t_ns"]
            assert record["t_ns"] >= 0
            # Only late DATA drained after the deadline may carry t_ns > D.
            assert record["t_ns"] <= deadline_ns or record["event"] == "data_received"
        issued = [event for event in decisions if event["outcome"] == "grant_issued"]
        assert issued[0]["grant_id"] == 1  # GatewayCore restarts grant IDs per round
        assert [event["grant_id"] for event in issued] == list(range(1, len(issued) + 1))
        for event in issued:
            assert event["timeout_ns"] > event["abs_t_ns"]  # absolute core clock, unconverted
            assert event["timeout_ns"] <= start + deadline_ns
        rtts = [event["rtt_ns"] for event in samples if event["event"] == "grant_resolved" and event["y"] == 1]
        assert rtts and all(isinstance(rtt, int) and rtt > 0 for rtt in rtts)
        assert sorted(result["rtt_samples_ns"]) == sorted(rtts)
    # Persistent client estimator state is the same object across rounds.
    assert runs[0][0]["gateway"] is runs[2][0]["gateway"]
    assert runs[2][0]["gateway"].client_states[0].srtt_ns != ClientState(0, 0.05, K=K).srtt_ns

    replay = _assert_replay_parity(runs, 20)
    assert {event["round_id"] for event in replay.resolutions} == {1, 2, 3}


def test_multi_round_simulation_is_deterministic():
    first = _run_rounds(3301, 2, selector="rr")
    second = _run_rounds(3301, 2, selector="rr")
    for (a_result, a_samples, a_decisions, a_events), (b_result, b_samples, b_decisions, b_events) in zip(
        first, second, strict=True
    ):
        assert a_events == b_events and a_decisions == b_decisions
        assert a_result["delivered_mask"] == b_result["delivered_mask"]
        assert a_result["rtt_samples_ns"] == b_result["rtt_samples_ns"]
        assert [(s["event"], s["client_id"], s["grant_id"], s["t_ns"]) for s in a_samples] == [
            (s["event"], s["client_id"], s["grant_id"], s["t_ns"]) for s in b_samples
        ]


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
