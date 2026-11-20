"""Bounded probe end to end (CHANGELOG 2026-11-13 rule (v) amendment).

A client whose prediction is always 0 (< THETA) passes rules (i)-(iv) but never
rule (v). Without the probe it would get no grants at all; with it, the real
simulator (run_net -> GatewayCore) and the production gateway (create_gateway ->
LiveClockGatewaySim) must each issue it exactly one probe grant per round, the
two must agree decision by decision, and the per-round counters that runs log
must match the decision records.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from fedgrant.mininet.production import GatewayContext, create_gateway
from fedgrant.policy.constants import K, L, THETA
from fedgrant.policy.types import PolicyConfig
from fedgrant.predictors.fallback import FallbackPredictor
from fedgrant.sim.run_net import run_network_round

N = 4
LOCKED = 3          # the client whose prediction is always 0
ROUNDS = 3
WEIGHTS = np.full(N, 1.0 / N, dtype=np.float32)
ENERGIES = np.tile(np.linspace(1.0, 0.01, K, dtype=np.float32), (N, 1))


class LockedOutPredictor:
    """p_hat = 0 for LOCKED, 0.9 for everyone else; deterministic and stateless."""

    def reset_round(self, _ctx) -> None:
        pass

    def observe(self, _client, _y, _t_ns, _obs) -> None:
        pass

    def predict(self, inp) -> np.ndarray:
        return np.where(np.asarray(inp.client_ids) == LOCKED, 0.0, 0.9).astype(np.float32)


def _fixed_clock_fallback() -> FallbackPredictor:
    # A constant clock: the 2 ms wall-clock limit must not switch either side to
    # P-EWMA under CPU load, which would change p_hat and the decisions.
    return FallbackPredictor(LockedOutPredictor(), clock=lambda: 0)


def _sim_rounds(policy: str):
    """ROUNDS consecutive M-sim rounds on one reused GatewaySim.

    The gateway is reused, so per-round state is snapshotted right after each
    round: the probe grant ids, the core's round counter and what run_grid logs.
    """
    from scripts.run_grid import _round_probe_grants

    selector = {"B6": "value", "B4": "rr"}[policy]
    gateway, out = None, []
    predictor = _fixed_clock_fallback()
    for round_id in range(1, ROUNDS + 1):
        decisions, events = [], []
        result = run_network_round(
            round_id=round_id, num_clients=N, client_weights=WEIGHTS, client_energies=ENERGIES, beta=0.4,
            scenario="L2", policy_config=PolicyConfig(selector=selector, predictor="ewma"), predictor=predictor,
            seed=int(np.random.SeedSequence([11, round_id - 1]).generate_state(1)[0]),
            gateway=gateway, decision_hook=decisions.append, event_hook=events.append)
        gateway = result["gateway"]
        core = gateway.core
        snapshot = {
            "round_probe_grants": core.round_probe_grants,
            "cumulative": core.stats.probe_grants,
            "logged": _round_probe_grants(result, result["metrics"]),
            "probe_ids": {(d["client_id"], d["grant_id"]) for d in decisions
                          if d["outcome"] == "grant_issued" and core.is_probe_grant(d["client_id"], d["grant_id"])},
        }
        out.append((snapshot, decisions, events))
    return out


def _issued(decisions):
    return [d for d in decisions if d["outcome"] == "grant_issued"]


@pytest.mark.parametrize("policy", ["B6", "B4"])
def test_simulator_issues_exactly_one_probe_per_round_to_a_locked_out_client(policy) -> None:
    for snapshot, decisions, _ in _sim_rounds(policy):
        issued = _issued(decisions)
        probes = [d for d in issued if d["probe"] == 1]
        # Exactly one probe, to the locked-out client, while its p_hat is below theta.
        assert [d["client_id"] for d in probes] == [LOCKED]
        assert probes[0]["p_hat"] < THETA
        # It is the client's only grant: after the probe, rule (v) excludes it again.
        assert Counter(d["client_id"] for d in issued)[LOCKED] == 1
        # Every other grant is normal and was issued at p_hat >= theta.
        assert all(d["probe"] == 0 and d["p_hat"] >= THETA for d in issued if d["client_id"] != LOCKED)
        # A pacing-blocked decision may carry the would-be probe flag, but never for others.
        assert all(d["client_id"] == LOCKED for d in decisions if d["probe"] == 1)
        # The counters runs log agree with the decision records.
        assert snapshot["round_probe_grants"] == snapshot["logged"] == 1
        assert snapshot["probe_ids"] == {(LOCKED, probes[0]["grant_id"])}


def test_probe_counters_accumulate_across_rounds() -> None:
    runs = _sim_rounds("B6")
    # The round counter resets at each ROUND_START; the cumulative one never does.
    assert [snapshot["round_probe_grants"] for snapshot, _, _ in runs] == [1] * ROUNDS
    assert [snapshot["cumulative"] for snapshot, _, _ in runs] == list(range(1, ROUNDS + 1))


@pytest.mark.parametrize("policy", ["B6", "B4"])
def test_production_gateway_issues_the_same_probes_as_the_simulator(policy) -> None:
    # Replay the simulator's observable inputs into the production core on an
    # unrelated monotonic origin; every decision, including its probe flag,
    # must match.
    origin = 555_000_111_222_333
    runs = _sim_rounds(policy)
    core = create_gateway(GatewayContext(num_clients=N, policy=policy, client_weights=WEIGHTS))
    core.predictor = _fixed_clock_fallback()
    core.eligibility.sim.predictor = core.predictor
    records: list[dict] = []
    core.decision_hook = records.append
    compared = ("event", "round_id", "client_id", "outcome", "p_hat", "probe", "grant_id", "send_pos",
                "candidate_client_ids", "pacing_allowed")
    for snapshot, decisions, events in runs:
        before = len(records)
        for event in events:
            t = origin + event["abs_t_ns"]
            kind = event["event"]
            if kind == "round_start":
                core.on_round_start(t, round_id=event["round_id"], deadline_ns=t + event["deadline_ns"], K=K, L=L)
            elif kind == "ready_received":
                core.on_ready(event["client_id"], np.asarray(event["energies"], np.float32), t)
            elif kind == "data_received":
                core.on_data(event["client_id"], event["grant_id"], event["send_pos"], event["block_idx"],
                             event["payload"], t, n_coords=event["n_coords"])
            elif kind == "tick":
                core.on_tick(t)
            elif kind == "deadline":
                core.on_deadline(t)
        produced = records[before:]
        assert len(produced) == len(decisions)
        for sim, live in zip(decisions, produced, strict=True):
            assert {k: live.get(k) for k in compared} == {k: sim.get(k) for k in compared}
        assert core.round_probe_grants == snapshot["round_probe_grants"] == 1
    assert core.predictor.n_fallbacks == 0
