"""Production Mininet gateway factories (design decisions D-01..D-05, D-17)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from fedgrant.core.gateway import GatewayCore
from fedgrant.core.state import ClientState
from fedgrant.mininet import production
from fedgrant.mininet.production import (
    GatewayContext,
    LiveClockGatewaySim,
    ProductionGatewayError,
    create_gateway,
    gateway_metadata,
)
from fedgrant.policy.constants import K, L
from fedgrant.policy.pacing import pacing_p
from fedgrant.policy.selectors import RoundRobinSelector, ValueSelector
from fedgrant.policy.types import CandidateView, PolicyConfig
from fedgrant.predictors.ewma import PEWMA
from fedgrant.predictors.fallback import FallbackPredictor
from fedgrant.predictors.gru_numpy import PGRUNumpy
from fedgrant.sim.run_net import compute_deadline_ns, run_network_round

N = 6
UNIFORM = np.full(N, 1.0 / N, dtype=np.float32)


def _ctx(policy: str, **kwargs) -> GatewayContext:
    return GatewayContext(num_clients=kwargs.pop("num_clients", N), policy=policy,
                          client_weights=kwargs.pop("client_weights", UNIFORM), **kwargs)


@pytest.mark.parametrize(("policy", "selector_type", "predictor_type"), [
    ("B4", RoundRobinSelector, PEWMA),
    ("B6", ValueSelector, PEWMA),
    ("FG-GRU", ValueSelector, PGRUNumpy),
])
def test_factory_builds_real_gatewaycore_with_policy_providers(policy, selector_type, predictor_type) -> None:
    core = create_gateway(_ctx(policy))
    assert isinstance(core, GatewayCore)
    adapter = core.eligibility.sim
    assert isinstance(adapter, LiveClockGatewaySim) and core.selector.sim is adapter
    assert isinstance(adapter.selector, selector_type)
    assert isinstance(core.predictor, FallbackPredictor)
    assert isinstance(core.predictor.primary, predictor_type)
    assert adapter.cfg == production.POLICY_CONFIGS[policy]
    assert core.decision_hook is None and core.retransmit is True
    assert len(core.clients) == N
    assert [state.w for _, state in sorted(core.clients.items())] == pytest.approx(UNIFORM)


@pytest.mark.parametrize(("policy", "message"), [
    ("ORACLE", "simulator-only"),
    ("B2", "baseline_proc"),
    ("B1", "baseline_proc"),
    ("FULL", "no GatewayCore configuration"),
])
def test_unsupported_policies_are_rejected_not_substituted(policy, message) -> None:
    with pytest.raises(ProductionGatewayError, match=message):
        create_gateway(_ctx(policy))


def test_exploration_and_bad_weights_are_rejected() -> None:
    with pytest.raises(ProductionGatewayError, match="explore_eps"):
        create_gateway(_ctx("B6", policy_config=PolicyConfig("value", "ewma", explore_eps=0.1)))
    with pytest.raises(ProductionGatewayError, match="sum to 1"):
        create_gateway(_ctx("B6", client_weights=np.full(N, 0.5, dtype=np.float32)))
    with pytest.raises(ProductionGatewayError, match="finite non-negative"):
        create_gateway(_ctx("B6", client_weights=np.ones(N + 1, dtype=np.float32) / (N + 1)))


def test_missing_checkpoint_fails_clearly_without_policy_fallback(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="gru_prod.npz"):
        create_gateway(_ctx("FG-GRU", models_dir=tmp_path))
    # EWMA policies need no checkpoint and are unaffected.
    assert isinstance(create_gateway(_ctx("B6", models_dir=tmp_path)), GatewayCore)


def test_metadata_records_policy_and_checkpoint_hash() -> None:
    meta = gateway_metadata(_ctx("FG-GRU"))
    expected = hashlib.sha256(open("models/gru_prod.npz", "rb").read()).hexdigest()
    assert meta["checkpoint"]["sha256"] == expected
    assert meta["policy_config"]["predictor"] == "gru" and meta["policy_config"]["explore_eps"] == 0.0
    assert gateway_metadata(_ctx("B4"))["checkpoint"] is None
    json.dumps(meta)


def _write_fixture_bank(path, n_clients: int = 20) -> np.ndarray:
    """Test-only bank fixture (never used by real runs)."""
    weights = np.arange(1, n_clients + 1, dtype=np.float32)
    weights /= weights.sum()
    provenance = path.parent / "provenance.json"
    provenance.write_text('{"fixture": true}\n')
    np.savez_compressed(
        path,
        energies_sorted=np.tile(np.arange(K, 0, -1, dtype=np.float32), (2, n_clients, 1)),
        client_weights=weights, round_ids=np.arange(1, 3), seed=np.asarray(0),
        num_rounds=np.asarray(2),
        dataset_metadata_json=np.asarray(json.dumps({
            "source": "test-fixture", "provenance_path": str(provenance),
            "provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
        })),
    )
    return weights


def test_policy_wrappers_require_the_real_bank_and_use_its_weights(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(production, "VALUE_BANK_PATH", tmp_path / "absent.npz")
    with pytest.raises(ProductionGatewayError, match="real value bank required"):
        production.create_b6_gateway(20)

    bank = tmp_path / "bank.npz"
    weights = _write_fixture_bank(bank)
    monkeypatch.setattr(production, "VALUE_BANK_PATH", bank)
    for factory in (production.create_b4_gateway, production.create_b6_gateway, production.create_fg_gru_gateway):
        core = factory(20)
        assert [core.clients[i].w for i in range(20)] == pytest.approx(weights)
    with pytest.raises(ProductionGatewayError, match="D-17"):
        production.create_b4_gateway(3)  # N=3 smoke has no bank profiles


def _sim_rounds(rounds: int, *, selector: str, predictor: str = "ewma", seed: int = 77):
    out, gateway = [], None
    weights = UNIFORM
    energies = np.tile(np.linspace(1.0, 0.01, K, dtype=np.float32), (N, 1))
    for round_id in range(1, rounds + 1):
        decisions, events = [], []
        result = run_network_round(
            round_id=round_id, num_clients=N, client_weights=weights, client_energies=energies,
            beta=0.4, scenario="L2", policy_config=PolicyConfig(selector=selector, predictor=predictor),
            seed=int(np.random.SeedSequence([seed, round_id - 1]).generate_state(1)[0]),
            gateway=gateway, decision_hook=decisions.append, event_hook=events.append,
        )
        gateway = result["gateway"]
        out.append((decisions, events))
    return out


# GatewaySim adds these to every decision record to keep the simulator's public
# decision schema (fedgrant/sim/run_net.py); production GatewayCore does not.
SIM_COMPAT_FIELDS = {"selection_method", "selected_client_id", "p_pacing", "p_target"}
GRANT_FIELDS = ("grant_id", "send_pos", "timeout_ns")


def assert_same_decision(sim: dict, live: dict, origin: int, selector: str) -> None:
    """Every field the production core emits matches; the simulator's compat fields are consistent."""
    expected = {k: v for k, v in sim.items()
                if k not in SIM_COMPAT_FIELDS | {"t_ns", "abs_t_ns", "timeout_ns"}
                and not (k in GRANT_FIELDS and v is None)}
    assert {k: v for k, v in live.items() if k not in {"t_ns", "timeout_ns"}} == expected
    assert live["t_ns"] == origin + sim["abs_t_ns"]
    if sim.get("timeout_ns") is not None:
        assert live["timeout_ns"] - origin == sim["timeout_ns"]
    # Compat fields are pure derivations of the GatewayCore decision.
    assert sim["selected_client_id"] == live["client_id"] and sim["selection_method"] == selector
    assert sim["p_pacing"] == pytest.approx(pacing_p(sim["p_hat"], PolicyConfig(selector, "ewma")))
    issued = live["outcome"] == "grant_issued"
    assert all((sim[k] is None) != issued for k in GRANT_FIELDS)


@pytest.mark.parametrize(("policy", "selector"), [("B6", "value"), ("B4", "rr")])
def test_production_core_reproduces_simulator_decisions_on_an_arbitrary_clock(policy, selector) -> None:
    # The daemon's monotonic origin has no relation to the simulator's; the
    # same observable inputs must yield the same decisions in every round.
    origin = 987_654_321_123_457
    runs = _sim_rounds(2, selector=selector)
    core = create_gateway(_ctx(policy))
    records: list[dict] = []
    core.decision_hook = records.append
    for decisions, events in runs:
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
        assert decisions, "scenario must issue grants"
        assert len(produced) == len(decisions)
        for sim, live in zip(decisions, produced, strict=True):
            assert_same_decision(sim, live, origin, selector)
        assert all(not state.outstanding for state in core.clients.values())


class _CountingPredictor:
    def __init__(self):
        self.batch_calls: list[int] = []

    def reset_round(self, ctx):
        pass

    def observe(self, client, y, t_ns, obs):
        pass

    def predict(self, inp):
        if len(inp.client_ids) > 1:
            self.batch_calls.append(int(inp.t_ns))
        return np.full(len(inp.client_ids), 0.9, dtype=np.float32)


class _NoCandidates:
    def candidates(self, clients, now_ns, deadline_ns, retransmit):
        return CandidateView(np.empty(0, np.int32), np.empty(0), np.empty(0, np.int32), 0)


def test_five_ms_batched_refresh_fires_on_unaligned_daemon_ticks() -> None:
    # Section 4.6: ticks from the UDP daemon have arbitrary phase relative to
    # the round start; the refresh must still happen once per 5 ms bucket.
    from fedgrant.policy.features import GatewayView

    states = {cid: ClientState(cid, 0.5, K=3) for cid in (0, 1)}
    predictor = _CountingPredictor()
    core = GatewayCore(states, selector=object(), eligibility=_NoCandidates(), pacing=object(),
                       timeout=object(), predictor=predictor, view=GatewayView(0, 1, 1.0))
    start = 1_000_000_000_000 + 333_333
    core.on_round_start(start, round_id=1, deadline_ns=start + 40_000_000, K=3, L=2)
    for cid in (0, 1):
        core.on_ready(cid, [4.0, 2.0, 1.0], start + 100)
    ticks = [start + 1_300_000 + k * 1_000_000 for k in range(12)]  # 1.3 ms, 2.3 ms, ... 12.3 ms
    for t in ticks:
        core.on_tick(t)
    assert predictor.batch_calls == [5_300_000, 10_300_000]
    # A new round restarts the bucket count.
    core.on_deadline(start + 40_000_000)
    second = start + 41_000_000
    core.on_round_start(second, round_id=2, deadline_ns=second + 40_000_000, K=3, L=2)
    for cid in (0, 1):
        core.on_ready(cid, [4.0, 2.0, 1.0], second + 100)
    core.on_tick(second + 5_100_000)
    assert predictor.batch_calls[-1] == 5_100_000


def test_live_clock_adapter_reports_live_round_window() -> None:
    core = create_gateway(_ctx("B6"))
    adapter = core.eligibility.sim
    D = compute_deadline_ns(0.10, N)
    core.on_round_start(5_000_000_000, round_id=1, deadline_ns=5_000_000_000 + D, K=K, L=L)
    assert adapter.round_start_abs_ns == 5_000_000_000 and adapter.D_ns == D
    core.on_deadline(5_000_000_000 + D)
    core.on_round_start(9_000_000_123, round_id=2, deadline_ns=9_000_000_123 + 2 * D, K=K, L=L)
    assert adapter.round_start_abs_ns == 9_000_000_123 and adapter.D_ns == 2 * D


def test_factory_accepts_the_runner_contract_context_shape(tmp_path) -> None:
    # Field list of MininetGatewayContext in docs/MININET_INTEGRATION_CONTRACT.md.
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class MininetGatewayContext:
        num_clients: int
        policy: str
        seed: int
        client_weights: np.ndarray
        deadline_ns: int
        K: int
        L: int
        models_dir: str
        log_dir: str

    ctx = MininetGatewayContext(N, "FG-GRU", 2, UNIFORM, compute_deadline_ns(0.10, N), K, L, "models", str(tmp_path))
    core = create_gateway(ctx)
    assert isinstance(core.predictor.primary, PGRUNumpy)
    assert core.eligibility.sim.D_ns == compute_deadline_ns(0.10, N)  # before the first round
    assert gateway_metadata(ctx)["seed"] == 2
    assert list(tmp_path.iterdir()) == []  # the factory writes no logs
