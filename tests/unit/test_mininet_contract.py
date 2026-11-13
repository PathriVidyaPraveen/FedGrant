"""Mininet integration contract: gateway context, bank profiles, process dispatch (no root)."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

import fedgrant.mininet.client_proc as client_proc
import fedgrant.mininet.gateway_proc as gateway_proc
import fedgrant.mininet.run_mininet as rm
from fedgrant.core.gateway import GatewayCore
from fedgrant.encoder.blocks import encode
from fedgrant.mininet import contract
from fedgrant.mininet.contract import MininetGatewayContext
from fedgrant.mininet.stub_policy import create_gateway
from fedgrant.policy.constants import K, L
from fedgrant.sim.value_bank import ValueBankInputs
from tests.unit.test_mininet_teardown import FakeNet, FakeSampler

PROD = "tests.unit.test_mininet_contract:production_like_factory"


def production_like_factory(ctx: MininetGatewayContext) -> GatewayCore:
    """Stand-in for Prajin's factory: honours the context like a real one must."""
    return create_gateway(ctx)


def _ctx(n=3, weights=None, **kwargs) -> MininetGatewayContext:
    fields = dict(num_clients=n, policy="B6", seed=1, policy_config=contract.POLICY_CONFIGS["B6"],
                  client_weights=weights if weights is not None else np.full(n, 1 / n, np.float32),
                  deadline_ns=398_000_000, K=K, L=L, models_dir="models", log_dir="/tmp")
    fields.update(kwargs)
    return MininetGatewayContext(**fields)


def _bank(n=3, rounds=4, seed=0) -> ValueBankInputs:
    rng = np.random.default_rng(seed)
    energies = -np.sort(-rng.uniform(1e-6, 1e-2, size=(rounds, n, K)), axis=2).astype(np.float32)
    weights = rng.uniform(1, 2, size=n)
    return ValueBankInputs(energies, (weights / weights.sum()).astype(np.float32), "0" * 64, "1" * 64)


# --- context and factory -----------------------------------------------------

def test_context_validates_weights_and_sizes():
    with pytest.raises(ValueError, match="one entry per client"):
        _ctx(n=3, weights=np.ones(2, np.float32) / 2)
    with pytest.raises(ValueError, match="sum to 1"):
        _ctx(n=2, weights=np.array([0.7, 0.7], np.float32))
    with pytest.raises(ValueError, match="positive"):
        _ctx(deadline_ns=0)
    assert _ctx().client_weights.dtype == np.float32
    with pytest.raises(ValueError, match="policy_config for B6"):
        _ctx(policy_config=contract.POLICY_CONFIGS["B4"])


def test_mininet_policy_configs_match_simulator_grid():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_grid_for_test", "scripts/run_grid.py")
    run_grid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_grid)
    for policy, config in contract.POLICY_CONFIGS.items():
        assert run_grid._SIMULATOR_POLICY_CONFIGS[policy] == config, policy
    assert contract.policy_config_for(rm.STUB_POLICY) is None
    with pytest.raises(ValueError):
        contract.policy_config_for("B2")


def test_stub_and_factory_use_context_weights():
    weights = np.array([0.5, 0.3, 0.2], np.float32)
    core = contract.build_gateway(PROD, _ctx(weights=weights))
    assert [core.clients[i].w for i in range(3)] == pytest.approx([0.5, 0.3, 0.2])
    assert isinstance(create_gateway(2), GatewayCore)  # int form kept for existing tests


def test_build_gateway_rejects_bad_factories(monkeypatch):
    import types
    fake = types.ModuleType("fake_factories")
    fake.not_core = lambda ctx: object()
    fake.wrong_clients = lambda ctx: create_gateway(ctx.num_clients + 1)
    fake.no_args = lambda: None
    monkeypatch.setitem(__import__("sys").modules, "fake_factories", fake)
    with pytest.raises(TypeError, match="not GatewayCore"):
        contract.build_gateway("fake_factories:not_core", _ctx())
    with pytest.raises(ValueError, match="register clients"):
        contract.build_gateway("fake_factories:wrong_clients", _ctx())
    with pytest.raises(TypeError, match="MininetGatewayContext"):
        contract.build_gateway("fake_factories:no_args", _ctx())
    with pytest.raises(ValueError, match="module:function"):
        contract.build_gateway("fake_factories", _ctx())


def _gateway_args(**overrides):
    args = dict(clients=3, policy="B6", seed=2, value_bank=None, duration_ns=398_000_000,
                blocks=K, block_length=L, models_dir="models", log_dir="/tmp")
    args.update(overrides)
    return argparse.Namespace(**args)


def test_gateway_proc_context_uses_bank_weights(monkeypatch):
    ctx = gateway_proc.build_context(_gateway_args())
    assert ctx.client_weights == pytest.approx(np.full(3, 1 / 3))
    bank = _bank()
    monkeypatch.setattr(gateway_proc, "load_bank", lambda path, n: bank)
    ctx = gateway_proc.build_context(_gateway_args(value_bank="bank.npz"))
    np.testing.assert_array_equal(ctx.client_weights, bank.client_weights)
    assert (ctx.policy, ctx.seed, ctx.deadline_ns) == ("B6", 2, 398_000_000)
    assert ctx.policy_config == contract.POLICY_CONFIGS["B6"]


# --- value-bank profiles ------------------------------------------------------

def test_bank_provider_matches_msim_profile_and_is_deterministic():
    bank = _bank()
    provider = contract.bank_update_provider(bank, master_seed=1, client_id=2)
    update = provider(3)
    expected = bank.energies_for_run_round(1, 3)[2]  # what run_grid gives the M-sim twin
    encoded = encode(update)
    np.testing.assert_allclose(encoded.energies_sorted, np.sort(expected)[::-1], rtol=2e-3, atol=1e-9)
    np.testing.assert_array_equal(update, contract.bank_update_provider(bank, 1, 2)(3))
    assert not np.array_equal(update, contract.bank_update_provider(bank, 1, 1)(3))


def test_payload_seed_is_stable_and_distinct():
    seeds = {contract.payload_seed(0, r, i) for r in (1, 2) for i in (0, 1)}
    assert len(seeds) == 4
    assert contract.payload_seed(0, 1, 0) == contract.payload_seed(0, 1, 0)
    with pytest.raises(ValueError):
        contract.payload_seed(0, 0, 0)


def test_client_proc_uses_bank_when_given(monkeypatch):
    base = dict(client_id=1, seed=0, clients=None, value_bank=None)
    synthetic = client_proc.make_update_provider(argparse.Namespace(**base))
    assert synthetic(1).shape == (117_694,)
    with pytest.raises(ValueError, match="--clients"):
        client_proc.make_update_provider(argparse.Namespace(**{**base, "value_bank": "b.npz"}))
    bank = _bank()
    monkeypatch.setattr(client_proc, "load_bank", lambda path, n: bank)
    provider = client_proc.make_update_provider(argparse.Namespace(**{**base, "value_bank": "b.npz", "clients": 3}))
    np.testing.assert_array_equal(provider(2), contract.bank_update_provider(bank, 0, 1)(2))


# --- process dispatch -------------------------------------------------------------

def _commands(policy, **kwargs):
    fields = dict(python="py", policy=policy, n=2, rounds=3, seed=1, duration_ns=398_000_000,
                  compute_window_s=1.0, out=rm.Path("/run"), round_state_path=rm.Path("/run/.rs"),
                  gateway_factory=PROD, value_bank="bank.npz", models_dir="models", profile_gateway=True)
    fields.update(kwargs)
    return rm.process_commands(**fields)


def test_grant_policy_commands():
    gw, clients = _commands("FG-GRU")
    assert gw[:3] == ["py", "-m", contract.GRANT_GATEWAY_MODULE]
    for flag, value in (("--gateway-factory", PROD), ("--policy", "FG-GRU"), ("--value-bank", "bank.npz"),
                        ("--seed", "1"), ("--models-dir", "models")):
        assert gw[gw.index(flag) + 1] == value
    assert "--profile-gateway" in gw
    assert clients[1][:3] == ["py", "-m", contract.GRANT_CLIENT_MODULE]
    assert clients[1][clients[1].index("--client-id") + 1] == "1"
    assert clients[1][clients[1].index("--clients") + 1] == "2"


@pytest.mark.parametrize("policy,mode", [("B1", "tcp"), ("B2", "push"), ("B3", "plan")])
def test_baseline_commands_use_baseline_entry_point(policy, mode):
    gw, clients = _commands(policy)
    assert gw[1:6] == ["-m", contract.BASELINE_MODULE, "gateway", "--mode", mode]
    assert "--gateway-factory" not in gw
    assert clients[0][1:6] == ["-m", contract.BASELINE_MODULE, "client", "--mode", mode]


def test_smoke_commands_have_no_bank():
    gw, clients = _commands(rm.STUB_POLICY, gateway_factory=rm.DEFAULT_STUB_FACTORY, value_bank=None,
                            profile_gateway=False)
    assert "--value-bank" not in gw and "--value-bank" not in clients[0]
    assert "--profile-gateway" not in gw


# --- runner validation ----------------------------------------------------------

@pytest.fixture
def fake_runtime(monkeypatch):
    FakeSampler.instances = []
    monkeypatch.setattr(rm, "cpu_roles", lambda: {"gateway": None, "telemetry": None, "clients": [],
                                                  "pinning_enabled": False})
    monkeypatch.setattr(rm, "ProcessSampler", FakeSampler)
    monkeypatch.setattr(rm, "_QueueSampler", FakeSampler)
    monkeypatch.setattr(rm, "verify_tc", lambda net, expected: None)
    return monkeypatch


def _run(tmp_path, **kwargs):
    fields = dict(n=2, scenario="L2", seed=0, rounds=1, beta=0.1, trace_dir=tmp_path / "traces",
                  output_dir=tmp_path / "run")
    fields.update(kwargs)
    return rm.run_experiment(**fields)


def test_m1_requires_value_bank(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no data/bank here
    with pytest.raises(FileNotFoundError, match="value bank"):
        _run(tmp_path, experiment_id="M1", policy="B6", gateway_factory=PROD)
    assert not (tmp_path / "run").exists()


def test_unimplemented_baseline_and_unknown_policy_fail_early(tmp_path, monkeypatch):
    bank = tmp_path / "bank.npz"
    bank.write_bytes(b"x")
    # baseline_proc now exists; simulate a checkout without it to keep the guard covered.
    real_find_spec = contract.importlib.util.find_spec
    monkeypatch.setattr(contract.importlib.util, "find_spec",
                        lambda name, *a: None if name == contract.BASELINE_MODULE else real_find_spec(name, *a))
    with pytest.raises(RuntimeError, match="baseline_proc"):
        _run(tmp_path, experiment_id="M1", policy="B2", value_bank=str(bank))
    with pytest.raises(ValueError, match="no Mininet implementation"):
        _run(tmp_path, experiment_id="M1", policy="B5", gateway_factory=PROD, value_bank=str(bank))
    assert not (tmp_path / "run").exists()


def test_m1_run_passes_bank_and_enables_profiling(tmp_path, fake_runtime):
    bank = tmp_path / "bank.npz"
    bank.write_bytes(b"bank")
    spawned = []

    def factory(host, cmd):
        spawned.append((host, cmd))
        raise OSError("stop after first spawn")

    fake_runtime.setattr(rm, "build", lambda **kwargs: FakeNet(factory))
    with pytest.raises(OSError):
        _run(tmp_path, experiment_id="M1", policy="B6", gateway_factory=PROD, value_bank=str(bank))
    host, cmd = spawned[0]
    assert host == "gw" and "--profile-gateway" in cmd
    assert cmd[cmd.index("--value-bank") + 1] == str(bank)
    failure = json.loads((tmp_path / "run" / "run_failure.json").read_text())
    assert (failure["experiment_id"], failure["policy"]) == ("M1", "B6")


def test_grant_round_protocol_sends_simulator_readiness_delays():
    """B4/B6/FG-GRU ROUND_STARTs carry the same per-client rho as M-sim and B1-B3."""
    from fedgrant.core.gateway import SendRoundStart
    from fedgrant.mininet.round_loop import readiness_delays_us

    class Tap:
        def __init__(self):
            self.sent = []

        def dispatch_actions(self, actions):
            self.sent += list(actions)

    core, tap = create_gateway(3), Tap()
    proto = gateway_proc.GrantRoundProtocol(core, tap, num_clients=3, K=K, L=L, seed=5)
    window = proto.begin_round(2, 1_000_000_000, 398_000_000)
    proto.send_round_start(window)
    sent = {a.client_id: a.rho_us for a in tap.sent if isinstance(a, SendRoundStart)}
    expected = readiness_delays_us(5, 2, 3, 398_000_000)
    assert sent == expected
    assert all(0 <= rho <= 39_800 for rho in sent.values()) and len(set(sent.values())) > 1
