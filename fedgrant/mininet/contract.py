"""Shared Mininet integration contract (docs/MININET_INTEGRATION_CONTRACT.md).

The runner, the gateway/client processes and the production providers import
these definitions so the process command lines, the gateway-factory input and
the value-bank sampling cannot drift apart.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fedgrant.policy.types import PolicyConfig

DEFAULT_VALUE_BANK = "data/bank/femnist_full_s0.npz"
DEFAULT_MODELS_DIR = "models"
STUB_POLICY = "STUB-FIRST-ELIGIBLE"

# Grant-mode policies run GatewayCore through a --gateway-factory (Section 10).
# These PolicyConfigs must equal scripts/run_grid.py::_SIMULATOR_POLICY_CONFIGS
# (tests/unit/test_mininet_contract.py enforces it), so M1 and its M-sim twin
# run identical policy code.
POLICY_CONFIGS: dict[str, PolicyConfig] = {
    "B4": PolicyConfig("rr", "ewma"),
    "B6": PolicyConfig("value", "ewma"),
    "FG-GRU": PolicyConfig("value", "gru"),
}
GRANT_POLICIES = frozenset(POLICY_CONFIGS)
# Baselines that are not GatewayCore run their own entry points (Prajin).
BASELINE_MODES = {"B1": "tcp", "B2": "push", "B3": "plan"}
GRANT_GATEWAY_MODULE = "fedgrant.mininet.gateway_proc"
GRANT_CLIENT_MODULE = "fedgrant.mininet.client_proc"
BASELINE_MODULE = "fedgrant.mininet.baseline_proc"
# Model files whose hashes are recorded for learned-predictor policies.
POLICY_MODEL_FILES = {"FG-GRU": ("gru_prod.npz", "norm_stats_gru.json")}


@dataclass(frozen=True)
class MininetGatewayContext:
    """Input to ``create_gateway(ctx)``; called once per run in the gateway process."""

    num_clients: int
    policy: str
    seed: int
    policy_config: PolicyConfig | None  # POLICY_CONFIGS[policy]; None only for the smoke stub
    client_weights: np.ndarray  # float32 [N], sums to 1
    deadline_ns: int
    K: int
    L: int
    models_dir: str
    log_dir: str

    def __post_init__(self) -> None:
        weights = np.asarray(self.client_weights, dtype=np.float32)
        if self.num_clients < 1 or weights.shape != (self.num_clients,):
            raise ValueError("client_weights must have one entry per client")
        if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(
                weights.sum(dtype=np.float64), 1.0, atol=1e-5):
            raise ValueError("client_weights must be finite, non-negative and sum to 1")
        if self.deadline_ns <= 0 or self.K <= 0 or self.L <= 0:
            raise ValueError("deadline_ns, K and L must be positive")
        if self.policy in POLICY_CONFIGS and self.policy_config != POLICY_CONFIGS[self.policy]:
            raise ValueError(f"policy_config for {self.policy} must be {POLICY_CONFIGS[self.policy]}")
        object.__setattr__(self, "client_weights", weights)


def policy_config_for(policy: str) -> PolicyConfig | None:
    """The simulator's PolicyConfig for a grant policy; None for the smoke stub."""
    if policy in POLICY_CONFIGS:
        return POLICY_CONFIGS[policy]
    if policy == STUB_POLICY:
        return None
    raise ValueError(f"{policy!r} is not a grant-mode policy")


def is_baseline(policy: str) -> bool:
    return policy in BASELINE_MODES


def entry_points(policy: str) -> tuple[list[str], list[str]]:
    """Return the (gateway, client) ``python -m`` argument prefixes for a policy."""
    if is_baseline(policy):
        mode = BASELINE_MODES[policy]
        return (["-m", BASELINE_MODULE, "gateway", "--mode", mode],
                ["-m", BASELINE_MODULE, "client", "--mode", mode])
    return ["-m", GRANT_GATEWAY_MODULE], ["-m", GRANT_CLIENT_MODULE]


def check_entry_points_available(policy: str) -> None:
    """Fail before building the network when a policy's process module is missing."""
    module = BASELINE_MODULE if is_baseline(policy) else GRANT_GATEWAY_MODULE
    if importlib.util.find_spec(module) is None:
        raise RuntimeError(
            f"policy {policy} needs the process module {module}, which is not implemented yet "
            "(see docs/MININET_INTEGRATION_CONTRACT.md Section 2)"
        )


def load_factory(spec: str) -> Callable[..., Any]:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("gateway factory must use module:function syntax")
    return getattr(importlib.import_module(module_name), attribute)


def build_gateway(spec: str, ctx: MininetGatewayContext):
    """Call a ``create_gateway(ctx)`` factory and check it returned a GatewayCore."""
    from fedgrant.core.gateway import GatewayCore

    factory = load_factory(spec)
    parameters = [p for p in inspect.signature(factory).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if not parameters:
        raise TypeError(f"gateway factory {spec} must accept a MininetGatewayContext")
    core = factory(ctx)
    if not isinstance(core, GatewayCore):
        raise TypeError(f"gateway factory {spec} returned {type(core).__name__}, not GatewayCore")
    if set(core.clients) != set(range(ctx.num_clients)):
        raise ValueError(f"gateway factory {spec} must register clients 0..{ctx.num_clients - 1}")
    return core


def uniform_weights(num_clients: int) -> np.ndarray:
    return np.full(num_clients, 1.0 / num_clients, dtype=np.float32)


def load_bank(path: str | Path, num_clients: int):
    """Load and validate the FULL seed-0 value bank (spec Section 5.2)."""
    from fedgrant.sim.value_bank import load_value_bank

    return load_value_bank(path, num_clients)


def payload_seed(master_seed: int, round_id: int, client_id: int) -> int:
    """Seed for the DATA payload layout of one client update (contract Section 3)."""
    if round_id < 1 or client_id < 0:
        raise ValueError("round_id must be >= 1 and client_id >= 0")
    return int(np.random.SeedSequence([int(master_seed), int(round_id) - 1, 2, int(client_id)])
               .generate_state(1)[0])


def bank_update_provider(bank, master_seed: int, client_id: int) -> Callable[[int], np.ndarray]:
    """Return ``round_id -> float32 update`` whose block energies are the bank profile.

    Client ``i`` in round ``r`` uses ``bank.energies_for_run_round(seed, r)[i]``,
    exactly the profile ``scripts/run_grid.py`` gives the M-sim twin.
    """
    from fedgrant.mininet.profile_update import synthetic_update

    def provider(round_id: int) -> np.ndarray:
        energies = bank.energies_for_run_round(int(master_seed), int(round_id))[int(client_id)]
        return synthetic_update(energies, seed=payload_seed(master_seed, round_id, client_id))

    return provider


__all__ = [
    "BASELINE_MODES", "BASELINE_MODULE", "POLICY_CONFIGS", "policy_config_for", "DEFAULT_MODELS_DIR", "DEFAULT_VALUE_BANK", "GRANT_POLICIES",
    "MininetGatewayContext", "POLICY_MODEL_FILES", "STUB_POLICY", "bank_update_provider", "build_gateway",
    "check_entry_points_available", "entry_points", "is_baseline", "load_bank", "load_factory",
    "payload_seed", "uniform_weights",
]
