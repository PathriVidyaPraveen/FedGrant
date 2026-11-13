"""Production GatewayCore factories for Mininet runs (B4, B5, B6, B7, FG-GRU, FG-MLP).

The daemon uses the *same* provider assembly as the simulator
(``fedgrant.sim.run_net.GatewaySim``: eligibility, selector, pacing, timeout and
estimator hooks), so Mininet and its simulator twins schedule identically
(specification Section 6). ``LiveClockGatewaySim`` only changes where the
providers read the round origin and deadline: from the live
``GatewayCore.round`` that the UDP daemon starts, instead of from
``GatewaySim.on_round_start`` (which the daemon never calls).

Factories return a long-lived ``GatewayCore``; estimator and predictor state
persist across rounds exactly as in the simulator. No construction-time
fallback to another policy exists: an unavailable checkpoint or value bank
raises. The runtime P-EWMA fallback of Section 9 is ``FallbackPredictor``,
applied automatically by ``GatewayCore``.

Runner contract (``fedgrant.mininet.gateway_proc`` with
``fedgrant.mininet.contract.build_gateway``): the gateway process builds one
``MininetGatewayContext`` (weights from the value bank, seed, D, K, L, models
directory) and calls ``--gateway-factory module:function`` once as
``factory(ctx)``; the returned ``GatewayCore`` serves every round. Use
``fedgrant.mininet.production:create_gateway`` there. ``create_b4_gateway`` /
``create_b6_gateway`` / ``create_fg_gru_gateway`` take a bare client count and
load the bank weights themselves, for direct callers only.

The factory opens no sockets and starts no event loop or thread; the gateway
process drives every lifecycle callback.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fedgrant.core.gateway import GatewayCore
from fedgrant.policy.constants import D_MODEL, K as K_DEFAULT, L as L_DEFAULT
from fedgrant.policy.types import PolicyConfig
from fedgrant.predictors.factory import PROD_FILES, make_predictor
from fedgrant.sim.run_net import GatewaySim, compute_deadline_ns
from fedgrant.sim.value_bank import load_value_bank

VALUE_BANK_PATH = Path("data/bank/femnist_full_s0.npz")
MODELS_DIR = Path("models")

# Section 10 baselines that use the FedGrant gateway (Mininet: B4, B6, FG-GRU).
POLICY_CONFIGS: dict[str, PolicyConfig] = {
    "B4": PolicyConfig(selector="rr", predictor="ewma"),
    "B5": PolicyConfig(selector="srpt", predictor="ewma"),
    "B6": PolicyConfig(selector="value", predictor="ewma"),
    "B7": PolicyConfig(selector="value", predictor="hmm"),
    "FG-GRU": PolicyConfig(selector="value", predictor="gru"),
    "FG-MLP": PolicyConfig(selector="value", predictor="mlp"),
}
SIMULATOR_ONLY_POLICIES = frozenset({"ORACLE"})


class ProductionGatewayError(ValueError):
    """A production gateway cannot be built as configured."""


class LiveClockGatewaySim(GatewaySim):
    """GatewaySim providers bound to the round window of a live GatewayCore."""

    @property
    def round_start_abs_ns(self) -> int:
        core = self.__dict__.get("core")
        if core is not None and core.round is not None:
            return int(core.round.start_ns)
        return int(self.__dict__.get("_fallback_round_start_ns", 0))

    @round_start_abs_ns.setter
    def round_start_abs_ns(self, value: int) -> None:
        self.__dict__["_fallback_round_start_ns"] = int(value)

    @property
    def D_ns(self) -> int:
        core = self.__dict__.get("core")
        if core is not None and core.round is not None:
            return int(core.round.deadline_ns - core.round.start_ns)
        return int(self.__dict__["_fallback_D_ns"])

    @D_ns.setter
    def D_ns(self, value: int) -> None:
        self.__dict__["_fallback_D_ns"] = int(value)


@dataclass(frozen=True)
class GatewayContext:
    """Factory input; field names follow the runner's draft ``MininetGatewayContext``
    (docs/MININET_INTEGRATION_CONTRACT.md). ``create_gateway`` accepts any object
    with these attributes, so the runner may define its own dataclass.
    """

    num_clients: int
    policy: str
    client_weights: np.ndarray
    seed: int = 0
    deadline_ns: int | None = None  # D; the live round window takes precedence once a round starts
    K: int = K_DEFAULT
    L: int = L_DEFAULT
    models_dir: str | Path = MODELS_DIR
    log_dir: str | None = None  # accepted for contract compatibility; the factory writes no logs
    d_model: int = D_MODEL
    policy_config: PolicyConfig | None = field(default=None)


def _resolved_config(ctx: Any) -> PolicyConfig:
    policy = str(ctx.policy)
    if policy in SIMULATOR_ONLY_POLICIES:
        raise ProductionGatewayError(f"{policy} is simulator-only (P-ORACLE needs true simulator state)")
    config = getattr(ctx, "policy_config", None) or POLICY_CONFIGS.get(policy)
    if config is None:
        raise ProductionGatewayError(
            f"policy {policy!r} has no GatewayCore configuration; "
            f"B1/B2/B3 use fedgrant.mininet.baseline_proc; known: {sorted(POLICY_CONFIGS)}"
        )
    if config.explore_eps != 0.0:
        raise ProductionGatewayError("explore_eps must be 0 in production (Section 9)")
    if config.predictor == "oracle":
        raise ProductionGatewayError("P-ORACLE is simulator-only")
    return config


def _validated_weights(ctx: Any) -> np.ndarray:
    n = ctx.num_clients
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ProductionGatewayError("num_clients must be a positive integer")
    weights = np.asarray(ctx.client_weights, dtype=np.float32)
    if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ProductionGatewayError(f"client_weights must be {n} finite non-negative values")
    if not np.isclose(float(weights.sum(dtype=np.float64)), 1.0):
        raise ProductionGatewayError("client_weights must sum to 1")
    return weights


def create_gateway(ctx: Any) -> GatewayCore:
    """Build the production GatewayCore for ``ctx.policy`` (raises; never substitutes).

    ``ctx`` is a ``GatewayContext`` or any object with the same attributes
    (``num_clients``, ``policy``, ``client_weights``; optional ``seed``,
    ``deadline_ns``, ``K``, ``L``, ``models_dir``, ``log_dir``).

    This is the runner's ``--gateway-factory`` entry point:
    ``fedgrant.mininet.gateway_proc`` builds a
    ``fedgrant.mininet.contract.MininetGatewayContext`` and calls
    ``create_gateway(ctx)`` once per run.
    """
    if not all(hasattr(ctx, name) for name in ("num_clients", "policy", "client_weights")):
        raise TypeError(
            f"create_gateway expects a gateway context (MininetGatewayContext), got "
            f"{type(ctx).__name__}; for a bare client count use create_b4_gateway, "
            "create_b6_gateway or create_fg_gru_gateway"
        )
    config = _resolved_config(ctx)
    weights = _validated_weights(ctx)
    K = int(getattr(ctx, "K", K_DEFAULT))
    L = int(getattr(ctx, "L", L_DEFAULT))
    predictor = make_predictor(config.predictor, getattr(ctx, "models_dir", MODELS_DIR))  # raises if absent
    initial_duration = getattr(ctx, "deadline_ns", None) or compute_deadline_ns(0.10, ctx.num_clients, K)
    adapter = LiveClockGatewaySim(
        num_clients=ctx.num_clients,
        client_weights=weights,
        predictor=predictor,
        selector=config.selector,
        policy_config=config,
        D_ns=int(initial_duration),
        K=K,
        L=L,
        d_model=int(getattr(ctx, "d_model", D_MODEL)),
        rng=np.random.default_rng(np.random.SeedSequence(int(getattr(ctx, "seed", 0)))),
    )
    core = adapter.core
    # The decision hook only feeds simulator sample recording; production
    # does not pay for per-grant feature construction it would discard.
    core.decision_hook = None
    return core


def gateway_metadata(ctx: Any) -> dict[str, Any]:
    """Provenance for meta.json: policy, configuration, checkpoint hash."""
    config = _resolved_config(ctx)
    checkpoint = None
    if config.predictor in PROD_FILES:
        path = Path(getattr(ctx, "models_dir", MODELS_DIR)) / PROD_FILES[config.predictor]
        checkpoint = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        }
    return {
        "policy": ctx.policy,
        "policy_config": {
            "selector": config.selector, "predictor": config.predictor,
            "pred_in_pacing": config.pred_in_pacing, "pred_in_eligibility": config.pred_in_eligibility,
            "retransmit": config.retransmit, "explore_eps": config.explore_eps,
        },
        "checkpoint": checkpoint,
        "client_weights_sha256": hashlib.sha256(_validated_weights(ctx).tobytes()).hexdigest(),
        "num_clients": ctx.num_clients,
        "seed": int(getattr(ctx, "seed", 0)),
        "gateway_logic": "fedgrant.sim.run_net.GatewaySim providers on a live GatewayCore",
    }


def bank_client_weights(num_clients: int, bank_path: str | Path | None = None) -> np.ndarray:
    """Real FEMNIST bank weights w_i (Section 5.2); raises if unavailable."""
    path = Path(VALUE_BANK_PATH if bank_path is None else bank_path)
    try:
        bank = load_value_bank(path, num_clients)
    except FileNotFoundError as exc:
        raise ProductionGatewayError(
            f"real value bank required for production weights and not found: {exc}"
        ) from exc
    except ValueError as exc:
        raise ProductionGatewayError(
            f"value bank {path} cannot supply weights for N={num_clients}: {exc} "
            "(the FULL seed-0 bank defines N=20; see design decision D-17)"
        ) from exc
    return bank.client_weights


def _from_bank(policy: str, num_clients: int) -> GatewayCore:
    if isinstance(num_clients, bool) or not isinstance(num_clients, (int, np.integer)):
        raise TypeError(
            f"create_{policy.lower().replace('-', '_')}_gateway takes a client count, got "
            f"{type(num_clients).__name__}; gateway_proc passes a MininetGatewayContext, so use "
            "--gateway-factory fedgrant.mininet.production:create_gateway"
        )
    return create_gateway(GatewayContext(
        num_clients=int(num_clients), policy=policy, client_weights=bank_client_weights(int(num_clients)),
    ))


def create_b4_gateway(num_clients: int) -> GatewayCore:
    """B4: selector rr, P-EWMA; weights from the real value bank."""
    return _from_bank("B4", num_clients)


def create_b6_gateway(num_clients: int) -> GatewayCore:
    """B6: selector value, P-EWMA; weights from the real value bank."""
    return _from_bank("B6", num_clients)


def create_fg_gru_gateway(num_clients: int) -> GatewayCore:
    """FG-GRU: selector value, P-GRU (models/gru_prod.npz); weights from the real bank."""
    return _from_bank("FG-GRU", num_clients)


__all__ = [
    "GatewayContext", "LiveClockGatewaySim", "POLICY_CONFIGS", "ProductionGatewayError",
    "VALUE_BANK_PATH", "bank_client_weights", "create_b4_gateway", "create_b6_gateway",
    "create_fg_gru_gateway", "create_gateway", "gateway_metadata",
]
