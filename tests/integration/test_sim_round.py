"""Integration test I1: one simulated round with Baseline B6 (Sections 10, 11, 17)."""

from __future__ import annotations

import numpy as np
import pytest

from fedgrant.policy.constants import K
from fedgrant.policy.types import PolicyConfig
from fedgrant.sim.run_net import compute_deadline_ns, run_network_round


def test_compute_deadline() -> None:
    # N=20: beta=0.05 -> 199ms, beta=0.10 -> 398ms, beta=0.20 -> 797ms (Section 3.1)
    assert compute_deadline_ns(beta=0.05, num_clients=20, K=169) == 199_000_000
    assert compute_deadline_ns(beta=0.10, num_clients=20, K=169) == 398_000_000
    assert compute_deadline_ns(beta=0.20, num_clients=20, K=169) == 797_000_000


def test_sim_round_b6_baseline() -> None:
    """Integration I1: Run 1 simulated round with Baseline B6 (FG-EWMA).

    Specification:
        - N = 20 clients
        - K = 169 blocks
        - beta = 0.10 (D = 398 ms)
        - Scenario L2 (GE pi=5%, burst 8)
        - Policy: selector="value", predictor="ewma"
    """
    cfg = PolicyConfig(selector="value", predictor="ewma")
    num_clients = 20

    # Load block energies from value bank if available, else synthetic
    bank_path = "data/bank/femnist_full_s0.npz"
    try:
        bank = np.load(bank_path)
        # Bank energies: shape (300, 20, 169), round 1 is index 0
        client_energies = bank["energies_sorted"][0]
        client_weights = bank["client_weights"]
    except Exception:
        client_energies = np.tile(np.linspace(1.0, 0.01, K, dtype=np.float32), (num_clients, 1))
        client_weights = np.full(num_clients, 1.0 / num_clients, dtype=np.float32)

    result = run_network_round(
        round_id=1,
        num_clients=num_clients,
        client_weights=client_weights,
        client_energies=client_energies,
        beta=0.10,
        scenario="L2",
        policy_config=cfg,
        seed=0,
    )

    # 1. Structure checks
    assert result["round_id"] == 1
    assert result["deadline_ms"] == 398.0
    delivered_mask = result["delivered_mask"]
    assert len(delivered_mask) == num_clients
    assert all(len(row) == K for row in delivered_mask)

    # 2. Transmission & throughput metrics
    delivered_counts = result["delivered_counts"]
    total_delivered = sum(delivered_counts)
    assert total_delivered > 0, "At least some blocks must be delivered before D=398ms"

    # Energy ratio eta must be in [0, 1]
    assert 0.0 < result["eta"] <= 1.0

    # Aggregate G has length 117,694
    assert result["aggregate_G"].shape == (117_694,)

    # 3. Check networking metrics
    metrics = result["metrics"]
    assert metrics["grants_issued"] > 0
    assert metrics["grants_delivered"] > 0
    assert metrics["goodput_bps"] > 0
    assert 0.0 < metrics["utilization"] <= 1.0


def test_sim_round_sample_hook_called() -> None:
    """Verify that sample_hook is invoked on grant issuance for ML data generation."""
    hook_calls: list[dict] = []

    def hook(entry: dict) -> None:
        hook_calls.append(entry)

    cfg = PolicyConfig(selector="value", predictor="ewma")
    result = run_network_round(
        round_id=1,
        num_clients=20,
        beta=0.10,
        scenario="L2",
        policy_config=cfg,
        seed=42,
        sample_hook=hook,
    )

    assert len(hook_calls) > 0
    first = hook_calls[0]
    assert first["event"] == "grant_issued"
    assert "client_id" in first
    assert "grant_id" in first
    assert "pos" in first
    assert "t_ns" in first
    assert first["x48"].shape == (48,)
    assert first["ctx"].shape == (16,)
    assert 0.0 <= first["p_hat"] <= 1.0
    assert 0.0 <= first["p_ewma"] <= 1.0
    assert 0.0 <= first["p_oracle"] <= 1.0
    assert 0.0 <= first["p_hmm"] <= 1.0

    resolved_events = [e for e in hook_calls if e["event"] == "grant_resolved"]
    assert len(resolved_events) > 0
    res0 = resolved_events[0]
    assert "y" in res0 and res0["y"] in (0, 1)
    assert res0["obs"].shape == (8,)
    assert res0["obs"][0] == float(res0["y"])


def test_sim_round_split_recorder_integration() -> None:
    """Verify that simulated round events plug directly into SplitRecorder."""
    from fedgrant.ml.schema import SplitRecorder

    rec = SplitRecorder("train", generator="test_sim_round")
    seed = 1001
    D_ns = 398_000_000
    rec.begin_run(seed, seed, burst=8, pi=0.05, beta=0.10, D_ns=D_ns, capacity="constant")

    handles: dict[int, int] = {}

    def hook(entry: dict) -> None:
        if entry["event"] == "grant_issued":
            h = rec.add_sample(
                seed,
                entry["client_id"],
                entry["abs_t_ns"],
                entry["x48"],
                entry["p_ewma"],
                entry["p_hmm"],
                entry["p_oracle"],
            )
            handles[entry["grant_id"]] = h
        elif entry["event"] == "grant_resolved":
            gid = entry["grant_id"]
            if gid in handles:
                rec.add_event(handles[gid], entry["y"], entry["abs_t_ns"], entry["obs"])

    cfg = PolicyConfig(selector="value", predictor="ewma")
    result = run_network_round(
        round_id=1,
        num_clients=20,
        beta=0.10,
        scenario="L2",
        policy_config=cfg,
        seed=seed,
        sample_hook=hook,
    )

    rec.end_run(seed)
    arrays = rec.arrays()
    assert arrays["events"].shape[1] == 8
    assert arrays["ctx"].shape[1] == 16
    assert arrays["x48"].shape[1] == 48
    assert len(arrays["y"]) > 0
