from __future__ import annotations

import time

import numpy as np
import pytest

from fedgrant.common.constants import L
from fedgrant.core.gateway import SendGrant, SendRoundStart
from fedgrant.mininet.profile_update import profile_bank, synthetic_update
from fedgrant.mininet.replay import RateReplay
from fedgrant.mininet.stub_policy import create_gateway
from fedgrant.mininet.run_mininet import DEFAULT_STUB_FACTORY, cpu_roles, smoke_duration_ns
from fedgrant.encoder.blocks import encode


def test_stub_gateway_can_issue_first_grant_without_production_wiring():
    core = create_gateway(2)
    actions = core.on_round_start(1, round_id=1, duration_ns=1_000_000_000, K=2, L=L)
    assert len(actions) == 2 and all(isinstance(action, SendRoundStart) for action in actions)
    actions = core.on_ready(0, np.array([2.0, 1.0], dtype=np.float32), t_ns=2)
    assert len(actions) == 1 and isinstance(actions[0], SendGrant)
    assert actions[0].client_id == 0


def test_synthetic_profile_update_encodes_finite_sorted_energies():
    profiles = profile_bank(3)
    assert profiles.shape == (8, 169)
    for energies in profiles:
        assert np.all(np.diff(energies) <= 0)
        update = synthetic_update(energies, seed=4)
        encoded = encode(update)
        assert encoded.blocks.shape == (169, 700)
        assert np.isfinite(encoded.energies_sorted).all()
        assert np.all(np.diff(encoded.energies_sorted) <= 0)


def test_stub_smoke_gets_minimum_exchange_deadline_but_custom_gateway_does_not():
    from fedgrant.sim.run_net import compute_deadline_ns

    small_deadline = compute_deadline_ns(0.1, 3)
    reference_deadline = compute_deadline_ns(0.1, 20)
    assert smoke_duration_ns(0.1, 3, DEFAULT_STUB_FACTORY) == reference_deadline
    assert smoke_duration_ns(0.1, 3, "custom.gateway:create") == small_deadline


def test_rate_replay_applies_in_order_and_stops():
    applied = []
    replay = RateReplay([1000, 2000, 3000], applied.append, interval_s=0.01).start()
    time.sleep(0.06)
    replay.stop()
    assert applied == [1000, 2000, 3000]


def test_cpu_roles_prefers_fast_cores_and_assigns_clients_to_efficiency_cores(tmp_path):
    for cpu, freq in ((0, 4_400_000), (1, 4_400_000), (2, 3_000_000), (3, 3_000_000)):
        path = tmp_path / f"cpu{cpu}" / "cpufreq" / "cpuinfo_max_freq"
        path.parent.mkdir(parents=True)
        path.write_text(str(freq))
    roles = cpu_roles({0, 1, 2, 3}, tmp_path)
    assert roles["gateway"] == 0
    assert roles["telemetry"] == 1
    assert roles["clients"] == [2, 3]
