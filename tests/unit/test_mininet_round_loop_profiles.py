"""Shared round loop, readiness draws and bank-backed client updates (D-06, D-10, D-11)."""

from __future__ import annotations

import asyncio
import hashlib
import json

import numpy as np
import pytest

from fedgrant.encoder.blocks import encode
from fedgrant.mininet.round_loop import (
    RoundSchedule,
    readiness_delays_ns,
    readiness_delays_us,
    run_round_loop,
)
from fedgrant.mininet.round_state import RoundStateFile
from fedgrant.mininet.profile_update import synthetic_update
from fedgrant.mininet.value_profiles import BankUpdateProvider, ValueProfileError, payload_seed
from fedgrant.policy.constants import K
from fedgrant.sim.links import access_delay_ns
from fedgrant.sim.run_net import compute_deadline_ns, run_network_round
from fedgrant.sim.seeding import network_round_seed
from fedgrant.sim.value_bank import load_value_bank


@pytest.mark.parametrize(("master_seed", "round_id", "n"), [(0, 1, 20), (2, 37, 20), (5, 3, 4)])
def test_ready_arrival_includes_both_control_path_legs_and_keeps_seeded_draws(master_seed, round_id, n) -> None:
    D = compute_deadline_ns(0.10, n)
    events: list[dict] = []
    run_network_round(round_id=round_id, num_clients=n, beta=0.10, scenario="L1",
                      seed=network_round_seed(master_seed, round_id), event_hook=events.append)
    ready = {e["client_id"]: e["t_ns"] for e in events if e["event"] == "ready_received"}
    # ROUND_START reaches the client after δ+5 ms; READY returns after ρ plus δ+5 ms.
    sim_rho = [ready[cid] - 2 * (access_delay_ns(cid) + 5_000_000) for cid in range(n)]
    assert readiness_delays_ns(master_seed, round_id, n, D) == sim_rho
    assert all(0 <= rho <= int(0.1 * D) for rho in sim_rho)
    assert readiness_delays_us(master_seed, round_id, n, D) == {cid: r // 1000 for cid, r in enumerate(sim_rho)}


def test_readiness_differs_by_round_and_is_shared_across_policies() -> None:
    D = compute_deadline_ns(0.10, 20)
    assert readiness_delays_ns(0, 1, 20, D) == readiness_delays_ns(0, 1, 20, D)
    assert readiness_delays_ns(0, 1, 20, D) != readiness_delays_ns(0, 2, 20, D)
    assert readiness_delays_ns(0, 1, 20, D) != readiness_delays_ns(1, 1, 20, D)


class _FakeClock:
    def __init__(self, start: int) -> None:
        self.now = start

    def __call__(self) -> int:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += int(round(seconds * 1e9))


class _Hooks:
    def __init__(self, clock, state: RoundStateFile) -> None:
        self.clock, self.state, self.calls = clock, state, []

    def start_round(self, round_id, start_ns, deadline_ns, rho_us):
        self.calls.append(("start", round_id, start_ns, deadline_ns, dict(rho_us), self.state.read()["active"]))

    async def on_deadline(self, round_id, deadline_ns):
        self.calls.append(("deadline", round_id, self.clock(), self.state.read()["active"]))

    def finalize_round(self, round_id):
        # Late drain has elapsed; the round is already inactive for samplers.
        self.calls.append(("finalize", round_id, self.clock(), self.state.read()["active"]))


def test_round_loop_schedule_compute_window_deadline_and_late_drain(tmp_path) -> None:
    clock = _FakeClock(1_000_000_000_000)
    state = RoundStateFile(tmp_path / "round_state.json")
    hooks = _Hooks(clock, state)
    schedule = RoundSchedule(duration_ns=398_000_000, late_drain_ns=100_000_000, inter_round_ns=5_000_000)
    timings = asyncio.run(run_round_loop(rounds=2, schedule=schedule, hooks=hooks, master_seed=4,
                                         num_clients=3, round_state=state, clock_ns=clock, sleep=clock.sleep))
    t0 = 1_000_000_000_000
    r1_start = t0 + 1_000_000_000
    assert hooks.calls[0] == ("start", 1, r1_start, r1_start + 398_000_000,
                              readiness_delays_us(4, 1, 3, 398_000_000), True)
    assert hooks.calls[1] == ("deadline", 1, r1_start + 398_000_000, True)
    assert hooks.calls[2] == ("finalize", 1, r1_start + 498_000_000, False)
    r2_start = r1_start + 498_000_000 + 5_000_000 + 1_000_000_000
    assert hooks.calls[3][:4] == ("start", 2, r2_start, r2_start + 398_000_000)
    assert [t.round_id for t in timings] == [1, 2]
    assert timings[1].drain_end_ns - timings[1].deadline_ns == 100_000_000
    assert state.read()["active"] is False


def test_schedule_validation_and_spec_defaults() -> None:
    schedule = RoundSchedule.for_experiment(0.10, 20)
    assert (schedule.duration_ns, schedule.compute_window_ns) == (398_000_000, 1_000_000_000)
    with pytest.raises(ValueError):
        RoundSchedule(duration_ns=0)
    with pytest.raises(ValueError):
        RoundSchedule(duration_ns=1, late_drain_ns=-1)


def _fixture_bank(tmp_path, n: int = 4):
    """Test-only bank; real runs must load data/bank/femnist_full_s0.npz."""
    rng = np.random.default_rng(3)
    energies = -np.sort(-rng.uniform(0.0, 2.0, size=(5, n, K)).astype(np.float32), axis=2)
    weights = np.arange(1, n + 1, dtype=np.float32)
    weights /= weights.sum()
    provenance = tmp_path / "provenance.json"
    provenance.write_text('{"fixture": true}\n')
    path = tmp_path / "bank.npz"
    np.savez_compressed(path, energies_sorted=energies, client_weights=weights,
                        round_ids=np.arange(1, 6), seed=np.asarray(0), num_rounds=np.asarray(5),
                        dataset_metadata_json=np.asarray(json.dumps({
                            "source": "test-fixture", "provenance_path": str(provenance),
                            "provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest()})))
    return path, energies, weights


def test_bank_provider_uses_real_profile_draw_and_simulator_payload_convention(tmp_path) -> None:
    path, _energies, weights = _fixture_bank(tmp_path)
    provider = BankUpdateProvider.from_bank(path, 4, master_seed=11, client_id=2)
    bank = load_value_bank(path, 4)
    for round_id in (1, 2, 7):
        profile = bank.energies_for_run_round(11, round_id)[2]
        np.testing.assert_array_equal(provider.energies(round_id), profile)
        update = provider(round_id)
        assert update.dtype == np.float32
        # Payload built by the existing helper with the contract's seed.
        np.testing.assert_array_equal(
            update, synthetic_update(profile, seed=payload_seed(11, 2, round_id)))
        # The encoder sees (to float32 accuracy) the stored profile.
        encoded = encode(update)
        np.testing.assert_allclose(encoded.energies_sorted, profile, rtol=1e-4, atol=1e-6)
    assert provider.weight == pytest.approx(weights[2])
    assert payload_seed(11, 2, 7) == int(np.random.SeedSequence([11, 6, 2, 2]).generate_state(1)[0])
    assert payload_seed(11, 2, 7) != payload_seed(11, 3, 7) != payload_seed(11, 2, 8)
    np.testing.assert_array_equal(provider(3), BankUpdateProvider.from_bank(path, 4, 11, 2)(3))
    meta = provider.metadata()
    assert meta["bank_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    json.dumps(meta)


def test_bank_provider_has_no_synthetic_fallback(tmp_path) -> None:
    with pytest.raises(ValueProfileError, match="value bank unavailable"):
        BankUpdateProvider.from_bank(tmp_path / "absent.npz", 20, 0, 0)
    path, _, _ = _fixture_bank(tmp_path)
    with pytest.raises(ValueProfileError, match="N=20"):
        BankUpdateProvider.from_bank(path, 20, 0, 0)  # bank defines 4 clients
    with pytest.raises(ValueProfileError, match="outside"):
        BankUpdateProvider.from_bank(path, 4, 0, 4)
