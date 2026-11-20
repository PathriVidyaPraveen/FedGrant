"""FL-over-simulator B2/B3 timing starts rho at ROUND_START receipt."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from fedgrant.policy.constants import K as K_DEFAULT
from fedgrant.sim.engine import EventType, SimulationEngine
from fedgrant.sim.links import AccessLink, access_delay_ns
from fedgrant.sim.run_baselines import run_baseline_round
from fedgrant.sim.run_net import compute_deadline_ns

MS = 1_000_000
N_CLIENTS = 4
SEED = 17
ROUND_START_NS = 2_000_000_000


def _rho_draws(seed: int, duration_ns: int, n: int) -> list[int]:
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(6)[2])
    return [int(rng.integers(0, int(0.10 * duration_ns) + 1)) for _ in range(n)]


def _control_leg_ns(client_id: int) -> int:
    return access_delay_ns(client_id) + 5 * MS


def test_fl_sim_b2_first_data_includes_round_start_downlink(monkeypatch) -> None:
    sent_by_client: dict[int, list[int]] = defaultdict(list)
    original_enqueue = AccessLink.enqueue

    def record_enqueue(link: AccessLink, frame) -> bool:
        sent_by_client[frame.client_id].append(frame.created_at_ns)
        return original_enqueue(link, frame)

    monkeypatch.setattr(AccessLink, "enqueue", record_enqueue)
    duration_ns = compute_deadline_ns(0.10, N_CLIENTS, K_DEFAULT)
    rho = _rho_draws(SEED, duration_ns, N_CLIENTS)

    run_baseline_round(
        "B2", num_clients=N_CLIENTS, beta=0.10, scenario="L1", seed=SEED,
        round_start_abs_ns=ROUND_START_NS, loss_rate=0.0, mean_burst_length=1.0,
    )

    observed_first_sends = [min(sent_by_client[cid]) for cid in range(N_CLIENTS)]
    expected = [ROUND_START_NS + _control_leg_ns(cid) + rho[cid] for cid in range(N_CLIENTS)]
    assert observed_first_sends == expected


def test_fl_sim_b3_ready_arrival_includes_downlink_rho_and_uplink(monkeypatch) -> None:
    ready_arrivals: list[int] = []
    original_schedule = SimulationEngine.schedule

    def record_ready_schedule(engine, delay_ns, event_type, callback, payload=None):
        if event_type is EventType.READY_ARRIVE:
            ready_arrivals.append(engine.now_ns + delay_ns)
        return original_schedule(engine, delay_ns, event_type, callback, payload)

    monkeypatch.setattr(SimulationEngine, "schedule", record_ready_schedule)
    duration_ns = compute_deadline_ns(0.10, N_CLIENTS, K_DEFAULT)
    # This simulator path keeps the seeded readiness draw at nanosecond precision.
    rho = _rho_draws(SEED, duration_ns, N_CLIENTS)

    run_baseline_round(
        "B3", num_clients=N_CLIENTS, beta=0.10, scenario="L1", seed=SEED,
        round_start_abs_ns=ROUND_START_NS, loss_rate=0.0, mean_burst_length=1.0,
    )

    expected = [
        ROUND_START_NS + _control_leg_ns(cid) + rho[cid] + _control_leg_ns(cid)
        for cid in range(N_CLIENTS)
    ]
    assert sorted(ready_arrivals) == sorted(expected)
