"""B2/B3 simulator-adapter timing follows the approved READY timing (CHANGELOG 2026-11-13).

Authoritative definition (fedgrant/sim/run_net.py, grant path):
    READY arrival = (delta_i + 5 ms) + rho_i + (delta_i + 5 ms)
i.e. the ROUND_START downlink, the client's readiness delay counted from its
receipt, and the READY uplink. The production clients (UdpClient, baseline_proc)
likewise start rho_i at ROUND_START receipt. The adapter must use the same legs:
B3 READY as above (rho_i from the wire, floored to whole us), B2's first DATA at
(delta_i + 5 ms) + rho_i.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedgrant.baselines.sim_adapter import BOTTLENECK_PROP_NS, control_leg_ns, run_baseline_network_round
from fedgrant.sim.ge import GilbertElliottModel
from fedgrant.sim.links import access_delay_ns
from fedgrant.sim.run_net import compute_deadline_ns, run_network_round

N = 4  # delta_i = 10, 20, 30, 40 ms
SEED = 11
MS = 1_000_000


@pytest.fixture(autouse=True)
def lossless(monkeypatch):
    monkeypatch.setattr(GilbertElliottModel, "for_scenario",
                        classmethod(lambda cls, scenario, seed=None: GilbertElliottModel(0.0, 1.0, seed=seed)))


def _rho_ns(seed: int, duration_ns: int, n: int) -> list[int]:
    """The adapter's and run_net's readiness draws: child 2 of SeedSequence(seed).spawn(6)."""
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(6)[2])
    return [int(rng.integers(0, int(0.10 * duration_ns) + 1)) for _ in range(n)]


def _leg(cid: int) -> int:
    return access_delay_ns(cid) + 5 * MS


def test_control_leg_is_access_delay_plus_bottleneck_propagation() -> None:
    assert BOTTLENECK_PROP_NS == 5 * MS
    assert [control_leg_ns(access_delay_ns(c)) // MS for c in range(N)] == [15, 25, 35, 45]


def test_b2_first_send_is_downlink_plus_rho() -> None:
    duration = 3_000 * MS
    result = run_baseline_network_round(round_id=1, num_clients=N, beta=0.1, scenario="L1", policy="B2",
                                        seed=SEED, d=1, deadline_ns=duration)
    rho = _rho_ns(SEED, duration, N)
    first_sends = [result["transfers"][c].send_times_ns[0] for c in range(N)]
    assert first_sends == [_leg(c) + rho[c] for c in range(N)]
    # The pre-amendment timing (first send at rho_i) must no longer hold.
    assert all(first != rho[c] for c, first in enumerate(first_sends))


def _ready_events(result) -> dict[int, dict]:
    events = [e for e in result["round_events"] if e["event"] in {"ready_received", "ready_rejected"}]
    assert sorted(e["client_id"] for e in events) == list(range(N))  # one READY per client
    return {e["client_id"]: e for e in events}


def test_b3_ready_arrival_is_downlink_rho_uplink() -> None:
    duration = 3_000 * MS  # t_plan = 0.1 D = 300 ms
    result = run_baseline_network_round(round_id=1, num_clients=N, beta=0.1, scenario="L1", policy="B3",
                                        seed=SEED, d=1, deadline_ns=duration)
    rho_us = [r // 1_000 for r in _rho_ns(SEED, duration, N)]  # as carried by ROUND_START
    events = _ready_events(result)
    expected = {c: _leg(c) + rho_us[c] * 1_000 + _leg(c) for c in range(N)}
    assert {c: e["t_ns"] for c, e in events.items()} == expected
    old = {c: rho_us[c] * 1_000 + _leg(c) for c in range(N)}  # pre-amendment: no downlink leg
    assert all(expected[c] != old[c] for c in range(N))
    # B3 registers a READY only if it arrives by t_plan (Section 10).
    plan_ns = result["protocol_result"].plan_ns
    assert {c: e["event"] for c, e in events.items()} == {
        c: "ready_received" if expected[c] <= plan_ns else "ready_rejected" for c in range(N)}


def test_b3_ready_times_match_run_net_ready_times_for_the_same_seed() -> None:
    # Cross-implementation check: identical seed, N, D -> identical rho draws. The
    # adapter takes rho from the wire (whole us); run_net keeps ns.
    duration = compute_deadline_ns(0.10, N)
    adapter = run_baseline_network_round(round_id=1, num_clients=N, beta=0.1, scenario="L1", policy="B3",
                                         seed=SEED)
    sim_events: list[dict] = []
    run_network_round(round_id=1, num_clients=N, beta=0.1, scenario="L1", seed=SEED,
                      client_energies=np.tile(np.linspace(1.0, 0.01, 169, dtype=np.float32), (N, 1)),
                      event_hook=sim_events.append)
    sim_ready = {e["client_id"]: e["t_ns"] for e in sim_events if e["event"] == "ready_received"}
    rho = _rho_ns(SEED, duration, N)
    adapter_ready = {c: e["t_ns"] for c, e in _ready_events(adapter).items()}
    assert set(sim_ready) == set(adapter_ready) == set(range(N))
    for c in range(N):
        assert adapter_ready[c] == sim_ready[c] - rho[c] % 1_000


def test_b3_plan_downlink_and_paced_data_are_unchanged() -> None:
    # PLAN reaches the client one control leg after it is sent; DATA starts there.
    duration = 3_000 * MS
    result = run_baseline_network_round(round_id=1, num_clients=N, beta=0.1, scenario="L1", policy="B3",
                                        seed=SEED, d=1, deadline_ns=duration)
    plans = result["plans"]
    for c in range(N):
        if plans[c].blocks:
            assert result["transfers"][c].send_times_ns[0] == plans[c].sent_ns + _leg(c)
