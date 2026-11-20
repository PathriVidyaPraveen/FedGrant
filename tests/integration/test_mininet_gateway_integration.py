"""Group 2 integration checks for the Mininet gateway path (no Mininet required).

* The real ``gateway_proc.serve`` loads a production factory through its
  ``--gateway-factory module:function`` path and serves rounds to real grant
  clients over localhost UDP. Only the bind/endpoint addresses are rewritten,
  because 10.0.0.x exists only inside the Mininet topology.
* Round-state semantics as seen by the actual consumers (``round_at`` used by
  the queue/process samplers, and ``RateReplay``).
* FG-GRU decision parity between the simulator and the production core.

The value bank here is a test-only fixture. None of this is an M1/M2 result.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import threading
from collections import Counter

import numpy as np
import pytest

from fedgrant.core.client_core import ClientCore
from fedgrant.core.gateway import GatewayCore
from fedgrant.mininet import gateway_proc, production
from fedgrant.mininet.contract import build_gateway
from fedgrant.mininet.production import GatewayContext, LiveClockGatewaySim, create_gateway
from fedgrant.mininet.replay import RateReplay
from fedgrant.mininet.round_loop import RoundSchedule, run_round_loop
from fedgrant.mininet.round_state import RoundStateFile
from fedgrant.mininet.value_profiles import BankUpdateProvider
from fedgrant.net.udp_client import UdpClient
from fedgrant.policy.constants import K, L
from fedgrant.policy.pacing import pacing_p
from fedgrant.policy.types import PolicyConfig
from fedgrant.predictors.factory import make_predictor
from fedgrant.predictors.fallback import FallbackPredictor
from fedgrant.predictors.gru_numpy import PGRUNumpy
from fedgrant.sim.run_net import run_network_round
from fedgrant.sim.value_bank import load_value_bank

N = 3
SEED = 4


def _fixture_bank(tmp_path, n: int = N):
    """Test-only bank; real runs require data/bank/femnist_full_s0.npz."""
    rng = np.random.default_rng(9)
    energies = -np.sort(-rng.uniform(0.05, 1.0, size=(4, n, K)).astype(np.float32), axis=2)
    weights = np.asarray([0.2, 0.3, 0.5][:n], dtype=np.float32)
    weights /= weights.sum()
    provenance = tmp_path / "provenance.json"
    provenance.write_text('{"fixture": true}\n')
    path = tmp_path / "bank.npz"
    np.savez_compressed(path, energies_sorted=energies, client_weights=weights, round_ids=np.arange(1, 5),
                        seed=np.asarray(0), num_rounds=np.asarray(4),
                        dataset_metadata_json=np.asarray(json.dumps({
                            "source": "test-fixture", "provenance_path": str(provenance),
                            "provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest()})))
    return path


def _runner_args(tmp_path, bank_path, policy: str, **overrides) -> argparse.Namespace:
    """The arguments run_mininet passes to gateway_proc for a grant policy."""
    log_dir = tmp_path / "run"
    values = dict(
        clients=N, rounds=2, duration_ns=300_000_000, blocks=K, block_length=L, port=9000, client_port=9001,
        # Gateway and clients share one event loop here. With seeded readiness
        # delays (rho_i <= 0.1 D) a 50 + 50 ms gap let round-1 DATA answered late
        # arrive after round 2 started (dropped and counted, Section 8). Real runs
        # use drain 0.1 s + Tc 1.0 s; keep a realistic gap.
        start_delay_s=0.05, compute_window_s=0.3, late_drain_s=0.1, inter_round_s=0.0,
        gateway_factory=PRODUCTION_FACTORY, log_dir=str(log_dir), round_state=str(log_dir / ".round_state.json"),
        profile_gateway=False, policy=policy, seed=SEED, value_bank=str(bank_path), models_dir="models",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


PRODUCTION_FACTORY = "fedgrant.mininet.production:create_gateway"


def test_factory_load_path_returns_a_passive_long_lived_core(tmp_path) -> None:
    bank_path = _fixture_bank(tmp_path)
    threads_before = threading.active_count()
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()  # called outside any event loop, as gateway_proc's caller is
    for policy, primary_type in (("B4", "PEWMA"), ("B6", "PEWMA"), ("FG-GRU", "PGRUNumpy")):
        args = _runner_args(tmp_path, bank_path, policy)
        ctx = gateway_proc.build_context(args)  # exactly gateway_proc's input
        core = build_gateway(PRODUCTION_FACTORY, ctx)  # exactly gateway_proc's call
        assert isinstance(core, GatewayCore) and type(core.predictor.primary).__name__ == primary_type
        assert core.eligibility.sim.cfg == ctx.policy_config
        assert [core.clients[i].w for i in range(N)] == pytest.approx(ctx.client_weights)
        assert core.round is None and core.phase.name == "IDLE"  # nothing started by the factory
    assert threading.active_count() == threads_before  # no background threads
    # The per-policy wrappers take a client count; given the runner's context
    # they fail with a pointer to the right factory, not an obscure error.
    with pytest.raises(TypeError, match="production:create_gateway"):
        build_gateway("fedgrant.mininet.production:create_b6_gateway", ctx)
    with pytest.raises(TypeError, match="MininetGatewayContext"):
        production.create_gateway(N)
    with pytest.raises(FileNotFoundError):
        gateway_proc.build_context(_runner_args(tmp_path, tmp_path / "absent.npz", "B6"))
    with pytest.raises(FileNotFoundError, match="gru_prod.npz"):
        build_gateway(PRODUCTION_FACTORY, gateway_proc.build_context(
            _runner_args(tmp_path, bank_path, "FG-GRU", models_dir=str(tmp_path / "no-models"))))


@pytest.mark.parametrize("policy", ["B6", "FG-GRU"])
def test_gateway_proc_serves_rounds_with_the_production_factory(policy, tmp_path, monkeypatch) -> None:
    bank_path = _fixture_bank(tmp_path)
    bank = load_value_bank(bank_path, N)
    bound: list = []
    gateway_ready: list[asyncio.Future] = []
    original_bind = gateway_proc.LoggingUdpGateway.bind.__func__

    async def localhost_bind(cls, core, *, local_addr, endpoints: dict, **kwargs):
        bound.append(core)
        assert local_addr == ("10.0.0.254", 9000)  # what gateway_proc asks for in Mininet
        # Kernel-assigned port: no probe-then-rebind race with other processes.
        gateway = await original_bind(cls, core, local_addr=("127.0.0.1", 0), endpoints={}, **kwargs)
        gateway_ready[0].set_result(gateway)
        return gateway

    monkeypatch.setattr(gateway_proc.LoggingUdpGateway, "bind", classmethod(localhost_bind))
    # D = 600 ms: gateway and clients share one event loop here, and under CPU
    # load a 300 ms round could pass with every DATA arriving after D.
    args = _runner_args(tmp_path, bank_path, policy, duration_ns=600_000_000)
    log_dir = tmp_path / "run"

    async def scenario():
        gateway_ready.append(asyncio.get_running_loop().create_future())
        server = asyncio.create_task(gateway_proc.serve(args))
        # Clients bind once the gateway is bound (its first ROUND_START waits
        # start_delay + Tc = 0.35 s), each on a kernel-assigned port that is then
        # registered as its endpoint.
        gateway = await asyncio.wait_for(asyncio.shield(gateway_ready[0]), 10.0)
        gw_port = gateway.transport.get_extra_info("sockname")[1]
        clients = []
        for cid in range(N):
            core = ClientCore(cid, error_feedback=False)  # network-only values (D-19)
            client = await UdpClient.bind(core, gateway_addr=("127.0.0.1", gw_port), local_addr=("127.0.0.1", 0),
                                          update_provider=BankUpdateProvider(bank, SEED, cid))
            gateway.set_endpoint(cid, ("127.0.0.1", client.transport.get_extra_info("sockname")[1]))
            clients.append(client)
        async def follow(client):
            # Same loop as client_proc: each client waits for its own rounds,
            # concurrently with the others (the round-end event is level-triggered).
            rounds = []
            for _ in range(2):
                rounds.append(await client.wait_for_round_end(timeout=5.0))
                client._round_end_event.clear()
            return rounds

        finished = await asyncio.gather(*(follow(client) for client in clients))
        await server
        for client in clients:
            await client.close()
        return finished

    finished = asyncio.run(scenario())
    assert finished == [[1, 2]] * N
    assert len(bound) == 1  # factory called once; one long-lived core served both rounds
    core = bound[0]
    assert isinstance(core.eligibility.sim, LiveClockGatewaySim) and core.last_round_id == 2
    assert [core.clients[i].w for i in range(N)] == pytest.approx(bank.client_weights)
    # READY energies received by the production core are the bank profiles.
    for cid in range(N):
        np.testing.assert_allclose(core.energies[cid], bank.energies_for_run_round(SEED, 2)[cid],
                                   rtol=1e-4, atol=1e-6)
    summary = json.loads((log_dir / "gateway_summary.json").read_text())
    assert [r["round_id"] for r in summary] == [1, 2]
    assert all(r["accepted_data"] > 0 for r in summary)
    # The only invalid packets allowed are previous-round DATA that arrives after
    # the next ROUND_START (Section 8: rejected and counted as stale). Under CPU
    # load the shared event loop can delay round-1 DATA past the 0.4 s gap.
    stats = core.stats
    assert stats.invalid_events == stats.rejected_data == stats.stale_round_data
    assert summary[0]["invalid_events"] == 0
    with (log_dir / "round_metrics.csv").open(newline="") as stream:
        metrics = list(csv.DictReader(stream))
    assert [int(m["round_id"]) for m in metrics] == [1, 2]
    assert all(0.0 < float(m["eta"]) <= 1.0 for m in metrics)
    # Probe logging: the issued rows flagged probe=1 in grants.csv are exactly the
    # round's probe_grants in round_metrics.csv, at most one per client per round,
    # and the flag is blank on every other grant event.
    with (log_dir / "grants.csv").open(newline="") as stream:
        grant_rows = list(csv.DictReader(stream))
    issued = [g for g in grant_rows if g["event"] == "issued"]
    assert issued and all(g["probe"] in {"0", "1"} for g in issued)
    assert all(g["probe"] == "" for g in grant_rows if g["event"] != "issued")
    probes = Counter((int(g["round_id"]), int(g["client_id"])) for g in issued if g["probe"] == "1")
    assert all(count == 1 for count in probes.values())
    for m in metrics:
        assert int(m["probe_grants"]) == sum(1 for (r, _) in probes if r == int(m["round_id"]))
    state = RoundStateFile(log_dir / ".round_state.json").read()
    assert (state["round_id"], state["active"]) == (2, False)


class _Hooks:
    def __init__(self, state: RoundStateFile, clock, fail_at: str | None = None) -> None:
        self.state, self.clock, self.fail_at, self.seen = state, clock, fail_at, []

    def start_round(self, round_id, start_ns, deadline_ns, rho_us):
        self.seen.append(("start", self.state.round_at(start_ns), self.state.round_at(deadline_ns),
                          self.state.round_at(deadline_ns + 1)))
        if self.fail_at == "start":
            raise RuntimeError("start hook failed")

    def on_deadline(self, round_id, deadline_ns):
        self.seen.append(("deadline", self.state.round_at(self.clock())))
        if self.fail_at == "deadline":
            raise RuntimeError("deadline hook failed")

    def finalize_round(self, round_id):
        # After the late drain: samplers label samples round 0 (outside [start, D]).
        self.seen.append(("finalize", self.state.round_at(self.clock()), self.state.read()["active"]))


class _Clock:
    def __init__(self) -> None:
        self.now = 5_000_000_000

    def __call__(self) -> int:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += int(round(seconds * 1e9))


def test_round_state_window_as_seen_by_sampler_consumers(tmp_path) -> None:
    state = RoundStateFile(tmp_path / "state.json")
    clock = _Clock()
    hooks = _Hooks(state, clock)
    schedule = RoundSchedule(duration_ns=200_000_000, compute_window_ns=10_000_000, late_drain_ns=50_000_000)
    asyncio.run(run_round_loop(rounds=2, schedule=schedule, hooks=hooks, master_seed=0, num_clients=2,
                               round_state=state, clock_ns=clock, sleep=clock.sleep))
    # Active (round id) on [start, D]; samples after D belong to no round.
    assert hooks.seen == [("start", 1, 1, 0), ("deadline", 1), ("finalize", 0, False),
                          ("start", 2, 2, 0), ("deadline", 2), ("finalize", 0, False)]


@pytest.mark.parametrize("fail_at", ["start", "deadline"])
def test_failing_hook_leaves_the_round_state_inactive(fail_at, tmp_path) -> None:
    state = RoundStateFile(tmp_path / "state.json")
    clock = _Clock()
    schedule = RoundSchedule(duration_ns=100_000_000, compute_window_ns=0, late_drain_ns=0)
    with pytest.raises(RuntimeError, match="hook failed"):
        asyncio.run(run_round_loop(rounds=2, schedule=schedule, hooks=_Hooks(state, clock, fail_at),
                                   master_seed=0, num_clients=1, round_state=state, clock_ns=clock,
                                   sleep=clock.sleep))
    assert state.read()["active"] is False and state.read()["round_id"] == 1
    assert state.round_at(clock()) == 0


def test_cancelled_round_leaves_the_round_state_inactive(tmp_path) -> None:
    state = RoundStateFile(tmp_path / "state.json")

    async def scenario():
        started = asyncio.Event()

        class Hooks:
            def start_round(self, *a):
                started.set()

            def on_deadline(self, *a):
                pass

            def finalize_round(self, *a):
                pass

        task = asyncio.create_task(run_round_loop(
            rounds=1, schedule=RoundSchedule(duration_ns=10_000_000_000, compute_window_ns=0), hooks=Hooks(),
            master_seed=0, num_clients=1, round_state=state))
        await started.wait()
        assert state.read()["active"] is True
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert state.read()["active"] is False


def test_trace_replay_follows_round_state_and_stops_at_the_deadline(tmp_path) -> None:
    # Real clock (replay is a thread on wall time): two 200 ms rounds, 50 ms bins.
    state_path = tmp_path / "state.json"
    changes: list[tuple] = []
    replay = RateReplay([1e6, 2e6, 3e6, 4e6, 5e6, 6e6], lambda rate: None, interval_s=0.05, repeat=True,
                        round_state_path=state_path, on_change=lambda *c: changes.append(c))

    class Hooks:
        windows: dict[int, tuple[int, int]] = {}

        def start_round(self, round_id, start_ns, deadline_ns, rho_us):
            self.windows[round_id] = (start_ns, deadline_ns)

        def on_deadline(self, *a):
            pass

        def finalize_round(self, *a):
            pass

    hooks = Hooks()
    RoundStateFile(state_path).write(round_id=0, active=False)
    replay.start()
    try:
        asyncio.run(run_round_loop(
            rounds=2, schedule=RoundSchedule(duration_ns=200_000_000, compute_window_ns=50_000_000,
                                             late_drain_ns=100_000_000),
            hooks=hooks, master_seed=0, num_clients=1, round_state=RoundStateFile(state_path)))
    finally:
        replay.stop()
    for round_id in (1, 2):
        start, deadline = hooks.windows[round_id]
        mine = [c for c in changes if c[1] == round_id]
        # Bins 0, 50, 100, 150 ms from this round's own start; nothing at/after D.
        assert [c[2] for c in mine] == [0, 1, 2, 3]
        assert [c[4] for c in mine] == [start + k * 50_000_000 for k in range(4)]
        assert all(c[4] < deadline for c in mine)
    assert {c[1] for c in changes} == {1, 2}  # no replay outside an active round


SIM_COMPAT_FIELDS = {"selection_method", "selected_client_id", "p_pacing", "p_target"}
GRANT_FIELDS = ("grant_id", "send_pos", "timeout_ns")


def test_fg_gru_production_core_matches_simulator_decisions_over_two_rounds() -> None:
    # The 2 ms predictor time limit is wall-clock (FallbackPredictor docstring);
    # it is lifted on both sides and both must report zero fallbacks, so every
    # decision below really used P-GRU from models/gru_prod.npz.
    n = 6
    weights = np.full(n, 1.0 / n, dtype=np.float32)
    energies = np.tile(np.linspace(1.0, 0.01, K, dtype=np.float32), (n, 1))
    sim_predictor = FallbackPredictor(make_predictor("gru"), time_limit_ns=10**15)
    runs, gateway = [], None
    for round_id in (1, 2):
        decisions, events = [], []
        result = run_network_round(
            round_id=round_id, num_clients=n, client_weights=weights, client_energies=energies, beta=0.4,
            scenario="L2", policy_config=PolicyConfig("value", "gru"), predictor=sim_predictor,
            seed=int(np.random.SeedSequence([31, round_id - 1]).generate_state(1)[0]),
            gateway=gateway, decision_hook=decisions.append, event_hook=events.append)
        gateway = result["gateway"]
        runs.append((decisions, events))

    core = create_gateway(GatewayContext(num_clients=n, policy="FG-GRU", client_weights=weights))
    assert isinstance(core.predictor.primary, PGRUNumpy)
    core.predictor.time_limit_ns = 10**15
    records: list[dict] = []
    core.decision_hook = records.append
    origin = 123_456_789_012_345
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
        assert decisions and len(produced) == len(decisions)
        for sim, live in zip(decisions, produced, strict=True):
            # GatewaySim adds compat fields to its public decision schema
            # (fedgrant/sim/run_net.py); every field GatewayCore emits must match.
            expected = {k: v for k, v in sim.items()
                        if k not in SIM_COMPAT_FIELDS | {"t_ns", "abs_t_ns", "timeout_ns"}
                        and not (k in GRANT_FIELDS and v is None)}
            assert {k: v for k, v in live.items() if k not in {"t_ns", "timeout_ns"}} == expected
            if sim.get("timeout_ns") is not None:
                assert live["timeout_ns"] - origin == sim["timeout_ns"]
            assert sim["selected_client_id"] == live["client_id"] and sim["selection_method"] == "value"
            assert sim["p_pacing"] == pytest.approx(pacing_p(sim["p_hat"], PolicyConfig("value", "gru")))
    assert sim_predictor.n_fallbacks == 0 and core.predictor.n_fallbacks == 0
