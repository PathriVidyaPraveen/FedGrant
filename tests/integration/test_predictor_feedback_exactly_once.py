"""Production predictor feedback: every grant is resolved and observed exactly once.

Path under test (no test code calls resolve_event or observe itself):

    production.create_gateway -> LiveClockGatewaySim (GatewaySim)
      -> GatewayCore(estimator=_CoreEstimator, predictor=FallbackPredictor(primary))
      GatewayCore DATA / timeout tick / deadline -> _CoreEstimator.resolve
      -> GatewaySim.resolve -> resolve_event(state, view, g, y, t)
                            -> core.predictor.observe(i, y, t, obs)

The real FG-GRU production predictor (models/gru_prod.npz) is used. The only
instrumentation is a pass-through spy on the core's FallbackPredictor.observe
(counts calls, then forwards) and GatewaySim.sample_hook, the adapter's
existing observation hook, which reports the grant id of each resolution.
"""

from __future__ import annotations

import asyncio
from collections import Counter

import numpy as np
import pytest

from fedgrant.common.constants import MessageType
from fedgrant.common.wire import encode_message
from fedgrant.core.gateway import GatewayCore, GatewayPhase, SendGrant
from fedgrant.mininet.production import GatewayContext, create_gateway
from fedgrant.policy.constants import K, L

N = 3
T0 = 1_000_000_000_000  # arbitrary monotonic origin, as on a live daemon
MS = 1_000_000
ENERGIES = tuple(float(v) for v in np.linspace(1.0, 0.01, K, dtype=np.float32))


class ObserveSpy:
    """Pass-through spy on the real predictor object GatewaySim.resolve calls."""

    def __init__(self, predictor) -> None:
        self.calls: list[tuple[int, int, int, np.ndarray]] = []
        original = predictor.observe

        def observe(client, y, t_ns, obs):
            self.calls.append((int(client), int(y), int(t_ns), np.array(obs, copy=True)))
            return original(client, y, t_ns, obs)

        predictor.observe = observe


class Round:
    """A production FG-GRU gateway driven through its wire/tick/deadline API."""

    def __init__(self, deadline_ms: int, policy: str = "FG-GRU") -> None:
        self.core: GatewayCore = create_gateway(GatewayContext(
            num_clients=N, policy=policy, client_weights=np.full(N, 1.0 / N, dtype=np.float32)))
        self.adapter = self.core.eligibility.sim
        self.resolutions: list[dict] = []
        self.adapter.sample_hook = self._sample
        self.spy = ObserveSpy(self.core.predictor)
        self.deadline = T0 + deadline_ms * MS
        self.core.on_round_start(T0, round_id=1, deadline_ns=self.deadline, K=K, L=L)

    def _sample(self, event: dict) -> None:
        if event["event"] == "grant_resolved":
            self.resolutions.append(event)

    def ready(self, client_id: int, at_ms: float) -> list[SendGrant]:
        packet = encode_message(MessageType.READY, round_id=1, client_id=client_id, t_send_ns=0,
                                body={"K": K, "energy": ENERGIES})
        return [a for a in self.core.on_ready_message(packet, received_at_ns=T0 + int(at_ms * MS))
                if isinstance(a, SendGrant)]

    def data(self, client_id: int, grant_id: int, send_pos: int, at_ms: float) -> None:
        packet = encode_message(MessageType.DATA, round_id=1, client_id=client_id, t_send_ns=0,
                                body={"grant_id": grant_id, "send_pos": send_pos, "block_idx": send_pos - 1,
                                      "n_coords": L, "payload": [0.0] * L})
        self.core.on_data_message(packet, received_at_ns=T0 + int(at_ms * MS))

    def tick(self, at_ms: float) -> None:
        self.core.on_tick(T0 + int(at_ms * MS))

    # -- observed quantities --------------------------------------------------
    def n_resolved(self) -> int:
        return sum(state.n_resolved for state in self.core.clients.values())

    def gru_buffer(self, client_id: int | None = None) -> int:
        """Observations ever appended to the P-GRU primary's per-client buffers."""
        primary = self.core.predictor.primary
        ids = self.core.clients if client_id is None else (client_id,)
        return sum(primary._buf(cid).total for cid in ids)

    def snapshot(self) -> tuple[int, int, int, int]:
        return self.n_resolved(), self.gru_buffer(), len(self.spy.calls), len(self.resolutions)

    def per_grant(self) -> Counter:
        return Counter((r["client_id"], r["grant_id"]) for r in self.resolutions)

    def assert_one_observation_per_resolution(self) -> None:
        # Each observe() call is the one made inside the resolution reported
        # with it, in the same order, with the same client, outcome, time and obs.
        assert len(self.spy.calls) == len(self.resolutions)
        for (client, y, t_ns, obs), resolution in zip(self.spy.calls, self.resolutions, strict=True):
            assert (client, y, t_ns) == (resolution["client_id"], resolution["y"], resolution["t_ns"])
            np.testing.assert_array_equal(obs, resolution["obs"])
            assert obs.shape == (8,) and obs[0] == y
        assert all(count == 1 for count in self.per_grant().values()), self.per_grant()


def test_data_arrival_resolves_and_observes_the_grant_once() -> None:
    r = Round(deadline_ms=400)
    grants = r.ready(0, at_ms=1)
    assert grants[0] == SendGrant(0, 1, 1)
    before = r.snapshot()
    evbuf_before = len(r.core.clients[0].evbuf)

    r.data(0, grant_id=1, send_pos=1, at_ms=21)

    assert 1 not in r.core.clients[0].outstanding and 1 in r.core.resolved_grants[0]
    assert r.snapshot() == tuple(v + 1 for v in before)  # n_resolved, GRU buffer, observe calls, resolutions
    assert len(r.core.clients[0].evbuf) == evbuf_before + 1
    client, y, t_ns, obs = r.spy.calls[-1]
    assert (client, y, t_ns) == (0, 1, 21 * MS)  # round-relative time of the DATA arrival
    assert r.resolutions[-1]["grant_id"] == 1 and r.resolutions[-1]["rtt_ns"] == 20 * MS
    assert r.core.stats.accepted_data == 1
    r.assert_one_observation_per_resolution()


def test_timeout_resolves_once_and_late_data_cannot_resolve_again() -> None:
    r = Round(deadline_ms=400)
    issued = r.ready(0, at_ms=1)
    timeout_ms = (r.core.clients[0].outstanding[1].timeout_ns - T0) / MS
    assert timeout_ms == 301  # srtt_0 100 ms + max(4 rttvar_0, 10 ms) after issue at 1 ms
    before = r.snapshot()

    r.tick(timeout_ms)  # every grant issued at 1 ms expires here

    expired = len(issued)
    assert r.snapshot() == tuple(v + expired for v in before)
    assert {(r_["grant_id"], r_["y"], r_["t_ns"]) for r_ in r.resolutions} == {
        (g.grant_id, 0, 301 * MS) for g in issued}
    assert 1 in r.core.timed_out_grants[0]
    after_timeout = r.snapshot()

    r.data(0, grant_id=1, send_pos=1, at_ms=305)  # the expired grant's DATA arrives late

    assert r.core.stats.late_after_timeout == 1 and r.core.stats.accepted_data == 1  # block kept
    assert r.snapshot() == after_timeout  # no second resolution or observation
    r.assert_one_observation_per_resolution()


def test_deadline_resolves_every_outstanding_grant_and_nothing_twice() -> None:
    # D = 250 ms caps every timeout at D, so grants stay outstanding until the deadline.
    r = Round(deadline_ms=250)
    issued = [g for cid in range(N) for g in r.ready(cid, at_ms=1)]
    assert len(issued) > 10 and all(
        o.timeout_ns == r.deadline for s in r.core.clients.values() for o in s.outstanding.values())
    r.data(0, issued[0].grant_id, issued[0].send_pos, at_ms=30)
    r.data(1, *next((g.grant_id, g.send_pos) for g in issued if g.client_id == 1), at_ms=31)
    delivered = {(r_["client_id"], r_["grant_id"]) for r_ in r.resolutions}
    assert len(delivered) == 2
    # The two DATA re-pumped new grants; count what is outstanding right before D.
    outstanding = {(cid, gid) for cid, s in r.core.clients.items() for gid in s.outstanding}
    before = r.snapshot()

    r.core.on_deadline(r.deadline)

    assert r.core.phase is GatewayPhase.DEADLINE
    newly = len(outstanding)
    assert r.snapshot() == tuple(v + newly for v in before)
    at_deadline = {(r_["client_id"], r_["grant_id"]) for r_ in r.resolutions[-newly:]}
    assert at_deadline == outstanding and not (at_deadline & delivered)
    assert all(r_["y"] == 0 and r_["t_ns"] == 250 * MS for r_ in r.resolutions[-newly:])
    assert all(not s.outstanding for s in r.core.clients.values()) and r.adapter.inflight.phi == 0.0
    issued_total = r.core.next_grant_id - 1
    assert len(r.resolutions) == issued_total  # every grant of the round resolved exactly once
    r.core.on_deadline(r.deadline)  # a repeated deadline is a no-op
    assert len(r.spy.calls) == issued_total
    r.assert_one_observation_per_resolution()


@pytest.mark.parametrize("sequence", [
    "data_then_timeout", "timeout_then_data", "data_then_deadline", "timeout_then_deadline", "duplicate_data",
])
def test_no_sequence_observes_a_grant_twice(sequence: str) -> None:
    r = Round(deadline_ms=400)
    r.ready(0, at_ms=1)
    grant = r.core.clients[0].outstanding[1]
    timeout_ms = (grant.timeout_ns - T0) / MS
    steps = {
        "data_then_timeout": [("data", 21), ("tick", timeout_ms + 1)],
        "timeout_then_data": [("tick", timeout_ms), ("data", timeout_ms + 1)],
        "data_then_deadline": [("data", 21), ("deadline", None)],
        "timeout_then_deadline": [("tick", timeout_ms), ("deadline", None)],
        "duplicate_data": [("data", 21), ("data", 22), ("data", 23)],
    }[sequence]
    for kind, at_ms in steps:
        if kind == "data":
            r.data(0, grant_id=1, send_pos=1, at_ms=at_ms)
        elif kind == "tick":
            r.tick(at_ms)
        else:
            r.core.on_deadline(r.deadline)

    grant_1 = [r_ for r_ in r.resolutions if (r_["client_id"], r_["grant_id"]) == (0, 1)]
    assert len(grant_1) == 1
    expected_y = 1 if sequence.startswith(("data", "duplicate")) else 0
    assert grant_1[0]["y"] == expected_y
    if sequence == "duplicate_data":
        assert r.core.stats.duplicate_data == 2 and r.core.stats.accepted_data == 1
    assert r.n_resolved() == len(r.resolutions) == r.gru_buffer() == len(r.spy.calls)
    r.assert_one_observation_per_resolution()


# -- acceptance: a real production round over localhost UDP ---------------------------


@pytest.mark.parametrize("policy", ["FG-GRU", "B6"])
def test_production_round_over_udp_observes_each_resolved_grant_once(policy: str) -> None:
    from fedgrant.core.client_core import ClientCore
    from fedgrant.mininet.contract import MininetGatewayContext, build_gateway, policy_config_for, uniform_weights
    from fedgrant.mininet.profile_update import profile_bank, synthetic_update
    from fedgrant.net.udp_client import UdpClient
    from fedgrant.net.udp_gateway import UdpGateway

    duration_ns = 400 * MS

    class LossyClient(UdpClient):
        """Never answers GRANTs for odd send positions (models lost DATA)."""

        def _handle_grant(self, message) -> None:
            if message.body["send_pos"] % 2:
                return
            super()._handle_grant(message)

    async def scenario():
        ctx = MininetGatewayContext(
            num_clients=N, policy=policy, seed=0, policy_config=policy_config_for(policy),
            client_weights=uniform_weights(N), deadline_ns=duration_ns, K=K, L=L, models_dir="models",
            log_dir=".")
        core = build_gateway("fedgrant.mininet.production:create_gateway", ctx)  # the runner's path
        adapter = core.eligibility.sim
        resolutions: list[dict] = []
        adapter.sample_hook = lambda e: resolutions.append(e) if e["event"] == "grant_resolved" else None
        spy = ObserveSpy(core.predictor)
        # Pass-through spy: how many grants are still outstanding when the
        # round's deadline fires (on the tick loop or otherwise). On localhost
        # this can legitimately be zero: every grant issued before the rule (iv)
        # cutoff may have been answered or timed out before D.
        at_deadline: list[int] = []
        on_deadline = core.on_deadline

        def deadline_spy(t_ns):
            if core.phase is GatewayPhase.ACTIVE:
                at_deadline.append(sum(len(s.outstanding) for s in core.clients.values()))
            return on_deadline(t_ns)

        core.on_deadline = deadline_spy
        gateway = await UdpGateway.bind(core, local_addr=("127.0.0.1", 0), aggregate_handler=lambda agg: None)
        port = gateway.transport.get_extra_info("sockname")[1]
        clients = []
        for cid in range(N):
            cls = LossyClient if cid == N - 1 else UdpClient
            update = synthetic_update(profile_bank(cid)[0], seed=cid)
            client = await cls.bind(ClientCore(cid), gateway_addr=("127.0.0.1", port),
                                    local_addr=("127.0.0.1", 0), update_provider=lambda r, u=update: u)
            gateway.set_endpoint(cid, ("127.0.0.1", client.transport.get_extra_info("sockname")[1]))
            clients.append(client)
        primary = core.predictor.primary
        before = (sum(s.n_resolved for s in core.clients.values()), len(spy.calls),
                  sum(primary._buf(c).total for c in core.clients) if policy == "FG-GRU" else None)
        try:
            gateway.start_round(round_id=1, duration_ns=duration_ns, K=K, L=L)
            # The gateway's own tick loop ends the round at D; clients finish on ROUND_END.
            await asyncio.wait_for(asyncio.gather(*(c.wait_for_round_end() for c in clients)), 10.0)
        finally:
            for client in clients:
                await client.close()
            await gateway.close()
        after = (sum(s.n_resolved for s in core.clients.values()), len(spy.calls),
                 sum(primary._buf(c).total for c in core.clients) if policy == "FG-GRU" else None)
        return core, spy, resolutions, before, after, [c.stats.round_end_timeouts for c in clients], at_deadline

    core, spy, resolutions, before, after, client_fallbacks, at_deadline = asyncio.run(scenario())
    start, deadline = core.round.start_ns, core.round.deadline_ns

    assert core.phase is GatewayPhase.DEADLINE and client_fallbacks == [0] * N
    newly_resolved = after[0] - before[0]
    observe_calls = after[1] - before[1]
    assert newly_resolved > 0
    assert observe_calls == newly_resolved == len(resolutions)
    if policy == "FG-GRU":
        assert after[2] - before[2] == newly_resolved  # P-GRU observation buffers grew by the same count
    # Every grant issued in the round was resolved and observed exactly once.
    per_grant = Counter((r["client_id"], r["grant_id"]) for r in resolutions)
    assert len(per_grant) == core.next_grant_id - 1 and set(per_grant.values()) == {1}
    for (client, y, t_ns, obs), r in zip(spy.calls, resolutions, strict=True):
        assert (client, y, t_ns) == (r["client_id"], r["y"], r["t_ns"]) and obs[0] == y
    # DATA and timeout resolutions are present, and the deadline resolved
    # exactly the grants still outstanding when it fired (possibly none; the
    # deterministic deadline path is covered above).
    d_rel = deadline - start
    assert any(r["y"] == 1 for r in resolutions)
    # A timeout *before* D is not guaranteed here: with min-RTT pacing (Eq. 12 amendment)
    # localhost RTT_min is < 1 ms, P_target falls to ~5-15 packets, and the lossy client's
    # first grant can come so late that its timeout (srtt0 + 4 rttvar0 = 300 ms) is past D.
    # The timeout-before-D path is covered deterministically by
    # test_no_sequence_observes_a_grant_twice; here every loss must be at or before D.
    assert all(r["t_ns"] <= d_rel for r in resolutions if r["y"] == 0)
    assert len(at_deadline) == 1
    assert sum(r["y"] == 0 and r["t_ns"] == d_rel for r in resolutions) == at_deadline[0]
    # Gateway accounting is unchanged by the observation path.
    delivered = sum(s.m for s in core.clients.values())
    assert core.stats.accepted_data == delivered == sum(len(b) for b in core.delivered_blocks.values())
    on_time = sum(r["y"] for r in resolutions)
    assert on_time + core.stats.late_after_timeout >= delivered >= on_time
    lossy = N - 1
    assert all(pos % 2 == 0 for pos in np.flatnonzero(core.clients[lossy].status == 2) + 1)
    assert core.stats.invalid_events == 0
