"""End-to-end B3 PLAN->DATA RTT: real producer -> baseline result -> FL history.

Measurement (specification Section 10, B3 mode 2): for each client and round,
one RTT sample = arrival time at the gateway of the client's first *accepted*
DATA of the round minus the gateway's send time of that client's PLAN. Both
timestamps are on the gateway's (simulator-absolute) nanosecond clock. The
producer is ``B3GatewayCore.accept_data``; it feeds the same value to the
RFC 6298 update. B3 has no retransmission. No sample exists for a client whose
PLAN was never handed off or whose DATA never arrived by D.

The producer is observed here through pass-through spies that call the real
``B3GatewayCore`` methods and only record their inputs; RTT values are never
mocked. Model training/evaluation, which is unrelated to RTT, is stubbed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fedgrant.baselines.b3_plan import B3GatewayCore
from fedgrant.common.wire import decode_message
from fedgrant.encoder.ef import ErrorFeedback
from fedgrant.fl import server as fl_server
from fedgrant.policy.constants import D_MODEL
from fedgrant.sim.run_baselines import run_baseline_round
from scripts import run_grid


@pytest.fixture
def b3_spy(monkeypatch):
    log: dict[str, list[dict]] = {"plans": [], "data": []}
    original_mark = B3GatewayCore.mark_plan_sent
    original_accept = B3GatewayCore.accept_data

    def mark_plan_sent(self, client_id, sent_ns):
        plan = original_mark(self, client_id, sent_ns)
        log["plans"].append({
            "round_id": self.round_id, "client_id": client_id,
            "sent_ns": sent_ns, "blocks": plan.blocks,
        })
        return plan

    def accept_data(self, packet, received_ns):
        original_accept(self, packet, received_ns)  # a rejected DATA raises and is not logged
        message = decode_message(packet, block_length=self.L)
        log["data"].append({
            "round_id": self.round_id, "client_id": message.client_id,
            "send_pos": message.body["send_pos"], "received_ns": received_ns,
            "t_send_ns": message.t_send_ns,
        })

    monkeypatch.setattr(B3GatewayCore, "mark_plan_sent", mark_plan_sent)
    monkeypatch.setattr(B3GatewayCore, "accept_data", accept_data)
    return log


def _expected_samples(log: dict, round_id: int) -> dict[int, int]:
    """Independent recomputation of the documented definition from events."""
    plan_sent = {p["client_id"]: p["sent_ns"] for p in log["plans"] if p["round_id"] == round_id}
    first_arrival: dict[int, int] = {}
    for event in log["data"]:
        if event["round_id"] == round_id:
            first_arrival.setdefault(event["client_id"], event["received_ns"])
    return {cid: arrival - plan_sent[cid] for cid, arrival in first_arrival.items()}


def _run_producer_rounds(mode: str, rounds: int, *, num_clients: int = 3, seed: int = 11):
    state = None
    results = []
    for round_id in range(1, rounds + 1):
        result = run_baseline_round(
            mode, round_id=round_id, num_clients=num_clients, beta=1.0,
            scenario="L1", seed=seed * 100 + round_id, state=state,
        )
        state = result["baseline_state"]
        results.append(result)
    return results


def test_real_b3_producer_samples_match_plan_and_first_data_events(b3_spy) -> None:
    results = _run_producer_rounds("B3", 2)
    for round_id, result in enumerate(results, start=1):
        samples = result["b3_rtt_samples_ns"]
        expected = _expected_samples(b3_spy, round_id)
        assert samples == expected and samples, round_id
        plans = {p["client_id"]: p for p in b3_spy["plans"] if p["round_id"] == round_id}
        start = (round_id - 1) * (1_000_000_000 + int(result["deadline_ms"] * 1_000_000))
        deadline = start + int(result["deadline_ms"] * 1_000_000)
        for cid, rtt in samples.items():
            assert isinstance(rtt, int) and rtt > 0
            first = next(d for d in b3_spy["data"] if d["round_id"] == round_id and d["client_id"] == cid)
            # Same client, same round, same clock domain, DATA after its PLAN.
            assert start <= plans[cid]["sent_ns"] < first["t_send_ns"] < first["received_ns"] <= deadline
            assert plans[cid]["blocks"] > 0
            # Propagation alone bounds the sample from below: PLAN reverse path
            # plus DATA forward path, each at least the 5 ms bottleneck delay.
            assert rtt >= 2 * 5_000_000
        # Clients that received a zero-block PLAN send no DATA: no sample.
        for cid, plan in plans.items():
            if plan["blocks"] == 0:
                assert cid not in samples
        # The same value is the srtt input: one sample per client per round.
        assert set(samples) <= set(plans)
    # Rounds keep separate sample sets even though gateway state persists.
    assert results[0]["b3_rtt_samples_ns"] is not results[1]["b3_rtt_samples_ns"]
    round_two = {(d["client_id"]) for d in b3_spy["data"] if d["round_id"] == 2}
    assert set(results[1]["b3_rtt_samples_ns"]) == round_two


def test_real_b3_producer_is_deterministic_and_b2_produces_no_b3_rtt() -> None:
    first = [r["b3_rtt_samples_ns"] for r in _run_producer_rounds("B3", 2)]
    second = [r["b3_rtt_samples_ns"] for r in _run_producer_rounds("B3", 2)]
    assert first == second and all(first)
    for result in _run_producer_rounds("B2", 2):
        assert result["b3_rtt_samples_ns"] == {}
        assert "plans_issued" not in result["metrics"]


def test_duplicate_data_cannot_create_a_second_sample() -> None:
    from fedgrant.baselines.b2_udp import encode_data_packet
    from fedgrant.baselines.b3_plan import B3PlanError
    from fedgrant.common.constants import MessageType
    from fedgrant.common.wire import encode_message

    core = B3GatewayCore(
        round_id=1, start_ns=1_000_000_000, deadline_ns=1_398_000_000, d=6, L=2,
        client_weights={1: 0.5, 2: 0.5}, bottleneck_bps=1_000_000,
    )
    for cid in (1, 2):
        ready = encode_message(MessageType.READY, round_id=1, client_id=cid, t_send_ns=0,
                               body={"K": 3, "energy": (9.0, 4.0, 1.0)})
        core.accept_ready(ready, core.plan_ns)
    core.compute_plans(core.plan_ns)
    core.mark_plan_sent(1, core.plan_ns + 1_000)  # client 2's PLAN is never handed off
    packet = encode_data_packet(1, 1, 0, 2, np.asarray([3.0, 0.0], np.float16), send_pos=1, t_send_ns=core.plan_ns + 2_000)
    core.accept_data(packet, core.plan_ns + 7_001_000)
    with pytest.raises(B3PlanError, match="duplicate"):
        core.accept_data(packet, core.plan_ns + 9_000_000)
    with pytest.raises(B3PlanError, match="before this client's PLAN"):
        core.accept_data(
            encode_data_packet(1, 2, 0, 2, np.asarray([3.0, 0.0], np.float16), send_pos=1),
            core.plan_ns + 8_000_000,
        )
    assert core.rtt_samples_ns == {1: 7_000_000}
    assert core.finish(core.deadline_ns).rtt_samples_ns == {1: 7_000_000}


class _FakeModel:
    def to(self, _device):
        return self


def _run_fl(mode: str, monkeypatch, *, rounds: int = 2, num_clients: int = 3):
    """Real run_fl_simulated + real run_baseline_round; only training is stubbed."""

    def train_client(client_id, _images, _labels, _model, *, round_idx, **_kwargs):
        rng = np.random.default_rng([client_id, round_idx])
        return rng.standard_normal(D_MODEL).astype(np.float32) * 1e-3

    monkeypatch.setattr(fl_server, "train_client", train_client)
    monkeypatch.setattr(fl_server, "get_flat_params", lambda _model: np.zeros(D_MODEL, dtype=np.float32))
    monkeypatch.setattr(fl_server, "set_flat_params", lambda *_args: None)
    monkeypatch.setattr(fl_server, "evaluate", lambda *_args, **_kwargs: (1.0, 0.5))
    assert fl_server.run_baseline_round is run_baseline_round  # producer not replaced
    return fl_server.run_fl_simulated(
        model=_FakeModel(),
        train_clients=[(None, None)] * num_clients,
        test_client_loader=lambda: [],
        client_weights=np.full(num_clients, 1.0 / num_clients, dtype=np.float32),
        loaders=[object()] * num_clients,
        error_feedback=[ErrorFeedback(D_MODEL) for _ in range(num_clients)],
        num_rounds=rounds,
        seed=7,
        beta=1.0,
        scenario="L1",
        transport_baseline=mode,
        eval_every=rounds,
        client_workers=1,
    )


def test_fl_history_records_real_b3_samples_per_round_with_client_ids(b3_spy, monkeypatch) -> None:
    history = _run_fl("B3", monkeypatch)
    assert [record["round"] for record in history] == [1, 2]
    for record in history:
        expected = _expected_samples(b3_spy, record["round"])
        assert expected, "the real producer must yield samples in this scenario"
        assert record["plan_data_rtt_client_ids"] == sorted(expected)
        assert record["plan_data_rtt_samples_ns"] == [expected[cid] for cid in sorted(expected)]
        # send_pos of the DATA that produced each sample (the first accepted
        # DATA observed by the producer spy), parallel to the client ids.
        first_pos: dict[int, int] = {}
        for event in b3_spy["data"]:
            if event["round_id"] == record["round"]:
                first_pos.setdefault(event["client_id"], event["send_pos"])
        assert record["plan_data_rtt_send_pos"] == [first_pos[cid] for cid in sorted(expected)]
        assert all(pos >= 1 for pos in record["plan_data_rtt_send_pos"])
        assert all(isinstance(v, int) and v > 0 for v in record["plan_data_rtt_samples_ns"])
        assert record["rtt_samples_ns"] == []  # never labelled as grant RTT
    # Values and attribution survive the JSON artifact written by run_fl.
    restored = json.loads(json.dumps(history))
    for before, after in zip(history, restored, strict=True):
        assert after["plan_data_rtt_client_ids"] == before["plan_data_rtt_client_ids"]
        assert after["plan_data_rtt_samples_ns"] == before["plan_data_rtt_samples_ns"]
        assert after["plan_data_rtt_send_pos"] == before["plan_data_rtt_send_pos"]
    # The grid reader carries the samples into its per-round rows unchanged.
    rows = run_grid._fl_round_metrics(restored)
    assert [row["plan_data_rtt_samples_ns"] for row in rows] == [
        record["plan_data_rtt_samples_ns"] for record in history
    ]


def test_fl_history_for_b2_has_no_b3_fields_and_legacy_records_still_read(monkeypatch) -> None:
    history = _run_fl("B2", monkeypatch)
    for record in history:
        assert "plan_data_rtt_samples_ns" not in record
        assert "plan_data_rtt_client_ids" not in record
        assert "plan_data_rtt_send_pos" not in record
        assert record["rtt_samples_ns"] == []
    legacy = [{"round": r, "eta": 0.5, "metrics": {}, "late_metrics": {}} for r in range(1, 23)]
    rows = run_grid._fl_round_metrics(legacy)
    assert all("plan_data_rtt_samples_ns" not in row for row in rows)
    summary = run_grid._network_summary(rows, "B3")
    assert "plan_data_rtt_p50_ms" not in summary and "plan_data_rtt_samples" not in summary


def test_empty_b3_round_records_empty_parallel_lists(monkeypatch) -> None:
    # A real B3 round whose DATA cannot arrive (deadline shorter than the PLAN
    # round trip) must record empty lists, not invented values.
    def short_round(mode, **kwargs):
        kwargs["beta"] = 0.02  # D = 8 ms for N=2; PLAN round trip alone exceeds it
        return run_baseline_round(mode, **kwargs)

    monkeypatch.setattr(fl_server, "run_baseline_round", short_round)
    monkeypatch.setattr(fl_server, "train_client",
                        lambda cid, *_a, round_idx, **_k: np.full(D_MODEL, 1e-3, dtype=np.float32))
    monkeypatch.setattr(fl_server, "get_flat_params", lambda _model: np.zeros(D_MODEL, dtype=np.float32))
    monkeypatch.setattr(fl_server, "set_flat_params", lambda *_args: None)
    monkeypatch.setattr(fl_server, "evaluate", lambda *_args, **_kwargs: (1.0, 0.5))
    history = fl_server.run_fl_simulated(
        model=_FakeModel(), train_clients=[(None, None)] * 2, test_client_loader=lambda: [],
        client_weights=np.full(2, 0.5, dtype=np.float32), loaders=[object()] * 2,
        error_feedback=[ErrorFeedback(D_MODEL) for _ in range(2)], num_rounds=1, seed=1,
        scenario="L1", transport_baseline="B3", eval_every=1, client_workers=1,
    )
    assert history[0]["plan_data_rtt_client_ids"] == []
    assert history[0]["plan_data_rtt_samples_ns"] == []
    assert history[0]["plan_data_rtt_send_pos"] == []
