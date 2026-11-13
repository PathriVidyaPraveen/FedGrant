"""B2/B3 measurement contract and accounting tests.

Contract (spec Sections 10 and 12; docs/SUMMARY_CSV_SCHEMA.md). "Round" = one
deadline window; all counters are snapshotted at D. Network-only rows come
from ``fedgrant/baselines/sim_adapter.py``; FL rows from
``fedgrant/sim/run_baselines.py`` (names in brackets where they differ).

| Summary field | Event that counts | Unit / scope | B2 | B3 | Not applicable |
|---|---|---|---|---|---|
| plan_messages_sent [plans_issued] | PLAN handed to transport (``mark_plan_sent``) | count / round | - | yes | blank for B2 |
| planned_blocks [blocks_planned] | sum of PLAN k_i | blocks / round | - | yes | blank for B2 |
| data_packets_sent | DATA handed to the client's access link/socket before D | count / round | yes | yes | |
| data_bytes_sent | DATA datagram bytes (32-B header + 2L payload), app layer | bytes / round | yes | yes | |
| delivered_blocks | unique (client, block) accepted by the gateway by D | blocks / round | yes | yes | |
| goodput_bps | unique delivered blocks x 2L x 8 / D (padded fp16 block, as run_net) | bit/s | yes | yes | |
| utilization | bottleneck frames sent by D x S x 8 / (C D) | ratio | yes | yes | |
| queue_drops_fg | bottleneck drop-tail drops of FedGrant DATA (``drops_queue_bn_client``) | frames / round | yes | yes | |
| queue_drops_data | access + bottleneck drops of DATA | frames / round | yes | yes | blank for grant policies |
| plan_data_rtt_* | first accepted DATA arrival - PLAN send, per client per round | ns (ms in summary) | - | yes | blank for B2 |
| grants_*, grant_rtt_samples, rtt_p50/p99, wasted_grants, deadline_completion, late_after_timeout | grant events | - | - | - | blank (None), never 0 |

Facts that bound the contract:
* B2/B3 have no retransmission (spec Section 10), so no retransmitted bytes.
* The simulated control path is lossless: every computed PLAN is handed off,
  so PLAN attempts == handoffs there. The UDP gateway keeps them separate
  (``plan_send_attempts`` / ``plan_local_handoffs``; tests in test_b3_plan.py).
* A failed UDP DATA ``sendto`` aborts the client's transfer
  (``B2TransferError``); counters only advance after a successful send. There
  is no separate DATA-attempt counter, and the spec defines none.
* Receiver delivery is never inferred from a send: delivery counts only
  packets the gateway accepts.

Unresolved (needs teammate confirmation): whether "delivered payload bits"
for goodput includes the zero padding of the final block (the code base uses
the padded 2L boundary everywhere); protocol_overhead stays blank because the
control path is not fully modeled.
"""

from __future__ import annotations

import asyncio
from collections import Counter

import numpy as np
import pytest

from fedgrant.analysis.runs_schema import parse_row
from fedgrant.baselines.b2_udp import (
    B2ClientCore,
    B2ProtocolError,
    B2RoundAccumulator,
    B2TransferError,
    B2UdpClient,
    encode_data_packet,
)
from fedgrant.baselines.sim_adapter import run_baseline_network_round
from fedgrant.policy.constants import C, D_MODEL, L, S
from scripts import run_grid

_HEADER_BYTES = 32  # 20-byte common header + 12-byte DATA body header
_GRANT_FIELDS = ("grants_issued", "grants_delivered", "grants_timed_out", "wasted_grants_ratio",
                 "deadline_completion", "late_after_timeout")


def _accumulator(L_: int = 2) -> B2RoundAccumulator:
    return B2RoundAccumulator(round_id=3, client_ids=(7, 9), start_ns=100, deadline_ns=1_000,
                              d=5, L=L_, client_weights={7: 0.25, 9: 0.75})


def _packet(client_id: int, block_idx: int, send_pos: int, n_coords: int = 2) -> bytes:
    return encode_data_packet(3, client_id, block_idx, n_coords,
                              np.asarray([1.5, 0.0], dtype=np.float16), send_pos=send_pos)


# Scenario 1 and 7: successful B2 delivery; duplicates are idempotent.
def test_b2_success_and_duplicate_are_counted_exactly_once() -> None:
    acc = _accumulator()
    first = _packet(7, 2, 1, n_coords=1)  # d=5, L=2: final block has 1 true coordinate
    assert len(first) == _HEADER_BYTES + 2 * 2  # padded on the wire: data_bytes_sent boundary
    acc.accept_packet(first, 200)
    acc.accept_packet(_packet(7, 0, 2), 210)
    with pytest.raises(B2ProtocolError, match="duplicate"):
        acc.accept_packet(first, 220)
    acc.stop(1_000, "deadline")
    result = acc.result()
    assert result.delivered_blocks == 2
    assert result.client_results[7].packets_received == 2
    assert result.client_results[7].delivered_indices == (0, 2)
    assert result.client_results[9].packets_received == 0
    assert result.rejected_duplicates == 1


# Scenario 2: a failed UDP send is not counted as sent or delivered.
def test_b2_failed_send_counts_nothing_beyond_successful_sends() -> None:
    class FailingTransport:
        def __init__(self):
            self.sent: list[bytes] = []

        def sendto(self, packet, _addr):
            if self.sent:
                raise OSError("simulated socket failure")
            self.sent.append(packet)

        def close(self):
            pass

    async def scenario() -> None:
        now = {"ns": 100}

        async def sleep(seconds):
            now["ns"] += int(round(seconds * 1e9))

        client = B2UdpClient(B2ClientCore(7, d=5, L=2), gateway_addr=("127.0.0.1", 9101),
                             clock_ns=lambda: now["ns"], sleep=sleep)
        transport = FailingTransport()
        client.connection_made(transport)  # type: ignore[arg-type]
        with pytest.raises(B2TransferError, match="send failed"):
            await client.send_round(round_id=3, deadline_ns=10**12,
                                    delta=np.asarray([1, 1, 4, 0, 2], np.float32), rate_bps=4_800.0)
        acc = _accumulator()
        for packet in transport.sent:
            acc.accept_packet(packet, 500)
        acc.stop(1_000)
        assert len(transport.sent) == 1 and acc.result().delivered_blocks == 1

    asyncio.run(scenario())


def _event_counts(result: dict, round_id: int) -> tuple[Counter, Counter]:
    counts: Counter = Counter()
    bytes_by_type: Counter = Counter()
    for event in result["packet_events"]:
        assert event["round_id"] == round_id  # no cross-round leakage
        counts[event["packet_type"]] += 1
        bytes_by_type[event["packet_type"]] += event["bytes"]
    return counts, bytes_by_type


# Scenarios 3, 5, 8 and C6: real adapter counters agree with its own event log
# for two clients-plus and two rounds, with persistent B3 state.
@pytest.mark.parametrize("policy", ["B2", "B3"])
def test_real_adapter_counters_match_event_logs_per_round(policy: str) -> None:
    gateway = None
    for round_id in (1, 2):
        result = run_baseline_network_round(
            round_id=round_id, num_clients=4, beta=0.5, scenario="L2", policy=policy,
            seed=40 + round_id, gateway=gateway, cross_mbps=3.0,
        )
        gateway = result["gateway"]
        metrics = result["metrics"]
        counts, bytes_by_type = _event_counts(result, round_id)
        duration_s = result["deadline_ms"] / 1000.0
        delivered = sum(result["delivered_counts"])

        assert metrics["data_packets_sent"] == counts["DATA"]
        assert metrics["data_bytes_sent"] == bytes_by_type["DATA"] == counts["DATA"] * (_HEADER_BYTES + 2 * L)
        assert delivered == result["round_result"].delivered_blocks <= counts["DATA"]
        assert metrics["goodput_bps"] == pytest.approx(delivered * 2 * L * 8 / duration_s)
        assert metrics["utilization"] == pytest.approx(metrics["frames_sent"] * S * 8 / (C * duration_s))
        assert metrics["drops_queue_data"] == metrics["drops_queue_access"] + metrics["drops_queue_bn_client"]
        assert metrics["clients_represented"] == sum(n > 0 for n in result["delivered_counts"])
        for name in _GRANT_FIELDS:
            assert metrics[name] is None, name  # not applicable, never 0
        transfers = result["transfers"]
        assert sum(t.packets_sent for t in transfers.values()) == counts["DATA"]
        if policy == "B2":
            assert counts["PLAN"] == 0 and metrics["plan_messages_sent"] is None
            assert metrics["planned_blocks"] is None and result["rtt_samples_ns"] == {}
        else:
            plans = result["plans"]
            assert metrics["plan_messages_sent"] == counts["PLAN"] == len(plans) == 4
            assert metrics["planned_blocks"] == sum(plan.blocks for plan in plans.values())
            assert counts["DATA"] <= metrics["planned_blocks"]
            for cid, transfer in transfers.items():
                assert transfer.packets_sent <= plans[cid].blocks
            # RTT samples come from the same producer, one per client with DATA.
            receivers = {cid for cid, n in enumerate(result["delivered_counts"]) if n > 0}
            assert set(result["rtt_samples_ns"]) == receivers
            assert all(isinstance(v, int) and v > 0 for v in result["rtt_samples_ns"].values())


# Scenario 9: an empty B3 round reports measured zeros and blank N/A fields.
def test_empty_b3_round_has_zero_counts_and_no_rtt() -> None:
    result = run_baseline_network_round(round_id=1, num_clients=2, beta=0.02, scenario="L1",
                                        policy="B3", seed=3)
    metrics = result["metrics"]
    assert metrics["plan_messages_sent"] == 2  # PLAN handed off even if no DATA follows
    assert metrics["data_packets_sent"] == metrics["data_bytes_sent"] == 0
    assert sum(result["delivered_counts"]) == 0 and metrics["goodput_bps"] == 0.0
    assert result["rtt_samples_ns"] == {}
    for name in _GRANT_FIELDS:
        assert metrics[name] is None


# Scenario 10: real adapters through the grid writer and schema.
def test_real_b2_b3_rows_follow_summary_schema(tmp_path) -> None:
    rows = {}
    for policy in ("B2", "B3"):
        job = {
            "experiment_id": "N-main", "config_id": "L1-b0.10-N20-c0", "scenario": "L1",
            "beta": 0.10, "N": 20, "cross_mbps": 0.0, "policy": policy, "seed": 0, "n_rounds": 21,
        }
        rows[policy] = run_grid.execute_job(job, tmp_path, baseline_runner=run_baseline_network_round)
        parsed = parse_row(rows[policy])
        assert parsed.status == "ok"
        for column in ("grants_issued", "grants_delivered", "grants_timed_out", "grant_rtt_samples",
                       "rtt_p50_ms", "rtt_p99_ms", "wasted_grants", "deadline_completion",
                       "late_after_timeout"):
            assert rows[policy][column] == "", (policy, column)
        for column in ("data_packets_sent", "data_bytes_sent", "goodput_bps", "queue_drops_fg",
                       "queue_drops_data", "delivered_blocks", "eta"):
            assert isinstance(rows[policy][column], float), (policy, column)
    assert rows["B2"]["plan_messages_sent"] == rows["B2"]["plan_data_rtt_p50_ms"] == ""
    assert rows["B3"]["plan_messages_sent"] == 20.0
    # A PLAN does not guarantee an RTT sample: one exists only for a client's first
    # accepted DATA. With no sample the p50 is empty, never zero.
    if rows["B3"]["plan_data_rtt_samples"] > 0:
        assert rows["B3"]["plan_data_rtt_p50_ms"] > 0
    else:
        assert rows["B3"]["plan_data_rtt_samples"] == 0.0 and rows["B3"]["plan_data_rtt_p50_ms"] == ""
    assert rows["B3"]["data_bytes_sent"] == rows["B3"]["data_packets_sent"] * (_HEADER_BYTES + 2 * L)
    assert D_MODEL > (D_MODEL // L) * L  # final block is padded: boundary is explicit


def test_goodput_boundary_matches_grant_simulator_for_padded_final_block() -> None:
    # Network-only B2/B3 and grant policies must use the same goodput boundary.
    # beta=8 gives one client time to reach the final block (seed 1: 166 delivered).
    result = run_baseline_network_round(round_id=1, num_clients=1, beta=8.0, scenario="L1",
                                        policy="B2", seed=1)
    delivered = result["round_result"].client_results[0].delivered_indices
    assert (D_MODEL - 1) // L in delivered  # the padded final block was delivered
    seconds = result["deadline_ms"] / 1000.0
    assert result["metrics"]["goodput_bps"] == pytest.approx(len(delivered) * 2 * L * 8 / seconds)


@pytest.mark.parametrize("policy", ["B2", "B3"])
def test_congested_round_counters_partition_without_double_counting(policy: str) -> None:
    # 20 clients pushing into the 10 Mbit/s bottleneck plus 3 Mbit/s cross
    # traffic: B2 overloads the drop-tail queue; B3 plans within the budget.
    result = run_baseline_network_round(round_id=1, num_clients=20, beta=0.10, scenario="L2",
                                        policy=policy, seed=7, cross_mbps=3.0)
    m = result["metrics"]
    delivered = sum(result["delivered_counts"])
    seconds = result["deadline_ms"] / 1000.0
    # Each sent DATA ends in at most one outcome: delivered by D, dropped in a
    # queue (counted at D), lost on the access link (GE) or late after D.
    assert delivered + m["drops_queue_data"] + m["late_after_deadline"] <= m["data_packets_sent"]
    assert m["drops_queue_data"] == m["drops_queue_access"] + m["drops_queue_bn_client"]
    # Every delivered DATA left the bottleneck by D; cross frames share it.
    assert m["frames_sent"] >= delivered and 0.0 <= m["utilization"] <= 1.0 + 1e-9
    assert m["goodput_bps"] == pytest.approx(delivered * 2 * L * 8 / seconds)
    if policy == "B2":
        assert m["drops_queue_bn_client"] > 0 and m["drops_queue_cross"] > 0 and m["late_after_deadline"] > 0
    else:
        assert delivered <= m["planned_blocks"] and m["drops_queue_bn_client"] == 0
