"""B1/B2/B3 Mininet processes over real localhost sockets (design decisions D-06..D-15, D-19).

These run the actual gateway and client coroutines of
``fedgrant.mininet.baseline_proc`` with real UDP/TCP sockets on 127.0.0.1.
They are not Mininet runs (no topology, shaping or loss) and are not M1
results. The value bank is a test-only fixture.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import socket

import numpy as np
import pytest

from fedgrant.baselines.b2_udp import encode_data_packet
from fedgrant.common.constants import MessageType, TransferMode
from fedgrant.common.wire import decode_message, encode_message
from fedgrant.encoder.ef import ErrorFeedback
from fedgrant.mininet import baseline_proc
from fedgrant.mininet.baseline_metrics import eta_from_run_dir, round_eta
from fedgrant.mininet.baseline_proc import BaselineConfig, run_client, run_gateway
from fedgrant.mininet.round_loop import RoundSchedule, readiness_delays_us
from fedgrant.mininet.round_state import RoundStateFile
from fedgrant.mininet.value_profiles import BankUpdateProvider
from fedgrant.policy.constants import K, L
from fedgrant.sim.value_bank import load_value_bank

N = 3
SEED = 5
SCHEDULE = RoundSchedule(duration_ns=250_000_000, compute_window_ns=60_000_000, late_drain_ns=80_000_000)
# Multi-round runs over real sockets: gateway and clients share one event loop,
# so under CPU load a 250 ms round can pass with no block delivered. The longer
# D and Tc keep that from happening without changing what is checked.
SOCKET_SCHEDULE = RoundSchedule(duration_ns=500_000_000, compute_window_ns=100_000_000, late_drain_ns=80_000_000)


def _free_port(kind: int) -> int:
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _fixture_bank(tmp_path):
    """Test-only bank with N clients; never used by real runs."""
    rng = np.random.default_rng(1)
    energies = -np.sort(-rng.uniform(0.01, 1.0, size=(4, N, K)).astype(np.float32), axis=2)
    weights = np.asarray([0.2, 0.3, 0.5], dtype=np.float32)
    provenance = tmp_path / "provenance.json"
    provenance.write_text('{"fixture": true}\n')
    path = tmp_path / "bank.npz"
    np.savez_compressed(path, energies_sorted=energies, client_weights=weights, round_ids=np.arange(1, 5),
                        seed=np.asarray(0), num_rounds=np.asarray(4),
                        dataset_metadata_json=np.asarray(json.dumps({
                            "source": "test-fixture", "provenance_path": str(provenance),
                            "provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest()})))
    return load_value_bank(path, N)


def _config(policy: str, rounds: int = 3, schedule: RoundSchedule = SCHEDULE) -> BaselineConfig:
    return BaselineConfig(
        policy=policy, num_clients=N, rounds=rounds, master_seed=SEED, schedule=schedule,
        client_endpoints={cid: ("127.0.0.1", _free_port(socket.SOCK_DGRAM)) for cid in range(N)},
        gateway_host="127.0.0.1", gateway_bind_host="127.0.0.1",
        udp_port=_free_port(socket.SOCK_DGRAM), tcp_port=_free_port(socket.SOCK_STREAM),
    )


def _csv(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def _jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


async def _run(cfg: BaselineConfig, bank, run_dir, *, extra=None):
    opened = asyncio.Event()
    original_open = baseline_proc.BaselineGateway.open

    async def open_and_signal(self) -> None:
        await original_open(self)
        opened.set()

    # Start the clients once the gateway's sockets are bound (not after a fixed
    # sleep); its first ROUND_START still waits Tc.
    baseline_proc.BaselineGateway.open = open_and_signal
    try:
        gateway = asyncio.create_task(run_gateway(cfg, bank.client_weights, run_dir,
                                                  RoundStateFile(run_dir / "round_state.json")))
        await asyncio.wait_for(opened.wait(), 10.0)
    finally:
        baseline_proc.BaselineGateway.open = original_open
    clients = [asyncio.create_task(run_client(cfg, cid, BankUpdateProvider(bank, SEED, cid),
                                              run_dir / f"client_{cid}.json", round_timeout_s=5.0))
               for cid in range(N)]
    others = [asyncio.create_task(task) for task in (extra or [])]
    timings = await gateway
    summaries = await asyncio.gather(*clients)
    await asyncio.gather(*others)
    return timings, summaries


@pytest.mark.parametrize("policy", ["B1", "B2", "B3"])
def test_baseline_processes_complete_rounds_over_real_sockets(policy, tmp_path) -> None:
    bank = _fixture_bank(tmp_path)
    cfg = _config(policy, schedule=SOCKET_SCHEDULE)
    run_dir = tmp_path / policy
    timings, summaries = asyncio.run(_run(cfg, bank, run_dir))

    assert [t.round_id for t in timings] == [1, 2, 3]
    for cid, summary in enumerate(summaries):
        assert summary["completed_rounds"] == [1, 2, 3], summary
        assert summary["round_end_timeouts"] == 0
        assert summary["client_weight"] == pytest.approx(float(bank.client_weights[cid]))
        assert summary["value_profile"]["bank_sha256"] == bank.sha256
    summary_rounds = json.loads((run_dir / "gateway_summary.json").read_text())
    assert [r["round_id"] for r in summary_rounds] == [1, 2, 3]
    assert all(r["invalid_events"] == 0 for r in summary_rounds)

    gateway_rounds = _jsonl(run_dir / "gateway_rounds.jsonl")
    metrics = _csv(run_dir / "round_metrics.csv")
    packets = _csv(run_dir / "packets.csv")
    assert [int(r["round_id"]) for r in metrics] == [1, 2, 3]
    D = cfg.schedule.duration_ns
    for record, row in zip(gateway_rounds, metrics, strict=True):
        r = record["round_id"]
        # Readiness delays are the simulator's seeded draws (D-06).
        assert {int(k): v for k, v in record["rho_us"].items()} == readiness_delays_us(SEED, r, N, D)
        delivered_total = sum(len(v) for v in record["delivered"].values())
        assert int(row["delivered_blocks"]) == delivered_total
        if policy == "B3":
            # A READY arriving after t_plan = 0.1 D gets k = 0 (Section 10), and
            # rho_i can reach 0.1 D, so a round may legitimately deliver nothing.
            assert delivered_total <= int(row["planned_blocks"])
        else:
            assert delivered_total > 0
        assert float(row["goodput_bps"]) == pytest.approx(delivered_total * 2 * L * 8 / (D / 1e9))
        accepted = [p for p in packets if int(p["round_id"]) == r and p["direction"] == "received"
                    and p["result"] == "accepted" and p["packet_type"] in {"DATA", "B1_BLOCK"}]
        assert len(accepted) == delivered_total
        assert all(int(p["t_mono_ns"]) <= record["deadline_ns"] for p in accepted)
        # Raw packets, summary and metrics reconcile: accepted_data counts DATA
        # only (never READY), and goodput is the padded 2L-byte payload of each
        # accepted DATA (1400 bytes at L = 700) over D.
        summary_row = summary_rounds[r - 1]
        assert summary_row["round_id"] == r and summary_row["accepted_data"] == delivered_total
        assert all(int(p["payload_bytes"]) == 2 * L for p in accepted)
        payload_bytes = sum(int(p["payload_bytes"]) for p in accepted)
        assert float(row["goodput_bps"]) == pytest.approx(payload_bytes * 8 / (D / 1e9))
        assert row["eta"] == "" and row["grants_issued"] == "" and row["grant_rtt_samples"] == ""
        sent_controls = {p["packet_type"] for p in packets if int(p["round_id"]) == r and p["direction"] == "sent"}
        assert {"ROUND_START", "ROUND_END"} <= sent_controls
        assert ("PLAN" in sent_controls) == (policy == "B3")

    for cid in range(N):
        client_rounds = {rec["round_id"]: rec for rec in _jsonl(run_dir / f"client_{cid}_rounds.jsonl")}
        sent = _csv(run_dir / f"client_{cid}_packets.csv")
        for record in gateway_rounds:
            r = record["round_id"]
            # ROUND_END told the client exactly what the gateway accepted.
            assert client_rounds[r]["delivered"] == record["delivered"][str(cid)]
            assert client_rounds[r]["status"] == "round_end"
            # Every baseline logs per-block energies (by block index) for the eta join (D-09).
            energies = client_rounds[r]["block_energies"]
            assert len(energies) == K and all(isinstance(x, float) and x >= 0 for x in energies)
            if policy in {"B2", "B3"}:
                sent_blocks = {int(p["block_idx"]) for p in sent if int(p["round_id"]) == r}
                assert set(record["delivered"][str(cid)]) <= sent_blocks
            if policy in {"B1", "B2"}:
                waited_ns = client_rounds[r]["readiness_ns"] - client_rounds[r]["round_start_rx_ns"]
                assert waited_ns >= record["rho_us"][str(cid)] * 1_000 - 1_000_000
            else:
                assert client_rounds[r]["ready_sent_ns"] - record["start_ns"] >= record["rho_us"][str(cid)] * 1_000

    assert sum(int(row["delivered_blocks"]) for row in metrics) > 0
    etas = eta_from_run_dir(run_dir, N)
    assert set(etas) == {1, 2, 3} and all(0.0 <= eta <= 1.0 for eta in etas.values())
    assert any(eta > 0.0 for eta in etas.values())

    if policy == "B3":
        rtts = [row for row in _csv(run_dir / "rtt_samples.csv")]
        events = [row for row in _csv(run_dir / "rounds.csv") if row["event"] == "plan_data_rtt"]
        assert rtts and len(rtts) == len(events)
        assert {row["rtt_type"] for row in rtts} == {"plan_data"}
        accepted_data = [p for p in packets if p["packet_type"] == "DATA" and p["result"] == "accepted"]
        plans_sent = [p for p in packets if p["packet_type"] == "PLAN" and p["direction"] == "sent"]
        seen: set[tuple[str, str]] = set()
        for rtt, event in zip(rtts, events, strict=True):
            detail = dict(item.split("=") for item in event["detail"].split(";"))
            key = (rtt["round_id"], rtt["client_id"])
            # Client and round stay paired between the RTT row and its event.
            assert key == (event["round_id"], event["client_id"]) and key not in seen
            seen.add(key)  # at most one sample per client per round
            assert int(detail["arrival_ns"]) - int(detail["plan_sent_ns"]) == int(detail["rtt_ns"]) > 0
            assert float(rtt["rtt_ms"]) == pytest.approx(int(detail["rtt_ns"]) / 1e6)
            assert int(rtt["t_mono_ns"]) == int(detail["arrival_ns"])
            # send_pos names the accepted DATA that produced the sample: it is
            # that client's earliest accepted DATA of the round, at that time.
            mine = [p for p in accepted_data if (p["round_id"], p["client_id"]) == key]
            first = min(mine, key=lambda p: int(p["t_mono_ns"]))
            assert (first["send_pos"], int(first["t_mono_ns"])) == (detail["send_pos"], int(detail["arrival_ns"]))
            # The PLAN was handed off for this client and round before the DATA arrived.
            plan_rows = [p for p in plans_sent if (p["round_id"], p["client_id"]) == key]
            assert len(plan_rows) == 1 and int(plan_rows[0]["t_mono_ns"]) >= int(detail["plan_sent_ns"])
            assert int(detail["plan_sent_ns"]) < int(detail["arrival_ns"])
        # Every client with accepted DATA in a round has exactly one sample.
        assert seen == {(p["round_id"], p["client_id"]) for p in accepted_data}
        for row in metrics:
            assert int(row["plan_messages_sent"]) == N
            assert int(row["plan_data_rtt_samples"]) == sum(1 for e in events if e["round_id"] == row["round_id"])
    else:
        assert _csv(run_dir / "rtt_samples.csv") == []


def test_b2_gateway_counts_late_data_during_drain_without_delivering_it(tmp_path) -> None:
    bank = _fixture_bank(tmp_path)
    cfg = _config("B2", rounds=1)
    run_dir = tmp_path / "late"
    state = run_dir / "round_state.json"

    async def late_sender():
        # Wait until the round has passed its deadline (inactive), then send.
        for _ in range(400):
            await asyncio.sleep(0.005)
            if state.exists():
                s = json.loads(state.read_text())
                if s["round_id"] == 1 and not s["active"]:
                    break
        packet = encode_data_packet(1, 0, 168, 94, np.zeros(L, dtype=np.float16), send_pos=169)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(packet, ("127.0.0.1", cfg.udp_port))

    asyncio.run(_run(cfg, bank, run_dir, extra=[late_sender()]))
    row = _csv(run_dir / "round_metrics.csv")[0]
    late = [p for p in _csv(run_dir / "packets.csv") if p["result"] == "late_after_deadline"]
    deadline = _jsonl(run_dir / "gateway_rounds.jsonl")[0]["deadline_ns"]
    # Genuine client DATA sent just before D may also arrive after D; every
    # late packet is counted, and the injected one is among them.
    assert int(row["late_after_deadline"]) == len(late) >= 1
    assert all(int(p["t_mono_ns"]) > deadline for p in late)
    assert any(p["client_id"] == "0" and p["block_idx"] == "168" and p["send_pos"] == "169" for p in late)
    # Block 168 is client 0's lowest-energy (last) block: never sent before D
    # at 2 Mbit/s, and the late copy must not be delivered or acknowledged.
    record = _jsonl(run_dir / "gateway_rounds.jsonl")[0]
    assert 168 not in record["delivered"]["0"]
    client_record = _jsonl(run_dir / "client_0_rounds.jsonl")[0]
    assert 168 not in client_record["delivered"]


def test_client_applies_round_end_fallback_when_gateway_is_silent(tmp_path) -> None:
    bank = _fixture_bank(tmp_path)
    cfg = _config("B2", rounds=1)

    async def scenario():
        gateway = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        gateway.bind(("127.0.0.1", cfg.udp_port))
        gateway.setblocking(False)
        client = asyncio.create_task(run_client(cfg, 0, BankUpdateProvider(bank, SEED, 0),
                                                tmp_path / "client_0.json", round_timeout_s=5.0))
        await asyncio.sleep(0.05)
        deadline = __import__("time").monotonic_ns() + 100_000_000
        gateway.sendto(encode_message(
            MessageType.ROUND_START, round_id=1, client_id=0, t_send_ns=0,
            body={"deadline_ns": deadline, "K": K, "L": L, "rho_us": 0, "mode": int(TransferMode.PUSH)}),
            cfg.client_endpoints[0])
        summary = await client  # no ROUND_END is ever sent
        gateway.close()
        return summary, deadline

    summary, deadline = asyncio.run(scenario())
    assert summary["completed_rounds"] == [1] and summary["round_end_timeouts"] == 1
    record = _jsonl(tmp_path / "client_0_rounds.jsonl")[0]
    assert record["status"] == "round_end_timeout" and record["delivered"] == []


def test_wrong_policy_and_missing_bank_fail_clearly(tmp_path) -> None:
    with pytest.raises(ValueError, match="gateway_proc"):
        BaselineConfig(policy="B6", num_clients=1, rounds=1, master_seed=0, schedule=SCHEDULE,
                       client_endpoints={0: ("127.0.0.1", 1)})
    with pytest.raises(FileNotFoundError):
        baseline_proc.main(["gateway", "--policy", "B2", "--clients", "3", "--log-dir", str(tmp_path),
                            "--round-state", str(tmp_path / "s.json"), "--bank", str(tmp_path / "absent.npz")])


def test_network_only_clients_discard_the_error_feedback_residual() -> None:
    cfg = BaselineConfig(policy="B2", num_clients=1, rounds=1, master_seed=0, schedule=SCHEDULE,
                         client_endpoints={0: ("127.0.0.1", 1)})

    class Core:
        error_feedback = ErrorFeedback(cfg.d, cfg.L)

    core = Core()
    baseline_proc._configure_error_feedback(core, cfg)
    assert core.error_feedback.enabled is False
    delta = np.random.default_rng(0).standard_normal(cfg.d).astype(np.float32)
    u = core.error_feedback.prepare(delta)
    np.testing.assert_array_equal(u, delta)  # u = Delta: no residual carried in (D-19)
    core.error_feedback.finish_round(u, np.arange(5))
    np.testing.assert_array_equal(core.error_feedback.prepare(delta), delta)  # none carried out
    fl_cfg = BaselineConfig(policy="B2", num_clients=1, rounds=1, master_seed=0, schedule=SCHEDULE,
                            client_endpoints={0: ("127.0.0.1", 1)}, network_only=False)
    kept = Core()
    kept.error_feedback = ErrorFeedback(cfg.d, cfg.L)
    baseline_proc._configure_error_feedback(kept, fl_cfg)
    assert kept.error_feedback.enabled is True


def test_round_eta_matches_equation_seven() -> None:
    eta = round_eta({1: {0: [0], 1: []}, 2: {0: [0, 1], 1: [1]}},
                    {1: {0: [3.0, 1.0], 1: [2.0, 2.0]}, 2: {0: [3.0, 1.0], 1: [2.0, 2.0]}},
                    {0: 0.5, 1: 0.5})
    assert eta[1] == pytest.approx((0.25 * 3) / (0.25 * 4 + 0.25 * 4))
    assert eta[2] == pytest.approx((0.25 * 4 + 0.25 * 2) / 2.0)
    assert round_eta({3: {0: [0]}}, {3: {}}, {0: 1.0}) == {3: None}


def test_b3_gateway_rho_us_defaults_to_zero_and_is_validated() -> None:
    from fedgrant.baselines.b3_plan import B3PlanError, B3UdpGateway

    async def scenario():
        sent = []

        class Transport:
            def sendto(self, packet, endpoint):
                sent.append(decode_message(packet, block_length=L))

            def close(self):
                pass

        gateway = B3UdpGateway(client_weights={0: 0.5, 1: 0.5},
                               endpoints={0: ("127.0.0.1", 1), 1: ("127.0.0.1", 2)},
                               d=117_694, duration_ns=50_000_000)
        gateway.connection_made(Transport())
        with pytest.raises(B3PlanError, match="exactly the configured clients"):
            gateway.start_round(1, rho_us={0: 5})
        with pytest.raises(B3PlanError, match="u32"):
            gateway.start_round(1, rho_us={0: -1, 1: 0})
        gateway.start_round(1, rho_us={0: 1234, 1: 0})
        assert [m.body["rho_us"] for m in sent] == [1234, 0]
        await gateway.close()

    asyncio.run(scenario())


def test_cli_follows_runner_contract_modes_and_bank_rules(tmp_path, monkeypatch) -> None:
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    _fixture_bank(bank_dir)
    bank_path = bank_dir / "bank.npz"
    assert baseline_proc._bank_client_count(str(bank_path)) == N
    assert baseline_proc.MODE_TO_POLICY == {"tcp": "B1", "push": "B2", "plan": "B3"}
    with pytest.raises(SystemExit):
        baseline_proc.main(["gateway", "--mode", "push", "--log-dir", str(tmp_path), "--round-state", "s"])
    with pytest.raises(SystemExit):  # --mode and --policy are alternatives
        baseline_proc.main(["client", "--mode", "push", "--policy", "B2", "--client-id", "0",
                            "--log-file", "x"])
    with pytest.raises(FileNotFoundError):  # no bank, no fallback
        baseline_proc.main(["client", "--mode", "plan", "--client-id", "0", "--log-file", str(tmp_path / "c.json"),
                            "--value-bank", str(tmp_path / "absent.npz")])
    captured = {}

    async def fake_client(cfg, client_id, provider, log_file, *, round_timeout_s):
        captured.update(cfg=cfg, client_id=client_id, provider=provider)

    monkeypatch.setattr(baseline_proc, "run_client", fake_client)
    baseline_proc.main(["client", "--mode", "tcp", "--client-id", "1", "--rounds", "4", "--seed", "2",
                        "--duration-ns", "199000000", "--compute-window-s", "1.0",
                        "--log-file", str(tmp_path / "c.json"), "--value-bank", str(bank_path)])
    cfg = captured["cfg"]
    assert (cfg.policy, cfg.num_clients, cfg.rounds, cfg.master_seed) == ("B1", N, 4, 2)
    assert cfg.schedule.duration_ns == 199_000_000 and cfg.schedule.compute_window_ns == 1_000_000_000
    assert (cfg.udp_port, cfg.tcp_port) == (9000, 9100)
    assert cfg.client_endpoints[1] == ("10.0.0.2", 9001)
    assert captured["provider"].client_id == 1 and captured["provider"].master_seed == 2


def test_b3_gateway_counts_only_data_in_data_counters(tmp_path) -> None:
    # READY is part of the B3 exchange but is not DATA: it must not inflate
    # accepted_data / rejected_data / late_after_deadline in gateway_summary.json.
    gateway = baseline_proc.BaselineGateway(_config("B3", rounds=1), np.asarray([0.2, 0.3, 0.5]), tmp_path)
    book = baseline_proc._RoundBook(round_id=2, start_ns=0, deadline_ns=1_000, rho_us={})
    gateway.current = gateway.books[2] = book
    ready = encode_message(MessageType.READY, round_id=2, client_id=0, t_send_ns=0,
                           body={"K": K, "energy": tuple([1.0] * K)})
    data = encode_data_packet(2, 0, 3, L, np.zeros(L, dtype=np.float16), send_pos=1)
    stale = encode_data_packet(1, 0, 3, L, np.zeros(L, dtype=np.float16), send_pos=1)

    gateway._log_b3_datagram(ready, 10, accepted=True)
    gateway._log_b3_datagram(ready, 20, accepted=False)  # duplicate READY
    gateway._log_b3_datagram(ready, 2_000, accepted=False)  # READY after D is not late DATA
    gateway._log_b3_datagram(data, 30, accepted=True)
    gateway._log_b3_datagram(data, 40, accepted=False)  # duplicate DATA
    gateway._log_b3_datagram(data, 1_500, accepted=False)  # same-round DATA in the drain
    gateway._log_b3_datagram(stale, 1_600, accepted=False)  # previous-round DATA: rejected, not late
    gateway.logs.close()

    assert (book.accepted, book.rejected, book.late_after_deadline, book.invalid) == (1, 2, 1, 0)
    rows = _csv(tmp_path / "packets.csv")
    assert [(r["packet_type"], r["result"]) for r in rows] == [
        ("READY", "accepted"), ("READY", "rejected"), ("READY", "rejected"),
        ("DATA", "accepted"), ("DATA", "rejected"), ("DATA", "late_after_deadline"), ("DATA", "rejected"),
    ]
    assert [int(r["payload_bytes"]) for r in rows] == [0, 0, 0, 2 * L, 2 * L, 2 * L, 2 * L]
    assert [r["round_id"] for r in rows][-1] == "1"  # stale DATA keeps its own round id
