"""Cross-traffic vs FedGrant bottleneck drop attribution (spec Section 12)."""

from __future__ import annotations

import csv
import struct

import numpy as np
import pytest

from fedgrant.common.logschema import QUEUE_COLUMNS, ROUND_METRIC_COLUMNS
from fedgrant.mininet import cross_drops as cd
from fedgrant.mininet.crosstraffic import cross_duration_s
from fedgrant.mininet.round_summary import enrich_round_metrics
from fedgrant.policy.constants import C, S

SERIAL_NS = int(S * 8 * 1e9 / C)


def _frame(seq: int, sent_ns: int, *, dport: int = 9999, proto: int = 17) -> bytes:
    sec, usec = divmod(sent_ns // 1000, 1_000_000)
    payload = struct.pack("!III", sec, usec, seq) + b"\0" * 20
    udp = struct.pack("!HHHH", 40000, dport, 8 + len(payload), 0) + payload
    ip = bytes([0x45, 0]) + struct.pack("!H", 20 + len(udp)) + b"\0" * 5 + bytes([proto]) + b"\0" * 10
    return b"\0" * 12 + b"\x08\x00" + ip + udp


def _pcap(path, records, *, nano=True, truncate_last=False):
    magic = 0xA1B23C4D if nano else 0xA1B2C3D4
    out = struct.pack("<IHHiIII", magic, 2, 4, 0, 0, 96, 1)
    for ts_ns, frame in records:
        sec, rest = divmod(ts_ns, 1_000_000_000)
        frac = rest if nano else rest // 1000
        out += struct.pack("<IIII", sec, frac, len(frame), len(frame)) + frame
    if truncate_last:
        out = out[:-5]
    path.write_bytes(out)
    return path


def _cbr(path, n=40, lost=(), *, clock_offset_ns=7_000_000_000, path_ns=5_000_000, gap_ns=3_930_000,
         queue_ns=None):
    """CBR stream: datagram k sent at k*gap; arrival = send + offset + path + serialization + queueing."""
    records = []
    for k in range(1, n + 1):
        sent = 1_000_000_000_000 + k * gap_ns
        if k in lost:
            continue
        q = queue_ns(k) if queue_ns else 0
        records.append((sent + clock_offset_ns + path_ns + SERIAL_NS + q, _frame(k, sent)))
    records.append((records[-1][0] + 1, _frame(999, 0, dport=5000)))      # other UDP flow
    records.append((records[-1][0] + 1, _frame(1000, 0, proto=6)))        # TCP control
    return _pcap(path, sorted(records))


def test_pcap_reader_handles_precisions_and_truncation(tmp_path):
    nano = _pcap(tmp_path / "n.pcap", [(1_500_000_123, _frame(1, 0)), (2_000_000_000, _frame(2, 0))],
                 truncate_last=True)
    assert [t for t, _ in cd.read_pcap(nano)] == [1_500_000_123]
    micro = _pcap(tmp_path / "u.pcap", [(1_500_000_123, _frame(1, 0))], nano=False)
    assert [t for t, _ in cd.read_pcap(micro)] == [1_500_000_000]
    (tmp_path / "bad.pcap").write_bytes(b"\x0a\x0d\x0d\x0a" + b"\0" * 40)
    with pytest.raises(ValueError, match="pcap"):
        list(cd.read_pcap(tmp_path / "bad.pcap"))


def test_missing_sequence_numbers_are_drops_at_their_queue_entry_time(tmp_path):
    # Queueing delay grows mid-stream; drop times must still follow send times.
    trace = cd.cbr_trace(_cbr(tmp_path / "c.pcap", lost=(10, 11, 25),
                              queue_ns=lambda k: 0 if k < 8 else 60_000_000))
    assert trace.seq.size == 37  # 40 sent - 3 lost; other flows ignored
    times, seqs = cd.drop_times(trace)
    assert seqs.tolist() == [10, 11, 25]
    for t, k in zip(times, seqs, strict=True):
        expected = 1_000_000_000_000 + k * 3_930_000 + 7_000_000_000 + 5_000_000
        assert abs(int(t) - expected) <= 1_000  # within 1 us
    assert cd.drop_times(cd.cbr_trace(_cbr(tmp_path / "none.pcap")))[1].size == 0


def test_write_cross_drops_converts_to_monotonic(tmp_path):
    pcap = _cbr(tmp_path / "c.pcap", lost=(5,))
    offset = 7_000_000_000 - 123
    assert cd.write_cross_drops(pcap, tmp_path / "cross_drops.csv", offset) == 1
    times = cd.read_cross_drop_times(tmp_path / "cross_drops.csv")
    assert times.tolist() == [1_000_000_000_000 + 5 * 3_930_000 + 5_000_000 + 123]
    assert cd.read_cross_drop_times(tmp_path / "absent.csv") is None


def test_split_drops_rules():
    times = np.array([100, 200, 300], np.int64)
    assert cd.split_drops(7, times, 150, 300, 3.0) == (5, 2)
    assert cd.split_drops(1, times, 0, 1000, 3.0) == (0, 3)  # never negative FedGrant drops
    assert cd.split_drops(4, None, 0, 1000, 0.0) == (4, 0)   # no cross traffic: all FedGrant
    assert cd.split_drops(4, None, 0, 1000, 3.0) is None      # cross traffic but no capture: unknown


def _write_csv(path, columns, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def test_round_summary_fills_split_from_capture(tmp_path):
    start, deadline = 10_000_000_000, 10_398_000_000
    _write_csv(tmp_path / "round_metrics.csv", ROUND_METRIC_COLUMNS,
               [{"round_id": 1, "round_start_mono_ns": start, "deadline_mono_ns": deadline, "goodput_bps": 1e6}])
    _write_csv(tmp_path / "queues.csv", QUEUE_COLUMNS, [
        {"t_mono_ns": start - 5_000_000, "round_id": 0, "interface": "s1-gw", "qdisc": "tbf",
         "packets": 100, "drops": 2, "queueing_delay_ms": 0},
        {"t_mono_ns": deadline - 1, "round_id": 1, "interface": "s1-gw", "qdisc": "tbf",
         "packets": 400, "drops": 12, "queueing_delay_ms": 70},
    ])
    _write_csv(tmp_path / "cross_drops.csv", cd.CROSS_DROP_COLUMNS,
               [{"t_mono_ns": t, "seq": i} for i, t in enumerate((start + 1, start + 2, start + 3, deadline + 9))])
    enrich_round_metrics(tmp_path, cross_mbps=3.0)
    row = next(csv.DictReader((tmp_path / "round_metrics.csv").open()))
    assert (row["queue_drops_fg"], row["queue_drops_cross"]) == ("7", "3")  # 10 total, 3 cross in [start, D]

    (tmp_path / "cross_drops.csv").unlink()
    _write_csv(tmp_path / "round_metrics.csv", ROUND_METRIC_COLUMNS,
               [{"round_id": 1, "round_start_mono_ns": start, "deadline_mono_ns": deadline}])
    enrich_round_metrics(tmp_path, cross_mbps=3.0)
    row = next(csv.DictReader((tmp_path / "round_metrics.csv").open()))
    assert row["queue_drops_fg"] == "" and row["queue_drops_cross"] == ""  # unattributable stays blank


def test_cross_duration_covers_every_round():
    assert cross_duration_s(100, 398_000_000, 1.0) == int(np.ceil(0.4 + 100 * 1.498 + 3.0))
    assert cross_duration_s(1, 398_000_000, 1.0) >= 5


def test_fill_baseline_eta_joins_client_energies_and_gateway_deliveries(tmp_path):
    """B1/B2 send no READY: eta comes from the post-run join (design decision D-09)."""
    import json
    from fedgrant.mininet.round_summary import fill_baseline_eta

    _write_csv(tmp_path / "round_metrics.csv", ROUND_METRIC_COLUMNS,
               [{"round_id": r, "round_start_mono_ns": 1, "deadline_mono_ns": 2} for r in (1, 2)])
    weights = {0: 0.25, 1: 0.75}
    energies = {0: [4.0, 1.0], 1: [2.0, 2.0]}
    with (tmp_path / "gateway_rounds.jsonl").open("w") as f:
        f.write(json.dumps({"round_id": 1, "delivered": {"0": [0], "1": [1]}}) + "\n")
        f.write(json.dumps({"round_id": 2, "delivered": {"0": [0, 1]}}) + "\n")
    for cid in (0, 1):
        (tmp_path / f"client_{cid}.json").write_text(json.dumps({"client_weight": weights[cid]}))
        with (tmp_path / f"client_{cid}_rounds.jsonl").open("w") as f:
            f.write(json.dumps({"round_id": 1, "block_energies": energies[cid]}) + "\n")
            if cid == 0:  # client 1 has no round-2 energies -> round 2 stays blank
                f.write(json.dumps({"round_id": 2, "block_energies": energies[cid]}) + "\n")
    assert fill_baseline_eta(tmp_path, 2) == 1
    rows = list(csv.DictReader((tmp_path / "round_metrics.csv").open()))
    total = 0.25**2 * 5 + 0.75**2 * 4
    assert float(rows[0]["eta"]) == pytest.approx((0.25**2 * 4 + 0.75**2 * 2) / total)
    assert rows[1]["eta"] == ""
