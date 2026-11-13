"""Attribute shared-bottleneck drops to cross traffic vs FedGrant (spec Section 12).

The bottleneck is one TBF queue, whose drop counter cannot say which flow a
dropped frame belonged to. Cross traffic, however, travels xt -> s1 -> gw with
no loss point except that queue, and iperf3 numbers every UDP datagram. So:

* every iperf3 sequence number missing at the gateway is one bottleneck drop
  of cross traffic;
* its drop time is its send time plus the fixed pre-queue delay. The send time
  comes from the iperf3 payload header. The offset to the capture clock is
  calibrated as the minimum (arrival - send) over received datagrams, minus
  one serialization time, which works whichever clock iperf3 uses;
* FedGrant drops in a window = total TBF drops in that window - cross drops.

The capture runs on the gateway host (``tcpdump -i gw-eth0 ... udp and dst port
9999``). Its timestamps are CLOCK_REALTIME; the runner records
``realtime - monotonic`` once so every drop is reported on the monotonic clock
used by all other logs.
"""

from __future__ import annotations

import csv
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fedgrant.policy.constants import C, S

CROSS_PORT = 9999
CROSS_DROP_COLUMNS = ("t_mono_ns", "seq")
_PCAP_MAGIC = {0xA1B2C3D4: ("<", 1000), 0xA1B23C4D: ("<", 1),
               0xD4C3B2A1: (">", 1000), 0x4D3CB2A1: (">", 1)}
_LINKTYPE_ETHERNET = 1


@dataclass(frozen=True)
class CbrTrace:
    arrival_ns: np.ndarray  # int64, capture clock (realtime)
    seq: np.ndarray         # int64, iperf3 packet count
    sent_ns: np.ndarray     # int64, iperf3 sender clock


def read_pcap(path: str | Path) -> Iterator[tuple[int, bytes]]:
    """Yield (timestamp_ns, frame bytes) from a classic little/big-endian pcap file."""
    data = Path(path).read_bytes()
    if len(data) < 24:
        raise ValueError(f"{path}: not a pcap file")
    magic = struct.unpack("<I", data[:4])[0]
    if magic not in _PCAP_MAGIC:
        raise ValueError(f"{path}: unsupported pcap magic {magic:#x} (pcapng is not supported)")
    endian, frac_to_ns = _PCAP_MAGIC[magic]
    linktype = struct.unpack(endian + "I", data[20:24])[0]
    if linktype != _LINKTYPE_ETHERNET:
        raise ValueError(f"{path}: link type {linktype} is not Ethernet")
    offset = 24
    while offset + 16 <= len(data):
        sec, frac, incl, _orig = struct.unpack(endian + "IIII", data[offset:offset + 16])
        offset += 16
        if offset + incl > len(data):
            break  # truncated final record (capture stopped mid-write)
        yield sec * 1_000_000_000 + frac * frac_to_ns, data[offset:offset + incl]
        offset += incl


def cbr_trace(pcap_path: str | Path, *, port: int = CROSS_PORT) -> CbrTrace:
    """Extract iperf3 UDP datagrams to ``port`` (IPv4 over Ethernet) from a capture."""
    arrivals, seqs, sent = [], [], []
    for ts_ns, frame in read_pcap(pcap_path):
        if len(frame) < 14 + 20 + 8 + 12 or frame[12:14] != b"\x08\x00":
            continue
        ihl = (frame[14] & 0x0F) * 4
        if frame[23] != 17:  # UDP
            continue
        udp = 14 + ihl
        if len(frame) < udp + 8 + 12:
            continue
        dst_port = struct.unpack("!H", frame[udp + 2:udp + 4])[0]
        if dst_port != port:
            continue
        sec, usec, count = struct.unpack("!III", frame[udp + 8:udp + 20])
        arrivals.append(ts_ns)
        seqs.append(count)
        sent.append(sec * 1_000_000_000 + usec * 1_000)
    return CbrTrace(np.asarray(arrivals, np.int64), np.asarray(seqs, np.int64), np.asarray(sent, np.int64))


def drop_times(trace: CbrTrace, *, frame_bytes: int = S, capacity_bps: int = C) -> tuple[np.ndarray, np.ndarray]:
    """Return (drop time on the capture clock, seq) for every missing datagram.

    Missing sequence numbers between two received ones are placed by linear
    interpolation of the neighbours' send times. Losses before the first or
    after the last received datagram cannot be detected and are not counted.
    """
    if trace.seq.size < 2:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    order = np.argsort(trace.seq, kind="stable")
    seq, sent, arrival = trace.seq[order], trace.sent_ns[order], trace.arrival_ns[order]
    keep = np.concatenate(([True], np.diff(seq) > 0))  # drop duplicates
    seq, sent, arrival = seq[keep], sent[keep], arrival[keep]
    # Capture-clock time of a send: arrival - send at an empty queue is the
    # fixed path delay plus one bottleneck serialization; the drop happens
    # when the frame reaches the TBF, i.e. before that serialization.
    serialization_ns = int(frame_bytes * 8 * 1e9 / capacity_bps)
    offset_ns = int(np.min(arrival - sent)) - serialization_ns
    gaps = np.flatnonzero(np.diff(seq) > 1)
    times, lost = [], []
    for i in gaps:
        missing = np.arange(seq[i] + 1, seq[i + 1], dtype=np.int64)
        frac = (missing - seq[i]) / (seq[i + 1] - seq[i])
        times.append(sent[i] + frac * (sent[i + 1] - sent[i]) + offset_ns)
        lost.append(missing)
    if not lost:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(times).astype(np.int64), np.concatenate(lost)


def write_cross_drops(pcap_path: str | Path, out_csv: str | Path, realtime_minus_mono_ns: int) -> int:
    """Write cross_drops.csv (monotonic drop time, seq); return the number of drops."""
    times_rt, seqs = drop_times(cbr_trace(pcap_path))
    with Path(out_csv).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(CROSS_DROP_COLUMNS)
        for t, s in zip(times_rt - int(realtime_minus_mono_ns), seqs, strict=True):
            writer.writerow((int(t), int(s)))
    return int(seqs.size)


def read_cross_drop_times(path: str | Path) -> np.ndarray | None:
    """Monotonic drop times from cross_drops.csv, or None when the file is absent."""
    path = Path(path)
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        return np.asarray([int(row["t_mono_ns"]) for row in csv.DictReader(stream)], np.int64)


def split_drops(total_drops: int, cross_drop_times: np.ndarray | None, start_ns: int, end_ns: int,
                cross_mbps: float) -> tuple[int, int] | None:
    """Return (fedgrant, cross) drops in [start, end], or None if they cannot be attributed."""
    if cross_mbps <= 0:
        return int(total_drops), 0
    if cross_drop_times is None:
        return None
    cross = int(np.count_nonzero((cross_drop_times >= start_ns) & (cross_drop_times <= end_ns)))
    # The TBF counter is sampled every 10 ms, so its window can miss a cross
    # drop at the very edge; never report negative FedGrant drops.
    return max(0, int(total_drops) - cross), cross


__all__ = ["CROSS_DROP_COLUMNS", "CROSS_PORT", "CbrTrace", "cbr_trace", "drop_times", "read_cross_drop_times",
           "read_pcap", "split_drops", "write_cross_drops"]
