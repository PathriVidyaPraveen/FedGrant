"""Fill per-round queue and protocol metrics from sampled/raw Mininet CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from fedgrant.common.logschema import ROUND_METRIC_COLUMNS
from fedgrant.policy.constants import C, S


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def enrich_round_metrics(run_dir: str | Path, *, cross_mbps: float, capacity_bps: int = C) -> None:
    """Add measured queue, utilization, and wire-overhead metrics to each round.

    A single shared qdisc cannot split drops by source flow. With no cross
    traffic all bottleneck drops are foreground; with cross traffic that split
    remains blank instead of being guessed.
    """
    root = Path(run_dir)
    rounds = _read(root / "round_metrics.csv")
    if not rounds:
        return
    queue = [r for r in _read(root / "queues.csv")
             if r.get("interface") == "s1-gw" and r.get("qdisc") == "tbf"]
    packets = _read(root / "packets.csv")
    for current in rounds:
        rid = int(current["round_id"])
        start_ns = int(current.get("round_start_mono_ns") or 0)
        deadline_ns = int(current.get("deadline_mono_ns") or 0)
        if start_ns <= 0 or deadline_ns <= start_ns:
            continue
        in_round = [r for r in queue if int(r.get("round_id") or 0) == rid
                    and int(r["t_mono_ns"]) <= deadline_ns]
        before = [r for r in queue if int(r["t_mono_ns"]) < start_ns]
        if in_round:
            delays = [float(r["queueing_delay_ms"]) for r in in_round
                      if r.get("queueing_delay_ms", "").strip()]
            if delays:
                current["queueing_delay_mean_ms"] = str(float(np.mean(delays)))
                current["queueing_delay_p99_ms"] = str(float(np.percentile(delays, 99)))
            if before:
                first, last = before[-1], in_round[-1]
                duration_s = (deadline_ns - start_ns) / 1e9
                if duration_s > 0 and first.get("packets", "").strip() and last.get("packets", "").strip():
                    packets_out = max(0, int(last.get("packets") or 0) - int(first.get("packets") or 0))
                    drops = max(0, int(last["drops"]) - int(first["drops"]))
                    current["utilization"] = str(packets_out * S * 8 / (capacity_bps * duration_s))
                    if cross_mbps <= 0:
                        current["queue_drops_fg"] = str(drops)
                        current["queue_drops_cross"] = "0"
        elif cross_mbps <= 0:
            current["queue_drops_fg"] = "0"
            current["queue_drops_cross"] = "0"

        transmitted = [r for r in packets if int(r.get("round_id") or 0) == rid
                       and r.get("direction") == "sent"
                       and r.get("packet_type") in {"READY", "GRANT", "DATA", "ROUND_END", "PLAN"}]
        nonpayload_bytes = sum(
            max(0, int(r.get("bytes") or 0) -
                (int(r.get("payload_bytes") or 0) if r.get("packet_type") == "DATA" else 0))
            for r in transmitted
        )
        goodput = float(current.get("goodput_bps") or 0)
        delivered_payload_bytes = goodput * ((deadline_ns - start_ns) / 1e9) / 8
        if delivered_payload_bytes > 0:
            current["protocol_overhead"] = str(nonpayload_bytes / delivered_payload_bytes)
    _write(root / "round_metrics.csv", ROUND_METRIC_COLUMNS, rounds)


__all__ = ["enrich_round_metrics"]
