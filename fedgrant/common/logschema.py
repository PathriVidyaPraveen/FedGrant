"""Canonical raw-log columns and CSV helpers shared by the Mininet runner."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, TextIO

ROUND_COLUMNS = ("t_mono_ns", "round_id", "event", "client_id", "detail")
PACKET_COLUMNS = ("t_mono_ns", "round_id", "client_id", "packet_seq", "packet_type", "bytes", "result")
QUEUE_COLUMNS = ("t_mono_ns", "interface", "qdisc", "backlog_bytes", "bytes", "drops", "overlimits")
PROCESS_COLUMNS = ("t_mono_ns", "process", "pid", "cpu_pct", "rss_bytes", "threads")
ROUND_METRIC_COLUMNS = (
    "round_id", "eta", "delivered_blocks", "clients_represented", "deadline_completion",
    "goodput_bps", "utilization", "queue_drops_fg", "queue_drops_cross", "wasted_grants",
    "late_after_timeout", "late_after_deadline", "protocol_overhead",
)
RTT_COLUMNS = ("t_mono_ns", "round_id", "rtt_ms")
SCHEDULER_COLUMNS = ("t_mono_ns", "round_id", "latency_us")
RUN_COLUMNS = ("run_id", "experiment_id", "config_id", "policy", "seed", "scenario", "beta", "n_clients", "cross_mbps", "status", "started_mono_ns", "ended_mono_ns")
SCHEMAS = {
    "rounds": ROUND_COLUMNS,
    "packets": PACKET_COLUMNS,
    "queues": QUEUE_COLUMNS,
    "processes": PROCESS_COLUMNS,
    "round_metrics": ROUND_METRIC_COLUMNS,
    "rtt_samples": RTT_COLUMNS,
    "scheduler_samples": SCHEDULER_COLUMNS,
    "run": RUN_COLUMNS,
}


def csv_writer(stream: TextIO, schema: str) -> csv.DictWriter:
    """Create a strict DictWriter and write its header exactly once."""
    if schema not in SCHEMAS:
        raise ValueError(f"unknown raw-log schema {schema!r}")
    writer = csv.DictWriter(stream, fieldnames=SCHEMAS[schema], extrasaction="raise")
    writer.writeheader()
    return writer


def write_rows(path: str | Path, schema: str, rows: Iterable[Mapping[str, object]]) -> Path:
    """Write a headered raw CSV, rejecting missing/extra columns before writing."""
    if schema not in SCHEMAS:
        raise ValueError(f"unknown raw-log schema {schema!r}")
    columns = SCHEMAS[schema]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for index, row in enumerate(rows):
            missing = set(columns) - row.keys()
            if missing:
                raise ValueError(f"{schema} row {index} missing columns {sorted(missing)}")
            writer.writerow(row)
    return p


__all__ = ["ROUND_COLUMNS", "PACKET_COLUMNS", "QUEUE_COLUMNS", "PROCESS_COLUMNS", "ROUND_METRIC_COLUMNS", "RTT_COLUMNS", "SCHEDULER_COLUMNS", "RUN_COLUMNS", "SCHEMAS", "csv_writer", "write_rows"]
