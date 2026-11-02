"""Build the shared runs.csv rows from per-run Mininet artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from fedgrant.analysis.runs_schema import make_row, write_runs_csv
from fedgrant.policy.constants import C, S

ROUND_AVERAGES = (
    "eta", "delivered_blocks", "clients_represented", "deadline_completion", "goodput_bps",
    "utilization", "queue_drops_fg", "queue_drops_cross", "wasted_grants",
    "late_after_timeout", "late_after_deadline", "queueing_delay_mean_ms",
    "queueing_delay_p99_ms", "protocol_overhead",
)


def _records(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _finite_values(rows: list[dict[str, str]], name: str) -> list[float]:
    result = []
    for row in rows:
        value = row.get(name, "")
        if value is None or value.strip() == "":
            continue
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite {name} in round_metrics.csv")
        result.append(number)
    return result


def _add_queue_metrics(root: Path, round_rows: list[dict[str, str]], metrics: dict[str, float],
                       cross_mbps: float) -> None:
    """Derive per-round queue metrics from cumulative 10 ms TBF samples."""
    queue = _records(root / "queues.csv")
    queue = [row for row in queue if row.get("interface") == "s1-gw" and row.get("qdisc") == "tbf"]
    samples_by_round: dict[int, list[dict[str, str]]] = {}
    for row in queue:
        rid = int(row.get("round_id") or 0)
        if rid:
            samples_by_round.setdefault(rid, []).append(row)
    util, fg_drops, all_delays = [], [], []
    for round_row in round_rows:
        rid = int(round_row["round_id"])
        try:
            start, deadline = int(round_row["round_start_mono_ns"]), int(round_row["deadline_mono_ns"])
        except (KeyError, ValueError, TypeError):
            continue
        in_window = samples_by_round.get(rid, [])
        if in_window:
            delays = _finite_values(in_window, "queueing_delay_ms")
            all_delays.extend(delays)
            # Cumulative tc counters: baseline is the final sample before t0;
            # endpoint is the final sample no later than D.
            before = [r for r in queue if int(r["t_mono_ns"]) < start]
            through = [r for r in in_window if int(r["t_mono_ns"]) <= deadline]
            if before and through:
                first = before[-1]
                last = through[-1]
                dt_s = (deadline - start) / 1e9
                if (dt_s > 0 and first.get("packets", "").strip()
                        and last.get("packets", "").strip()):
                    packets_out = max(0, int(last.get("packets") or 0) - int(first.get("packets") or 0))
                    drops = max(0, int(last["drops"]) - int(first["drops"]))
                    util.append(packets_out * S * 8 / (float(C) * dt_s))
                    if cross_mbps == 0:
                        fg_drops.append(float(drops))
    if util:
        metrics["utilization"] = float(np.mean(util))
    if fg_drops:
        metrics["queue_drops_fg"] = float(np.mean(fg_drops))
        metrics["queue_drops_cross"] = 0.0
    elif cross_mbps == 0:
        metrics["queue_drops_fg"] = 0.0
        metrics["queue_drops_cross"] = 0.0
    if all_delays:
        metrics["queueing_delay_mean_ms"] = float(np.mean(all_delays))
        metrics["queueing_delay_p99_ms"] = float(np.percentile(all_delays, 99))


def _protocol_overhead(root: Path, rounds: list[dict[str, str]], metrics: dict[str, float]) -> None:
    """Count transmitted non-payload protocol bytes from sender-side packet records."""
    packets = _records(root / "packets.csv")
    round_ids = {int(r["round_id"]) for r in rounds}
    transmitted = [r for r in packets if int(r.get("round_id") or 0) in round_ids and r.get("direction") == "sent"]
    counted_types = {"READY", "GRANT", "DATA", "ROUND_END", "PLAN"}
    nonpayload = 0
    for row in transmitted:
        if row.get("packet_type") in counted_types:
            size = int(row.get("bytes") or 0)
            payload = int(row.get("payload_bytes") or 0) if row.get("packet_type") == "DATA" else 0
            nonpayload += max(0, size - payload)
    delivered_bytes = 0.0
    for row in rounds:
        if not row.get("deadline_mono_ns") or not row.get("round_start_mono_ns"):
            continue
        duration_ns = int(row["deadline_mono_ns"]) - int(row["round_start_mono_ns"])
        goodput = _finite_values([row], "goodput_bps")
        if duration_ns > 0 and goodput:
            delivered_bytes += goodput[0] * (duration_ns / 1e9) / 8
    if delivered_bytes > 0:
        metrics["protocol_overhead"] = nonpayload / delivered_bytes


def run_row(run_dir: str | Path) -> dict[str, Any]:
    """Return one schema-validated runs.csv row for a run directory.

    Round aggregates use rounds 21..last as specified. RTT and scheduler
    percentiles are calculated from individual samples after warm-up.
    """
    root = Path(run_dir)
    meta_path = root / "run.json"
    if not meta_path.exists():
        meta_path = root / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{root}: expected run.json or meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # Summary-schema JSON uses `N`; the older logschema RUN_COLUMNS CSV uses
    # `n_clients`. Prefer the canonical `N` spelling, but read legacy metadata.
    client_count = meta.get("N", meta.get("n_clients"))
    if "N" in meta and "n_clients" in meta and int(meta["N"]) != int(meta["n_clients"]):
        raise ValueError(f"{meta_path}: N and n_clients disagree")
    required = ("experiment_id", "scenario", "beta", "policy", "seed")
    missing = [key for key in required if key not in meta]
    if client_count is None:
        missing.append("N (or legacy n_clients)")
    if missing:
        raise ValueError(f"{meta_path}: missing run identity fields {missing}")
    round_rows = _records(root / "round_metrics.csv")
    rounds = [row for row in round_rows if int(row["round_id"]) > 20]
    status = str(meta.get("status", "ok"))
    metrics: dict[str, float] = {}
    for name in ROUND_AVERAGES:
        values = _finite_values(rounds, name)
        if values:
            metrics[name] = float(np.mean(values, dtype=np.float64))

    _add_queue_metrics(root, rounds, metrics, float(meta.get("cross_mbps", 0.0)))
    _protocol_overhead(root, rounds, metrics)

    rtt = _records(root / "rtt_samples.csv")
    rtt = [row for row in rtt if int(row["round_id"]) > 20]
    rtt_values = _finite_values(rtt, "rtt_ms")
    if rtt_values:
        metrics["rtt_p50_ms"] = float(np.percentile(rtt_values, 50))
        metrics["rtt_p99_ms"] = float(np.percentile(rtt_values, 99))

    scheduler = _records(root / "scheduler_samples.csv")
    scheduler = [row for row in scheduler if int(row["round_id"]) > 20]
    latencies = _finite_values(scheduler, "latency_us")
    if latencies:
        metrics["sched_lat_p50_us"] = float(np.percentile(latencies, 50))
        metrics["sched_lat_p99_us"] = float(np.percentile(latencies, 99))

    processes = [row for row in _records(root / "processes.csv") if row.get("process") == "gateway"]
    warm_processes = [row for row in processes if int(row.get("round_id") or 0) > 20]
    cpu_values = _finite_values(warm_processes, "cpu_pct")
    rss_values = _finite_values(processes, "rss_bytes")
    if cpu_values:
        metrics["gw_cpu_pct"] = float(np.mean(cpu_values))
    if rss_values:
        metrics["gw_rss_mb"] = float(max(rss_values) / (1024 * 1024))

    n_rounds = int(meta.get("n_rounds", meta.get("rounds", len(round_rows))))
    if status != "missing" and not rounds:
        raise ValueError(f"{root}: no round metrics remain after warm-up round 20")
    return make_row(
        str(meta["experiment_id"]), str(meta["scenario"]), float(meta["beta"]), int(client_count),
        str(meta["policy"]), int(meta["seed"]), n_rounds,
        cross_mbps=float(meta.get("cross_mbps", 0.0)), status=status, **metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="run directories containing run.json and raw CSVs")
    parser.add_argument("--output", default="results/summary/runs.csv")
    args = parser.parse_args()
    rows = [run_row(path) for path in args.run_dirs]
    write_runs_csv(rows, args.output)
    print(f"wrote {len(rows)} run rows to {args.output}")


if __name__ == "__main__":
    main()
