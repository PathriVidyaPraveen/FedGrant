"""Queue and process telemetry samplers using monotonic timestamps."""

from __future__ import annotations

import json
import csv
import os
import threading
import subprocess
import time
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fedgrant.common.logschema import PROCESS_COLUMNS
from fedgrant.mininet.round_state import RoundStateFile
from fedgrant.policy.constants import C


def _tc_integer(value: object, field: str) -> int:
    """Accept iproute2 JSON counters as either integers or suffixed strings."""
    if isinstance(value, bool):
        raise ValueError(f"invalid tc {field}: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits:
            return int(digits)
    raise ValueError(f"invalid tc {field}: {value!r}")


def sample_qdisc(host: Any, iface: str, *, round_id: int = 0,
                 clock_ns: Callable[[], int] = time.monotonic_ns) -> list[dict[str, object]]:
    """Read qdisc backlog/drop counters using tc JSON output."""
    raw = host.cmd(f"tc -s -j qdisc show dev {iface}")
    try:
        qdiscs = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"tc returned invalid JSON for {iface}: {raw[:200]!r}") from exc
    now = clock_ns()
    out = []
    for qdisc in qdiscs:
        stats = qdisc.get("stats", {})
        stats2 = qdisc.get("stats2", {})
        basic_stats = stats2.get("basic", {}) if isinstance(stats2, dict) else {}
        queue_stats = stats2.get("queue", {}) if isinstance(stats2, dict) else {}

        def counter(field: str, *, section: dict[str, object] | None = None) -> int:
            # iproute2 releases expose qdisc counters in different JSON forms:
            # nested under `stats`, flattened on the qdisc object, or grouped
            # in `stats2.basic` / `stats2.queue`.
            for values in (stats, qdisc, section or {}):
                if isinstance(values, dict) and field in values:
                    return _tc_integer(values[field], field)
            return 0

        backlog = counter("backlog", section=queue_stats)
        kind = str(qdisc.get("kind", "unknown"))
        out.append({
            "t_mono_ns": now,
            "round_id": int(round_id),
            "interface": iface,
            "qdisc": kind,
            "backlog_bytes": backlog,
            "bytes": counter("bytes", section=basic_stats),
            "packets": counter("packets", section=basic_stats),
            "drops": counter("drops", section=queue_stats),
            "overlimits": counter("overlimits", section=queue_stats),
            # TBF backlog is in bytes. Convert to serialization time at the
            # configured bottleneck rate C (bits/s), as required by the spec.
            "queueing_delay_ms": (backlog * 8_000.0 / C if iface == "s1-gw" and kind == "tbf" else ""),
        })
    return out


def sample_process(process: Any, name: str, *, round_id: int = 0,
                   clock_ns: Callable[[], int] = time.monotonic_ns) -> dict[str, object]:
    """Sample cumulative CPU time and resident memory for one local process."""
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("psutil is required for process telemetry") from exc
    proc = process if isinstance(process, psutil.Process) else psutil.Process(
        int(process.pid if hasattr(process, "pid") else process)
    )
    with proc.oneshot():
        cpu_times = proc.cpu_times()
        cpu_time_s = float(cpu_times.user + cpu_times.system)
        rss = proc.memory_info().rss
        threads = proc.num_threads()
    return {"t_mono_ns": clock_ns(), "round_id": int(round_id), "process": name, "pid": proc.pid,
            "cpu_time_s": cpu_time_s, "wall_interval_s": "",
            "cpu_pct": "", "rss_bytes": rss, "threads": threads}


def cpu_usage_percent(previous_cpu_time_s: float, current_cpu_time_s: float, wall_interval_s: float) -> float:
    """Return process CPU-time delta as percent of one CPU over wall time."""
    values = (previous_cpu_time_s, current_cpu_time_s, wall_interval_s)
    if any(not np.isfinite(value) for value in values):
        raise ValueError("CPU and wall-time samples must be finite")
    if previous_cpu_time_s < 0 or current_cpu_time_s < previous_cpu_time_s:
        raise ValueError("cumulative process CPU time must be non-negative and monotonic")
    if wall_interval_s <= 0:
        raise ValueError("wall_interval_s must be positive")
    return 100.0 * (current_cpu_time_s - previous_cpu_time_s) / wall_interval_s


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _percentiles(values: list[float], *, unit: str) -> dict[str, object]:
    count = len(values)
    return {
        "unit": unit,
        "sample_count": count,
        "p50": float(np.percentile(values, 50)) if count >= 1 else None,
        "p95": float(np.percentile(values, 95)) if count >= 20 else None,
        "p99": float(np.percentile(values, 99)) if count >= 100 else None,
    }


def summarize_gateway_profile(
    log_dir: str | Path,
    *,
    warmup_rounds: int = 20,
) -> dict[str, object]:
    """Summarize gateway samples, excluding warm-up rounds without inventing zeros."""
    if isinstance(warmup_rounds, bool) or not isinstance(warmup_rounds, int) or warmup_rounds < 0:
        raise ValueError("warmup_rounds must be a non-negative integer")
    root = Path(log_dir)
    rounds_path = root / "gateway_summary.json"
    round_records = (
        json.loads(rounds_path.read_text(encoding="utf-8"))
        if rounds_path.is_file()
        else []
    )
    if not isinstance(round_records, list):
        raise ValueError(f"{rounds_path}: expected a JSON list")
    first_measured_round = warmup_rounds + 1
    boundary = next(
        (
            int(record["start_mono_ns"])
            for record in round_records
            if int(record["round_id"]) == first_measured_round
        ),
        None,
    )

    process_rows = _read_csv(root / "processes.csv")
    if boundary is None:
        process_rows = []
    else:
        process_rows = [
            row for row in process_rows
            if row.get("process") == "gateway"
            and row.get("t_mono_ns", "").strip()
            and int(row["t_mono_ns"]) >= boundary
        ]
    cpu_values = [
        float(row["cpu_pct"])
        for row in process_rows
        if row.get("cpu_pct", "").strip()
    ]
    rss_values = [
        float(row["rss_bytes"])
        for row in process_rows
        if row.get("rss_bytes", "").strip()
    ]
    if any(not np.isfinite(value) or value < 0 for value in (*cpu_values, *rss_values)):
        raise ValueError("gateway profile samples must be finite and non-negative")
    if boundary is not None:
        cpu_values = [
            float(row["cpu_pct"])
            for row in process_rows
            if row.get("cpu_pct", "").strip()
            and int(row["t_mono_ns"]) - int(float(row["wall_interval_s"]) * 1_000_000_000) >= boundary
        ]

    scheduler_rows = [
        row for row in _read_csv(root / "scheduler_samples.csv")
        if row.get("round_id", "").strip()
        and int(row["round_id"]) > warmup_rounds
    ]
    scheduler_values = [
        float(row["latency_us"])
        for row in scheduler_rows
        if row.get("latency_us", "").strip()
    ]
    if any(not np.isfinite(value) or value < 0 for value in scheduler_values):
        raise ValueError("scheduler latency samples must be finite and non-negative")

    rss_mib_values = [value / (1024 * 1024) for value in rss_values]
    return {
        "profiling_enabled": True,
        "warmup_rounds_excluded": warmup_rounds,
        "process_sample_boundary_mono_ns": boundary,
        "cpu_usage": {
            **_percentiles(cpu_values, unit="percent_of_one_cpu"),
            "definition": "100 * process CPU-time delta / monotonic wall-time delta",
            "cpu_time_unit": "seconds (user + system, cumulative in processes.csv)",
            "wall_time_unit": "seconds (per-sample monotonic interval in processes.csv)",
        },
        "rss": {
            **_percentiles(rss_values, unit="bytes"),
            "p50_mib": float(np.percentile(rss_mib_values, 50)) if rss_values else None,
            "p95_mib": float(np.percentile(rss_mib_values, 95)) if len(rss_values) >= 20 else None,
            "p99_mib": float(np.percentile(rss_mib_values, 99)) if len(rss_values) >= 100 else None,
        },
        "scheduler_latency": {
            **_percentiles(scheduler_values, unit="microseconds"),
            "definition": (
                "UDP datagram callback or scheduler tick entry to generated GRANT action dispatch; "
                "trigger is recorded per sample"
            ),
        },
    }


class ProcessSampler:
    """Sample a fixed set of local process handles to a headered CSV."""
    def __init__(self, processes: Sequence[tuple[Any, str]], path: str | Path, interval_s: float = 0.1,
                 cpu_id: int | None = None, round_state_path: str | Path | None = None):
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.processes = tuple(processes)
        self.path = Path(path)
        self.interval_s = interval_s
        self.cpu_id = cpu_id
        self.round_state = RoundStateFile(round_state_path) if round_state_path is not None else None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="fedgrant-process-sampler", daemon=True)
        self.error: BaseException | None = None
        self._previous_samples: dict[tuple[int, str], tuple[int, float]] = {}

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()
        return self

    def _run(self):
        try:
            import psutil
            if self.cpu_id is not None and hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(threading.get_native_id(), {self.cpu_id})
            with self.path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=PROCESS_COLUMNS)
                writer.writeheader()
                tracked = [(psutil.Process(int(process.pid if hasattr(process, "pid") else process)), name)
                           for process, name in self.processes]
                interval_ns = max(1, int(self.interval_s * 1e9))
                next_sample_ns = time.monotonic_ns()
                while not self.stop_event.is_set():
                    for process, name in tracked:
                        try:
                            try:
                                now_ns = time.monotonic_ns()
                                round_id = self.round_state.round_at(now_ns) if self.round_state else 0
                                record = sample_process(process, name, round_id=round_id,
                                                        clock_ns=lambda: now_ns)
                                sample_key = (int(record["pid"]), name)
                                cpu_time_s = float(record["cpu_time_s"])
                                previous = self._previous_samples.get(sample_key)
                                if previous is not None:
                                    previous_ns, previous_cpu_s = previous
                                    interval_s = (now_ns - previous_ns) / 1_000_000_000.0
                                    record["wall_interval_s"] = interval_s
                                    record["cpu_pct"] = cpu_usage_percent(
                                        previous_cpu_s, cpu_time_s, interval_s
                                    )
                                self._previous_samples[sample_key] = (now_ns, cpu_time_s)
                                writer.writerow(record)
                            except psutil.NoSuchProcess:
                                continue
                        except psutil.Error:
                            continue
                    stream.flush()
                    next_sample_ns += interval_ns
                    now_ns = time.monotonic_ns()
                    while next_sample_ns <= now_ns:
                        next_sample_ns += interval_ns
                    self.stop_event.wait((next_sample_ns - now_ns) / 1e9)
        except BaseException as exc:
            self.error = exc
            self.stop_event.set()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise TimeoutError("process sampler did not stop")
        if self.error is not None:
            raise RuntimeError("process sampler failed") from self.error


__all__ = [
    "sample_qdisc",
    "sample_process",
    "cpu_usage_percent",
    "summarize_gateway_profile",
    "ProcessSampler",
]
