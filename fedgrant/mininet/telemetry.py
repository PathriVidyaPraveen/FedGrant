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
    """Sample one local process via psutil, importing it only when requested."""
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("psutil is required for process telemetry") from exc
    proc = process if isinstance(process, psutil.Process) else psutil.Process(
        int(process.pid if hasattr(process, "pid") else process)
    )
    with proc.oneshot():
        cpu_pct = proc.cpu_percent(interval=None)
        rss = proc.memory_info().rss
        threads = proc.num_threads()
    return {"t_mono_ns": clock_ns(), "round_id": int(round_id), "process": name, "pid": proc.pid,
            "cpu_pct": cpu_pct, "rss_bytes": rss, "threads": threads}


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
                                writer.writerow(sample_process(process, name, round_id=round_id,
                                                                clock_ns=lambda: now_ns))
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


__all__ = ["sample_qdisc", "sample_process", "ProcessSampler"]
