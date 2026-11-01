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


def sample_qdisc(host: Any, iface: str, *, clock_ns: Callable[[], int] = time.monotonic_ns) -> list[dict[str, object]]:
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
        out.append({
            "t_mono_ns": now,
            "interface": iface,
            "qdisc": str(qdisc.get("kind", "unknown")),
            "backlog_bytes": int(stats.get("backlog", 0)),
            "bytes": int(stats.get("bytes", 0)),
            "drops": int(stats.get("drops", 0)),
            "overlimits": int(stats.get("overlimits", 0)),
        })
    return out


def sample_process(process: Any, name: str, *, clock_ns: Callable[[], int] = time.monotonic_ns) -> dict[str, object]:
    """Sample one local process via psutil, importing it only when requested."""
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("psutil is required for process telemetry") from exc
    proc = psutil.Process(int(process.pid if hasattr(process, "pid") else process))
    with proc.oneshot():
        cpu_pct = proc.cpu_percent(interval=None)
        rss = proc.memory_info().rss
        threads = proc.num_threads()
    return {"t_mono_ns": clock_ns(), "process": name, "pid": proc.pid,
            "cpu_pct": cpu_pct, "rss_bytes": rss, "threads": threads}


class ProcessSampler:
    """Sample a fixed set of local process handles to a headered CSV."""
    def __init__(self, processes: Sequence[tuple[Any, str]], path: str | Path, interval_s: float = 0.1,
                 cpu_id: int | None = None):
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.processes = tuple(processes)
        self.path = Path(path)
        self.interval_s = interval_s
        self.cpu_id = cpu_id
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
                while not self.stop_event.is_set():
                    for process, name in self.processes:
                        if process.poll() is None:
                            try:
                                writer.writerow(sample_process(process, name))
                            except psutil.NoSuchProcess:
                                continue
                    stream.flush()
                    self.stop_event.wait(self.interval_s)
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
