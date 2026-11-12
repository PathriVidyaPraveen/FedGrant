"""Shared Mininet round lifecycle for every gateway process (contract Section 2).

``run_rounds`` owns the timing, the round-state file, the deadline, the late
drain and the round_metrics.csv / gateway_summary.json lifecycle. A
``RoundProtocol`` supplies only the protocol-specific steps, so the grant
gateway (``gateway_proc``) and the B1/B2/B3 gateways (``baseline_proc``) share
one implementation.

Per round:

1. Wait ``compute_window_s`` (spec Tc).
2. ``protocol.begin_round(round_id, now_ns, duration_ns)`` prepares state and
   returns the round window. Nothing has been sent yet.
3. The round-state file is published as *active* for [start, D], so trace replay
   and the samplers use the real round origin.
4. ``protocol.send_round_start(window)`` sends ROUND_START (or the mode's
   equivalent).
5. Sleep until D, then call ``protocol.at_deadline(window)``.
6. The round state moves to *drain* until ``D + late_drain_s``. Sockets, packet
   logs and samplers keep running, so late arrivals are still logged and counted
   against this round. ``RoundStateFile.round_at`` deliberately still returns 0
   after D, because the spec's per-round metrics cover [0, D]; use
   ``phase_at`` to label drain samples.
7. ``protocol.finish_round(window)`` returns the round_metrics row and the
   gateway_summary record. Then the round state becomes idle.
"""

from __future__ import annotations

import asyncio
import csv
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from fedgrant.common.logschema import ROUND_METRIC_COLUMNS
from fedgrant.mininet.round_state import RoundStateFile


@dataclass(frozen=True)
class RoundTiming:
    rounds: int
    duration_ns: int
    compute_window_s: float = 1.0
    late_drain_s: float = 0.1
    inter_round_s: float = 0.0
    start_delay_s: float = 0.4

    def __post_init__(self) -> None:
        if self.rounds < 1 or self.duration_ns < 1:
            raise ValueError("rounds and duration_ns must be positive")
        if min(self.compute_window_s, self.late_drain_s, self.inter_round_s, self.start_delay_s) < 0:
            raise ValueError("wait intervals must be non-negative")


@dataclass(frozen=True)
class RoundWindow:
    round_id: int
    start_ns: int
    deadline_ns: int


class RoundProtocol(Protocol):
    def begin_round(self, round_id: int, now_ns: int, duration_ns: int) -> RoundWindow: ...
    def send_round_start(self, window: RoundWindow) -> None: ...
    def at_deadline(self, window: RoundWindow) -> None: ...
    def finish_round(self, window: RoundWindow) -> tuple[dict[str, Any], dict[str, Any]]: ...


async def run_rounds(
    protocol: RoundProtocol,
    timing: RoundTiming,
    *,
    round_state: RoundStateFile,
    log_dir: str | Path,
    clock: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> list[dict[str, Any]]:
    """Run every round and return the gateway_summary records."""
    log_dir = Path(log_dir)
    round_state.write(round_id=0, active=False)
    await sleep(timing.start_delay_s)
    records: list[dict[str, Any]] = []
    with (log_dir / "round_metrics.csv").open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=ROUND_METRIC_COLUMNS, extrasaction="raise")
        writer.writeheader()
        metrics_file.flush()
        for round_id in range(1, timing.rounds + 1):
            await sleep(timing.compute_window_s)
            window = protocol.begin_round(round_id, clock(), timing.duration_ns)
            if window.round_id != round_id or window.deadline_ns <= window.start_ns:
                raise ValueError(f"protocol returned an invalid window for round {round_id}: {window}")
            round_state.write(round_id=round_id, start_ns=window.start_ns,
                              deadline_ns=window.deadline_ns, active=True)
            protocol.send_round_start(window)
            await sleep(max(0, window.deadline_ns - clock()) / 1e9)
            protocol.at_deadline(window)
            drain_until_ns = window.deadline_ns + int(timing.late_drain_s * 1e9)
            round_state.write(round_id=round_id, start_ns=window.start_ns, deadline_ns=window.deadline_ns,
                              active=False, drain_until_ns=drain_until_ns)
            await sleep(max(0, drain_until_ns - clock()) / 1e9)
            row, record = protocol.finish_round(window)
            writer.writerow({"round_id": round_id, "round_start_mono_ns": window.start_ns,
                             "deadline_mono_ns": window.deadline_ns, **row})
            metrics_file.flush()
            records.append({"round_id": round_id, "start_mono_ns": window.start_ns,
                            "deadline_mono_ns": window.deadline_ns, **record})
            round_state.write(round_id=round_id, active=False)
            if round_id < timing.rounds:
                await sleep(timing.inter_round_s)
    (log_dir / "gateway_summary.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


__all__ = ["RoundProtocol", "RoundTiming", "RoundWindow", "run_rounds"]
