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

Per-client readiness delays (``readiness_delays_ns`` / ``readiness_delays_us``)
reproduce the simulator exactly: the readiness stream is child 2 of
``SeedSequence(network_round_seed(seed, r)).spawn(6)`` and each client, in id
order, draws ``integers(0, floor(0.1 * D) + 1)`` ns. Mininet runs and their
simulator twins therefore share readiness draws (Section 14, "Pairing").

``run_round_loop`` is the hook-based form of the same lifecycle used by the
B1/B2/B3 processes (``baseline_proc``), whose round steps are asynchronous. It
computes and passes the per-client readiness delays and publishes the same
active -> drain -> idle round-state phases as ``run_rounds``.
"""

from __future__ import annotations

import asyncio
import csv
import inspect
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from fedgrant.common.logschema import ROUND_METRIC_COLUMNS
from fedgrant.mininet.round_state import RoundStateFile
from fedgrant.sim.run_net import compute_deadline_ns
from fedgrant.sim.seeding import network_round_seed


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


# ---------------------------------------------------------------------------
# Readiness delays and the hook-based loop (design decisions D-06 and D-11).

COMPUTE_WINDOW_NS = 1_000_000_000  # Tc = 1.0 s (Section 3.1)
DEFAULT_LATE_DRAIN_NS = 100_000_000  # same default as RoundTiming.late_drain_s; not specified (A6-1)
READINESS_FRACTION = 0.10  # rho_i ~ U(0, 0.1 D)
READINESS_STREAM = 2  # SeedSequence(...).spawn(6) index used by the simulator


@dataclass(frozen=True)
class RoundSchedule:
    duration_ns: int
    compute_window_ns: int = COMPUTE_WINDOW_NS
    late_drain_ns: int = DEFAULT_LATE_DRAIN_NS
    inter_round_ns: int = 0

    def __post_init__(self) -> None:
        if self.duration_ns < 1:
            raise ValueError("duration_ns must be positive")
        if min(self.compute_window_ns, self.late_drain_ns, self.inter_round_ns) < 0:
            raise ValueError("schedule waits must be non-negative")

    @classmethod
    def for_experiment(cls, beta: float, num_clients: int, **kwargs: int) -> RoundSchedule:
        return cls(duration_ns=compute_deadline_ns(beta, num_clients), **kwargs)


def readiness_delays_ns(master_seed: int, round_id: int, num_clients: int, duration_ns: int) -> list[int]:
    """Per-client rho_i in ns, identical to the simulator's readiness draws."""
    if round_id < 1 or num_clients < 1 or duration_ns < 1:
        raise ValueError("round_id, num_clients and duration_ns must be positive")
    streams = np.random.SeedSequence(network_round_seed(master_seed, round_id)).spawn(6)
    rng = np.random.default_rng(streams[READINESS_STREAM])
    max_rho_ns = int(READINESS_FRACTION * duration_ns)
    return [int(rng.integers(0, max_rho_ns + 1)) if max_rho_ns > 0 else 0 for _ in range(num_clients)]


def readiness_delays_us(master_seed: int, round_id: int, num_clients: int, duration_ns: int) -> dict[int, int]:
    """``rho_us`` field values for ROUND_START (floor to whole microseconds)."""
    return {cid: rho // 1_000 for cid, rho in enumerate(
        readiness_delays_ns(master_seed, round_id, num_clients, duration_ns)
    )}


class RoundHooks(Protocol):
    def start_round(self, round_id: int, start_ns: int, deadline_ns: int, rho_us: dict[int, int]) -> Any: ...
    def on_deadline(self, round_id: int, deadline_ns: int) -> Any: ...
    def finalize_round(self, round_id: int) -> Any: ...


@dataclass(frozen=True)
class RoundOutcome:
    """Timestamps of one completed ``run_round_loop`` round."""

    round_id: int
    start_ns: int
    deadline_ns: int
    deadline_handled_ns: int
    drain_end_ns: int


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def run_round_loop(
    *,
    rounds: int,
    schedule: RoundSchedule,
    hooks: RoundHooks,
    master_seed: int,
    num_clients: int,
    round_state: RoundStateFile | None = None,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> list[RoundOutcome]:
    if rounds < 1:
        raise ValueError("rounds must be positive")
    outcomes: list[RoundOutcome] = []
    if round_state is not None:
        round_state.write(round_id=0, active=False)
    for round_id in range(1, rounds + 1):
        await sleep(schedule.compute_window_ns / 1e9)
        start_ns = clock_ns()
        deadline_ns = start_ns + schedule.duration_ns
        rho_us = readiness_delays_us(master_seed, round_id, num_clients, schedule.duration_ns)
        if round_state is not None:
            # Publish the window before ROUND_START leaves, as run_rounds does.
            round_state.write(round_id=round_id, start_ns=start_ns, deadline_ns=deadline_ns, active=True)
        try:
            await _maybe_await(hooks.start_round(round_id, start_ns, deadline_ns, rho_us))
            remaining_ns = deadline_ns - clock_ns()
            if remaining_ns > 0:
                await sleep(remaining_ns / 1e9)
            await _maybe_await(hooks.on_deadline(round_id, deadline_ns))
        except BaseException:
            # A failing hook or a cancelled task never leaves a stale active
            # window for trace replay or the samplers.
            if round_state is not None:
                round_state.write(round_id=round_id, active=False)
            raise
        handled_ns = clock_ns()
        # Same drain semantics as run_rounds: drain until D + late drain.
        drain_until_ns = deadline_ns + schedule.late_drain_ns
        if round_state is not None:
            round_state.write(round_id=round_id, start_ns=start_ns, deadline_ns=deadline_ns,
                              active=False, drain_until_ns=drain_until_ns)
        try:
            await sleep(max(0, drain_until_ns - clock_ns()) / 1e9)
            await _maybe_await(hooks.finalize_round(round_id))
            outcomes.append(RoundOutcome(round_id, start_ns, deadline_ns, handled_ns, clock_ns()))
        finally:
            if round_state is not None:
                round_state.write(round_id=round_id, active=False)
        if round_id < rounds and schedule.inter_round_ns:
            await sleep(schedule.inter_round_ns / 1e9)
    return outcomes


__all__ = [
    "COMPUTE_WINDOW_NS", "RoundHooks", "RoundOutcome", "RoundProtocol", "RoundSchedule", "RoundTiming",
    "RoundWindow", "readiness_delays_ns", "readiness_delays_us", "run_round_loop", "run_rounds",
]
