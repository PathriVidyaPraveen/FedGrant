"""Gateway profiling harness: production GatewayCore + UdpGateway over localhost UDP.

Profiling boundary
------------------
The gateway process runs exactly the production grant gateway used by
``fedgrant.mininet.gateway_proc``: ``contract.build_gateway`` with
``fedgrant.mininet.production:create_gateway`` (policy providers, predictor and
aggregate of ``GatewaySim``), the asyncio ``fedgrant.net.udp_gateway.UdpGateway``
with its 1 ms tick loop, and the shared round lifecycle
(``round_loop.run_rounds`` + ``gateway_proc.GrantRoundProtocol``: Tc, ROUND_START
with seeded rho_i, deadline, late drain). Excluded from the measurement:

* Mininet, tc shaping, GE loss and queueing: this is **localhost UDP**, never a
  Mininet measurement.
* ``LoggingUdpGateway``'s per-packet CSV logging (experiment instrumentation).
* FL model broadcast and aggregate transfer: the aggregate is computed by the
  production provider at the deadline, then discarded.

Synthetic clients (``fedgrant.net.udp_client.UdpClient``, synthetic updates as
in ``client_proc``'s smoke path) run in a *separate* process, so their CPU never
counts as gateway CPU. Each client delays its uplink datagrams by the spec's
access RTT 2(delta_i + 5 ms), delta_i = 10/20/30/40 ms by i mod 4, so the
gateway sees READY at rho_i + RTT and DATA one RTT after its GRANT, as on the
testbed; there is no bandwidth limit, loss or queueing.

Instrumentation (``ProfilingUdpGateway``) is a subclass plus per-instance
wrappers of ``GatewayCore._pump``, ``on_tick`` and ``on_data_message``; samples
go to in-memory lists and are written once after the run (no file I/O in the
packet path). Durations use ``time.perf_counter_ns``; tick scheduling delay
uses ``time.monotonic_ns``, the clock of the tick grid.

Usage::

    python -m fedgrant.profiling.gateway run                       # N = 5 10 20 40 80
    python -m fedgrant.profiling.gateway run --clients 5 --rounds 3 --warmup-rounds 1
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import platform
import secrets
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fedgrant.core.gateway import GatewayCore, GatewayPhase, SendGrant
from fedgrant.net.udp_client import UdpClient
from fedgrant.net.udp_gateway import DEFAULT_TICK_INTERVAL_NS, UdpGateway
from fedgrant.policy.constants import D_MODEL, L as L_SPEC, SRTT0_NS
from fedgrant.policy.eligibility import DEADLINE_MARGIN_NS

MODULE = "fedgrant.profiling.gateway"
PRODUCTION_FACTORY = "fedgrant.mininet.production:create_gateway"
DEFAULT_CLIENT_COUNTS = (5, 10, 20, 40, 80)
DEFAULT_OUTPUT_ROOT = "results/raw/gateway_profile"
WORKLOAD_TYPE = "localhost_udp_synthetic_clients"
DELAY_MODELS = ("spec_rtt", "none")
# Percentiles are reported only with enough samples to be meaningful; otherwise null.
PERCENTILES = (("p50", 50, 1), ("p90", 90, 10), ("p95", 95, 20), ("p99", 99, 100))
CLIENT_SATURATION_PCT = 90.0  # client process CPU p95 at/above this flags a harness bottleneck

TICK_COLUMNS = ("round_id", "active", "scheduled_mono_ns", "actual_mono_ns", "exec_ns", "core_ns", "pump_ns",
                "grants")
PUMP_COLUMNS = ("round_id", "trigger", "duration_ns", "grants")
GRANT_COLUMNS = ("round_id", "client_id", "grant_id", "sendto_ns", "decision_to_sent_ns")
DATA_COLUMNS = ("round_id", "rx_to_core_done_ns", "core_ns", "pump_ns", "total_ns", "outcome")


# ---------------------------------------------------------------------------
# Configuration and run identity


@dataclass(frozen=True)
class ProfileConfig:
    client_counts: tuple[int, ...] = DEFAULT_CLIENT_COUNTS
    rounds: int = 12
    warmup_rounds: int = 2
    policy: str = "B6"
    seed: int = 0
    beta: float = 0.10
    compute_window_s: float = 1.0  # M1 Tc
    late_drain_s: float = 0.1  # M1 drain
    start_delay_s: float = 0.4  # RoundTiming default
    tick_interval_ns: int = DEFAULT_TICK_INTERVAL_NS
    sample_interval_s: float = 0.05
    delay_model: str = "spec_rtt"
    block_length: int = L_SPEC
    models_dir: str = "models"
    output_root: str = DEFAULT_OUTPUT_ROOT

    def __post_init__(self) -> None:
        from fedgrant.mininet.contract import POLICY_CONFIGS

        counts = tuple(self.client_counts)
        if not counts or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts):
            raise ValueError("client_counts must be positive integers")
        if len(set(counts)) != len(counts):
            raise ValueError("client_counts must not repeat")
        object.__setattr__(self, "client_counts", counts)
        if self.rounds < 1:
            raise ValueError("rounds must be positive")
        if not 0 <= self.warmup_rounds < self.rounds:
            raise ValueError("warmup_rounds must be in [0, rounds - 1] so at least one round is measured")
        if self.policy not in POLICY_CONFIGS:
            raise ValueError(f"policy must be a GatewayCore grant policy: {sorted(POLICY_CONFIGS)}")
        if not 0.0 < self.beta <= 1.0:
            raise ValueError("beta must be in (0, 1]")
        if min(self.compute_window_s, self.late_drain_s, self.start_delay_s) < 0:
            raise ValueError("Tc, drain and start delay must be non-negative")
        if self.tick_interval_ns < 1:
            raise ValueError("tick_interval_ns must be positive")
        if not 0.005 <= self.sample_interval_s <= 10.0:
            raise ValueError("sample_interval_s must be in [0.005, 10] s")
        if self.delay_model not in DELAY_MODELS:
            raise ValueError(f"delay_model must be one of {DELAY_MODELS}")
        if not 1 <= self.block_length <= 0xFFFF:
            raise ValueError("block_length must be in [1, 65535]")

    @property
    def blocks(self) -> int:
        return -(-D_MODEL // self.block_length)

    def deadline_ns(self, num_clients: int) -> int:
        from fedgrant.sim.run_net import compute_deadline_ns

        return compute_deadline_ns(self.beta, num_clients, self.blocks)


def uplink_delay_ns(client_id: int, delay_model: str) -> int:
    """Client-side uplink delay emulating the spec access RTT 2(delta_i + 5 ms)."""
    if delay_model == "none":
        return 0
    if delay_model == "spec_rtt":
        from fedgrant.sim.links import access_delay_ns

        return 2 * (access_delay_ns(client_id) + 5_000_000)
    raise ValueError(f"unknown delay model {delay_model!r}")


def make_run_id(now: datetime, revision: str | None, token: str) -> str:
    if now.tzinfo is None:
        raise ValueError("run timestamps must be timezone-aware")
    if not token or not token.isalnum():
        raise ValueError("token must be a non-empty alphanumeric string")
    rev = (revision or "norev")[:8]
    return f"{now.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{rev}-{token}"


def create_run_dir(root: str | Path, run_id: str) -> Path:
    """Create a fresh run directory; never reuses (and so never overwrites) one."""
    path = Path(root) / run_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path


def git_metadata(cwd: str | Path = ".") -> dict[str, Any]:
    """Read-only git revision and dirty flag; None when unavailable."""
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    revision = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"revision": revision, "dirty": None if status is None else bool(status)}


def environment_metadata() -> dict[str, Any]:
    import psutil

    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    release = platform.release()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "kernel_release": release,
        "wsl": "microsoft" in release.lower(),
        "logical_cpus": os.cpu_count(),
        "cpu_model": cpu_model,
        "total_memory_bytes": psutil.virtual_memory().total,
        "numpy": np.__version__,
        "psutil": psutil.__version__,
    }


# ---------------------------------------------------------------------------
# Statistics


def latency_summary(values_ns: Iterable[Any], *, allow_negative: bool = False) -> dict[str, Any]:
    """Distribution in microseconds; invalid samples are counted, never imputed."""
    valid: list[float] = []
    invalid = 0
    for value in values_ns:
        try:
            number = float(value)
        except (TypeError, ValueError):
            invalid += 1
            continue
        if not math.isfinite(number) or (number < 0 and not allow_negative):
            invalid += 1
            continue
        valid.append(number / 1_000.0)
    out: dict[str, Any] = {"unit": "us", "sample_count": len(valid), "invalid_count": invalid}
    array = np.asarray(valid, dtype=np.float64)
    for name, q, minimum in PERCENTILES:
        out[name] = float(np.percentile(array, q)) if len(valid) >= minimum else None
    out["max"] = float(array.max()) if valid else None
    out["min"] = float(array.min()) if valid else None
    out["mean"] = float(array.mean()) if valid else None
    return out


def cpu_summary(rows: Sequence[dict[str, str]], window: tuple[int, int],
                active_windows: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Process CPU in percent of one logical CPU (100 * CPU-time delta / wall delta).

    ``rows`` are one process's ``processes.csv`` samples (cumulative CPU time).
    Only intervals that lie entirely inside ``window`` count; active-phase CPU
    uses intervals entirely inside one round's [start, D].
    """
    start, end = window
    samples = sorted((int(r["t_mono_ns"]), float(r["cpu_time_s"])) for r in rows
                     if r.get("t_mono_ns", "").strip() and r.get("cpu_time_s", "").strip())
    intervals = []
    for (t0, c0), (t1, c1) in zip(samples, samples[1:]):
        if t1 <= t0 or c1 < c0:
            continue  # non-monotonic sample pair: dropped, counted below
        intervals.append((t0, t1, c1 - c0))
    dropped = max(0, len(samples) - 1) - len(intervals)
    inside = [(t0, t1, cpu) for t0, t1, cpu in intervals if start <= t0 and t1 <= end]
    active = [(t0, t1, cpu) for t0, t1, cpu in inside
              if any(a <= t0 and t1 <= b for a, b in active_windows)]

    def mean_pct(items):
        wall = sum(t1 - t0 for t0, t1, _ in items) / 1e9
        return (100.0 * sum(cpu for _, _, cpu in items) / wall) if wall > 0 else None

    per_sample = [100.0 * cpu / ((t1 - t0) / 1e9) for t0, t1, cpu in inside]
    array = np.asarray(per_sample, dtype=np.float64)
    return {
        "unit": "percent_of_one_logical_cpu",
        "scope": "process",
        "interval_mean_pct": mean_pct(inside),
        "interval_wall_s": sum(t1 - t0 for t0, t1, _ in inside) / 1e9,
        "interval_sample_count": len(inside),
        "active_mean_pct": mean_pct(active),
        "active_wall_s": sum(t1 - t0 for t0, t1, _ in active) / 1e9,
        "active_sample_count": len(active),
        "per_sample_p50_pct": float(np.percentile(array, 50)) if len(per_sample) >= 1 else None,
        "per_sample_p95_pct": float(np.percentile(array, 95)) if len(per_sample) >= 20 else None,
        "per_sample_max_pct": float(array.max()) if per_sample else None,
        "dropped_non_monotonic_intervals": dropped,
    }


def rss_summary(rows: Sequence[dict[str, str]], baseline: tuple[int, int] | None,
                window: tuple[int, int]) -> dict[str, Any]:
    """Resident set size (not virtual size) before and during the measured rounds."""
    points = [(int(r["t_mono_ns"]), int(float(r["rss_bytes"]))) for r in rows
              if r.get("t_mono_ns", "").strip() and r.get("rss_bytes", "").strip()]
    before = [v for t, v in points if baseline is not None and baseline[0] <= t < baseline[1]]
    during = [v for t, v in points if window[0] <= t <= window[1]]
    mib = 1024 * 1024
    base = float(np.median(before)) if before else None
    peak = max(during) if during else None
    return {
        "unit": "bytes",
        "baseline_bytes": base,
        "baseline_sample_count": len(before),
        "steady_median_bytes": float(np.median(during)) if during else None,
        "peak_bytes": peak,
        "measured_sample_count": len(during),
        "growth_bytes": (peak - base) if peak is not None and base is not None else None,
        "baseline_mib": base / mib if base is not None else None,
        "peak_mib": peak / mib if peak is not None else None,
    }


def classify_data_outcome(before: dict[str, int], after: dict[str, int]) -> str:
    """Outcome of one DATA datagram from GatewayCore counter deltas."""
    for name, label in (("stale_round_data", "stale_round"), ("late_after_deadline", "late_after_deadline"),
                        ("accepted_data", "accepted"), ("duplicate_data", "duplicate"),
                        ("rejected_data", "rejected")):
        if after.get(name, 0) > before.get(name, 0):
            return label
    return "other"


_OUTCOME_COUNTERS = ("stale_round_data", "late_after_deadline", "accepted_data", "duplicate_data", "rejected_data")


# ---------------------------------------------------------------------------
# Instrumented gateway


@dataclass
class SampleRecorder:
    """In-memory sample buffers; written once after the run."""

    ticks: list[tuple] = field(default_factory=list)
    pumps: list[tuple] = field(default_factory=list)
    grants: list[tuple] = field(default_factory=list)
    data: list[tuple] = field(default_factory=list)

    FILES = (("tick_samples.csv", "ticks", TICK_COLUMNS), ("pump_samples.csv", "pumps", PUMP_COLUMNS),
             ("grant_samples.csv", "grants", GRANT_COLUMNS), ("data_samples.csv", "data", DATA_COLUMNS))

    def write(self, directory: str | Path) -> dict[str, int]:
        counts = {}
        for name, attribute, columns in self.FILES:
            rows = getattr(self, attribute)
            with (Path(directory) / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(columns)
                writer.writerows(["" if value is None else value for value in row] for row in rows)
            counts[attribute] = len(rows)
        return counts


def _discard_aggregate(aggregate: Any) -> None:
    """The production aggregate is computed at D; its FL transfer is out of scope."""


class ProfilingUdpGateway(UdpGateway):
    """UdpGateway with timing probes; protocol behaviour is unchanged."""

    def __init__(self, core: GatewayCore, *, recorder: SampleRecorder,
                 perf_ns: Callable[[], int] = time.perf_counter_ns, **kwargs: Any) -> None:
        super().__init__(core, **kwargs)
        self.recorder = recorder
        self.perf_ns = perf_ns
        self._trigger = "other"
        self._pump_ns = 0
        self._last_pump_end_ns: int | None = None
        self._core_span: tuple[int, int] | None = None
        self._tick_core: tuple[int, int] | None = None
        self._deadline_snapshot: tuple[int, int] | None = None
        self._wrap_core()

    @classmethod
    async def bind(cls, core: GatewayCore, *, recorder: SampleRecorder, local_addr: tuple[str, int],
                   **kwargs: Any) -> "ProfilingUdpGateway":
        protocol = cls(core, recorder=recorder, **kwargs)
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(lambda: protocol, local_addr=local_addr)
        return protocol

    def _round_id(self) -> int:
        return self.core.round.round_id if self.core.round is not None else 0

    def _wrap_core(self) -> None:
        core, perf = self.core, self.perf_ns
        pump, on_tick, on_data = core._pump, core.on_tick, core.on_data_message

        def timed_pump(t_ns: int):
            start = perf()
            actions = pump(t_ns)
            end = perf()
            grants = sum(isinstance(action, SendGrant) for action in actions)
            self._pump_ns += end - start
            self._last_pump_end_ns = end
            self.recorder.pumps.append((self._round_id(), self._trigger, end - start, grants))
            return actions

        def timed_on_tick(t_ns: int):
            start = perf()
            actions = on_tick(t_ns)
            self._tick_core = (perf() - start, sum(isinstance(a, SendGrant) for a in actions))
            return actions

        def timed_on_data(data, **kwargs):
            start = perf()
            actions = on_data(data, **kwargs)
            self._core_span = (start, perf())
            return actions

        core._pump = timed_pump
        core.on_tick = timed_on_tick
        core.on_data_message = timed_on_data

    def datagram_received(self, data: bytes, addr) -> None:
        rx = self.perf_ns()  # callback entry; kernel queueing before it is not visible here
        previous = self._trigger
        self._trigger, self._pump_ns, self._core_span = "datagram", 0, None
        stats = self.core.stats
        before = {name: getattr(stats, name) for name in _OUTCOME_COUNTERS}
        round_id = self._round_id()
        try:
            super().datagram_received(data, addr)
        finally:
            end = self.perf_ns()
            self._trigger = previous
        span = self._core_span
        if span is None:
            return  # READY, or dropped by the transport before reaching the core
        after = {name: getattr(stats, name) for name in _OUTCOME_COUNTERS}
        core_start, core_end = span
        self.recorder.data.append((round_id, core_end - rx, core_end - core_start, self._pump_ns, end - rx,
                                   classify_data_outcome(before, after)))

    def _run_tick(self) -> None:
        core = self.core
        if (core.phase is GatewayPhase.ACTIVE and core.round is not None
                and self.clock_ns() >= core.round.deadline_ns):
            self.capture_deadline_state()  # as LoggingUdpGateway: snapshot before the tick ends the round
        scheduled = self.scheduled_tick_ns
        actual = self.clock_ns()
        active = core.phase is GatewayPhase.ACTIVE
        round_id = self._round_id()
        previous = self._trigger
        self._trigger, self._pump_ns, self._tick_core = "tick", 0, None
        start = self.perf_ns()
        try:
            super()._run_tick()
        finally:
            end = self.perf_ns()
            self._trigger = previous
        core_ns, grants = self._tick_core if self._tick_core is not None else (None, 0)
        self.recorder.ticks.append((round_id, int(active), scheduled, actual, end - start, core_ns,
                                    self._pump_ns, grants))

    def _send_action(self, action) -> None:
        if not isinstance(action, SendGrant):
            super()._send_action(action)
            return
        start = self.perf_ns()
        sent_before = self.stats.sent
        super()._send_action(action)
        end = self.perf_ns()
        if self.stats.sent > sent_before:
            decision_end = self._last_pump_end_ns
            self.recorder.grants.append((self._round_id(), action.client_id, action.grant_id, end - start,
                                         None if decision_end is None else end - decision_end))

    # Hooks used by gateway_proc.GrantRoundProtocol (same semantics as LoggingUdpGateway).
    def capture_deadline_state(self) -> int:
        if self.core.round is None:
            return 0
        round_id = self.core.round.round_id
        if self._deadline_snapshot is None or self._deadline_snapshot[0] != round_id:
            unresolved = sum(len(state.outstanding) for state in self.core.clients.values())
            self._deadline_snapshot = (round_id, unresolved)
        return self._deadline_snapshot[1]

    def log_timeouts(self) -> None:
        """No grant log in the harness."""


# ---------------------------------------------------------------------------
# Child processes


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


async def _gateway_main(cfg: dict[str, Any]) -> None:
    from fedgrant.mininet.contract import MininetGatewayContext, build_gateway, policy_config_for, uniform_weights
    from fedgrant.mininet.gateway_proc import GrantRoundProtocol
    from fedgrant.mininet.round_loop import RoundTiming, run_rounds
    from fedgrant.mininet.round_state import RoundStateFile

    n_dir, n = Path(cfg["n_dir"]), int(cfg["num_clients"])
    ctx = MininetGatewayContext(
        num_clients=n, policy=cfg["policy"], seed=int(cfg["seed"]), policy_config=policy_config_for(cfg["policy"]),
        client_weights=uniform_weights(n), deadline_ns=int(cfg["deadline_ns"]), K=int(cfg["blocks"]),
        L=int(cfg["block_length"]), models_dir=cfg["models_dir"], log_dir=str(n_dir),
    )
    core = build_gateway(PRODUCTION_FACTORY, ctx)
    recorder = SampleRecorder()
    gateway = await ProfilingUdpGateway.bind(
        core, recorder=recorder, local_addr=("127.0.0.1", int(cfg["gateway_port"])),
        endpoints={cid: ("127.0.0.1", int(port)) for cid, port in enumerate(cfg["client_ports"])},
        tick_interval_ns=int(cfg["tick_interval_ns"]), aggregate_handler=_discard_aggregate,
    )
    _write_json(n_dir / "gateway_ready.json", {"t_mono_ns": time.monotonic_ns(), "pid": os.getpid()})
    try:
        await run_rounds(
            GrantRoundProtocol(core, gateway, num_clients=n, K=int(cfg["blocks"]), L=int(cfg["block_length"]),
                               seed=int(cfg["seed"])),
            RoundTiming(rounds=int(cfg["rounds"]), duration_ns=int(cfg["deadline_ns"]),
                        compute_window_s=float(cfg["compute_window_s"]), late_drain_s=float(cfg["late_drain_s"]),
                        start_delay_s=float(cfg["start_delay_s"])),
            round_state=RoundStateFile(n_dir / "round_state.json"), log_dir=n_dir)
    finally:
        await gateway.close()
        counts = recorder.write(n_dir)
        _write_json(n_dir / "gateway_stats.json", {
            "core": asdict(core.stats), "transport": asdict(gateway.stats), "sample_counts": counts,
        })


class DelayedUdpClient(UdpClient):
    """UdpClient whose uplink datagrams leave after a fixed emulated path delay."""

    uplink_delay_s: float = 0.0

    def _send(self, packet: bytes) -> bool:
        if self.uplink_delay_s <= 0:
            return super()._send(packet)
        if self.transport is None or self._closing:
            return False
        asyncio.get_running_loop().call_later(self.uplink_delay_s, self._send_now, packet)
        self.stats.sent += 1
        return True

    def _send_now(self, packet: bytes) -> None:
        if self.transport is not None and not self._closing:
            self.transport.sendto(packet, self.gateway_addr)


def synthetic_update_provider(seed: int, client_id: int, block_length: int) -> Callable[[int], np.ndarray]:
    """Same synthetic updates as client_proc's smoke path (no value bank)."""
    from fedgrant.mininet.profile_update import profile_bank, synthetic_update

    profiles = profile_bank(seed + client_id, block_length=block_length)

    def provider(round_id: int) -> np.ndarray:
        profile = profiles[(round_id - 1) % len(profiles)]
        return synthetic_update(profile, seed=seed + client_id * 1009 + round_id, block_length=block_length)

    return provider


async def _clients_main(cfg: dict[str, Any]) -> None:
    from fedgrant.core.client_core import ClientCore
    from fedgrant.mininet.client_proc import PrefetchedUpdates

    n_dir, n, rounds = Path(cfg["n_dir"]), int(cfg["num_clients"]), int(cfg["rounds"])
    round_timeout_s = (float(cfg["start_delay_s"]) + float(cfg["compute_window_s"])
                       + int(cfg["deadline_ns"]) / 1e9 + float(cfg["late_drain_s"]) + 10.0)
    clients: list[tuple[DelayedUdpClient, Any]] = []
    for cid in range(n):
        updates = PrefetchedUpdates(synthetic_update_provider(int(cfg["seed"]), cid, int(cfg["block_length"])),
                                    rounds)
        updates.prefetch(1)
        client = await DelayedUdpClient.bind(
            ClientCore(cid, L=int(cfg["block_length"])), gateway_addr=("127.0.0.1", int(cfg["gateway_port"])),
            local_addr=("127.0.0.1", 0), update_provider=updates)  # kernel-assigned: no collisions
        client.uplink_delay_s = uplink_delay_ns(cid, cfg["delay_model"]) / 1e9
        clients.append((client, updates))
    ports = [int(client.transport.get_extra_info("sockname")[1]) for client, _ in clients]
    _write_json(n_dir / "clients_ready.json", {"t_mono_ns": time.monotonic_ns(), "pid": os.getpid(),
                                               "client_ports": ports})

    async def drive(client: DelayedUdpClient, updates) -> dict[str, Any]:
        completed: list[int] = []
        for _ in range(rounds):
            try:
                round_id = await client.wait_for_round_end(timeout=round_timeout_s)
            except asyncio.TimeoutError:
                break
            if not completed or round_id > completed[-1]:
                completed.append(round_id)
                updates.prefetch(round_id + 1)
            client._round_end_event.clear()
        stats = client.stats
        return {"client_id": client.core.client_id, "completed_rounds": completed,
                "uplink_delay_s": client.uplink_delay_s, **asdict(stats)}

    try:
        results = await asyncio.gather(*(drive(client, updates) for client, updates in clients))
    finally:
        for client, _ in clients:
            await client.close()
    _write_json(n_dir / "clients.json", results)


# ---------------------------------------------------------------------------
# Orchestration and summaries


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def count_skipped_ticks(scheduled: Sequence[int], interval_ns: int) -> int:
    """Grid ticks the loop skipped after an overrun (gap > one interval)."""
    ordered = sorted(scheduled)
    return sum(max(0, (b - a) // interval_ns - 1) for a, b in zip(ordered, ordered[1:]))


def summarize_n(n_dir: Path, cfg: ProfileConfig, num_clients: int) -> dict[str, Any]:
    """Summary for one client count from its raw artifacts (no imputed values)."""
    issues: list[str] = []
    records = _read_json(n_dir / "gateway_summary.json") or []
    drain_ns = int(round(cfg.late_drain_s * 1e9))
    measured = [r for r in records if int(r["round_id"]) > cfg.warmup_rounds]
    if len(records) != cfg.rounds:
        issues.append(f"gateway completed {len(records)} of {cfg.rounds} rounds")
    result: dict[str, Any] = {"num_clients": num_clients, "deadline_ns": cfg.deadline_ns(num_clients),
                              "measured_rounds": [int(r["round_id"]) for r in measured]}
    if not measured:
        result.update(status="failed", issues=issues + ["no measured (post-warm-up) rounds"])
        return result
    window = (int(measured[0]["start_mono_ns"]), int(measured[-1]["deadline_mono_ns"]) + drain_ns)
    active_windows = [(int(r["start_mono_ns"]), int(r["deadline_mono_ns"])) for r in measured]
    ready = _read_json(n_dir / "gateway_ready.json")
    first_start = int(records[0]["start_mono_ns"])
    baseline = (int(ready["t_mono_ns"]), first_start) if ready else None
    if baseline is None:
        issues.append("gateway_ready.json missing: no pre-workload RSS baseline")

    processes = _read_csv(n_dir / "processes.csv")
    gateway_rows = [r for r in processes if r["process"] == "gateway"]
    client_rows = [r for r in processes if r["process"] == "clients"]
    gateway_cpu = cpu_summary(gateway_rows, window, active_windows)
    client_cpu = cpu_summary(client_rows, window, active_windows)
    result["measurement_window_mono_ns"] = list(window)
    result["measurement_window_s"] = (window[1] - window[0]) / 1e9
    result["gateway_cpu"] = gateway_cpu
    result["gateway_rss"] = rss_summary(gateway_rows, baseline, window)
    result["client_process_cpu"] = client_cpu

    warm = cfg.warmup_rounds
    ticks = [r for r in _read_csv(n_dir / "tick_samples.csv") if int(r["round_id"]) > warm]
    active_ticks = [r for r in ticks if r["active"] == "1"]
    scheduled = [int(r["scheduled_mono_ns"]) for r in _read_csv(n_dir / "tick_samples.csv")
                 if r["scheduled_mono_ns"].strip()]
    pumps = [r for r in _read_csv(n_dir / "pump_samples.csv") if int(r["round_id"]) > warm]
    grants = [r for r in _read_csv(n_dir / "grant_samples.csv") if int(r["round_id"]) > warm]
    data = [r for r in _read_csv(n_dir / "data_samples.csv") if int(r["round_id"]) > warm]
    granting = [r for r in pumps if int(r["grants"]) > 0]
    result["latency"] = {
        "tick_scheduling_delay": latency_summary(
            (int(r["actual_mono_ns"]) - int(r["scheduled_mono_ns"]) if r["scheduled_mono_ns"].strip() else None
             for r in active_ticks), allow_negative=True),
        "tick_execution": latency_summary(r["exec_ns"] for r in active_ticks),
        "tick_core": latency_summary(r["core_ns"] for r in active_ticks),
        "grant_decision_pump": latency_summary(r["duration_ns"] for r in granting),
        "grant_decision_per_grant": latency_summary(int(r["duration_ns"]) / int(r["grants"]) for r in granting),
        "pump_without_grant": latency_summary(r["duration_ns"] for r in pumps if int(r["grants"]) == 0),
        "grant_sendto": latency_summary(r["sendto_ns"] for r in grants),
        "grant_decision_to_sent": latency_summary(r["decision_to_sent_ns"] for r in grants),
        "data_rx_to_core_done": latency_summary(r["rx_to_core_done_ns"] for r in data),
        "data_core": latency_summary(r["core_ns"] for r in data),
        "data_validation": latency_summary(int(r["core_ns"]) - int(r["pump_ns"]) for r in data),
        "data_callback_total": latency_summary(r["total_ns"] for r in data),
    }
    outcomes: dict[str, int] = {}
    for row in data:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    measured_ids = {int(r["round_id"]) for r in measured}
    metric_rows = [r for r in _read_csv(n_dir / "round_metrics.csv") if int(r["round_id"]) in measured_ids]
    result["counts"] = {
        "active_ticks": len(active_ticks), "idle_ticks": len(ticks) - len(active_ticks),
        "skipped_ticks_all_rounds": count_skipped_ticks(scheduled, cfg.tick_interval_ns),
        "grants_sent": len(grants), "granting_pumps": len(granting), "data_datagrams": len(data),
        "data_outcomes": outcomes,
        "delivered_blocks_per_round": [int(r["delivered_blocks"]) for r in metric_rows],
    }
    clients = _read_json(n_dir / "clients.json")
    if clients is None:
        issues.append("clients.json missing")
    else:
        incomplete = [c["client_id"] for c in clients if len(c["completed_rounds"]) != cfg.rounds]
        timeouts = sum(int(c["round_end_timeouts"]) for c in clients)
        result["counts"]["client_round_end_timeouts"] = timeouts
        if incomplete:
            issues.append(f"clients without all rounds completed: {incomplete}")
        if timeouts:
            issues.append(f"{timeouts} client ROUND_END fallbacks")
    if any(v == 0 for v in result["counts"]["delivered_blocks_per_round"]):
        issues.append("a measured round delivered no blocks")
    if result["deadline_ns"] < SRTT0_NS + DEADLINE_MARGIN_NS:
        # Section 3.9 rule (iv) t + srtt + 2 ms <= D with srtt_0 = 100 ms: no client is ever
        # eligible before its first RTT sample, and none can arrive without a grant.
        issues.append(f"D = {result['deadline_ns'] / 1e6:.0f} ms < srtt_0 + 2 ms: eligibility rule (iv) "
                      "blocks every grant (spec parameters, not a harness failure)")
    p95 = client_cpu["per_sample_p95_pct"]
    if p95 is not None and p95 >= CLIENT_SATURATION_PCT:
        issues.append(f"client process CPU p95 {p95:.0f}% >= {CLIENT_SATURATION_PCT:.0f}%: harness may limit load")
    for name in ("interval_sample_count", "active_sample_count"):
        if gateway_cpu[name] == 0:
            issues.append(f"gateway CPU {name} is 0")
    result["issues"] = issues
    result["status"] = "ok" if not issues else "completed_with_issues"
    return result


def run_one(cfg: ProfileConfig, num_clients: int, run_dir: Path) -> dict[str, Any]:
    from fedgrant.mininet.telemetry import ProcessSampler

    n_dir = run_dir / f"n{num_clients:03d}"
    n_dir.mkdir()
    child = {
        **asdict(cfg), "client_counts": list(cfg.client_counts), "num_clients": num_clients, "n_dir": str(n_dir),
        "blocks": cfg.blocks, "deadline_ns": cfg.deadline_ns(num_clients), "gateway_port": _free_port(),
    }
    config_path = n_dir / "child_config.json"
    _write_json(config_path, child)
    gateway_config_path = n_dir / "gateway_config.json"
    timeout_s = cfg.start_delay_s + cfg.rounds * (cfg.compute_window_s + child["deadline_ns"] / 1e9
                                                   + cfg.late_drain_s + 1.0) + 60.0
    failure = None
    with (n_dir / "clients.log").open("w") as clients_log, (n_dir / "gateway.log").open("w") as gateway_log:
        clients = subprocess.Popen([sys.executable, "-m", MODULE, "clients", "--config", str(config_path)],
                                   stdout=clients_log, stderr=subprocess.STDOUT)
        gateway = sampler = None
        try:
            deadline = time.monotonic() + 120.0
            while not (n_dir / "clients_ready.json").exists():
                if clients.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("client process did not become ready")
                time.sleep(0.05)
            client_ports = _read_json(n_dir / "clients_ready.json")["client_ports"]
            if len(client_ports) != num_clients or len(set(client_ports)) != num_clients:
                raise RuntimeError(f"client process reported invalid ports {client_ports}")
            if child["gateway_port"] in client_ports:  # clients already target this port: fail, never move it
                raise RuntimeError("a client was assigned the gateway port; rerun")
            _write_json(gateway_config_path, {**child, "client_ports": client_ports})
            gateway = subprocess.Popen([sys.executable, "-m", MODULE, "gateway", "--config",
                                        str(gateway_config_path)],
                                       stdout=gateway_log, stderr=subprocess.STDOUT)
            sampler = ProcessSampler([(gateway, "gateway"), (clients, "clients")], n_dir / "processes.csv",
                                     interval_s=cfg.sample_interval_s,
                                     round_state_path=n_dir / "round_state.json").start()
            if gateway.wait(timeout=timeout_s) != 0:
                failure = f"gateway process exited {gateway.returncode}"
            if clients.wait(timeout=60) != 0:
                failure = failure or f"client process exited {clients.returncode}"
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            failure = str(exc)
        finally:
            for proc in (gateway, clients):
                if proc is not None and proc.poll() is None:
                    proc.kill()
                    proc.wait()
            if sampler is not None:
                sampler.stop()
    summary = summarize_n(n_dir, cfg, num_clients)
    if failure is not None:
        summary["status"] = "failed"
        summary.setdefault("issues", []).insert(0, failure)
    _write_json(n_dir / "summary.json", summary)
    return summary


LATENCY_ROWS = ("tick_scheduling_delay", "tick_execution", "tick_core", "grant_decision_pump",
                "grant_decision_per_grant", "pump_without_grant", "grant_sendto", "grant_decision_to_sent",
                "data_rx_to_core_done", "data_core", "data_validation", "data_callback_total")
RESOURCE_COLUMNS = ("num_clients", "status", "deadline_ms", "measured_rounds", "measurement_window_s",
                    "gateway_cpu_interval_mean_pct", "gateway_cpu_active_mean_pct", "gateway_cpu_active_samples",
                    "gateway_cpu_per_sample_max_pct", "gateway_rss_baseline_mib", "gateway_rss_peak_mib",
                    "gateway_rss_growth_mib", "client_cpu_per_sample_p95_pct", "issues")
LATENCY_COLUMNS = ("num_clients", "metric", "unit", "sample_count", "invalid_count", "p50", "p90", "p95", "p99",
                   "max", "min", "mean")


def write_summaries(run_dir: Path, results: Sequence[dict[str, Any]]) -> None:
    mib = 1024 * 1024
    with (run_dir / "summary_resources.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESOURCE_COLUMNS)
        writer.writeheader()
        for r in results:
            cpu, rss, client = r.get("gateway_cpu", {}), r.get("gateway_rss", {}), r.get("client_process_cpu", {})
            writer.writerow({
                "num_clients": r["num_clients"], "status": r["status"], "deadline_ms": r["deadline_ns"] / 1e6,
                "measured_rounds": len(r.get("measured_rounds", [])),
                "measurement_window_s": r.get("measurement_window_s"),
                "gateway_cpu_interval_mean_pct": cpu.get("interval_mean_pct"),
                "gateway_cpu_active_mean_pct": cpu.get("active_mean_pct"),
                "gateway_cpu_active_samples": cpu.get("active_sample_count"),
                "gateway_cpu_per_sample_max_pct": cpu.get("per_sample_max_pct"),
                "gateway_rss_baseline_mib": rss.get("baseline_mib"),
                "gateway_rss_peak_mib": rss.get("peak_mib"),
                "gateway_rss_growth_mib": (rss["growth_bytes"] / mib if rss.get("growth_bytes") is not None
                                           else None),
                "client_cpu_per_sample_p95_pct": client.get("per_sample_p95_pct"),
                "issues": "; ".join(r.get("issues", [])),
            })
    with (run_dir / "summary_latency.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LATENCY_COLUMNS)
        writer.writeheader()
        for r in results:
            for metric in LATENCY_ROWS:
                stats = r.get("latency", {}).get(metric)
                if stats is not None:
                    writer.writerow({"num_clients": r["num_clients"], "metric": metric,
                                     **{k: stats.get(k) for k in LATENCY_COLUMNS[2:]}})
    _write_json(run_dir / "summary.json", list(results))
    (run_dir / "summary.md").write_text(summary_markdown(results), encoding="utf-8")


def _fmt(value: Any, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def summary_markdown(results: Sequence[dict[str, Any]]) -> str:
    lines = [
        f"Workload: {WORKLOAD_TYPE} (not Mininet). CPU in % of one logical CPU (process-level); "
        "latencies in microseconds, post-warm-up rounds only; n/a = too few samples.",
        "",
        "| N | status | CPU active mean % | CPU interval mean % | RSS base / peak MiB | tick delay p50 / p99 "
        "| tick exec p50 / p99 | grant decision p50 / p99 | grant sendto p50 / p99 | DATA core p50 / p99 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        cpu, rss, lat = r.get("gateway_cpu", {}), r.get("gateway_rss", {}), r.get("latency", {})

        def pair(metric: str) -> str:
            s = lat.get(metric) or {}
            return f"{_fmt(s.get('p50'))} / {_fmt(s.get('p99'))}"

        lines.append(
            f"| {r['num_clients']} | {r['status']} | {_fmt(cpu.get('active_mean_pct'))} "
            f"| {_fmt(cpu.get('interval_mean_pct'))} | {_fmt(rss.get('baseline_mib'))} / {_fmt(rss.get('peak_mib'))} "
            f"| {pair('tick_scheduling_delay')} | {pair('tick_execution')} | {pair('grant_decision_pump')} "
            f"| {pair('grant_sendto')} | {pair('data_core')} |")
    issues = [f"- N={r['num_clients']}: {issue}" for r in results for issue in r.get("issues", [])]
    if issues:
        lines += ["", "Issues:", *issues]
    return "\n".join(lines) + "\n"


def build_manifest(cfg: ProfileConfig, run_id: str, created: datetime) -> dict[str, Any]:
    from fedgrant.mininet.contract import POLICY_CONFIGS

    return {
        "run_id": run_id,
        "created_utc": created.astimezone(timezone.utc).isoformat(),
        "git": git_metadata(),
        "workload_type": WORKLOAD_TYPE,
        "is_mininet": False,
        "gateway": {
            "factory": PRODUCTION_FACTORY,
            "transport": "fedgrant.net.udp_gateway.UdpGateway (ProfilingUdpGateway probes)",
            "round_lifecycle": "fedgrant.mininet.round_loop.run_rounds + gateway_proc.GrantRoundProtocol",
            "policy": cfg.policy,
            "policy_config": asdict(POLICY_CONFIGS[cfg.policy]),
            "client_weights": "uniform 1/N",
            "aggregate": "computed by the production provider at D, then discarded",
            "excluded": ["Mininet/tc/GE loss", "LoggingUdpGateway per-packet CSV logging",
                         "FL model broadcast and aggregate transfer"],
        },
        "clients": {
            "implementation": "fedgrant.net.udp_client.UdpClient (DelayedUdpClient uplink delay), one process",
            "updates": "synthetic (client_proc smoke-path formula), no value bank",
            "delay_model": cfg.delay_model,
            "delay_definition": "uplink datagrams delayed by 2*(delta_i + 5 ms); no bandwidth limit, loss or queue"
                                if cfg.delay_model == "spec_rtt" else "none",
        },
        "config": {**asdict(cfg), "client_counts": list(cfg.client_counts), "blocks": cfg.blocks,
                   "d_model": D_MODEL, "data_payload_bytes": 2 * cfg.block_length,
                   "spec_payload": cfg.block_length == L_SPEC,
                   "deadline_ns_by_n": {str(n): cfg.deadline_ns(n) for n in cfg.client_counts}},
        "measurement": {
            "warmup": f"rounds 1..{cfg.warmup_rounds} excluded",
            "measurement_window": "start of first measured round to last measured deadline + drain",
            "cpu": "process-level; 100 * CPU-time delta / monotonic wall delta; percent of one logical CPU",
            "rss": "process RSS (not VMS); baseline = median between gateway bind and round 1 start",
            "sample_interval_s": cfg.sample_interval_s,
            "durations_clock": "time.perf_counter_ns", "tick_grid_clock": "time.monotonic_ns",
            "percentile_min_samples": {name: minimum for name, _, minimum in PERCENTILES},
        },
        "environment": environment_metadata(),
    }


def run_profile(cfg: ProfileConfig, *, now: datetime | None = None) -> Path:
    created = now or datetime.now(timezone.utc)
    git = git_metadata()
    run_dir = create_run_dir(cfg.output_root, make_run_id(created, git["revision"], secrets.token_hex(3)))
    manifest = build_manifest(cfg, run_dir.name, created)
    manifest["status"] = "running"
    _write_json(run_dir / "manifest.json", manifest)
    results = []
    for n in cfg.client_counts:
        print(f"[profile] N={n} ...", flush=True)
        results.append(run_one(cfg, n, run_dir))
        print(f"[profile] N={n}: {results[-1]['status']}", flush=True)
    write_summaries(run_dir, results)
    manifest["status"] = "complete"
    manifest["results"] = {str(r["num_clients"]): {"status": r["status"], "issues": r.get("issues", [])}
                           for r in results}
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="profile the gateway for each client count")
    defaults = ProfileConfig()
    run.add_argument("--clients", type=int, nargs="+", default=list(DEFAULT_CLIENT_COUNTS))
    run.add_argument("--rounds", type=int, default=defaults.rounds)
    run.add_argument("--warmup-rounds", type=int, default=defaults.warmup_rounds)
    run.add_argument("--policy", default=defaults.policy, help="GatewayCore grant policy (B4, B6, FG-GRU)")
    run.add_argument("--seed", type=int, default=defaults.seed)
    run.add_argument("--beta", type=float, default=defaults.beta, help="D = beta * T_full(N)")
    run.add_argument("--compute-window-s", type=float, default=defaults.compute_window_s, help="Tc")
    run.add_argument("--late-drain-s", type=float, default=defaults.late_drain_s)
    run.add_argument("--tick-interval-ns", type=int, default=defaults.tick_interval_ns)
    run.add_argument("--sample-interval-s", type=float, default=defaults.sample_interval_s)
    run.add_argument("--delay-model", choices=DELAY_MODELS, default=defaults.delay_model)
    run.add_argument("--block-length", type=int, default=defaults.block_length,
                     help="L (FP16 coordinates per DATA; payload 2L bytes). Non-default departs from the spec")
    run.add_argument("--models-dir", default=defaults.models_dir)
    run.add_argument("--output-root", default=defaults.output_root)
    for role in ("gateway", "clients"):
        sub.add_parser(role, help=f"internal: {role} child process").add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command in ("gateway", "clients"):
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        asyncio.run(_gateway_main(cfg) if args.command == "gateway" else _clients_main(cfg))
        return
    cfg = ProfileConfig(
        client_counts=tuple(args.clients), rounds=args.rounds, warmup_rounds=args.warmup_rounds,
        policy=args.policy, seed=args.seed, beta=args.beta, compute_window_s=args.compute_window_s,
        late_drain_s=args.late_drain_s, tick_interval_ns=args.tick_interval_ns,
        sample_interval_s=args.sample_interval_s, delay_model=args.delay_model, block_length=args.block_length,
        models_dir=args.models_dir, output_root=args.output_root,
    )
    run_dir = run_profile(cfg)
    print(f"[profile] results in {run_dir}")
    print((run_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
