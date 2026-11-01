"""Cellular uplink rate trace reader and dynamic rate manager (Section 5.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from fedgrant.sim.links import mean_access_rate_bps
from fedgrant.traces.client_trace import POOL_ID, POOL_OOD

ID_TRACE_NAMES = POOL_ID
OOD_TRACE_NAMES = POOL_OOD


class TraceSegment:
    """Preprocessed cellular trace segment with (t_ms, rate_bps) columns.

    Each row represents a 100 ms bin.
    """

    def __init__(self, t_ms: np.ndarray, rate_bps: np.ndarray) -> None:
        if len(t_ms) == 0 or len(t_ms) != len(rate_bps):
            raise ValueError("t_ms and rate_bps must be non-empty and of equal length")
        self.t_ms = np.ascontiguousarray(t_ms, dtype=np.int64)
        self.rate_bps = np.ascontiguousarray(rate_bps, dtype=np.float64)
        if not np.isfinite(self.rate_bps).all() or np.any(self.rate_bps < 0):
            raise ValueError("rate_bps values must be finite and non-negative")
        self.bin_ms = int(self.t_ms[1] - self.t_ms[0]) if len(self.t_ms) > 1 else 100
        if self.bin_ms <= 0 or (len(self.t_ms) > 1 and np.any(np.diff(self.t_ms) != self.bin_ms)):
            raise ValueError("t_ms must be strictly increasing with a constant bin width")
        self.duration_ms = int(len(self.t_ms) * self.bin_ms)
        self.mean_rate_bps = float(np.mean(self.rate_bps))
        if self.mean_rate_bps <= 0:
            self.mean_rate_bps = 1_000_000.0

    @classmethod
    def load_csv(cls, path: str | Path) -> TraceSegment:
        """Load a CSV with columns (t_ms, rate_bps)."""
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        if data.dtype.names is None or not {"t_ms", "rate_bps"}.issubset(data.dtype.names):
            raise ValueError(f"{path}: expected t_ms,rate_bps columns")
        if data.ndim == 0:
            data = np.asarray([data])
        return cls(data["t_ms"], data["rate_bps"])

    def rate_at(self, offset_ms: int) -> float:
        """Get rate for an offset wrapped inside the segment duration."""
        wrapped_ms = int(offset_ms) % self.duration_ms
        idx = min(int(wrapped_ms // self.bin_ms), len(self.rate_bps) - 1)
        return float(self.rate_bps[idx])


class ClientTrace:
    """Manages rate queries for a single client with trace wrapping and scaling.

    Specification (Section 5.3):
        (3) Client i starts at an offset drawn uniformly inside the split segment
            and wrapping within the segment.
        (4) Scale by b_bar_i / mean(segment).
        (5) Floor at 0.1 Mbit/s.
    """

    def __init__(
        self,
        client_id: int,
        segment: TraceSegment | None = None,
        offset_ms: int = 0,
        floor_bps: float = 100_000.0,
    ) -> None:
        self.client_id = client_id
        self.target_mean_bps = mean_access_rate_bps(client_id)
        self.floor_bps = floor_bps
        self.segment = segment
        self.offset_ms = int(offset_ms)

        if segment is not None and segment.mean_rate_bps > 0:
            self.scale = self.target_mean_bps / segment.mean_rate_bps
        else:
            self.scale = 1.0

    def get_rate_bps(self, time_ns: int) -> float:
        """Query rate b_i(t) in bps at time_ns."""
        if self.segment is None:
            return self.target_mean_bps

        time_ms = int(time_ns // 1_000_000)
        raw_rate = self.segment.rate_at(self.offset_ms + time_ms)
        scaled_rate = raw_rate * self.scale
        return max(self.floor_bps, scaled_rate)


class TraceManager:
    """Manages trace assignments and offsets across all N clients.

    Specification (Sections 5.3, 6):
        - ID pool (L4, train, val, test): Verizon-LTE-driving, Verizon-LTE-short, ATT-LTE-driving.
        - OOD pool (L5): TMobile-UMTS-driving, Verizon-EVDO-driving.
        - Client i uses trace (i mod pool_size).
        - Offsets are drawn using the trace random stream.
    """

    def __init__(
        self,
        num_clients: int = 20,
        scenario: str = "L2",
        split: str = "test",
        trace_dir: str | Path = "data/traces",
        rng: np.random.Generator | None = None,
    ) -> None:
        self.num_clients = num_clients
        self.scenario = scenario.upper()
        self.split = split
        self.trace_dir = Path(trace_dir)
        self.rng = rng or np.random.default_rng()

        self.client_traces: list[ClientTrace] = []
        self._setup_traces()

    def _setup_traces(self) -> None:
        # Constant capacity scenarios L1, L2, L3, X2 do not use trace files
        if self.scenario in ("L1", "L2", "L3", "X2"):
            for i in range(self.num_clients):
                self.client_traces.append(ClientTrace(client_id=i, segment=None))
            return

        if self.scenario not in {"L4", "L5"}:
            raise ValueError(f"unknown network scenario {self.scenario!r}")

        # Scenario L4 uses ID split segments; L5 uses OOD split traces. Every
        # trace in the selected pool is required so no client silently gets a
        # constant-rate fallback.
        pool_names = OOD_TRACE_NAMES if self.scenario == "L5" else ID_TRACE_NAMES
        trace_split = "ood" if self.scenario == "L5" else self.split
        paths = [self.trace_dir / f"{name}_{trace_split}.csv" for name in pool_names]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            missing_list = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"scenario {self.scenario} requires all {self.split!r} traces; missing: {missing_list}"
            )

        segments: list[TraceSegment] = []

        for path in paths:
            segments.append(TraceSegment.load_csv(path))

        # Draw a fixed-size offset vector once. The first N offsets therefore
        # remain identical when an experiment changes its client count.
        if self.num_clients > 50:
            raise ValueError("trace assignment supports at most 50 clients")
        fractions = self.rng.random(50)
        n_pool = len(pool_names)
        for i in range(self.num_clients):
            seg = segments[i % n_pool]
            offset = int(fractions[i] * len(seg.rate_bps)) * seg.bin_ms
            self.client_traces.append(
                ClientTrace(client_id=i, segment=seg, offset_ms=offset)
            )

    def get_client_rate(self, client_id: int, time_ns: int) -> float:
        """Query client uplink rate at time_ns."""
        if 0 <= client_id < len(self.client_traces):
            return self.client_traces[client_id].get_rate_bps(time_ns)
        return mean_access_rate_bps(client_id)


__all__ = [
    "TraceSegment",
    "ClientTrace",
    "TraceManager",
    "ID_TRACE_NAMES",
    "OOD_TRACE_NAMES",
]
