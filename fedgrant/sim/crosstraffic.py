"""Background UDP cross-traffic generator for shared bottleneck link (Sections 5.4, 6)."""

from __future__ import annotations

import numpy as np

from fedgrant.policy.constants import S
from fedgrant.sim.engine import EventType, SimulationEngine
from fedgrant.sim.links import BottleneckLink, Frame


class CrossTrafficGenerator:
    """Generates Constant Bit Rate (CBR) background UDP cross-traffic into the bottleneck queue.

    Specification (Scenario X2):
        Rate: 3 Mbit/s CBR UDP.
        Frame size: S = 1474 bytes.
        Emission interval: 1474 * 8 / (3 * 10^6) s (~3.93067 ms).
        Initial phase: drawn from cross-traffic random stream.
    """

    def __init__(
        self,
        engine: SimulationEngine,
        bottleneck: BottleneckLink,
        rate_bps: float = 3_000_000.0,  # 3 Mbit/s
        frame_size_bytes: int = S,       # S = 1474 bytes
        rng: np.random.Generator | None = None,
        phase_ns: int | None = None,
    ) -> None:
        self.engine = engine
        self.bottleneck = bottleneck
        self.rate_bps = max(0.0, float(rate_bps))
        self.frame_size_bytes = int(frame_size_bytes)
        self.seq: int = 0
        self.is_running: bool = False
        self.rng = rng or np.random.default_rng()

        if self.rate_bps > 0:
            self.interval_ns = int(round((self.frame_size_bytes * 8 * 1_000_000_000) / self.rate_bps))
        else:
            self.interval_ns = 0

        if phase_ns is not None:
            self.phase_ns = int(phase_ns)
        elif self.interval_ns > 0 and rng is not None:
            self.phase_ns = int(self.rng.integers(0, self.interval_ns))
        else:
            self.phase_ns = 0

    def start(self) -> None:
        """Start injecting background UDP cross-traffic."""
        if self.rate_bps <= 0 or self.interval_ns <= 0:
            return
        self.is_running = True
        # Schedule first emit after the initial phase delay
        self.engine.schedule(
            delay_ns=self.phase_ns,
            event_type=EventType.CBR_EMIT,
            callback=self._on_emit,
        )

    def stop(self) -> None:
        """Stop background cross-traffic generation."""
        self.is_running = False

    def _schedule_next_emit(self) -> None:
        if not self.is_running or self.interval_ns <= 0:
            return

        self.engine.schedule(
            delay_ns=self.interval_ns,
            event_type=EventType.CBR_EMIT,
            callback=self._on_emit,
        )

    def _on_emit(self, _event) -> None:
        if not self.is_running:
            return

        frame = Frame(
            frame_id=f"crosstraffic_{self.seq}",
            client_id=-1,  # Marker for background cross-traffic
            size_bytes=self.frame_size_bytes,
            created_at_ns=self.engine.now_ns,
            payload={"type": "crosstraffic"},
        )
        self.seq += 1
        self.bottleneck.enqueue(frame)

        self._schedule_next_emit()


__all__ = ["CrossTrafficGenerator"]
