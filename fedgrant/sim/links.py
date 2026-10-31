"""Network access links, shared drop-tail bottleneck queue, and wire parameters (Sections 3.1, 3.5, 6)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from fedgrant.policy.constants import C, Q, S
from fedgrant.sim.engine import EventType, SimulationEngine
from fedgrant.sim.ge import GilbertElliottModel


def access_delay_ns(client_id: int) -> int:
    """One-way access delay delta_i: 10, 20, 30, 40 ms for i mod 4 = 0, 1, 2, 3 (Section 3.1)."""
    delays_ms = (10, 20, 30, 40)
    return delays_ms[client_id % 4] * 1_000_000


def mean_access_rate_bps(client_id: int) -> float:
    """Mean access rate b_bar_i: 2, 4, 6 Mbit/s for i mod 3 = 0, 1, 2 (Section 3.1)."""
    rates_mbps = (2.0, 4.0, 6.0)
    return rates_mbps[client_id % 3] * 1_000_000.0


@dataclass
class Frame:
    """Network frame payload representation (DATA wire size S = 1474 bytes, Section 3.1)."""

    frame_id: str
    client_id: int
    size_bytes: int = S
    created_at_ns: int = 0
    payload: dict[str, Any] = field(default_factory=dict)


class AccessLink:
    """Client access link with rate-limiting, drop-tail FIFO queue (32 frames), and GE loss.

    Specification (Sections 3.5, 6):
        - On ACCESS_ENQUEUE: advance the GE chain; drop if Bad; otherwise schedule arrival
          at the FIFO after delta_i.
        - The FIFO holds 32 frames, is drop-tail, and serializes at b_i(t).
        - Rate changes apply to the remaining bits of a frame in service.
    """

    def __init__(
        self,
        client_id: int,
        engine: SimulationEngine,
        prop_delay_ns: int | None = None,
        ge_model: GilbertElliottModel | None = None,
        max_queue_frames: int = 32,
        initial_rate_bps: float | None = None,
    ) -> None:
        self.client_id = client_id
        self.engine = engine
        self.prop_delay_ns = prop_delay_ns if prop_delay_ns is not None else access_delay_ns(client_id)
        self.ge_model = ge_model if ge_model is not None else GilbertElliottModel.for_scenario("L2")
        self.max_queue_frames = max_queue_frames
        self.rate_bps: float = initial_rate_bps if initial_rate_bps is not None else mean_access_rate_bps(client_id)

        self.queue: deque[Frame] = deque()
        self.current_frame: Frame | None = None
        self.bits_remaining: float = 0.0
        self.last_rate_update_ns: int = 0
        self.tx_gen: int = 0
        self.next_hop_cb: Callable[[Frame], None] | None = None

        self.drops_ge: int = 0
        self.drops_queue: int = 0
        self.frames_sent: int = 0

    @property
    def is_busy(self) -> bool:
        return self.current_frame is not None

    @property
    def backlog_bits(self) -> float:
        """Total bits queued in FIFO plus in-service remaining bits (used by P-ORACLE, Sec 4.6)."""
        queued = sum(f.size_bytes * 8 for f in self.queue)
        in_service = self._current_remaining_bits() if self.is_busy else 0.0
        return queued + in_service

    @property
    def backlog_bytes(self) -> int:
        """Total backlog in bytes (ceil of remaining bits)."""
        return int(math.ceil(self.backlog_bits / 8.0))

    def _current_remaining_bits(self) -> float:
        if not self.is_busy:
            return 0.0
        elapsed_ns = self.engine.now_ns - self.last_rate_update_ns
        bits_transmitted = (elapsed_ns * self.rate_bps) / 1_000_000_000.0
        return max(0.0, self.bits_remaining - bits_transmitted)

    def set_rate(self, new_rate_bps: float) -> None:
        """Update client access transmission rate b_i(t).

        Applies rate change to the remaining bits of a frame in service (Section 6).
        """
        new_rate = max(1.0, float(new_rate_bps))
        if self.is_busy:
            self.bits_remaining = self._current_remaining_bits()
            self.rate_bps = new_rate
            self.last_rate_update_ns = self.engine.now_ns

            # Invalidate current depart event and reschedule with new rate
            self.tx_gen += 1
            rem_ns = int(round((self.bits_remaining * 1_000_000_000.0) / self.rate_bps))
            gen = self.tx_gen
            self.engine.schedule(
                delay_ns=rem_ns,
                event_type=EventType.ACCESS_DEPART,
                callback=lambda _evt, g=gen: self._on_depart(g),
            )
        else:
            self.rate_bps = new_rate
            self.last_rate_update_ns = self.engine.now_ns

    def enqueue(self, frame: Frame) -> bool:
        """Present a frame to the access link.

        Advances GE chain. If Bad: drops frame immediately.
        If Good: schedules arrival at the access FIFO after delta_i.
        """
        if self.ge_model.should_drop():
            self.drops_ge += 1
            return False

        self.engine.schedule(
            delay_ns=self.prop_delay_ns,
            event_type=EventType.ACCESS_ENQUEUE,
            callback=lambda _evt, f=frame: self._on_fifo_arrival(f),
        )
        return True

    def _on_fifo_arrival(self, frame: Frame) -> None:
        """Arrival at the 32-frame access drop-tail FIFO after delta_i propagation."""
        if len(self.queue) >= self.max_queue_frames:
            self.drops_queue += 1
            return

        self.queue.append(frame)
        self._maybe_start_tx()

    def _maybe_start_tx(self) -> None:
        if self.is_busy or not self.queue:
            return

        self.current_frame = self.queue.popleft()
        self.bits_remaining = float(self.current_frame.size_bytes * 8)
        self.last_rate_update_ns = self.engine.now_ns

        self.tx_gen += 1
        gen = self.tx_gen
        tx_delay_ns = int(round((self.bits_remaining * 1_000_000_000.0) / self.rate_bps))

        self.engine.schedule(
            delay_ns=tx_delay_ns,
            event_type=EventType.ACCESS_DEPART,
            callback=lambda _evt, g=gen: self._on_depart(g),
        )

    def _on_depart(self, gen: int) -> None:
        if gen != self.tx_gen or not self.is_busy:
            return  # Superseded by rate change reschedule

        frame = self.current_frame
        self.current_frame = None
        self.bits_remaining = 0.0
        self.frames_sent += 1

        if self.next_hop_cb and frame is not None:
            self.next_hop_cb(frame)

        self._maybe_start_tx()


class BottleneckLink:
    """Shared bottleneck link with capacity C=10 Mbps, 5ms delay, and drop-tail FIFO Q=64.

    Specification (Sections 3.1, 3.5, 6):
        - After 5 ms delay from access depart: FIFO of Q = 64 frames (64 x 1474 bytes), drop-tail.
        - Rate C = 10^7 bit/s.
        - Cross traffic enters this FIFO.
    """

    def __init__(
        self,
        engine: SimulationEngine,
        capacity_bps: float = C,
        prop_delay_ns: int = 5_000_000,  # 5 ms
        max_queue_frames: int = Q,        # 64 frames
    ) -> None:
        self.engine = engine
        self.capacity_bps = float(capacity_bps)
        self.prop_delay_ns = int(prop_delay_ns)
        self.max_queue_frames = int(max_queue_frames)

        self.queue: deque[Frame] = deque()
        self.current_frame: Frame | None = None
        self.bits_remaining: float = 0.0
        self.last_tx_start_ns: int = 0
        self.dest_cb: Callable[[Frame], None] | None = None

        self.drops_queue_client: int = 0
        self.drops_queue_cross: int = 0
        self.frames_sent: int = 0

    @property
    def is_busy(self) -> bool:
        return self.current_frame is not None

    @property
    def drops_queue(self) -> int:
        """Total bottleneck drop-tail drops."""
        return self.drops_queue_client + self.drops_queue_cross

    @property
    def backlog_bits(self) -> float:
        """Total bits queued in bottleneck FIFO plus in-service remaining bits (Sec 4.6)."""
        queued = sum(f.size_bytes * 8 for f in self.queue)
        in_service = self._current_remaining_bits() if self.is_busy else 0.0
        return queued + in_service

    @property
    def backlog_bytes(self) -> int:
        """Total backlog in bytes."""
        return int(math.ceil(self.backlog_bits / 8.0))

    def _current_remaining_bits(self) -> float:
        if not self.is_busy:
            return 0.0
        elapsed_ns = self.engine.now_ns - self.last_tx_start_ns
        bits_transmitted = (elapsed_ns * self.capacity_bps) / 1_000_000_000.0
        return max(0.0, self.bits_remaining - bits_transmitted)

    def route_from_access(self, frame: Frame) -> None:
        """Frame departs access link; travels 5 ms to bottleneck FIFO (Section 6)."""
        self.engine.schedule(
            delay_ns=self.prop_delay_ns,
            event_type=EventType.BN_ENQUEUE,
            callback=lambda _evt, f=frame: self.enqueue(f),
        )

    def enqueue(self, frame: Frame) -> bool:
        """Enqueue frame directly into bottleneck FIFO (used by cross traffic or after 5ms delay)."""
        if len(self.queue) >= self.max_queue_frames:
            if frame.client_id >= 0:
                self.drops_queue_client += 1
            else:
                self.drops_queue_cross += 1
            return False

        self.queue.append(frame)
        self._maybe_start_tx()
        return True

    def _maybe_start_tx(self) -> None:
        if self.is_busy or not self.queue:
            return

        self.current_frame = self.queue.popleft()
        self.bits_remaining = float(self.current_frame.size_bytes * 8)
        self.last_tx_start_ns = self.engine.now_ns

        tx_delay_ns = int(round((self.bits_remaining * 1_000_000_000.0) / self.capacity_bps))

        self.engine.schedule(
            delay_ns=tx_delay_ns,
            event_type=EventType.BN_DEPART,
            callback=self._on_depart,
        )

    def _on_depart(self, _event: Any) -> None:
        if not self.is_busy:
            return

        frame = self.current_frame
        self.current_frame = None
        self.bits_remaining = 0.0
        self.frames_sent += 1

        if self.dest_cb and frame is not None:
            self.dest_cb(frame)

        self._maybe_start_tx()


__all__ = [
    "Frame",
    "AccessLink",
    "BottleneckLink",
    "access_delay_ns",
    "mean_access_rate_bps",
]
