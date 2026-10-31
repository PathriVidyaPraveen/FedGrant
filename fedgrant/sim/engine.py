"""Discrete-event simulation engine for FedGrant (nanosecond precision)."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


class EventType(Enum):
    ROUND_START = auto()
    READY_ARRIVE = auto()
    GRANT_ARRIVE_CLIENT = auto()
    ACCESS_ENQUEUE = auto()
    ACCESS_DEPART = auto()
    BN_ENQUEUE = auto()
    BN_DEPART = auto()
    GW_ARRIVE = auto()
    CBR_EMIT = auto()
    TICK_1MS = auto()
    DEADLINE = auto()


@dataclass(order=True)
class Event:
    time_ns: int
    seq: int
    event_type: EventType = field(compare=False)
    callback: Callable[[Event], None] = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)


class SimulationEngine:
    """Nanosecond-precision discrete-event simulator using a priority queue (min-heap)."""

    def __init__(self) -> None:
        self.now_ns: int = 0
        self._queue: list[Event] = []
        self._seq: int = 0

    def schedule(
        self,
        delay_ns: int,
        event_type: EventType,
        callback: Callable[[Event], None],
        payload: dict[str, Any] | None = None,
    ) -> Event:
        """Schedule a callback event `delay_ns` nanoseconds from the current simulation time."""
        if delay_ns < 0:
            raise ValueError(f"delay_ns cannot be negative, got {delay_ns}")
        time_ns = self.now_ns + delay_ns
        event = Event(
            time_ns=time_ns,
            seq=self._seq,
            event_type=event_type,
            callback=callback,
            payload=payload or {},
        )
        self._seq += 1
        heapq.heappush(self._queue, event)
        return event

    def run_until(self, until_ns: int) -> None:
        """Process all queued events up to `until_ns` nanoseconds."""
        while self._queue and self._queue[0].time_ns <= until_ns:
            event = heapq.heappop(self._queue)
            self.now_ns = event.time_ns
            event.callback(event)
        self.now_ns = max(self.now_ns, until_ns)

    def run_all(self) -> None:
        """Process queued events until the event queue is empty."""
        while self._queue:
            event = heapq.heappop(self._queue)
            self.now_ns = event.time_ns
            event.callback(event)

    def clear(self) -> None:
        """Reset the simulation clock and clear pending events."""
        self.now_ns = 0
        self._queue.clear()
        self._seq = 0


__all__ = ["EventType", "Event", "SimulationEngine"]
