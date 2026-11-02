"""Monotonic 100 ms rate replay for client TBF qdiscs."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from fedgrant.mininet.round_state import RoundStateFile


class RateReplay:
    """Apply a sequence of bps rates at fixed monotonic deadlines in a thread."""

    def __init__(self, rates_bps: Sequence[float], apply_rate: Callable[[int], None], *,
                 interval_s: float = 0.1, repeat: bool = False,
                 clock_ns: Callable[[], int] = time.monotonic_ns,
                 round_state_path: str | Path | None = None,
                 on_change: Callable[[int, int, int, int, int], None] | None = None):
        self.rates = tuple(float(rate) for rate in rates_bps)
        if not self.rates or any(not (rate > 0) for rate in self.rates):
            raise ValueError("rates_bps must be a non-empty sequence of positive values")
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.apply_rate = apply_rate
        self.repeat = bool(repeat)
        self.interval_ns = int(interval_s * 1e9)
        self.clock_ns = clock_ns
        self.round_state = RoundStateFile(round_state_path) if round_state_path is not None else None
        self.on_change = on_change
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def start(self, start_ns: int | None = None) -> "RateReplay":
        if self._thread is not None:
            raise RuntimeError("replay already started")
        origin = self.clock_ns() if start_ns is None else int(start_ns)
        self._thread = threading.Thread(target=self._run, args=(origin,), name="fedgrant-rate-replay", daemon=True)
        self._thread.start()
        return self

    def _run(self, origin_ns: int) -> None:
        try:
            if self.round_state is not None:
                self._run_round_scoped()
                return
            i = 0
            while self.repeat or i < len(self.rates):
                rate = self.rates[i % len(self.rates)]
                deadline = origin_ns + i * self.interval_ns
                while not self._stop.is_set():
                    remaining = deadline - self.clock_ns()
                    if remaining <= 0:
                        break
                    self._stop.wait(remaining / 1e9)
                if self._stop.is_set():
                    break
                self.apply_rate(int(round(rate)))
                if self.on_change is not None:
                    self.on_change(self.clock_ns(), 0, i, int(round(rate)), deadline)
                i += 1
        except BaseException as exc:
            self.error = exc
            self._stop.set()

    def _run_round_scoped(self) -> None:
        """Restart the trace offset at every gateway ROUND_START boundary."""
        last_round = 0
        while not self._stop.is_set():
            state = self.round_state.read()
            if not state.get("active") or int(state["round_id"]) == last_round:
                self._stop.wait(0.001)
                continue
            round_id = int(state["round_id"])
            start_ns, deadline_ns = int(state["start_ns"]), int(state["deadline_ns"])
            last_round = round_id
            index = 0
            while not self._stop.is_set():
                scheduled_ns = start_ns + index * self.interval_ns
                if scheduled_ns >= deadline_ns:
                    break
                while not self._stop.is_set():
                    now = self.clock_ns()
                    remaining = scheduled_ns - now
                    if remaining <= 0:
                        break
                    self._stop.wait(min(remaining / 1e9, 0.001))
                    current = self.round_state.read()
                    if not current.get("active") or int(current["round_id"]) != round_id:
                        break
                if self._stop.is_set():
                    return
                current = self.round_state.read()
                if not current.get("active") or int(current["round_id"]) != round_id:
                    break
                rate = int(round(self.rates[index % len(self.rates)]))
                self.apply_rate(rate)
                if self.on_change is not None:
                    self.on_change(self.clock_ns(), round_id, index, rate, scheduled_ns)
                index += 1
            # Do not replay the same round after its active flag is cleared.

    def stop(self, timeout: float | None = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("rate replay thread did not stop")
        if self.error is not None:
            raise RuntimeError("rate replay failed") from self.error


__all__ = ["RateReplay"]
