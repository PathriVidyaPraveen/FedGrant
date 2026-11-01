"""Monotonic 100 ms rate replay for client TBF qdiscs."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence


class RateReplay:
    """Apply a sequence of bps rates at fixed monotonic deadlines in a thread."""

    def __init__(self, rates_bps: Sequence[float], apply_rate: Callable[[int], None], *,
                 interval_s: float = 0.1, repeat: bool = False,
                 clock_ns: Callable[[], int] = time.monotonic_ns):
        self.rates = tuple(float(rate) for rate in rates_bps)
        if not self.rates or any(not (rate > 0) for rate in self.rates):
            raise ValueError("rates_bps must be a non-empty sequence of positive values")
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.apply_rate = apply_rate
        self.repeat = bool(repeat)
        self.interval_ns = int(interval_s * 1e9)
        self.clock_ns = clock_ns
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
                i += 1
        except BaseException as exc:
            self.error = exc
            self._stop.set()

    def stop(self, timeout: float | None = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("rate replay thread did not stop")
        if self.error is not None:
            raise RuntimeError("rate replay failed") from self.error


__all__ = ["RateReplay"]
