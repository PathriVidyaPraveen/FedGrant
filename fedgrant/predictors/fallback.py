"""Predictor fallback (Section 9, Fallback).

If Predictor.predict raises, returns a non-finite value, or takes more than 2 ms
for one call, the gateway switches to P-EWMA for the rest of the round, logs the
event, and restores the configured predictor at the next round.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fedgrant.predictors.base import PredInput, Predictor, RoundCtx
from fedgrant.predictors.ewma import PEWMA

log = logging.getLogger(__name__)

TIME_LIMIT_NS: int = 2_000_000          # 2 ms per predict call


@dataclass(frozen=True, slots=True)
class FallbackEvent:
    """One switch to P-EWMA, for the run log."""

    round_id: int | None
    t_ns: int
    reason: str            # "exception" | "timeout" | "non_finite" | "bad_output" | "observe_exception" | "reset_exception"
    detail: str


class FallbackPredictor:
    """Wraps the configured predictor; falls back to a warm P-EWMA (implements Predictor).

    observe() is always forwarded to BOTH the primary and the EWMA, so the EWMA
    is warm when needed and the primary keeps its state while it is bypassed.

    SPEC-NOTE: besides the three triggers in the spec, an output of the wrong
    shape or with finite values outside [0, 1] also triggers the fallback: the
    Predictor contract requires [0, 1], and the pacing code rejects p > 1.
    An exception in primary.observe or primary.reset_round also switches to
    P-EWMA for the round instead of crashing the gateway.

    The 2 ms limit uses wall-clock time (perf_counter_ns), so whether it fires
    is not deterministic across machines; it is meant as a safety net.
    """

    def __init__(
        self,
        primary: Predictor,
        ewma: PEWMA | None = None,
        time_limit_ns: int = TIME_LIMIT_NS,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.primary = primary
        self.ewma = ewma if ewma is not None else PEWMA()
        self.time_limit_ns = int(time_limit_ns)
        self._clock = clock
        self.in_fallback: bool = False
        self.n_fallbacks: int = 0
        self.last_reason: str | None = None
        self.events: list[FallbackEvent] = []
        self._round_id: int | None = None

    @property
    def active(self) -> str:
        """'ewma' while falling back, else 'primary'."""
        return "ewma" if self.in_fallback else "primary"

    def _trip(self, reason: str, t_ns: int, detail: str = "") -> None:
        if self.in_fallback:
            return
        self.in_fallback = True
        self.n_fallbacks += 1
        self.last_reason = reason
        ev = FallbackEvent(self._round_id, int(t_ns), reason, detail)
        self.events.append(ev)
        log.warning("predictor fallback to P-EWMA: round=%s t_ns=%d reason=%s %s",
                    ev.round_id, ev.t_ns, reason, detail)

    def reset_round(self, ctx: RoundCtx) -> None:
        """Restore the primary for the new round (Section 9) and forward the reset."""
        self._round_id = ctx.round_id
        self.in_fallback = False
        self.ewma.reset_round(ctx)
        try:
            self.primary.reset_round(ctx)
        except Exception as e:  # noqa: BLE001 - any primary failure must not stop the gateway
            self._trip("reset_exception", 0, repr(e))

    def observe(self, client: int, y: int, t_ns: int, obs: np.ndarray) -> None:
        """Forward to both predictors."""
        self.ewma.observe(client, y, t_ns, obs)
        try:
            self.primary.observe(client, y, t_ns, obs)
        except Exception as e:  # noqa: BLE001
            self._trip("observe_exception", t_ns, repr(e))

    def predict(self, inp: PredInput) -> np.ndarray:
        """Primary output if it is valid and took <= 2 ms; otherwise P-EWMA, for the rest of the round."""
        if not self.in_fallback:
            t0 = self._clock()
            try:
                out = self.primary.predict(inp)
            except Exception as e:  # noqa: BLE001
                self._trip("exception", inp.t_ns, repr(e))
            else:
                elapsed = self._clock() - t0
                try:
                    arr = np.asarray(out, dtype=np.float64)
                except (TypeError, ValueError) as e:     # e.g. strings, None, ragged lists
                    arr = None
                    self._trip("bad_output", inp.t_ns, f"not numeric: {e!r}")
                if arr is None:
                    pass
                elif arr.shape != (inp.n,):
                    self._trip("bad_output", inp.t_ns, f"shape {arr.shape}, expected ({inp.n},)")
                elif not np.all(np.isfinite(arr)):
                    self._trip("non_finite", inp.t_ns, "")
                elif inp.n and (arr.min() < 0.0 or arr.max() > 1.0):
                    self._trip("bad_output", inp.t_ns, f"values outside [0, 1]: [{arr.min()}, {arr.max()}]")
                elif elapsed > self.time_limit_ns:
                    self._trip("timeout", inp.t_ns, f"{elapsed} ns")
                else:
                    return arr.astype(np.float32)
        return self.ewma.predict(inp)