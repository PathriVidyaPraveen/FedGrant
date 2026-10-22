"""Tests for predictors/fallback.py (Section 9, Fallback)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from fedgrant.predictors.base import PredInput, Predictor, RoundCtx
from fedgrant.predictors.ewma import PEWMA
from fedgrant.predictors.fallback import FallbackPredictor

Z = np.zeros(8, np.float32)


def _inp(ids: list[int], t: int = 0) -> PredInput:
    return PredInput.from_x48(np.array(ids, np.int32), np.zeros((len(ids), 48), np.float32), t)


def _ctx(r: int) -> RoundCtx:
    return RoundCtx(r, 398_000_000, 20, 169)


class Fake:
    """Primary predictor whose behaviour per call is scripted."""

    def __init__(self, mode: str = "ok", value: float = 0.7) -> None:
        self.mode, self.value = mode, value
        self.calls = self.observed = self.resets = 0

    def reset_round(self, ctx: RoundCtx) -> None:
        self.resets += 1

    def observe(self, client: int, y: int, t_ns: int, obs: np.ndarray) -> None:
        self.observed += 1

    def predict(self, inp: PredInput) -> np.ndarray:
        self.calls += 1
        if self.mode == "raise":
            raise RuntimeError("boom")
        if self.mode == "nan":
            out = np.full(inp.n, self.value, np.float32)
            out[-1] = np.nan
            return out
        if self.mode == "inf":
            return np.full(inp.n, np.inf, np.float32)
        if self.mode == "neg_inf":
            return np.full(inp.n, -np.inf, np.float32)
        if self.mode == "shape":
            return np.full(inp.n + 1, self.value, np.float32)
        if self.mode == "range":
            return np.full(inp.n, 1.5, np.float32)
        if self.mode == "strings":
            return np.array(["a"] * inp.n)
        if self.mode == "none":
            return None  # type: ignore[return-value]
        if self.mode == "float64":
            return np.full(inp.n, self.value, np.float64)
        return np.full(inp.n, self.value, np.float32)


def _warm_ewma() -> PEWMA:
    e = PEWMA()
    e.observe(1, 0, 0, Z)                    # client 1 -> 0.81, client 2 -> 0.9
    return e


def test_ok_path_returns_primary() -> None:
    fp = FallbackPredictor(Fake("ok"), _warm_ewma())
    assert isinstance(fp, Predictor)
    fp.reset_round(_ctx(1))
    out = fp.predict(_inp([1, 2]))
    assert out.dtype == np.float32 and np.all(out == np.float32(0.7))
    assert fp.n_fallbacks == 0 and fp.last_reason is None and fp.active == "primary"


@pytest.mark.parametrize(
    "mode,reason", [("raise", "exception"), ("nan", "non_finite"), ("inf", "non_finite"),
                    ("neg_inf", "non_finite"),
                    ("shape", "bad_output"), ("range", "bad_output"), ("strings", "bad_output"),
                    ("none", "bad_output")],
)
def test_bad_primary_triggers_fallback(mode: str, reason: str) -> None:
    prim = Fake(mode)
    fp = FallbackPredictor(prim, _warm_ewma())
    fp.reset_round(_ctx(3))
    out = fp.predict(_inp([1, 2], t=5_000_000))
    np.testing.assert_array_equal(out, np.float32([0.81, 0.9]))      # EWMA output for this call
    assert fp.in_fallback and fp.n_fallbacks == 1 and fp.last_reason == reason
    ev = fp.events[-1]
    assert (ev.round_id, ev.t_ns, ev.reason) == (3, 5_000_000, reason)


def test_slow_call_over_2ms_triggers_fallback_with_fake_clock() -> None:
    ticks = iter([10, 2_000_011])
    fp = FallbackPredictor(Fake("ok"), _warm_ewma(), clock=lambda: next(ticks))
    fp.reset_round(_ctx(1))
    out = fp.predict(_inp([1]))
    assert fp.last_reason == "timeout" and out[0] == np.float32(0.81)


def test_time_limit_boundary_with_fake_clock() -> None:
    ticks = iter([0, 2_000_000, 10, 10 + 2_000_001])
    fp = FallbackPredictor(Fake("ok"), _warm_ewma(), clock=lambda: next(ticks))
    fp.reset_round(_ctx(1))
    assert fp.predict(_inp([1]))[0] == np.float32(0.7)              # exactly 2 ms: allowed
    assert fp.predict(_inp([1]))[0] == np.float32(0.81)             # 2 ms + 1 ns: fallback
    assert fp.last_reason == "timeout"


def test_stays_on_ewma_for_rest_of_round_then_restored() -> None:
    prim = Fake("raise")
    fp = FallbackPredictor(prim, _warm_ewma())
    fp.reset_round(_ctx(1))
    fp.predict(_inp([1]))
    prim.mode = "ok"                                                  # primary healthy again ...
    for _ in range(3):
        assert fp.predict(_inp([1]))[0] == np.float32(0.81)          # ... but not used this round
    assert prim.calls == 1 and fp.n_fallbacks == 1
    fp.reset_round(_ctx(2))                                           # next round: restored
    assert fp.active == "primary" and prim.resets == 2
    assert fp.predict(_inp([1]))[0] == np.float32(0.7)
    assert fp.last_reason == "exception"                              # last event kept for logs


def test_observe_goes_to_both_even_in_fallback() -> None:
    prim = Fake("raise")
    e = PEWMA()
    fp = FallbackPredictor(prim, e)
    fp.reset_round(_ctx(1))
    fp.observe(4, 0, 1, Z)
    fp.predict(_inp([4]))
    fp.observe(4, 0, 2, Z)
    assert prim.observed == 2 and e.value(4) == pytest.approx(0.9 * 0.9 * 0.9)


def test_primary_observe_exception_falls_back_without_crashing() -> None:
    class BadObserve(Fake):
        def observe(self, client: int, y: int, t_ns: int, obs: np.ndarray) -> None:
            raise ValueError("bad")

    fp = FallbackPredictor(BadObserve(), _warm_ewma())
    fp.reset_round(_ctx(1))
    fp.observe(1, 1, 7, Z)
    assert fp.in_fallback and fp.last_reason == "observe_exception"
    assert fp.predict(_inp([1]))[0] == np.float32(0.9 * 0.81 + 0.1)


def test_second_failure_in_same_round_not_counted_twice() -> None:
    class BadBoth(Fake):
        def observe(self, client: int, y: int, t_ns: int, obs: np.ndarray) -> None:
            raise ValueError("bad")

    fp = FallbackPredictor(BadBoth("raise"), _warm_ewma())
    fp.reset_round(_ctx(1))
    fp.predict(_inp([1]))                                             # trips: exception
    fp.observe(1, 1, 5, Z)                                            # fails again: ignored
    assert fp.n_fallbacks == 1 and fp.last_reason == "exception" and len(fp.events) == 1


def test_one_event_per_round_and_logging(caplog: pytest.LogCaptureFixture) -> None:
    fp = FallbackPredictor(Fake("nan"), _warm_ewma())
    with caplog.at_level(logging.WARNING, logger="fedgrant.predictors.fallback"):
        for rnd in (1, 2):
            fp.reset_round(_ctx(rnd))
            fp.predict(_inp([1]))
            fp.predict(_inp([1]))
    assert fp.n_fallbacks == 2 and [e.round_id for e in fp.events] == [1, 2]
    assert sum("predictor fallback" in r.message for r in caplog.records) == 2


def test_empty_request() -> None:
    fp = FallbackPredictor(Fake("ok"), _warm_ewma())
    fp.reset_round(_ctx(1))
    assert fp.predict(_inp([])).shape == (0,) and not fp.in_fallback


def test_float64_output_is_accepted_and_returned_as_float32() -> None:
    fp = FallbackPredictor(Fake("float64"), _warm_ewma())
    fp.reset_round(_ctx(1))
    out = fp.predict(_inp([1, 2]))
    assert out.dtype == np.float32 and np.all(out == np.float32(0.7)) and not fp.in_fallback
