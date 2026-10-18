"""Tests for the Section 6.1 interfaces: predictors/base.py and policy/types.py."""

from __future__ import annotations

import numpy as np
import pytest

from fedgrant.policy.types import CandidateView, PolicyConfig, Selector
from fedgrant.predictors.base import PredInput, Predictor, RoundCtx

# --------------------------------------------------------------------------- RoundCtx


def test_roundctx_ok_and_frozen() -> None:
    r = RoundCtx(round_id=1, D_ns=398_000_000, N=20, K=169)
    assert (r.round_id, r.D_ns, r.N, r.K) == (1, 398_000_000, 20, 169)
    with pytest.raises(AttributeError):
        r.N = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    "kw",
    [
        dict(round_id=0, D_ns=1, N=1, K=1),
        dict(round_id=1, D_ns=0, N=1, K=1),
        dict(round_id=1, D_ns=1, N=0, K=1),
        dict(round_id=1, D_ns=1, N=1, K=0),
    ],
)
def test_roundctx_rejects_bad_values(kw: dict) -> None:
    with pytest.raises(ValueError):
        RoundCtx(**kw)


def test_roundctx_rejects_float_time() -> None:
    with pytest.raises(TypeError):
        RoundCtx(round_id=1, D_ns=0.398, N=1, K=1)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- PredInput


def _x48(n: int) -> np.ndarray:
    return np.arange(n * 48, dtype=np.float32).reshape(n, 48)


def test_predinput_from_x48_ctx_is_view() -> None:
    x = _x48(3)
    p = PredInput.from_x48(np.array([0, 4, 7]), x, t_ns=5_000_000)
    assert p.n == 3 and p.client_ids.dtype == np.int32 and p.t_ns == 5_000_000
    np.testing.assert_array_equal(p.ctx, x[:, 32:48])
    assert np.shares_memory(p.ctx, p.x48)


def test_predinput_empty_batch() -> None:
    p = PredInput.from_x48(np.zeros(0, np.int32), np.zeros((0, 48), np.float32), 0)
    assert p.n == 0 and p.ctx.shape == (0, 16)


def test_predinput_rejects_mismatched_ctx() -> None:
    x = _x48(2)
    bad = x[:, 32:48].copy()
    bad[0, 0] += 1
    with pytest.raises(ValueError):
        PredInput(np.array([0, 1], np.int32), x, bad, 0)


@pytest.mark.parametrize(
    "ids,x,ctx",
    [
        (np.array([0, 1], np.int64), _x48(2), _x48(2)[:, 32:48]),                 # id dtype
        (np.array([0, 1], np.int32), _x48(2).astype(np.float64), _x48(2)[:, 32:48]),  # x dtype
        (np.array([0, 1], np.int32), _x48(3), _x48(3)[:, 32:48]),                 # n mismatch
        (np.array([0, 1], np.int32), _x48(2), _x48(2)[:, 31:47]),                 # wrong slice
    ],
)
def test_predinput_rejects_bad_arrays(ids: np.ndarray, x: np.ndarray, ctx: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        PredInput(ids, x, ctx, 0)


def test_predinput_rejects_float_time() -> None:
    x = _x48(1)
    with pytest.raises(TypeError):
        PredInput(np.array([0], np.int32), x, x[:, 32:48], 1.5)  # type: ignore[arg-type]


def test_predictor_protocol_runtime_check() -> None:
    class Const:
        def reset_round(self, ctx: RoundCtx) -> None: ...
        def observe(self, client: int, y: int, t_ns: int, obs: np.ndarray) -> None: ...
        def predict(self, inp: PredInput) -> np.ndarray:
            return np.full(inp.n, 0.9, np.float32)

    assert isinstance(Const(), Predictor)
    assert not isinstance(object(), Predictor)


# --------------------------------------------------------------------------- CandidateView / Selector


def test_candidate_view_ok() -> None:
    cv = CandidateView(
        client_id=np.array([1, 2], np.int32),
        next_value=np.array([np.inf, 0.5]),
        undelivered=np.array([10, 3], np.int32),
        rr_pointer=np.int64(4),
    )
    assert len(cv) == 2 and cv.rr_pointer == 4 and isinstance(cv.rr_pointer, int)


def test_candidate_view_rejects_bad_dtypes() -> None:
    with pytest.raises(TypeError):
        CandidateView(np.array([1], np.int32), np.array([0.5], np.float32), np.array([1], np.int32), 0)
    with pytest.raises(TypeError):
        CandidateView(np.array([1], np.int32), np.array([0.5]), np.array([1, 2], np.int32), 0)


def test_selector_protocol_runtime_check() -> None:
    class First:
        def select(self, cand: CandidateView) -> int:
            return int(cand.client_id[0])

    assert isinstance(First(), Selector)


# --------------------------------------------------------------------------- PolicyConfig


def test_policy_config_defaults_section_6_1() -> None:
    p = PolicyConfig(selector="value", predictor="gru")
    assert (p.pred_in_pacing, p.pred_in_eligibility, p.retransmit, p.explore_eps) == (True, True, True, 0.0)


def test_policy_config_from_dict() -> None:
    p = PolicyConfig.from_dict(
        {"selector": "rr", "predictor": "ewma", "retransmit": False, "explore_eps": 0.1}
    )
    assert (p.selector, p.predictor, p.retransmit, p.explore_eps) == ("rr", "ewma", False, 0.1)
    assert PolicyConfig.from_dict({"selector": "value", "predictor": "hmm", "explore_eps": 0}).explore_eps == 0.0


@pytest.mark.parametrize(
    "d",
    [
        {"selector": "greedy", "predictor": "gru"},
        {"selector": "value", "predictor": "lstm"},
        {"selector": "value", "predictor": "gru", "explore_eps": 1.5},
        {"selector": "value", "predictor": "gru", "explore_eps": -0.1},
        {"selector": "value", "predictor": "gru", "explore_eps": float("nan")},
        {"selector": "value", "predictor": "gru", "unknown_key": 1},
        {"selector": "value"},
    ],
)
def test_policy_config_rejects_bad_values(d: dict) -> None:
    with pytest.raises(ValueError):
        PolicyConfig.from_dict(d)


@pytest.mark.parametrize(
    "d",
    [
        {"selector": "value", "predictor": "gru", "retransmit": 1},
        {"selector": "value", "predictor": "gru", "pred_in_pacing": "yes"},
        {"selector": "value", "predictor": "gru", "explore_eps": True},
        {"selector": "value", "predictor": "gru", "explore_eps": "0.1"},
    ],
)
def test_policy_config_rejects_bad_types(d: dict) -> None:
    with pytest.raises(TypeError):
        PolicyConfig.from_dict(d)


def test_policy_config_direct_construction_validates() -> None:
    with pytest.raises(ValueError):
        PolicyConfig(selector="VALUE", predictor="gru")