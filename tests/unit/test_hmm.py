"""Tests for predictors/ewma.py and predictors/hmm.py (Section 4.6)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from fedgrant.predictors.base import PredInput, Predictor, RoundCtx
from fedgrant.predictors.ewma import PEWMA
from fedgrant.predictors.hmm import (
    E_B,
    E_G,
    P0,
    R0,
    REFIT_EVERY,
    WINDOW,
    PHMM,
    baum_welch,
    forward_filter,
    stationary_g,
)


def _inp(ids: list[int]) -> PredInput:
    return PredInput.from_x48(np.array(ids, np.int32), np.zeros((len(ids), 48), np.float32), 0)


def _emit(y: int, s: int) -> float:
    pg = E_G if s == 0 else E_B
    return pg if y == 1 else 1.0 - pg


def gen_hmm(n: int, p: float, r: float, rng: np.random.Generator) -> np.ndarray:
    """Outcomes generated FROM the P-HMM model (emissions 0.98 / 0.02)."""
    u = rng.random((n, 2))
    s = 0 if rng.random() < r / (p + r) else 1
    y = np.empty(n, np.int8)
    for t in range(n):
        y[t] = u[t, 0] < (E_G if s == 0 else E_B)
        s = int(u[t, 1] < p) if s == 0 else int(u[t, 1] >= r)
    return y


def gen_ge(n: int, p: float, r: float, rng: np.random.Generator) -> np.ndarray:
    """Gilbert-Elliott outcomes as in the network model: loss prob 1 in B, 0 in G (Section 3.5)."""
    u = rng.random(n)
    s = 0 if rng.random() < r / (p + r) else 1
    y = np.empty(n, np.int8)
    for t in range(n):
        y[t] = 1 if s == 0 else 0
        s = int(u[t] < p) if s == 0 else int(u[t] >= r)
    return y


def ge_params(burst: int, pi: float) -> tuple[float, float]:
    """(p, r) from mean burst length 1/r and average loss pi = p / (p + r)."""
    r = 1.0 / burst
    return pi * r / (1.0 - pi), r


# --------------------------------------------------------------------------- P-EWMA


def test_pewma_update_and_persistence() -> None:
    e = PEWMA()
    assert isinstance(e, Predictor)
    np.testing.assert_array_equal(e.predict(_inp([3, 5])), np.float32([0.9, 0.9]))
    e.observe(3, 0, 10, np.zeros(8, np.float32))
    e.observe(3, 1, 20, np.zeros(8, np.float32))
    e.reset_round(RoundCtx(2, 398_000_000, 20, 169))         # persists across rounds
    assert e.value(3) == pytest.approx(0.9 * (0.9 * 0.9) + 0.1)
    out = e.predict(_inp([5, 3]))
    assert out.dtype == np.float32 and out[0] == np.float32(0.9)
    assert out[1] == np.float32(0.9 * 0.81 + 0.1)
    with pytest.raises(ValueError):
        e.observe(3, 2, 0, np.zeros(8, np.float32))
    assert e.predict(_inp([])).shape == (0,)


# --------------------------------------------------------------------------- forward filter


def test_forward_filter_matches_unscaled_computation() -> None:
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0], np.int8)
    p, r, g0 = 0.07, 0.3, 0.8
    A = np.array([[1 - p, p], [r, 1 - r]])
    alpha = np.array([g0, 1 - g0]) * [_emit(int(y[0]), 0), _emit(int(y[0]), 1)]
    exp_bel = [alpha[0] / alpha.sum()]
    for t in range(1, len(y)):
        alpha = (alpha @ A) * [_emit(int(y[t]), 0), _emit(int(y[t]), 1)]     # unscaled
        exp_bel.append(alpha[0] / alpha.sum())
    bel, ll = forward_filter(y, p, r, g0)
    np.testing.assert_allclose(bel, exp_bel, rtol=1e-12)
    assert ll == pytest.approx(np.log(alpha.sum()), rel=1e-12)


def test_forward_filter_long_sequence_no_underflow() -> None:
    y = gen_hmm(20000, 0.01, 0.1, np.random.default_rng(0))
    bel, ll = forward_filter(y, 0.01, 0.1, stationary_g(0.01, 0.1))
    assert np.all(np.isfinite(bel)) and np.isfinite(ll) and ll < -100
    assert np.all((bel > 0) & (bel < 1))


# --------------------------------------------------------------------------- Baum-Welch


def _brute_counts(y: np.ndarray, p: float, r: float, g0: float) -> tuple[float, float, float]:
    """Exact log-likelihood and one EM update of (p, r) by enumerating all state paths."""
    A = [[1 - p, p], [r, 1 - r]]
    tot, n = 0.0, np.zeros((2, 2))
    for path in itertools.product((0, 1), repeat=len(y)):
        w = (g0 if path[0] == 0 else 1 - g0) * _emit(int(y[0]), path[0])
        for t in range(1, len(y)):
            w *= A[path[t - 1]][path[t]] * _emit(int(y[t]), path[t])
        tot += w
        for t in range(len(y) - 1):
            n[path[t], path[t + 1]] += w
    return np.log(tot), n[0, 1] / n[0].sum(), n[1, 0] / n[1].sum()


@pytest.mark.parametrize("seed", range(4))
def test_baum_welch_step_matches_path_enumeration(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 10).astype(np.int8)
    p, r, g0 = float(rng.uniform(0.01, 0.4)), float(rng.uniform(0.05, 0.9)), float(rng.uniform(0.2, 0.9))
    ll, p_exp, r_exp = _brute_counts(y, p, r, g0)
    p1, r1, lls = baum_welch(y, p, r, g0, iters=1)
    assert p1 == pytest.approx(p_exp, rel=1e-12) and r1 == pytest.approx(r_exp, rel=1e-12)
    assert lls[0] == pytest.approx(ll, rel=1e-12)


def test_baum_welch_likelihood_non_decreasing() -> None:
    y = gen_hmm(5000, 0.02, 0.15, np.random.default_rng(1))
    _, _, lls = baum_welch(y, 0.2, 0.9, stationary_g(0.2, 0.9), iters=40)
    d = np.diff(lls)
    assert np.all(d >= -1e-10 * abs(lls[-1]))           # EM: monotone up to float rounding
    assert lls[-1] > lls[0]


def test_baum_welch_short_input() -> None:
    assert baum_welch(np.array([1], np.int8), 0.1, 0.2, 0.5) == (0.1, 0.2, [])


@pytest.mark.parametrize("burst", [1, 8, 32])
@pytest.mark.parametrize("pi", [0.02, 0.05, 0.10])
def test_baum_welch_recovers_parameters(burst: int, pi: float) -> None:
    """Data generated from the model; enough data for ~300 bad episodes;
    EM from the spec's (0.01, 0.2) to convergence. p and r within 25%."""
    p, r = ge_params(burst, pi)
    n = max(40_000, int(np.ceil(300 * burst / pi)))
    y = gen_hmm(n, p, r, np.random.default_rng(burst * 10 + int(pi * 100)))
    ph, rh = P0, R0
    for _ in range(60):
        ph2, rh2, _ = baum_welch(y, ph, rh, stationary_g(P0, R0), iters=5)
        done = abs(ph2 - ph) <= 1e-4 * ph and abs(rh2 - rh) <= 1e-4 * rh
        ph, rh = ph2, rh2
        if done:
            break
    assert abs(ph - p) / p < 0.25 and abs(rh - r) / r < 0.25, (ph, p, rh, r)


# --------------------------------------------------------------------------- P-HMM predictor


def test_phmm_new_client_and_predict_formula() -> None:
    h = PHMM()
    assert isinstance(h, Predictor)
    g = stationary_g(P0, R0)
    out = h.predict(_inp([4, 9]))
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.float32([g * E_G + (1 - g) * E_B] * 2))
    h.observe(4, 0, 0, np.zeros(8, np.float32))
    st = h._c[4]
    pg = g
    bg = pg * (1 - E_G) / (pg * (1 - E_G) + (1 - pg) * (1 - E_B))
    assert st.belief_g == pytest.approx(bg, rel=1e-15)
    nxt = bg * (1 - P0) + (1 - bg) * R0
    assert h.predict(_inp([4]))[0] == np.float32(nxt * E_G + (1 - nxt) * E_B)
    assert h.predict(_inp([9]))[0] == out[1]                     # clients are independent
    assert set(h._c) == {4}                                       # reads never create state
    assert h.params(9) == (P0, R0)


def test_phmm_refit_schedule_window_clip_and_refresh() -> None:
    h = PHMM()
    y = gen_ge(1000, *ge_params(8, 0.10), np.random.default_rng(3))
    for k, v in enumerate(y, start=1):
        h.observe(0, int(v), 0, np.zeros(8, np.float32))
        st = h._c[0]
        assert st.n_fits == k // 64
        assert len(st.window) == min(k, 256)
        assert 1e-4 <= st.p <= 0.5 and 0.01 <= st.r <= 1.0
        if k % 64 == 0:
            w = np.array(st.window, np.int8)
            np.testing.assert_array_equal(w, y[max(0, k - 256):k])
            # refit = 5 BW iterations from (0.01, 0.2), clipped, then the filter over the window
            p, r, _ = baum_welch(w, P0, R0, stationary_g(P0, R0), 5)
            p, r = min(max(p, 1e-4), 0.5), min(max(r, 0.01), 1.0)
            assert (st.p, st.r) == (p, r)
            bel, _ = forward_filter(w, p, r, stationary_g(p, r))
            assert st.belief_g == bel[-1]


def test_spec_constants() -> None:
    import fedgrant.predictors.hmm as H
    assert (H.E_G, H.E_B, H.P0, H.R0) == (0.98, 0.02, 0.01, 0.2)
    assert (H.WINDOW, H.REFIT_EVERY, H.BW_ITERS) == (256, 64, 5)
    assert H.P_CLIP == (1e-4, 0.5) and H.R_CLIP == (0.01, 1.0)


def test_phmm_r_lower_clip() -> None:
    h = PHMM()
    for v in [1] * 192 + [0] * 64:                               # ends in one long loss run
        h.observe(0, v, 0, np.zeros(8, np.float32))
    raw = baum_welch(np.array([1] * 192 + [0] * 64, np.int8), P0, R0, stationary_g(P0, R0), 5)[1]
    assert raw < 0.01 and h.params(0)[1] == 0.01


def test_phmm_clipping_applied() -> None:
    h = PHMM()
    for _ in range(REFIT_EVERY):                                 # no losses at all
        h.observe(1, 1, 0, np.zeros(8, np.float32))
    p, r = h.params(1)
    assert p == 1e-4                                              # clipped at the lower bound
    assert 0.01 <= r <= 1.0


def test_r_equal_1_is_absorbing_for_warm_start_but_not_default() -> None:
    """Documents the SPEC-NOTE: a quiet window drives r to 1; with warm start EM
    can never leave r = 1, so a later 40-loss burst is not learned."""
    quiet = [1] * WINDOW
    burst = [1] * 100 + [0] * 40 + [1] * 116                      # 256 outcomes with a long burst
    for warm, expect_stuck in ((True, True), (False, False)):
        h = PHMM(warm_start=warm)
        for v in quiet + burst:
            h.observe(0, v, 0, np.zeros(8, np.float32))
        _, r = h.params(0)
        assert (r == 1.0) == expect_stuck, (warm, r)
    assert h.params(0)[1] < 0.2                                   # default learns persistence


@pytest.mark.parametrize("pi", [0.05, 0.10])
def test_phmm_brier_beats_pewma_on_bursty_losses(pi: float) -> None:
    y = gen_ge(30_000, *ge_params(8, pi), np.random.default_rng(int(pi * 100)))
    h, e = PHMM(), PEWMA()
    bh = be = 0.0
    z = np.zeros(8, np.float32)
    for v in y:
        bh += (h.value(0) - v) ** 2
        be += (e.value(0) - v) ** 2
        h.observe(0, int(v), 0, z)
        e.observe(0, int(v), 0, z)
    assert bh < be, (bh / len(y), be / len(y))


def test_phmm_deterministic() -> None:
    y = gen_ge(800, *ge_params(8, 0.05), np.random.default_rng(9))
    a, b = PHMM(), PHMM()
    for v in y:
        a.observe(2, int(v), 0, np.zeros(8, np.float32))
        b.observe(2, int(v), 0, np.zeros(8, np.float32))
    assert a.predict(_inp([2])).tobytes() == b.predict(_inp([2])).tobytes()
    assert a.params(2) == b.params(2)


def test_phmm_rejects_bad_y() -> None:
    with pytest.raises(ValueError):
        PHMM().observe(0, 2, 0, np.zeros(8, np.float32))