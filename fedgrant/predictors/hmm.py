"""P-HMM, the two-state HMM baseline predictor (Section 4.6).

States G (good) and B (bad). Emissions are FIXED: Pr[y=1|G] = 0.98, Pr[y=1|B] = 0.02.
Transitions p = Pr[G->B], r = Pr[B->G], initially (0.01, 0.2). Every 64 new
outcomes, Baum-Welch (5 iterations, scaled forward-backward) re-estimates ONLY p
and r on the last 256 outcomes; then p is clipped to [1e-4, 0.5], r to [0.01, 1],
and the forward filter is re-run over the window to refresh the belief.
Output: p_hat = Pr[G_next] * 0.98 + Pr[B_next] * 0.02.

Model per outcome t: hidden state s_t emits y_t, then s_{t+1} ~ T(s_t).
Every forward step is normalised, so nothing underflows; with the clipped
transitions and fixed emissions no probability is ever 0.

SPEC-NOTE: the spec does not give the initial state distribution. The
stationary distribution of the current (p, r) is used, both for a new client
(so its first p_hat is the model's long-run delivery probability) and as the
start of the window in Baum-Welch and in the refresh filter.

SPEC-NOTE (needs team sign-off): the spec does not say where each Baum-Welch
run starts. Starting from the current (p, r) ("warm start") has a failure mode:
r = 1 is an absorbing fixed point of EM (with r = 1 the expected B->B count is
exactly 0, so the M-step returns r = 1 again), and the spec's clip range for r
includes 1. One window without losses drives r to 1 and P-HMM then never learns
bursts again (measured: Brier at burst 8, pi = 0.10 was 0.088 vs 0.031). The
default here starts every refit from the spec's initial values (0.01, 0.2)
("cold start"), which changes no parameter; warm_start=True gives the other
reading.

The recursions are sequential in time, so they run as scalar float loops (much
faster than NumPy on 2-element arrays); predict() is vectorised over clients.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from fedgrant.core.estimators import check_y
from fedgrant.predictors.base import PredInput, RoundCtx

E_G: float = 0.98                  # Pr[y = 1 | G]
E_B: float = 0.02                  # Pr[y = 1 | B]
P0: float = 0.01                   # initial Pr[G -> B]
R0: float = 0.2                    # initial Pr[B -> G]
P_CLIP: tuple[float, float] = (1e-4, 0.5)
R_CLIP: tuple[float, float] = (0.01, 1.0)
WINDOW: int = 256
REFIT_EVERY: int = 64
BW_ITERS: int = 5


def stationary_g(p: float, r: float) -> float:
    """Stationary Pr[G] = r / (p + r)."""
    return r / (p + r)


def _emit(y: int) -> tuple[float, float]:
    """(Pr[y | G], Pr[y | B])."""
    return (E_G, E_B) if y == 1 else (1.0 - E_G, 1.0 - E_B)


def forward_filter(
    y: np.ndarray, p: float, r: float, init_g: float
) -> tuple[np.ndarray, float]:
    """Scaled forward filter.

    Returns (beliefs, loglik): beliefs[t] = Pr[s_t = G | y_1..y_t] (float64 [T]),
    loglik = log Pr[y_1..y_T]. init_g = Pr[s_1 = G].
    """
    T = len(y)
    out = np.empty(T, dtype=np.float64)
    loglik = 0.0
    pg, pb = init_g, 1.0 - init_g                  # prior of s_t
    for t in range(T):
        eg, eb = _emit(int(y[t]))
        ag, ab = pg * eg, pb * eb
        c = ag + ab
        loglik += np.log(c)
        bg = ag / c
        out[t] = bg
        bb = 1.0 - bg
        pg = bg * (1.0 - p) + bb * r               # prior of s_{t+1}
        pb = 1.0 - pg
    return out, float(loglik)


def baum_welch(
    y: np.ndarray, p: float, r: float, init_g: float, iters: int = BW_ITERS
) -> tuple[float, float, list[float]]:
    """EM re-estimation of p and r only (emissions and init_g fixed).

    Scaled forward-backward; M-step p = sum_t xi_t(G,B) / sum_{t<T} gamma_t(G),
    r = sum_t xi_t(B,G) / sum_{t<T} gamma_t(B). No clipping here.
    Returns (p, r, logliks), logliks[k] = log-likelihood under the parameters
    used in iteration k (non-decreasing, by EM).
    """
    T = len(y)
    if T < 2:
        return p, r, []
    ys = [int(v) for v in y]
    em = [_emit(v) for v in ys]
    logliks: list[float] = []
    fg = [0.0] * T
    c = [0.0] * T
    for _ in range(iters):
        # forward (scaled): fg[t] = Pr[s_t = G | y_1..t], c[t] = Pr[y_t | y_1..t-1]
        pg = init_g
        ll = 0.0
        for t in range(T):
            eg, eb = em[t]
            ag, ab = pg * eg, (1.0 - pg) * eb
            ct = ag + ab
            c[t] = ct
            ll += np.log(ct)
            g = ag / ct
            fg[t] = g
            pg = g * (1.0 - p) + (1.0 - g) * r
        logliks.append(float(ll))
        # backward (scaled): beta_t = Pr[y_{t+1..T} | s_t] / prod_{u>t} c[u]
        bg, bb = 1.0, 1.0
        n_gb = n_bg = 0.0                         # expected transition counts
        d_g = d_b = 0.0                           # expected occupancy for t < T
        for t in range(T - 2, -1, -1):
            eg, eb = em[t + 1]
            wg, wb = eg * bg, eb * bb             # Pr[y_{t+1}|s_{t+1}] beta_{t+1}
            a_g, a_b = fg[t], 1.0 - fg[t]
            inv = 1.0 / c[t + 1]
            x_gg = a_g * (1.0 - p) * wg * inv
            x_gb = a_g * p * wb * inv
            x_bg = a_b * r * wg * inv
            x_bb = a_b * (1.0 - r) * wb * inv
            n_gb += x_gb
            n_bg += x_bg
            d_g += x_gg + x_gb                    # = gamma_t(G)
            d_b += x_bg + x_bb                    # = gamma_t(B)
            bg = ((1.0 - p) * wg + p * wb) * inv
            bb = (r * wg + (1.0 - r) * wb) * inv
        if d_g > 0.0:
            p = n_gb / d_g
        if d_b > 0.0:
            r = n_bg / d_b
    return p, r, logliks


@dataclass(slots=True)
class _ClientHMM:
    p: float = P0
    r: float = R0
    belief_g: float | None = None                 # Pr[s_t = G | outcomes so far]; None = none yet
    window: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    since_fit: int = 0
    n_fits: int = 0

    def prior_next_g(self) -> float:
        """Pr[G_next]."""
        if self.belief_g is None:
            return stationary_g(self.p, self.r)
        b = self.belief_g
        return b * (1.0 - self.p) + (1.0 - b) * self.r


class PHMM:
    """Two-state HMM forward filter with periodic Baum-Welch (implements Predictor).

    warm_start: start each Baum-Welch run from the current (p, r) instead of the
    spec's initial (0.01, 0.2); see the module SPEC-NOTE. Default False.
    """

    name = "hmm"

    def __init__(self, warm_start: bool = False) -> None:
        self.warm_start = bool(warm_start)
        self._c: dict[int, _ClientHMM] = {}

    def _get(self, client: int) -> _ClientHMM:
        c = int(client)
        st = self._c.get(c)
        if st is None:
            st = self._c[c] = _ClientHMM()
        return st

    def reset_round(self, ctx: RoundCtx) -> None:
        """No per-round state: the filter persists across rounds."""

    def observe(self, client: int, y: int, t_ns: int, obs: np.ndarray) -> None:
        """One forward step; every 64 outcomes refit (p, r) on the last 256 and refilter."""
        yy = check_y(y)
        st = self._get(client)
        pg = st.prior_next_g()
        eg, eb = _emit(yy)
        ag, ab = pg * eg, (1.0 - pg) * eb
        st.belief_g = ag / (ag + ab)
        st.window.append(yy)
        st.since_fit += 1
        if st.since_fit >= REFIT_EVERY:
            st.since_fit = 0
            self._refit(st)

    def _refit(self, st: _ClientHMM) -> None:
        """Baum-Welch on the window, clip, then re-run the forward filter over the window."""
        w = np.fromiter(st.window, dtype=np.int8, count=len(st.window))
        p0, r0 = (st.p, st.r) if self.warm_start else (P0, R0)
        p, r, _ = baum_welch(w, p0, r0, stationary_g(p0, r0), BW_ITERS)
        st.p = min(max(p, P_CLIP[0]), P_CLIP[1])
        st.r = min(max(r, R_CLIP[0]), R_CLIP[1])
        beliefs, _ = forward_filter(w, st.p, st.r, stationary_g(st.p, st.r))
        st.belief_g = float(beliefs[-1])
        st.n_fits += 1

    def _prior_next_g(self, client: int) -> float:
        """Pr[G_next] without creating state for an unseen client (reads are side-effect free)."""
        st = self._c.get(int(client))
        return stationary_g(P0, R0) if st is None else st.prior_next_g()

    def params(self, client: int) -> tuple[float, float]:
        """Current (p, r) of a client ((0.01, 0.2) if never observed)."""
        st = self._c.get(int(client))
        return (P0, R0) if st is None else (st.p, st.r)

    def value(self, client: int) -> float:
        """p_hat of one client."""
        g = self._prior_next_g(client)
        return g * E_G + (1.0 - g) * E_B

    def predict(self, inp: PredInput) -> np.ndarray:
        """p_hat = Pr[G_next] 0.98 + Pr[B_next] 0.02 for each requested client, float32 [n]."""
        g = np.fromiter((self._prior_next_g(c) for c in inp.client_ids), dtype=np.float64, count=inp.n)
        return (g * E_G + (1.0 - g) * E_B).astype(np.float32)