"""P-GRU, NumPy float32 inference from models/gru_prod.npz (Sections 4.2, 4.6).

Per client the predictor keeps (Section 4.6): a ring buffer of the last 16 RAW
observations (float32 [16, 8]) with a count, a cached hidden state h (float32
[32]) and a stale flag. All of it persists across rounds; reset_round clears
nothing. The predictor gets its events only through observe(); it does not
read ClientState.evbuf or ClientState.h.

observe(i, y, t, o) appends o and marks h_i stale. predict(inp) first
recomputes h for the stale requested clients in ONE batched pass (windows
left-aligned, oldest first; at step j only rows with more than j events are
updated, which equals running each window alone; n = 0 gives h = 0), then
evaluates the head on [h; standardized ctx] for every requested client.

GRU equations are PyTorch's (gate order r, z, n):
  r = sig(W_ir x + b_ir + W_hr h + b_hr)
  z = sig(W_iz x + b_iz + W_hz h + b_hz)
  n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
  h' = (1 - z) * n + z * h
Standardization is computed in float64 and rounded once to float32, exactly as
in training; every other operation is float32.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from fedgrant.core.estimators import check_y
from fedgrant.policy.constants import GRU_HIDDEN, N_OBS, T_EVBUF
from fedgrant.predictors.base import PredInput, RoundCtx
from fedgrant.predictors.mlp_numpy import _weights, stable_sigmoid, standardize

H = GRU_HIDDEN


class _ClientBuf:
    """Ring buffer of raw observations + cached h for one client."""

    __slots__ = ("raw", "total", "h", "stale")

    def __init__(self) -> None:
        self.raw = np.zeros((T_EVBUF, N_OBS), np.float32)
        self.total = 0                       # observations ever appended
        self.h = np.zeros(H, np.float32)
        self.stale = False                   # no events: h = 0 is already correct

    @property
    def n(self) -> int:
        return min(self.total, T_EVBUF)

    def append(self, o: np.ndarray) -> None:
        self.raw[self.total % T_EVBUF] = o
        self.total += 1
        self.stale = True

    def window(self) -> np.ndarray:
        """Last n raw observations, oldest first, float32 [n, 8]."""
        n = self.n
        idx = (self.total - n + np.arange(n)) % T_EVBUF
        return self.raw[idx]


class PGRUNumpy:
    """P-GRU predictor (implements Predictor)."""

    name = "gru"

    def __init__(self, npz: str | Path | Mapping[str, np.ndarray]) -> None:
        d = _weights(npz, "gru")
        if "gate_order" in d and str(d["gate_order"]) != "r,z,n":
            raise ValueError(f"unsupported gate order {str(d['gate_order'])!r}")
        self.w_ih_t = np.ascontiguousarray(d["weight_ih_l0"].T, dtype=np.float32)   # [8, 96]
        self.w_hh_t = np.ascontiguousarray(d["weight_hh_l0"].T, dtype=np.float32)   # [32, 96]
        self.b_ih = np.asarray(d["bias_ih_l0"], np.float32)
        self.b_hh = np.asarray(d["bias_hh_l0"], np.float32)
        self.w1t = np.ascontiguousarray(d["head_w1"].T, dtype=np.float32)           # [48, 32]
        self.b1 = np.asarray(d["head_b1"], np.float32)
        self.w2t = np.ascontiguousarray(d["head_w2"].T, dtype=np.float32)           # [32, 1]
        self.b2 = np.asarray(d["head_b2"], np.float32)
        self.mu_obs = np.asarray(d["mu_obs"], np.float64)
        self.s_obs = np.asarray(d["s_obs"], np.float64)
        self.mu_ctx = np.asarray(d["mu_ctx"], np.float64)
        self.s_ctx = np.asarray(d["s_ctx"], np.float64)
        self.seed = int(d["seed"]) if "seed" in d else -1
        self._c: dict[int, _ClientBuf] = {}
        self.n_recomputed = 0                # hidden states recomputed so far (diagnostics)

    # ------------------------------------------------------------------ Predictor
    def reset_round(self, ctx: RoundCtx) -> None:
        """Nothing is cleared: buffers and h persist across rounds (Section 4.6)."""

    def observe(self, client: int, y: int, t_ns: int, obs: np.ndarray) -> None:
        """Append the raw observation o (float32 [8]) of client i and mark h_i stale."""
        yy = check_y(y)
        o = np.asarray(obs, dtype=np.float32)
        if o.shape != (N_OBS,) or not np.all(np.isfinite(o)):
            raise ValueError("obs must be 8 finite values")
        if float(o[0]) != float(yy):
            raise ValueError(f"obs[0] = {o[0]} does not match y = {yy}")
        self._buf(client).append(o)

    def predict(self, inp: PredInput) -> np.ndarray:
        """p_hat float32 [inp.n]: refresh stale h (one batch), then the head for all requested."""
        if inp.n == 0:
            return np.zeros(0, np.float32)
        bufs = [self._buf(c) for c in inp.client_ids.tolist()]
        stale = [b for b in dict.fromkeys(bufs) if b.stale]        # unique, in request order
        if stale:
            self._recompute(stale)
        h = np.stack([b.h for b in bufs])                            # [n, 32]
        return stable_sigmoid(self._head(h, inp.ctx))

    # ------------------------------------------------------------------ internals
    def _buf(self, client: int) -> _ClientBuf:
        c = int(client)
        b = self._c.get(c)
        if b is None:
            b = self._c[c] = _ClientBuf()
        return b

    def _recompute(self, bufs: list[_ClientBuf]) -> None:
        """Batched masked GRU over the windows of `bufs`; stores h and clears stale."""
        B = len(bufs)
        lens = np.fromiter((b.n for b in bufs), dtype=np.int64, count=B)
        T = int(lens.max()) if B else 0
        h = np.zeros((B, H), np.float32)
        if T > 0:
            X = np.zeros((B, T, N_OBS), np.float32)
            for r, b in enumerate(bufs):
                if lens[r]:
                    X[r, : lens[r]] = standardize(b.window(), self.mu_obs, self.s_obs)
            gi_all = X @ self.w_ih_t + self.b_ih                       # [B, T, 96]
            w_hh_t, b_hh = self.w_hh_t, self.b_hh
            for j in range(T):
                gi = gi_all[:, j]
                gh = h @ w_hh_t
                gh += b_hh
                rz = stable_sigmoid(gi[:, : 2 * H] + gh[:, : 2 * H])  # r and z in one call
                n = gh[:, 2 * H :] * rz[:, :H]
                n += gi[:, 2 * H :]
                np.tanh(n, out=n)
                z = rz[:, H:]
                h_new = n - z * n + z * h                             # (1 - z) n + z h
                if j < lens.min():
                    h = h_new                                         # every row has an event at step j
                else:
                    h = np.where((lens > j)[:, None], h_new, h)       # only rows with an event at step j
        for r, b in enumerate(bufs):
            b.h = h[r].copy()
            b.stale = False
        self.n_recomputed += B

    def _head(self, h: np.ndarray, ctx: np.ndarray) -> np.ndarray:
        """Logits float32 [n] from h [n, 32] and raw ctx [n, 16]."""
        c = standardize(ctx, self.mu_ctx, self.s_ctx)
        a = np.maximum(np.concatenate([h, c], axis=1) @ self.w1t + self.b1, 0.0)
        return (a @ self.w2t + self.b2)[:, 0]

    # ------------------------------------------------------------------ inspection
    def hidden(self, client: int) -> np.ndarray:
        """Current h of a client, recomputing it if stale (copy)."""
        b = self._buf(client)
        if b.stale:
            self._recompute([b])
        return b.h.copy()

    def window(self, client: int) -> np.ndarray:
        """Raw window of a client, oldest first (copy)."""
        return self._buf(client).window().copy()

    def logits(self, inp: PredInput) -> np.ndarray:
        """Logits instead of probabilities (tests, analysis)."""
        bufs = [self._buf(c) for c in inp.client_ids.tolist()]
        stale = [b for b in dict.fromkeys(bufs) if b.stale]
        if stale:
            self._recompute(stale)
        return self._head(np.stack([b.h for b in bufs]), inp.ctx)