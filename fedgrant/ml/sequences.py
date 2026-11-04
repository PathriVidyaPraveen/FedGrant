"""Standardization and window batching for P-GRU / P-MLP training (Sections 4.2-4.5).

Loading is done by fedgrant.ml.schema.load_split (re-exported here). This module
only computes the train-split standardization statistics, standardizes, and
gathers minibatch windows by index; sequences are never materialized.

SPEC-NOTE: the splits are compressed .npz files, so memory-mapping is not
possible; load_split reads them into RAM (about 0.5 GB for the 4M-sample train
split per the spec's estimate). Each model keeps only the arrays it needs.

SPEC-NOTE: window indices are computed per batch from (sample_end, sample_len)
instead of a precomputed [ns, 16] index matrix (512 MB at 4M samples). Padding
gathers a zero row appended after the last event, so the gather is one fancy-index.

No PyTorch here: NumPy only, so the gateway side can import the stats helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fedgrant.ml.schema import load_split
from fedgrant.policy.constants import N_CTX, N_OBS, N_X48, T_EVBUF

S_MIN: float = 1e-6                    # features with s < 1e-6 use s = 1 (Section 4.2)
NORM_GRU_FILE = "norm_stats_gru.json"
NORM_MLP_FILE = "norm_stats_mlp.json"
NORM_VERSION = 1

__all__ = [
    "NormStats",
    "GRUData",
    "MLPData",
    "S_MIN",
    "compute_norm_stats",
    "save_norm_stats",
    "load_norm_stats",
    "window_index",
    "load_split",
    "resolve_split_path",
]


# --------------------------------------------------------------------------- statistics


def _mu_s(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and population std in float64; s < 1e-6 -> 1."""
    x64 = np.asarray(x, dtype=np.float64)
    if x64.shape[0] == 0:
        raise ValueError("cannot compute statistics on zero rows")
    mu = x64.mean(axis=0)
    s = x64.std(axis=0)
    s = np.where(s < S_MIN, 1.0, s)
    return mu, s


@dataclass(frozen=True)
class NormStats:
    """mu / s per feature, stored as float64 and applied in float32.

    obs: the 8 event observations, over all train event rows.
    ctx: the 16 context features, over all train samples.
    x48: the 48 P-MLP features, over all train samples.
    """

    mu_obs: np.ndarray
    s_obs: np.ndarray
    mu_ctx: np.ndarray
    s_ctx: np.ndarray
    mu_x48: np.ndarray
    s_x48: np.ndarray

    def __post_init__(self) -> None:
        for name, n in (("obs", N_OBS), ("ctx", N_CTX), ("x48", N_X48)):
            for kind in ("mu", "s"):
                a = np.asarray(getattr(self, f"{kind}_{name}"), dtype=np.float64)
                if a.shape != (n,) or not np.all(np.isfinite(a)):
                    raise ValueError(f"{kind}_{name} must be {n} finite values")
                if kind == "s" and np.any(a <= 0):
                    raise ValueError(f"s_{name} must be > 0")
                object.__setattr__(self, f"{kind}_{name}", a)

    # standardization, always returning float32
    def obs(self, o: np.ndarray) -> np.ndarray:
        return ((np.asarray(o, np.float64) - self.mu_obs) / self.s_obs).astype(np.float32)

    def ctx(self, c: np.ndarray) -> np.ndarray:
        return ((np.asarray(c, np.float64) - self.mu_ctx) / self.s_ctx).astype(np.float32)

    def x48(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, np.float64) - self.mu_x48) / self.s_x48).astype(np.float32)

    def gru_dict(self) -> dict[str, Any]:
        return {"version": NORM_VERSION, "s_min": S_MIN,
                "mu_obs": self.mu_obs.tolist(), "s_obs": self.s_obs.tolist(),
                "mu_ctx": self.mu_ctx.tolist(), "s_ctx": self.s_ctx.tolist()}

    def mlp_dict(self) -> dict[str, Any]:
        return {"version": NORM_VERSION, "s_min": S_MIN,
                "mu_x48": self.mu_x48.tolist(), "s_x48": self.s_x48.tolist()}


def compute_norm_stats(train: dict[str, np.ndarray]) -> NormStats:
    """Standardization statistics from the TRAIN split only (Section 4.2, 4.5)."""
    mu_o, s_o = _mu_s(train["events"])
    mu_c, s_c = _mu_s(train["ctx"])
    mu_x, s_x = _mu_s(train["x48"])
    return NormStats(mu_o, s_o, mu_c, s_c, mu_x, s_x)


def save_norm_stats(stats: NormStats, out_dir: str | Path) -> tuple[Path, Path]:
    """Write models/norm_stats_gru.json (events + ctx) and models/norm_stats_mlp.json (x48)."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    pg, pm = d / NORM_GRU_FILE, d / NORM_MLP_FILE
    pg.write_text(json.dumps(stats.gru_dict(), indent=1))
    pm.write_text(json.dumps(stats.mlp_dict(), indent=1))
    return pg, pm


def load_norm_stats(out_dir: str | Path) -> NormStats:
    """Inverse of save_norm_stats."""
    d = Path(out_dir)
    g = json.loads((d / NORM_GRU_FILE).read_text())
    m = json.loads((d / NORM_MLP_FILE).read_text())
    return NormStats(np.array(g["mu_obs"]), np.array(g["s_obs"]), np.array(g["mu_ctx"]),
                     np.array(g["s_ctx"]), np.array(m["mu_x48"]), np.array(m["s_x48"]))


def norm_stats_from_dicts(gru: dict[str, Any], mlp: dict[str, Any]) -> NormStats:
    """Rebuild NormStats from the two dicts stored in a checkpoint."""
    return NormStats(np.array(gru["mu_obs"]), np.array(gru["s_obs"]), np.array(gru["mu_ctx"]),
                     np.array(gru["s_ctx"]), np.array(mlp["mu_x48"]), np.array(mlp["s_x48"]))


# --------------------------------------------------------------------------- windows


def window_index(end: np.ndarray, length: np.ndarray, pad: int, T: int = T_EVBUF) -> np.ndarray:
    """int64 [B, T] event indices, oldest first, right-padded with `pad`.

    Row b holds events[end_b - len_b : end_b] in columns 0..len_b-1 (Section 4.5).
    Right padding is what pack_padded_sequence expects.
    """
    e = np.asarray(end, dtype=np.int64)[:, None]
    n = np.asarray(length, dtype=np.int64)[:, None]
    j = np.arange(T, dtype=np.int64)[None, :]
    return np.where(j < n, e - n + j, pad)


def resolve_split_path(data_dir: str | Path, split: str) -> Path:
    """DIR/<split>.npz if it exists, else the shard directory DIR/<split>/."""
    d = Path(data_dir)
    f = d / f"{split}.npz"
    if f.is_file():
        return f
    if (d / split).is_dir():
        return d / split
    raise FileNotFoundError(f"neither {f} nor {d / split}/ exists")


class GRUData:
    """Standardized P-GRU inputs of one split, gathered by sample index.

    Holds: standardized events with one trailing zero row (padding target),
    sample_end / sample_len, standardized ctx, y. Nothing per-window is stored.
    """

    def __init__(self, split: dict[str, np.ndarray], stats: NormStats) -> None:
        ev = stats.obs(split["events"])
        self.events = np.concatenate([ev, np.zeros((1, N_OBS), np.float32)])   # row E = padding
        self.pad = int(ev.shape[0])
        self.end = np.ascontiguousarray(split["sample_end"], dtype=np.int64)
        self.len = np.ascontiguousarray(split["sample_len"], dtype=np.int64)
        self.ctx = stats.ctx(split["ctx"])
        self.y = np.ascontiguousarray(split["y"], dtype=np.float32)
        if not (self.end.shape == self.len.shape == self.y.shape == (self.ctx.shape[0],)):
            raise ValueError("inconsistent sample arrays")

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def batch(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(seq float32 [B,16,8], lengths int64 [B], ctx float32 [B,16], y float32 [B])."""
        idx = np.asarray(idx, dtype=np.int64)
        w = window_index(self.end[idx], self.len[idx], self.pad)
        return self.events[w], self.len[idx], self.ctx[idx], self.y[idx]


class MLPData:
    """Standardized P-MLP inputs (x48) of one split."""

    def __init__(self, split: dict[str, np.ndarray], stats: NormStats) -> None:
        self.x48 = stats.x48(split["x48"])
        self.y = np.ascontiguousarray(split["y"], dtype=np.float32)
        if self.x48.shape != (self.y.shape[0], N_X48):
            raise ValueError("inconsistent sample arrays")

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def batch(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(x48 float32 [B,48], y float32 [B])."""
        idx = np.asarray(idx, dtype=np.int64)
        return self.x48[idx], self.y[idx]