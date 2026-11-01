"""Gilbert–Elliott parameter conversion and observed loss/burst statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LossStats:
    n: int
    lost: int
    loss_rate: float
    mean_burst: float
    n_bursts: int
    complete_bursts: int


def _complete_loss_bursts(lost_mask: np.ndarray) -> np.ndarray:
    """Return loss-run lengths not censored by the observation boundaries."""
    starts = np.flatnonzero(lost_mask & ~np.r_[False, lost_mask[:-1]])
    ends = np.flatnonzero(lost_mask & ~np.r_[lost_mask[1:], False]) + 1
    lengths = (ends - starts).astype(np.int64)
    left_censored = bool(lost_mask[0])
    right_censored = bool(lost_mask[-1])
    left = int(left_censored)
    right = len(lengths) - int(right_censored)
    return lengths[left:right]


def ge_params(pi: float, mean_burst: float) -> tuple[float, float]:
    """Convert stationary loss probability and mean bad-run length to `(p, r)`."""
    pi = float(pi)
    mean_burst = float(mean_burst)
    if not np.isfinite(pi) or not 0.0 < pi < 1.0:
        raise ValueError("pi must be finite and strictly between 0 and 1")
    if not np.isfinite(mean_burst) or mean_burst < 1.0:
        raise ValueError("mean_burst must be finite and at least one packet")
    r = 1.0 / mean_burst
    p = pi * r / (1.0 - pi)
    if p > 1.0:
        raise ValueError("pi and mean_burst imply an invalid Good-to-Bad probability")
    return p, r


def analyze_loss(received: np.ndarray | list[bool]) -> LossStats:
    """Summarize a sequence where True means received and False means lost."""
    values = np.asarray(received)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("received must be a non-empty one-dimensional sequence")
    if values.dtype.kind != "b":
        if not np.isin(values, (0, 1)).all():
            raise ValueError("received values must be booleans (or 0/1)")
        values = values.astype(bool)
    lost_mask = ~values
    lost = int(lost_mask.sum())
    starts = lost_mask & ~np.r_[False, lost_mask[:-1]]
    n_bursts = int(starts.sum())
    burst_lengths = _complete_loss_bursts(lost_mask)
    return LossStats(
        n=int(values.size), lost=lost, loss_rate=lost / int(values.size),
        mean_burst=(float(np.mean(burst_lengths)) if burst_lengths.size else 0.0),
        n_bursts=n_bursts, complete_bursts=int(burst_lengths.size),
    )


def bootstrap_loss_intervals(
    received: np.ndarray | list[bool], *, replicates: int = 400, block_size: int = 256,
    confidence: float = 0.95, seed: int = 0,
) -> dict[str, float | int | str]:
    """Bootstrap loss-rate and complete-burst percentile intervals.

    The loss-rate interval uses circular moving blocks to preserve temporal
    dependence. Mean bad-run length uses resampling of complete runs, because
    cutting runs at arbitrary block boundaries biases that statistic downward.
    """
    values = np.asarray(received)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("received must be a non-empty one-dimensional sequence")
    if values.dtype.kind != "b":
        if not np.isin(values, (0, 1)).all():
            raise ValueError("received values must be booleans (or 0/1)")
        values = values.astype(bool)
    if isinstance(replicates, bool) or int(replicates) != replicates or replicates < 100:
        raise ValueError("replicates must be an integer of at least 100")
    if isinstance(block_size, bool) or int(block_size) != block_size or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and in (0, 1)")
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed must be an integer")

    n = int(values.size)
    block_size = min(int(block_size), n)
    n_blocks = (n + block_size - 1) // block_size
    offsets = np.arange(block_size, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    loss_rates = np.empty(int(replicates), dtype=np.float64)
    burst_lengths = _complete_loss_bursts(~values)
    mean_bursts = np.full(int(replicates), np.nan, dtype=np.float64)
    for index in range(int(replicates)):
        starts = rng.integers(0, n, size=n_blocks, dtype=np.int64)
        sample = values[((starts[:, None] + offsets) % n).ravel()[:n]]
        loss_rates[index] = float((~sample).mean())
        if burst_lengths.size:
            burst_sample = rng.choice(burst_lengths, size=burst_lengths.size, replace=True)
            mean_bursts[index] = float(burst_sample.mean())
    tail = (1.0 - confidence) / 2.0
    loss_low, loss_high = np.quantile(loss_rates, [tail, 1.0 - tail])
    if burst_lengths.size:
        burst_low, burst_high = np.quantile(mean_bursts, [tail, 1.0 - tail])
    else:
        burst_low = burst_high = math.nan
    return {
        "loss_rate_ci_low": float(loss_low),
        "loss_rate_ci_high": float(loss_high),
        "mean_burst_ci_low": float(burst_low),
        "mean_burst_ci_high": float(burst_high),
        "bootstrap_replicates": int(replicates),
        "bootstrap_block_size": block_size,
        "complete_bursts_bootstrapped": int(burst_lengths.size),
        "loss_interval_method": "circular_moving_block",
        "burst_interval_method": "complete_bad_run_resampling",
        "confidence": float(confidence),
    }


__all__ = ["LossStats", "ge_params", "analyze_loss", "bootstrap_loss_intervals"]
