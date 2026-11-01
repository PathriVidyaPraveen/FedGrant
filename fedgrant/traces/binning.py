"""Pure helpers and CLI support for the cellular ``.up`` trace pipeline."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

TRACE_FILES = (
    "Verizon-LTE-driving",
    "Verizon-LTE-short",
    "ATT-LTE-driving",
    "TMobile-UMTS-driving",
    "Verizon-EVDO-driving",
)
BIN_MS = 100
BYTES_PER_OPPORTUNITY = 1500
RATE_PER_OPPORTUNITY = BYTES_PER_OPPORTUNITY * 8 * 1000 / BIN_MS


def read_up(path: str | Path) -> np.ndarray:
    """Read a whitespace-delimited ``.up`` timestamp file in milliseconds."""
    p = Path(path)
    try:
        values = np.loadtxt(p, dtype=np.float64, comments="#", ndmin=1)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read .up timestamps from {p}: {exc}") from exc
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"trace {p} is empty")
    if not np.isfinite(values).all():
        raise ValueError(f"trace {p} contains non-finite timestamps")
    if np.any(values < 0):
        raise ValueError(f"trace {p} contains negative timestamps")
    if np.any(np.diff(values) < 0):
        raise ValueError(f"trace {p} timestamps are not monotonic")
    return values


def bin_opportunities(timestamps_ms: Iterable[float] | np.ndarray, *, bin_ms: int = BIN_MS) -> tuple[np.ndarray, np.ndarray]:
    """Return full bins as ``(t_ms, rate_bps)``; discard the partial final bin.

    An opportunity is one 1500-byte delivery in the source trace.  Bins are
    anchored at zero, so their rates are exactly ``count * 120000`` at 100 ms.
    """
    times = np.asarray(list(timestamps_ms) if not isinstance(timestamps_ms, np.ndarray) else timestamps_ms,
                       dtype=np.float64).reshape(-1)
    if isinstance(bin_ms, bool) or int(bin_ms) != bin_ms or bin_ms <= 0:
        raise ValueError("bin_ms must be a positive integer")
    if times.size == 0:
        raise ValueError("timestamps must not be empty")
    if not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("timestamps must be finite and non-negative")
    if np.any(np.diff(times) < 0):
        raise ValueError("timestamps must be monotonically nondecreasing")
    # The timestamp extent does not establish a complete bin at the maximum
    # timestamp.  Keep only [0, floor(max/bin)*bin), dropping the tail.
    n_bins = int(math.floor(float(times[-1]) / bin_ms))
    if n_bins < 1:
        raise ValueError(f"trace is shorter than one complete {bin_ms} ms bin")
    counts = np.bincount(np.floor(times / bin_ms).astype(np.int64), minlength=n_bins)
    counts = counts[:n_bins]
    t_ms = np.arange(n_bins, dtype=np.int64) * int(bin_ms)
    rates = counts.astype(np.float64) * (BYTES_PER_OPPORTUNITY * 8 * 1000.0 / bin_ms)
    return t_ms, rates


def split_rates(t_ms: np.ndarray, rates_bps: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split one ID trace contiguously into 70%/15%/15% train/val/test."""
    t = np.asarray(t_ms, dtype=np.int64)
    r = np.asarray(rates_bps, dtype=np.float64)
    if t.ndim != 1 or r.ndim != 1 or t.size != r.size or t.size == 0:
        raise ValueError("t_ms and rates_bps must be non-empty 1-D arrays of equal length")
    if not np.isfinite(r).all() or np.any(r < 0):
        raise ValueError("rates_bps must be finite and non-negative")
    n_train = int(math.floor(0.70 * t.size))
    n_val = int(math.floor(0.15 * t.size))
    edges = (0, n_train, n_train + n_val, t.size)
    return {name: (t[a:b] - (t[a] if b > a else 0), r[a:b].copy())
            for name, a, b in zip(("train", "val", "test"), edges[:-1], edges[1:], strict=True)}


def _write_trace(path: Path, t_ms: np.ndarray, rates_bps: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("t_ms", "rate_bps"))
        writer.writerows(zip(t_ms.tolist(), rates_bps.tolist(), strict=True))


def process_directory(source_dir: str | Path, output_dir: str | Path) -> dict[str, dict[str, float | int]]:
    """Process the five named traces into split CSVs and a deterministic summary."""
    source, output = Path(source_dir), Path(output_dir)
    summary: dict[str, dict[str, float | int]] = {}
    for name in TRACE_FILES:
        raw = read_up(source / f"{name}.up")
        t_ms, rates = bin_opportunities(raw)
        outputs = split_rates(t_ms, rates) if name in TRACE_FILES[:3] else {"ood": (t_ms, rates)}
        empty = [split for split, (_, values) in outputs.items() if values.size == 0]
        if empty:
            raise ValueError(f"{name}: empty {', '.join(empty)} split; source trace has only {len(rates)} bins")
        for split, (split_t, split_rates_bps) in outputs.items():
            filename = f"{name}_{split}.csv"
            _write_trace(output / filename, split_t, split_rates_bps)
            summary[filename] = {
                "n_bins": int(split_rates_bps.size),
                "duration_ms": int(split_rates_bps.size * BIN_MS),
                "mean_bps": float(np.mean(split_rates_bps)),
                "p05_bps": float(np.percentile(split_rates_bps, 5)),
                "p95_bps": float(np.percentile(split_rates_bps, 95)),
            }
    output.mkdir(parents=True, exist_ok=True)
    (output / "traces_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


__all__ = ["TRACE_FILES", "BIN_MS", "read_up", "bin_opportunities", "split_rates", "process_directory"]
