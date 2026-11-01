"""Deterministic trace selection, wrapping, scaling, and flooring."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

POOL_ID = ("Verizon-LTE-driving", "Verizon-LTE-short", "ATT-LTE-driving")
POOL_OOD = ("TMobile-UMTS-driving", "Verizon-EVDO-driving")
MEAN_BPS = (2_000_000.0, 4_000_000.0, 6_000_000.0)
FLOOR_BPS = 100_000.0
MAX_CLIENTS_FOR_OFFSETS = 50


def offset_fraction(seed: int, client_id: int) -> float:
    """The stable per-client offset draw, independent of the experiment's N."""
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if not 0 <= client_id < MAX_CLIENTS_FOR_OFFSETS:
        raise ValueError(f"client_id must be in [0, {MAX_CLIENTS_FOR_OFFSETS - 1}]")
    rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(6)[0])
    return float(rng.random(MAX_CLIENTS_FOR_OFFSETS)[client_id])


def load_rate_csv(path: str | Path) -> np.ndarray:
    """Read a trace CSV and validate its 100 ms rate column."""
    p = Path(path)
    try:
        with p.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not {"t_ms", "rate_bps"}.issubset(reader.fieldnames):
                raise ValueError(f"{p} must contain t_ms,rate_bps columns")
            rows = list(reader)
        times = np.asarray([int(row["t_ms"]) for row in rows], dtype=np.int64)
        rates = np.asarray([float(row["rate_bps"]) for row in rows], dtype=np.float64)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"cannot read trace CSV {p}: {exc}") from exc
    if not rates.size or not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError(f"{p} rates must be non-empty, finite, and non-negative")
    if times.size != rates.size or np.any(np.diff(times) != 100):
        raise ValueError(f"{p} t_ms must be contiguous 100 ms bins")
    return rates


def client_trace(
    client_id: int,
    split: str,
    seed: int,
    *,
    data_dir: str | Path = "data/traces",
    mean_rate_bps: float | None = None,
    pool: str | None = None,
    floor_bps: float = FLOOR_BPS,
) -> np.ndarray:
    """Load, rotate, scale, and floor the selected client's rate trace.

    ``split`` is train/val/test for the ID pool and ``ood`` for the OOD pool.
    If ``pool`` is omitted, ``split='ood'`` selects OOD and all other valid
    splits select ID. An explicit pool must be ``id`` or ``ood``.
    """
    if isinstance(client_id, bool) or int(client_id) != client_id or client_id < 0 or client_id >= MAX_CLIENTS_FOR_OFFSETS:
        raise ValueError(f"client_id must be in [0, {MAX_CLIENTS_FOR_OFFSETS - 1}]")
    selected_pool = pool or ("ood" if split == "ood" else "id")
    if selected_pool not in {"id", "ood"}:
        raise ValueError("pool must be 'id' or 'ood'")
    if selected_pool == "id" and split not in {"train", "val", "test"}:
        raise ValueError("ID split must be train, val, or test")
    if selected_pool == "ood" and split not in {"ood", "train", "val", "test"}:
        raise ValueError("OOD split must be ood (or a legacy split name)")
    names = POOL_OOD if selected_pool == "ood" else POOL_ID
    file_split = "ood" if selected_pool == "ood" else split
    path = Path(data_dir) / f"{names[client_id % len(names)]}_{file_split}.csv"
    segment = load_rate_csv(path)
    target_mean = float(MEAN_BPS[client_id % 3] if mean_rate_bps is None else mean_rate_bps)
    if not np.isfinite(target_mean) or target_mean <= 0:
        raise ValueError("mean_rate_bps must be finite and positive")
    if not np.isfinite(floor_bps) or floor_bps < 0:
        raise ValueError("floor_bps must be finite and non-negative")
    start = int(offset_fraction(seed, client_id) * segment.size)
    rotated = np.roll(segment, -start)
    scaled = rotated * (target_mean / float(segment.mean())) if segment.mean() > 0 else np.full_like(segment, target_mean)
    return np.maximum(scaled, floor_bps)


__all__ = ["POOL_ID", "POOL_OOD", "MEAN_BPS", "FLOOR_BPS", "offset_fraction", "client_trace"]
