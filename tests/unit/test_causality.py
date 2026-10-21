"""Causality and split-isolation checks on a generated ML split (Section 4.5).

The same checks run on synthetic data now and on the real simulator's data later:
set FEDGRANT_ML_SPLITS to a comma-separated list of split files or shard
directories, e.g.
    FEDGRANT_ML_SPLITS=data/mldata/train.npz,data/mldata/val.npz pytest tests/unit/test_causality.py
Without it, a small synthetic split is generated into a temporary directory.

These checks recompute everything from the raw arrays, independently of the
recorder that wrote them.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from fedgrant.ml.schema import SPLITS, load_split, meta
from fedgrant.ml.synthetic import generate_split

T = 16


@pytest.fixture(scope="module")
def split_paths(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    env = os.environ.get("FEDGRANT_ML_SPLITS")
    if env:
        return [Path(p) for p in env.split(",") if p]
    out = tmp_path_factory.mktemp("mldata")
    return [
        generate_split("val", runs=2, seed=2000, out=out, n_rounds=3, verbose=False),
        generate_split("test_id", runs=2, seed=3000, out=out, n_rounds=3, verbose=False),
    ]


@pytest.fixture(scope="module")
def splits(split_paths: list[Path]) -> list[dict[str, np.ndarray]]:
    out = [load_split(p) for p in split_paths]
    for d in out:                                   # the checks below must never pass vacuously
        assert d["y"].size > 0 and d["events"].shape[0] > 0
        assert np.any(d["sample_len"] == T) and np.any(d["sample_len"] < T)
    return out


def _window_ranges(d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    end = d["sample_end"].astype(np.int64)
    return end - d["sample_len"].astype(np.int64), end


def test_window_events_precede_the_grant_and_match_run_client(splits) -> None:
    for d in splits:
        start, end = _window_ranges(d)
        n = (end - start).astype(np.int64)
        assert np.all((n >= 0) & (n <= T))
        idx = np.repeat(np.arange(len(n)), n)                     # sample of each window slot
        ev = np.concatenate([np.arange(s, e) for s, e in zip(start, end)]) if n.sum() else np.zeros(0, int)
        assert np.all(d["event_t_ns"][ev] < d["sample_t_ns"][idx])          # strictly before t_g
        assert np.array_equal(d["event_run"][ev], d["sample_run"][idx])     # never crosses runs
        assert np.array_equal(d["event_client"][ev], d["sample_client"][idx])


def test_sample_len_is_min_16_prior_events(splits) -> None:
    for d in splits:
        key_e = d["event_run"].astype(np.int64) * 65536 + d["event_client"].astype(np.int64)
        key_s = d["sample_run"].astype(np.int64) * 65536 + d["sample_client"].astype(np.int64)
        for k in np.unique(key_s):
            ev = np.flatnonzero(key_e == k)
            sm = np.flatnonzero(key_s == k)
            t_ev = d["event_t_ns"][ev]
            assert np.all(np.diff(t_ev) >= 0)                     # resolution order
            prior = np.searchsorted(t_ev, d["sample_t_ns"][sm], side="left")   # count of t_e < t_g
            assert np.array_equal(d["sample_len"][sm].astype(np.int64), np.minimum(T, prior))
            if ev.size:
                # the window is the LAST min(16, prior) events before t_g
                assert np.array_equal(d["sample_end"][sm], ev[0] + prior)


def test_events_contiguous_per_run_client(splits) -> None:
    for d in splits:
        key = d["event_run"].astype(np.int64) * 65536 + d["event_client"].astype(np.int64)
        change = np.flatnonzero(np.diff(key)) + 1
        starts = np.concatenate([[0], change]) if key.size else np.zeros(0, int)
        assert len(np.unique(key)) == len(starts)                  # each (run, client) is one block


def test_one_event_per_sample_and_labels_agree(splits) -> None:
    """Every grant resolves exactly once, so each (run, client) has as many events as
    samples, and the outcome counts match (a late DATA never adds or edits an event)."""
    for d in splits:
        for r in np.unique(d["sample_run"]):
            for c in np.unique(d["sample_client"][d["sample_run"] == r]):
                sm = (d["sample_run"] == r) & (d["sample_client"] == c)
                ev = (d["event_run"] == r) & (d["event_client"] == c)
                assert sm.sum() == ev.sum()
                assert d["y"][sm].sum() == d["events"][ev, 0].sum()


def test_split_isolation(splits) -> None:
    seen: dict[int, str] = {}
    for d in splits:
        m = meta(d)
        lo, hi = SPLITS[m["split"]].start, SPLITS[m["split"]].stop
        assert np.all((d["run_seed"] >= lo) & (d["run_seed"] < hi))
        for s in d["run_seed"].tolist():
            assert seen.setdefault(s, m["split"]) == m["split"]    # no run in two splits


def test_times_monotone_within_run(splits) -> None:
    """Run-absolute clock: a later round never has earlier times."""
    for d in splits:
        for r in np.unique(d["sample_run"]):
            t = d["sample_t_ns"][d["sample_run"] == r]
            assert np.all(np.diff(t) >= 0)                          # samples recorded in issue order


def test_features_computed_before_the_grant(splits) -> None:
    """x48 is taken before the grant changes any state: a client's first grant of a run
    sees no outstanding grant of its own (feature 40 = 0), and the first grant of a run
    sees Phi = 0 (feature 41) and nothing delivered yet (feature 48)."""
    for d in splits:
        key = d["sample_run"].astype(np.int64) * 65536 + d["sample_client"].astype(np.int64)
        _, first = np.unique(key, return_index=True)
        assert np.all(d["x48"][first, 39] == 0.0)
        for r in np.unique(d["sample_run"]):
            i0 = np.flatnonzero(d["sample_run"] == r)[0]
            assert d["x48"][i0, 40] == 0.0 and d["x48"][i0, 47] == 0.0