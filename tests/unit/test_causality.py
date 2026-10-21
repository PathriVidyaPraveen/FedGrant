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


def _keys(d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(run, client) as one int64 key for events and for samples."""
    ke = d["event_run"].astype(np.int64) * 65536 + d["event_client"].astype(np.int64)
    ks = d["sample_run"].astype(np.int64) * 65536 + d["sample_client"].astype(np.int64)
    return ke, ks


def _prior_counts(d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Independent recount, per sample: (start of its (run, client) block in events,
    number of that block's events with resolution time < t_g). Fully vectorised:
    events are ordered by the composite (key, time) after the checks below."""
    ke, ks = _keys(d)
    times = np.unique(np.concatenate([d["event_t_ns"], d["sample_t_ns"]]))
    te = np.searchsorted(times, d["event_t_ns"])           # dense time ranks
    ts = np.searchsorted(times, d["sample_t_ns"])
    width = np.int64(times.size + 1)
    ce = ke * width + te                                   # lexicographic (key, time)
    cs = ks * width + ts
    assert np.all(np.diff(ce) >= 0)                        # grouped by key, time order inside
    lo = np.searchsorted(ke, ks, side="left")
    return lo, np.searchsorted(ce, cs, side="left") - lo


def test_events_grouped_and_in_resolution_order(splits) -> None:
    for d in splits:
        ke, _ = _keys(d)
        assert np.all(np.diff(ke) >= 0)                    # each (run, client) is one sorted block
        same = np.diff(ke) == 0
        assert np.all(np.diff(d["event_t_ns"])[same] >= 0)


def test_sample_len_is_min_16_prior_events(splits) -> None:
    for d in splits:
        lo, prior = _prior_counts(d)
        assert np.array_equal(d["sample_len"].astype(np.int64), np.minimum(T, prior))
        # the window is the LAST min(16, prior) events before t_g of the same block
        assert np.array_equal(d["sample_end"], lo + prior)


def test_window_events_precede_the_grant_and_match_run_client(splits) -> None:
    for d in splits:
        ke, ks = _keys(d)
        end = d["sample_end"].astype(np.int64)
        n = d["sample_len"].astype(np.int64)
        start = end - n
        assert np.all((n >= 0) & (n <= T)) and np.all(start >= 0) and np.all(end <= ke.size)
        w = n > 0
        # blocks are contiguous and time-sorted (checked above), so first/last slot suffice
        assert np.array_equal(ke[start[w]], ks[w]) and np.array_equal(ke[end[w] - 1], ks[w])
        assert np.all(d["event_t_ns"][end[w] - 1] < d["sample_t_ns"][w])     # strictly before t_g
        # and a direct slot-by-slot check on a sample of windows
        rng = np.random.default_rng(0)
        pick = rng.choice(np.flatnonzero(w), size=min(20_000, int(w.sum())), replace=False)
        slots = np.repeat(start[pick], n[pick]) + (np.arange(int(n[pick].sum()))
                                                   - np.repeat(np.cumsum(n[pick]) - n[pick], n[pick]))
        owner = np.repeat(pick, n[pick])
        assert np.all(d["event_t_ns"][slots] < d["sample_t_ns"][owner])
        assert np.array_equal(d["event_run"][slots], d["sample_run"][owner])        # never crosses runs
        assert np.array_equal(d["event_client"][slots], d["sample_client"][owner])


def test_one_event_per_sample_and_labels_agree(splits) -> None:
    """Every grant resolves exactly once, so each (run, client) has as many events as
    samples, and the outcome counts match (a late DATA never adds or edits an event)."""
    for d in splits:
        ke, ks = _keys(d)
        u, inv = np.unique(np.concatenate([ke, ks]), return_inverse=True)
        ie, is_ = inv[: ke.size], inv[ke.size:]
        assert np.array_equal(np.bincount(ie, minlength=u.size), np.bincount(is_, minlength=u.size))
        y_ev = np.bincount(ie, weights=d["events"][:, 0].astype(np.float64), minlength=u.size)
        y_s = np.bincount(is_, weights=d["y"].astype(np.float64), minlength=u.size)
        assert np.array_equal(y_ev, y_s)


def test_split_isolation(splits) -> None:
    seen: dict[int, str] = {}
    for d in splits:
        m = meta(d)
        lo, hi = SPLITS[m["split"]].start, SPLITS[m["split"]].stop
        assert np.all((d["run_seed"] >= lo) & (d["run_seed"] < hi))
        for s in d["run_seed"].tolist():
            assert seen.setdefault(s, m["split"]) == m["split"]    # no run in two splits


def test_times_monotone_within_run(splits) -> None:
    """Run-absolute clock and issue order: within a run, sample times never decrease."""
    for d in splits:
        order = np.argsort(d["sample_run"], kind="stable")
        same = np.diff(d["sample_run"][order]) == 0
        assert np.all(np.diff(d["sample_t_ns"][order])[same] >= 0)


def test_features_computed_before_the_grant(splits) -> None:
    """x48 is taken before the grant changes any state: a client's first grant of a run
    sees no outstanding grant of its own (feature 40 = 0), and the first grant of a run
    sees Phi = 0 (feature 41) and nothing delivered yet (feature 48)."""
    for d in splits:
        _, ks = _keys(d)
        _, first = np.unique(ks, return_index=True)
        assert np.all(d["x48"][first, 39] == 0.0)
        _, first_run = np.unique(d["sample_run"], return_index=True)
        assert np.all(d["x48"][first_run, 40] == 0.0) and np.all(d["x48"][first_run, 47] == 0.0)