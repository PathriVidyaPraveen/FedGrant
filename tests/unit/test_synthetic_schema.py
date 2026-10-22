"""Tests for ml/schema.py (recorder, writer, loader) and ml/synthetic.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fedgrant.ml import synthetic as syn
from fedgrant.ml.schema import (
    ALL_FIELDS,
    EVENT_FIELDS,
    RUN_FIELDS,
    SAMPLE_FIELDS,
    SplitRecorder,
    load_split,
    meta,
    validate_split,
)

MS = 1_000_000
OBS1 = np.array([1, .2, .1, .5, .1, .3, .2, 0], np.float32)
OBS0 = np.array([0, 1, .1, .5, .1, .3, .2, 0], np.float32)
X = np.zeros(48, np.float32)


def _small(tmp: Path, split: str = "val", seed: int = 2000, runs: int = 1) -> Path:
    # N = 20 keeps D = beta * T_full realistic (fewer clients shrink D below the initial srtt)
    p = syn.generate_split(split, runs=runs, seed=seed, out=tmp, n_clients=20, n_rounds=2, verbose=False)
    with np.load(p, allow_pickle=False) as z:
        assert z["y"].size > 200, "test data must not be empty"
    return p


# --------------------------------------------------------------------------- scenario helpers


def test_deadline_matches_spec_values() -> None:
    assert [syn.deadline_ns(b) // MS for b in (0.05, 0.10, 0.20)] == [199, 398, 797]
    assert syn.deadline_ns(0.10, 50) // MS == 996


def test_ge_params() -> None:
    p, r = syn.ge_params(8, 0.05)
    assert r == 0.125 and p / (p + r) == pytest.approx(0.05)
    assert syn.ge_params(32, 0.05)[0] == pytest.approx(0.001645, rel=1e-3)   # L3 in Section 5.4


# --------------------------------------------------------------------------- recorder


def test_recorder_builds_windows_from_times() -> None:
    rec = SplitRecorder("val")
    rec.begin_run(2000, 2000, burst=8, pi=0.05, beta=0.1, D_ns=398 * MS, capacity="constant")
    h1 = rec.add_sample(2000, 3, 10, X, 0.9, 0.9, 0.5)
    h2 = rec.add_sample(2000, 3, 20, X, 0.9, 0.9, 0.5)
    rec.add_event(h1, 1, 20, OBS1)                      # resolves exactly at t_g of h2: excluded
    h3 = rec.add_sample(2000, 3, 21, X, 0.9, 0.9, 0.5)
    h4 = rec.add_sample(2000, 1, 22, X, 0.9, 0.9, 0.5)  # another client: its own group
    rec.add_event(h2, 0, 30, OBS0)
    rec.add_event(h3, 1, 30, OBS1)
    rec.add_event(h4, 1, 40, OBS1)
    rec.end_run(2000)
    d = rec.arrays()
    validate_split(d)
    # groups sorted by (run, client): client 1 first (1 event), then client 3 (3 events)
    assert d["event_client"].tolist() == [1, 3, 3, 3]
    assert d["event_t_ns"].tolist() == [40, 20, 30, 30]
    assert d["sample_len"].tolist() == [0, 0, 1, 0]
    assert d["sample_end"].tolist() == [1, 1, 2, 0]
    assert d["y"].tolist() == [1, 0, 1, 1]


def test_recorder_rejects_misuse() -> None:
    rec = SplitRecorder("val")
    with pytest.raises(ValueError):
        rec.begin_run(1, 1500, burst=8, pi=0.05, beta=0.1, D_ns=1, capacity="constant")   # train seed
    rec.begin_run(2001, 2001, burst=8, pi=0.05, beta=0.1, D_ns=1, capacity="constant")
    with pytest.raises(ValueError):
        rec.begin_run(2001, 2001, burst=8, pi=0.05, beta=0.1, D_ns=1, capacity="constant")
    h = rec.add_sample(2001, 0, 100, X, 0.9, 0.9, 0.9)
    with pytest.raises(ValueError):
        rec.add_event(h, 1, 50, OBS1)                     # before issue
    with pytest.raises(ValueError):
        rec.add_event(h, 0, 150, OBS1)                    # obs[0] != y
    rec.add_event(h, 1, 150, OBS1)
    with pytest.raises(ValueError):
        rec.add_event(h, 1, 160, OBS1)                    # second resolution of one grant
    h2 = rec.add_sample(2001, 0, 120, X, 0.9, 0.9, 0.9)
    with pytest.raises(ValueError):
        rec.add_event(h2, 1, 140, OBS1)                   # out of resolution order for the client
    with pytest.raises(ValueError):
        rec.end_run(2001)                                 # h2 unresolved
    with pytest.raises(ValueError):
        rec.arrays()                                      # the run is still open after that error
    with pytest.raises(ValueError):
        rec.add_sample(2001, 0, 130, X, 1.5, 0.9, 0.9)    # prediction outside [0, 1]
    with pytest.raises(ValueError):
        SplitRecorder("dev")
    with pytest.raises(ValueError):
        rec.arrays()                                      # run still open


# --------------------------------------------------------------------------- written files


def test_written_file_has_exact_schema(tmp_path: Path) -> None:
    p = _small(tmp_path)
    with np.load(p, allow_pickle=False) as z:
        assert set(z.files) == set(ALL_FIELDS)
        for group in (EVENT_FIELDS, SAMPLE_FIELDS, RUN_FIELDS):
            for k, (dt, tail) in group.items():
                assert z[k].dtype == dt and z[k].shape[1:] == tail, k
    d = load_split(p)
    m = meta(d)
    assert m["split"] == "val" and m["seed_range"] == [2000, 2100] and m["n_runs"] == 1
    assert d["run_seed"].tolist() == [2000] and d["run_id"].tolist() == [2000]
    assert d["run_capacity"].tolist() == ["constant"]
    assert d["run_D_ns"][0] in (199 * MS, 398 * MS, 797 * MS)
    assert d["y"].size > 100 and 0.0 < d["y"].mean() < 1.0
    np.testing.assert_array_equal(d["ctx"], d["x48"][:, 32:48])


def test_deterministic_given_seed(tmp_path: Path) -> None:
    a = load_split(_small(tmp_path / "a"))
    b = load_split(_small(tmp_path / "b"))
    for k in ALL_FIELDS:
        assert np.array_equal(a[k], b[k]), k
    c = load_split(_small(tmp_path / "c", seed=2001))
    assert c["y"].size != a["y"].size or not np.array_equal(c["x48"], a["x48"])


def test_shards_concatenate_like_one_file(tmp_path: Path) -> None:
    one = load_split(_small(tmp_path / "one", runs=2))
    sh = tmp_path / "sharded"
    sh.mkdir()
    for k, seed in enumerate((2000, 2001)):
        p = _small(tmp_path / f"s{k}", seed=seed)
        p.rename(sh / f"shard_{k:03d}.npz")
    two = load_split(sh)
    for k in (*EVENT_FIELDS, *SAMPLE_FIELDS, *RUN_FIELDS):
        assert np.array_equal(one[k], two[k]), k
    assert meta(two)["n_shards"] == 2


def test_oracle_consistent_with_network(tmp_path: Path) -> None:
    d = load_split(_small(tmp_path, split="test_id", seed=3000, runs=2))
    po = d["p_oracle"]
    assert np.all(d["y"][po == 0] == 0)           # predicted late -> never delivered in time
    assert np.any(po == 0) and np.any(po > 0)
    for k, rid in enumerate(d["run_id"]):
        p, r = syn.ge_params(int(d["run_burst"][k]), float(d["run_pi"][k]))
        run = d["sample_run"] == rid
        vals = set(np.unique(po[run][po[run] > 0]).tolist())
        assert vals <= {float(np.float32(1 - p)), float(np.float32(r))}   # Pr[next G | true state]


def test_ood_scenarios_and_seed_ranges(tmp_path: Path) -> None:
    d = load_split(_small(tmp_path, split="test_ood", seed=4000, runs=3))
    assert set(d["run_burst"].tolist()) <= {8, 32} and set(d["run_pi"].tolist()) <= {0.05, 0.15}
    with pytest.raises(ValueError):
        syn.generate_split("val", runs=2, seed=2099, out=tmp_path, verbose=False)    # leaves the range


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    syn.main(["--split", "val", "--runs", "1", "--seed", "2005", "--out", str(tmp_path),
              "--rounds", "1"])
    d = load_split(tmp_path / "val.npz")
    assert d["run_seed"].tolist() == [2005]
    assert "wrote" in capsys.readouterr().out


def test_validate_catches_corruption(tmp_path: Path) -> None:
    d = load_split(_small(tmp_path))
    bad = dict(d)
    bad["sample_len"] = d["sample_len"].copy()
    bad["sample_len"][0] = 17
    with pytest.raises(ValueError):
        validate_split(bad)
    bad = dict(d)
    bad["ctx"] = d["ctx"] + 1
    with pytest.raises(ValueError):
        validate_split(bad)
    bad = dict(d)
    bad["event_t_ns"] = d["event_t_ns"][::-1].copy()
    with pytest.raises(ValueError):
        validate_split(bad)