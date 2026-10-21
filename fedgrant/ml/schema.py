"""ML dataset schema (Section 4.5): field definitions, recorder, writer, loader, validator.

The single source of truth for docs/ML_DATASET_SCHEMA.md. The real generator
(ml/datagen.py) and the synthetic one (ml/synthetic.py) both write through
SplitRecorder, so the on-disk layout cannot drift between them.

No PyTorch here: this module is imported by the simulator side as well.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fedgrant.policy.constants import CTX_SLICE, N_CTX, N_OBS, N_X48, T_EVBUF

SCHEMA_VERSION: int = 1

SPLITS: dict[str, range] = {           # disjoint seed ranges, Section 4.5
    "train": range(1000, 2000),
    "val": range(2000, 2100),
    "test_id": range(3000, 3100),
    "test_ood": range(4000, 4100),
}

# name -> (dtype, trailing shape); leading dimension is E (events) or ns (samples) or R (runs)
EVENT_FIELDS: dict[str, tuple[np.dtype, tuple[int, ...]]] = {
    "events": (np.dtype(np.float32), (N_OBS,)),
    "event_t_ns": (np.dtype(np.int64), ()),
    "event_run": (np.dtype(np.int32), ()),
    "event_client": (np.dtype(np.int16), ()),
}
SAMPLE_FIELDS: dict[str, tuple[np.dtype, tuple[int, ...]]] = {
    "sample_end": (np.dtype(np.int64), ()),
    "sample_len": (np.dtype(np.int8), ()),
    "sample_t_ns": (np.dtype(np.int64), ()),
    "sample_run": (np.dtype(np.int32), ()),
    "sample_client": (np.dtype(np.int16), ()),
    "ctx": (np.dtype(np.float32), (N_CTX,)),
    "x48": (np.dtype(np.float32), (N_X48,)),
    "y": (np.dtype(np.uint8), ()),
    "p_ewma": (np.dtype(np.float32), ()),
    "p_hmm": (np.dtype(np.float32), ()),
    "p_oracle": (np.dtype(np.float32), ()),
}
RUN_FIELDS: dict[str, tuple[np.dtype, tuple[int, ...]]] = {
    "run_id": (np.dtype(np.int32), ()),
    "run_seed": (np.dtype(np.int64), ()),
    "run_burst": (np.dtype(np.int32), ()),
    "run_pi": (np.dtype(np.float64), ()),
    "run_beta": (np.dtype(np.float64), ()),
    "run_D_ns": (np.dtype(np.int64), ()),
    "run_capacity": (np.dtype("<U32"), ()),
}
META_FIELD = "meta_json"               # 0-d unicode array: {"schema_version", "split", "seed_range", "generator", ...}
ALL_FIELDS = (*EVENT_FIELDS, *SAMPLE_FIELDS, *RUN_FIELDS, META_FIELD)


# --------------------------------------------------------------------------- recorder


@dataclass(slots=True)
class _Sample:
    run: int
    client: int
    t_ns: int
    x48: np.ndarray
    p_ewma: float
    p_hmm: float
    p_oracle: float
    y: int | None = None


@dataclass(slots=True)
class _Group:
    """Events of one (run, client), in resolution order."""

    t: list[int] = field(default_factory=list)
    obs: list[np.ndarray] = field(default_factory=list)


class SplitRecorder:
    """Collects one split while a generator runs; save() writes the schema.

    Usage (all times are run-absolute ns, monotonic within a run):
        rec.begin_run(run_id, seed, burst=..., pi=..., beta=..., D_ns=..., capacity=...)
        h = rec.add_sample(run_id, client, t_g, x48, p_ewma, p_hmm, p_oracle)   # at grant issue
        rec.add_event(h, y, t_e, obs)                                          # at its resolution
        rec.end_run(run_id)
        rec.save(path)

    Every sample gets exactly one event (its own grant's resolution); the event
    sets the sample's label y. Windows are built at save time from event times,
    so the caller cannot get them wrong.
    """

    def __init__(self, split: str, seed_range: range | None = None, generator: str = "") -> None:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLITS)}")
        self.split = split
        self.seed_range = SPLITS[split] if seed_range is None else seed_range
        self.generator = generator
        self._samples: list[_Sample] = []
        self._groups: dict[tuple[int, int], _Group] = {}
        self._runs: dict[int, dict[str, Any]] = {}
        self._open: set[int] = set()
        self._done: set[int] = set()

    # -- runs
    def begin_run(self, run_id: int, seed: int, *, burst: int, pi: float, beta: float,
                  D_ns: int, capacity: str) -> None:
        run_id, seed = int(run_id), int(seed)
        if run_id in self._runs:
            raise ValueError(f"run {run_id} already recorded")
        if seed not in SPLITS[self.split]:
            raise ValueError(f"seed {seed} outside the {self.split} range {SPLITS[self.split]}")
        self._runs[run_id] = dict(run_id=run_id, run_seed=seed, run_burst=int(burst), run_pi=float(pi),
                                  run_beta=float(beta), run_D_ns=int(D_ns), run_capacity=str(capacity))
        self._open.add(run_id)

    def end_run(self, run_id: int) -> None:
        """Close a run; every one of its samples must have been resolved."""
        run_id = int(run_id)
        if run_id not in self._open:
            raise ValueError(f"run {run_id} is not open")
        missing = sum(1 for s in self._samples if s.run == run_id and s.y is None)
        if missing:
            raise ValueError(f"run {run_id}: {missing} samples were never resolved")
        self._open.remove(run_id)
        self._done.add(run_id)

    # -- samples and events
    def add_sample(self, run_id: int, client: int, t_ns: int, x48: np.ndarray,
                   p_ewma: float, p_hmm: float, p_oracle: float) -> int:
        """Record a grant issued at t_ns (features computed BEFORE the grant changes any state).
        Returns a handle to pass to add_event when the grant resolves."""
        run_id = int(run_id)
        if run_id not in self._open:
            raise ValueError(f"run {run_id} is not open")
        x = np.asarray(x48, dtype=np.float32)
        if x.shape != (N_X48,) or not np.all(np.isfinite(x)):
            raise ValueError("x48 must be 48 finite values")
        ps = (float(p_ewma), float(p_hmm), float(p_oracle))
        if not all(0.0 <= p <= 1.0 for p in ps):
            raise ValueError(f"predictions must be in [0, 1], got {ps}")
        self._samples.append(_Sample(run_id, int(client), int(t_ns), x.copy(), *ps))
        return len(self._samples) - 1

    def add_event(self, handle: int, y: int, t_ns: int, obs: np.ndarray) -> None:
        """Record the resolution of the grant `handle` (obs from features.resolve_event)."""
        s = self._samples[handle]
        if s.y is not None:
            raise ValueError(f"sample {handle} already resolved; a late DATA never edits an event")
        if y not in (0, 1):
            raise ValueError(f"y must be 0 or 1, got {y!r}")
        t = int(t_ns)
        if t < s.t_ns:
            raise ValueError(f"resolution at {t} before issue at {s.t_ns}")
        o = np.asarray(obs, dtype=np.float32)
        if o.shape != (N_OBS,) or not np.all(np.isfinite(o)) or float(o[0]) != float(y):
            raise ValueError("obs must be 8 finite values with obs[0] == y")
        g = self._groups.setdefault((s.run, s.client), _Group())
        if g.t and t < g.t[-1]:
            raise ValueError(f"events of (run {s.run}, client {s.client}) must arrive in resolution order")
        g.t.append(t)
        g.obs.append(o.copy())
        s.y = int(y)

    # -- output
    def arrays(self) -> dict[str, np.ndarray]:
        """Build all schema arrays (see docs/ML_DATASET_SCHEMA.md)."""
        if self._open:
            raise ValueError(f"runs still open: {sorted(self._open)}")
        if any(s.y is None for s in self._samples):
            raise ValueError("unresolved samples")
        keys = sorted(self._groups)                       # (run, client) order
        start: dict[tuple[int, int], int] = {}
        ev_obs, ev_t, ev_run, ev_cli = [], [], [], []
        pos = 0
        for k in keys:
            g = self._groups[k]
            start[k] = pos
            ev_obs.extend(g.obs)
            ev_t.extend(g.t)
            ev_run.extend([k[0]] * len(g.t))
            ev_cli.extend([k[1]] * len(g.t))
            pos += len(g.t)
        E = pos
        out: dict[str, np.ndarray] = {
            "events": np.array(ev_obs, dtype=np.float32).reshape(E, N_OBS),
            "event_t_ns": np.array(ev_t, dtype=np.int64),
            "event_run": np.array(ev_run, dtype=np.int32),
            "event_client": np.array(ev_cli, dtype=np.int16),
        }
        ns = len(self._samples)
        end = np.empty(ns, np.int64)
        ln = np.empty(ns, np.int8)
        for i, s in enumerate(self._samples):
            k = (s.run, s.client)
            grp = self._groups.get(k)
            if grp is None:                                # (run, client) without events: empty window
                j = bisect_left(keys, k)                   # at the position its group would occupy
                end[i] = start[keys[j]] if j < len(keys) else E
                ln[i] = 0
                continue
            n_prior = bisect_left(grp.t, s.t_ns)           # events with resolution time < t_g
            end[i] = start[k] + n_prior
            ln[i] = min(T_EVBUF, n_prior)
        x48 = np.array([s.x48 for s in self._samples], dtype=np.float32).reshape(ns, N_X48)
        out.update(
            sample_end=end,
            sample_len=ln,
            sample_t_ns=np.array([s.t_ns for s in self._samples], dtype=np.int64),
            sample_run=np.array([s.run for s in self._samples], dtype=np.int32),
            sample_client=np.array([s.client for s in self._samples], dtype=np.int16),
            ctx=np.ascontiguousarray(x48[:, CTX_SLICE]),
            x48=x48,
            y=np.array([s.y for s in self._samples], dtype=np.uint8),
            p_ewma=np.array([s.p_ewma for s in self._samples], dtype=np.float32),
            p_hmm=np.array([s.p_hmm for s in self._samples], dtype=np.float32),
            p_oracle=np.array([s.p_oracle for s in self._samples], dtype=np.float32),
        )
        runs = [self._runs[r] for r in sorted(self._runs)]
        for name, (dt, _) in RUN_FIELDS.items():
            out[name] = np.array([r[name] for r in runs], dtype=dt)
        meta = dict(schema_version=SCHEMA_VERSION, split=self.split,
                    seed_range=[self.seed_range.start, self.seed_range.stop],
                    generator=self.generator, n_runs=len(runs), n_events=E, n_samples=ns)
        out[META_FIELD] = np.array(json.dumps(meta, sort_keys=True))
        return out

    def save(self, path: str | Path) -> Path:
        """Write one compressed .npz (path ending in .npz) and validate it."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        arrs = self.arrays()
        validate_split(arrs)
        np.savez_compressed(p, **arrs)  # type: ignore[arg-type]  # numpy stub types **kwds as bool
        return p


# --------------------------------------------------------------------------- loading


def _load_npz(p: Path) -> dict[str, np.ndarray]:
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_split(path: str | Path) -> dict[str, np.ndarray]:
    """Load a split: one .npz, or a directory of shard_*.npz files (each self-contained:
    whole runs, sample_end indexing its own events). Shards are concatenated in
    sorted file-name order with sample_end offset. The result is validated."""
    p = Path(path)
    files = sorted(p.glob("shard_*.npz")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"no shard_*.npz in {p}")
    parts = [_load_npz(f) for f in files]
    if len(parts) == 1:
        d = parts[0]
    else:
        metas = [json.loads(str(x[META_FIELD])) for x in parts]
        if len({m["split"] for m in metas}) != 1:
            raise ValueError("shards belong to different splits")
        d = {}
        offset = 0
        ends = []
        for x in parts:
            ends.append(x["sample_end"] + offset)
            offset += x["events"].shape[0]
        for k in (*EVENT_FIELDS, *SAMPLE_FIELDS, *RUN_FIELDS):
            d[k] = np.concatenate([x[k] for x in parts], axis=0)
        d["sample_end"] = np.concatenate(ends)
        meta = dict(metas[0])
        meta.update(n_runs=int(d["run_id"].size), n_events=int(d["events"].shape[0]),
                    n_samples=int(d["y"].size), n_shards=len(parts))
        d[META_FIELD] = np.array(json.dumps(meta, sort_keys=True))
    validate_split(d)
    return d


def meta(d: dict[str, np.ndarray]) -> dict[str, Any]:
    return json.loads(str(d[META_FIELD]))


# --------------------------------------------------------------------------- validation


def validate_split(d: dict[str, np.ndarray]) -> None:
    """Structural checks of the schema (dtypes, shapes, ranges, grouping, window bounds).
    The causality rules are checked independently in tests/unit/test_causality.py."""
    missing = [k for k in ALL_FIELDS if k not in d]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    E = d["events"].shape[0]
    ns = d["y"].shape[0]
    R = d["run_id"].shape[0]
    for fields_, n in ((EVENT_FIELDS, E), (SAMPLE_FIELDS, ns), (RUN_FIELDS, R)):
        for k, (dt, tail) in fields_.items():
            a = d[k]
            if a.dtype != dt or a.shape != (n, *tail):
                raise ValueError(f"{k}: expected {dt} {(n, *tail)}, got {a.dtype} {a.shape}")
    m = meta(d)
    if m.get("schema_version") != SCHEMA_VERSION or m.get("split") not in SPLITS:
        raise ValueError(f"bad meta: {m}")
    lo, hi = SPLITS[m["split"]].start, SPLITS[m["split"]].stop
    if R and not np.all((d["run_seed"] >= lo) & (d["run_seed"] < hi)):
        raise ValueError(f"run seeds outside the {m['split']} range [{lo}, {hi})")
    if len(np.unique(d["run_id"])) != R:
        raise ValueError("duplicate run ids")
    for a, k in ((d["event_run"], "event_run"), (d["sample_run"], "sample_run")):
        if a.size and not np.all(np.isin(a, d["run_id"])):
            raise ValueError(f"{k} refers to an unknown run")
    for k in ("events", "x48", "ctx", "p_ewma", "p_hmm", "p_oracle"):
        if not np.all(np.isfinite(d[k])):
            raise ValueError(f"{k} has non-finite values")
    if not np.array_equal(d["ctx"], d["x48"][:, CTX_SLICE]):
        raise ValueError("ctx != x48[:, 32:48]")
    if not np.all(d["y"] <= 1) or not np.all(np.isin(d["events"][:, 0], (0.0, 1.0))):
        raise ValueError("labels must be 0/1")
    for k in ("p_ewma", "p_hmm", "p_oracle"):
        if not np.all((d[k] >= 0) & (d[k] <= 1)):
            raise ValueError(f"{k} outside [0, 1]")
    if E:
        key = d["event_run"].astype(np.int64) * 65536 + d["event_client"].astype(np.int64)
        if np.any(np.diff(key) < 0):
            raise ValueError("events are not grouped by (run, client) in sorted order")
        same = np.diff(key) == 0
        if np.any(np.diff(d["event_t_ns"])[same] < 0):
            raise ValueError("events of a (run, client) are not in resolution order")
    ln = d["sample_len"].astype(np.int64)
    end = d["sample_end"]
    if np.any(ln < 0) or np.any(ln > T_EVBUF) or np.any(end < ln) or np.any(end > E):
        raise ValueError("sample_end / sample_len out of bounds")