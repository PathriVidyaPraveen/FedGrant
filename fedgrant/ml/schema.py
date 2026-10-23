"""ML dataset schema (Section 4.5): field definitions, recorder, writer, loader, validator.

The single source of truth for docs/ML_DATASET_SCHEMA.md. The real generator
(ml/datagen.py) and the synthetic one (ml/synthetic.py) both write through
SplitRecorder, so the on-disk layout cannot drift between them.

No PyTorch here: this module is imported by the simulator side as well.
"""

from __future__ import annotations

import json
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


class _Buf:
    """Growable NumPy buffer (amortised O(1) append, no per-row Python objects)."""

    __slots__ = ("a", "n")

    def __init__(self, dtype: np.dtype | type, tail: tuple[int, ...] = (), cap: int = 1024) -> None:
        self.a = np.empty((cap, *tail), dtype=dtype)
        self.n = 0

    def append(self, row: object) -> int:
        if self.n == self.a.shape[0]:
            grown = np.empty((2 * self.n, *self.a.shape[1:]), dtype=self.a.dtype)
            grown[: self.n] = self.a
            self.a = grown
        self.a[self.n] = row
        self.n += 1
        return self.n - 1

    def view(self) -> np.ndarray:
        return self.a[: self.n]


_UNRESOLVED = 255                      # y sentinel until the grant resolves


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

    Memory: about 260 bytes per sample (x48 dominates), held in NumPy buffers.
    For the 4M-sample train split, write shards (one recorder per group of runs,
    saved as <split>/shard_000.npz, ...) to bound peak memory; see the schema doc.
    """

    def __init__(self, split: str, generator: str = "") -> None:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLITS)}")
        self.split = split
        self.generator = generator
        # samples
        self._s_run = _Buf(np.int32)
        self._s_client = _Buf(np.int16)
        self._s_t = _Buf(np.int64)
        self._s_x48 = _Buf(np.float32, (N_X48,))
        self._s_p = _Buf(np.float32, (3,))
        self._s_y = _Buf(np.uint8)
        # events, in arrival order (regrouped at save time)
        self._e_obs = _Buf(np.float32, (N_OBS,))
        self._e_t = _Buf(np.int64)
        self._e_run = _Buf(np.int32)
        self._e_client = _Buf(np.int16)
        self._last_t: dict[tuple[int, int], int] = {}      # last event time per (run, client)
        self._runs: dict[int, dict[str, Any]] = {}
        self._unresolved: dict[int, int] = {}              # open run -> unresolved samples

    # -- runs
    def begin_run(self, run_id: int, seed: int, *, burst: int, pi: float, beta: float,
                  D_ns: int, capacity: str) -> None:
        run_id, seed = int(run_id), int(seed)
        if run_id in self._runs:
            raise ValueError(f"run {run_id} already recorded")
        if seed not in SPLITS[self.split]:
            raise ValueError(f"seed {seed} outside the {self.split} range {SPLITS[self.split]}")
        if not 0 <= run_id <= np.iinfo(np.int32).max:
            raise ValueError(f"run id {run_id} does not fit int32")
        self._runs[run_id] = dict(run_id=run_id, run_seed=seed, run_burst=int(burst), run_pi=float(pi),
                                  run_beta=float(beta), run_D_ns=int(D_ns), run_capacity=str(capacity))
        self._unresolved[run_id] = 0

    def end_run(self, run_id: int) -> None:
        """Close a run; every one of its samples must have been resolved."""
        run_id = int(run_id)
        if run_id not in self._unresolved:
            raise ValueError(f"run {run_id} is not open")
        if self._unresolved[run_id]:
            raise ValueError(f"run {run_id}: {self._unresolved[run_id]} samples were never resolved")
        del self._unresolved[run_id]

    # -- samples and events
    def add_sample(self, run_id: int, client: int, t_ns: int, x48: np.ndarray,
                   p_ewma: float, p_hmm: float, p_oracle: float) -> int:
        """Record a grant issued at t_ns (features computed BEFORE the grant changes any state).
        Returns a handle to pass to add_event when the grant resolves."""
        run_id, client = int(run_id), int(client)
        if run_id not in self._unresolved:
            raise ValueError(f"run {run_id} is not open")
        if not 0 <= client <= np.iinfo(np.int16).max:
            raise ValueError(f"client id {client} does not fit int16")
        x = np.asarray(x48, dtype=np.float32)
        if x.shape != (N_X48,) or not np.all(np.isfinite(x)):
            raise ValueError("x48 must be 48 finite values")
        ps = (float(p_ewma), float(p_hmm), float(p_oracle))
        if not all(0.0 <= p <= 1.0 for p in ps):
            raise ValueError(f"predictions must be in [0, 1], got {ps}")
        self._s_run.append(run_id)
        self._s_client.append(client)
        self._s_t.append(int(t_ns))
        self._s_x48.append(x)
        self._s_p.append(ps)
        self._unresolved[run_id] += 1
        return self._s_y.append(_UNRESOLVED)

    def add_event(self, handle: int, y: int, t_ns: int, obs: np.ndarray) -> None:
        """Record the resolution of the grant `handle` (obs from features.resolve_event)."""
        h = int(handle)
        if not 0 <= h < self._s_y.n:
            raise ValueError(f"unknown sample handle {handle}")
        if self._s_y.a[h] != _UNRESOLVED:
            raise ValueError(f"sample {h} already resolved; a late DATA never edits an event")
        if isinstance(y, (bool, np.bool_)) or y not in (0, 1):
            raise ValueError(f"y must be 0 or 1, got {y!r}")
        run, client, t_g = int(self._s_run.a[h]), int(self._s_client.a[h]), int(self._s_t.a[h])
        if run not in self._unresolved:
            raise ValueError(f"run {run} is not open")
        t = int(t_ns)
        if t < t_g:
            raise ValueError(f"resolution at {t} before issue at {t_g}")
        o = np.asarray(obs, dtype=np.float32)
        if o.shape != (N_OBS,) or not np.all(np.isfinite(o)) or float(o[0]) != float(y):
            raise ValueError("obs must be 8 finite values with obs[0] == y")
        last = self._last_t.get((run, client))
        if last is not None and t < last:
            raise ValueError(f"events of (run {run}, client {client}) must arrive in resolution order")
        self._last_t[(run, client)] = t
        self._e_obs.append(o)
        self._e_t.append(t)
        self._e_run.append(run)
        self._e_client.append(client)
        self._s_y.a[h] = int(y)
        self._unresolved[run] -= 1

    # -- output
    def arrays(self) -> dict[str, np.ndarray]:
        """Build all schema arrays (see docs/ML_DATASET_SCHEMA.md)."""
        if self._unresolved:
            raise ValueError(f"runs still open: {sorted(self._unresolved)}")
        e_run, e_cli, e_t = self._e_run.view(), self._e_client.view(), self._e_t.view()
        # group events by (run, client); stable sort keeps resolution order inside a group
        order = np.lexsort((e_cli, e_run))
        e_run, e_cli, e_t = e_run[order], e_cli[order], e_t[order]
        E = int(order.size)
        out: dict[str, np.ndarray] = {
            "events": self._e_obs.view()[order],
            "event_t_ns": e_t.copy(),
            "event_run": e_run.copy(),
            "event_client": e_cli.copy(),
        }
        s_run, s_cli, s_t = self._s_run.view(), self._s_client.view(), self._s_t.view()
        ns = int(s_t.size)
        e_key = e_run.astype(np.int64) * 65536 + e_cli.astype(np.int64)    # sorted ascending
        s_key = s_run.astype(np.int64) * 65536 + s_cli.astype(np.int64)
        g_lo = np.searchsorted(e_key, s_key, side="left")                 # group start (or insertion point)
        g_hi = np.searchsorted(e_key, s_key, side="right")
        end = np.empty(ns, np.int64)
        s_order = np.argsort(s_key, kind="stable")                        # samples grouped by (run, client)
        cuts = np.flatnonzero(np.diff(s_key[s_order])) + 1
        for idx in np.split(s_order, cuts) if ns else []:                  # one slice per (run, client)
            lo, hi = int(g_lo[idx[0]]), int(g_hi[idx[0]])
            # number of this group's events with resolution time < t_g
            end[idx] = lo + np.searchsorted(e_t[lo:hi], s_t[idx], side="left")
        n_prior = end - g_lo
        x48 = self._s_x48.view().copy()
        p = self._s_p.view()
        out.update(
            sample_end=end,
            sample_len=np.minimum(T_EVBUF, n_prior).astype(np.int8),
            sample_t_ns=s_t.copy(),
            sample_run=s_run.copy(),
            sample_client=s_cli.copy(),
            ctx=np.ascontiguousarray(x48[:, CTX_SLICE]),
            x48=x48,
            y=self._s_y.view().copy(),
            p_ewma=np.ascontiguousarray(p[:, 0]),
            p_hmm=np.ascontiguousarray(p[:, 1]),
            p_oracle=np.ascontiguousarray(p[:, 2]),
        )
        runs = [self._runs[r] for r in sorted(self._runs)]
        for name, (dt, _) in RUN_FIELDS.items():
            out[name] = np.array([r[name] for r in runs], dtype=dt)
        rng = SPLITS[self.split]
        meta = dict(schema_version=SCHEMA_VERSION, split=self.split, seed_range=[rng.start, rng.stop],
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