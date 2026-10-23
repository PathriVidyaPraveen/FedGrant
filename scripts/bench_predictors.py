"""NumPy predictor latency benchmark (Section 4.6), single thread.

    python scripts/bench_predictors.py --models models/
    python scripts/bench_predictors.py --random          # no trained models needed

Cases (1000 timed repetitions each, after a warm-up; only predict() is timed):
  a  gru_single_recompute   1 client, 16 events, h stale  -> GRU over 16 steps + head
  b  gru_tick20_all_stale   20 clients, 16 events each, all stale (batched recompute + head)
  c  gru_head20             20 clients, nothing stale (head only)
  d  mlp20                  P-MLP for 20 clients
Spec reference (Section 4.6): a 0.18 ms, b 0.36 ms, c 6 us (median).

Prints a table and writes results/summary/pred_latency.csv
(case, n_clients, reps, median_us, p99_us, mean_us, spec_median_us, model_source).

The gateway currently calls predict() for ONE client per grant (case a when the
client has a new event, otherwise a 1-client head). The 2 ms fallback limit
applies to each call.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"                                   # must precede the NumPy import

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere without installing

from fedgrant.ml.export import load_prod  # noqa: E402
from fedgrant.predictors.base import PredInput  # noqa: E402
from fedgrant.predictors.gru_numpy import PGRUNumpy  # noqa: E402
from fedgrant.predictors.mlp_numpy import PMLPNumpy  # noqa: E402

SPEC_US = {"gru_single_recompute": 180.0, "gru_tick20_all_stale": 360.0, "gru_head20": 6.0, "mlp20": None}


def random_exports(seed: int = 0) -> tuple[dict, dict]:
    """GRU / MLP export dicts with random weights (same layout as *_prod.npz)."""
    r = np.random.default_rng(seed)

    def w(*s):
        return (r.standard_normal(s) * 0.2).astype(np.float32)

    gru = dict(weight_ih_l0=w(96, 8), weight_hh_l0=w(96, 32), bias_ih_l0=w(96), bias_hh_l0=w(96),
               head_w1=w(32, 48), head_b1=w(32), head_w2=w(1, 32), head_b2=w(1),
               mu_obs=np.zeros(8), s_obs=np.ones(8), mu_ctx=np.zeros(16), s_ctx=np.ones(16),
               model=np.array("gru"), gate_order=np.array("r,z,n"), seed=np.array(-1))
    mlp = dict(w1=w(64, 48), b1=w(64), w2=w(64, 64), b2=w(64), w3=w(1, 64), b3=w(1),
               mu_x48=np.zeros(48), s_x48=np.ones(48), model=np.array("mlp"), seed=np.array(-1))
    return gru, mlp


def _obs(rng: np.random.Generator) -> np.ndarray:
    o = rng.random(8).astype(np.float32)
    o[0] = float(rng.integers(0, 2))
    return o


def _inp(n: int, rng: np.random.Generator) -> PredInput:
    return PredInput.from_x48(np.arange(n, dtype=np.int32), rng.random((n, 48)).astype(np.float32), 0)


def _time(fn, before, reps: int, warmup: int) -> np.ndarray:
    """Run before() untimed and fn() timed, reps times; return ns per call."""
    for _ in range(warmup):
        before()
        fn()
    out = np.empty(reps, np.int64)
    for k in range(reps):
        before()
        t0 = time.perf_counter_ns()
        fn()
        out[k] = time.perf_counter_ns() - t0
    return out


def run(gru_src, mlp_src, reps: int = 1000, warmup: int = 50, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    g = PGRUNumpy(gru_src)
    for c in range(20):                                     # full 16-event windows
        for _ in range(16):
            o = _obs(rng)
            g.observe(c, int(o[0]), 0, o)
    inp1, inp20 = _inp(1, rng), _inp(20, rng)

    def stale_one():
        o = _obs(rng)
        g.observe(0, int(o[0]), 0, o)

    def stale_all():
        for c in range(20):
            o = _obs(rng)
            g.observe(c, int(o[0]), 0, o)

    def nothing():
        pass

    g.predict(inp20)                                        # all fresh for the head-only case
    m = PMLPNumpy(mlp_src)
    cases = [
        ("gru_single_recompute", 1, lambda: g.predict(inp1), stale_one),
        ("gru_tick20_all_stale", 20, lambda: g.predict(inp20), stale_all),
        ("gru_head20", 20, lambda: g.predict(inp20), nothing),
        ("mlp20", 20, lambda: m.predict(inp20), nothing),
    ]
    rows = []
    for name, n, fn, before in cases:
        if name == "gru_head20":
            g.predict(inp20)                                # clear staleness left by earlier cases
        ns = _time(fn, before, reps, warmup)
        rows.append(dict(case=name, n_clients=n, reps=reps, median_us=float(np.median(ns)) / 1e3,
                         p99_us=float(np.percentile(ns, 99)) / 1e3, mean_us=float(ns.mean()) / 1e3,
                         spec_median_us=SPEC_US[name]))
    return rows


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description="NumPy predictor latency benchmark (Section 4.6).")
    ap.add_argument("--models", default="models/")
    ap.add_argument("--random", action="store_true", help="use random weights instead of *_prod.npz")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", default="results/summary/pred_latency.csv")
    a = ap.parse_args(argv)
    if a.random:
        gru_src, mlp_src = random_exports()
        source = "random"
    else:
        md = Path(a.models)
        gru_src, mlp_src = load_prod(md / "gru_prod.npz"), load_prod(md / "mlp_prod.npz")
        source = f"{md}/gru_prod.npz seed {int(gru_src['seed'])}, mlp_prod.npz seed {int(mlp_src['seed'])}"
    rows = run(gru_src, mlp_src, a.reps, a.warmup)
    print(f"NumPy {np.__version__}, single thread, {a.reps} reps, models: {source}")
    print(f"{'case':24s} {'n':>3s} {'median_us':>10s} {'p99_us':>10s} {'spec_us':>8s}")
    for r in rows:
        spec = "-" if r["spec_median_us"] is None else f"{r['spec_median_us']:.0f}"
        print(f"{r['case']:24s} {r['n_clients']:3d} {r['median_us']:10.1f} {r['p99_us']:10.1f} {spec:>8s}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "n_clients", "reps", "median_us", "p99_us", "mean_us",
                                          "spec_median_us", "model_source"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "spec_median_us": "" if r["spec_median_us"] is None else r["spec_median_us"],
                        "model_source": source})
    print(f"wrote {out}")
    return rows


if __name__ == "__main__":
    main()