"""PyTorch vs NumPy parity of P-GRU (Sections 4.4, 4.6): max |p_torch - p_numpy| < 1e-5."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fedgrant.ml import synthetic  # noqa: E402
from fedgrant.ml.export import checkpoint_to_arrays  # noqa: E402
from fedgrant.ml.models import PGRU  # noqa: E402
from fedgrant.ml.schema import load_split  # noqa: E402
from fedgrant.ml.sequences import GRUData, compute_norm_stats  # noqa: E402
from fedgrant.predictors.base import PredInput, Predictor, RoundCtx  # noqa: E402
from fedgrant.predictors.factory import make_predictor  # noqa: E402
from fedgrant.predictors.gru_numpy import PGRUNumpy  # noqa: E402

TOL = 1e-5


@pytest.fixture(scope="module")
def split(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    p = synthetic.generate_split("train", 1, 1000, d, n_clients=20, n_rounds=3, verbose=False)
    return load_split(p)


@pytest.fixture(scope="module")
def setup(split):
    """(torch model, exported npz dict, stats) for a randomly initialised P-GRU with real norm stats."""
    torch.manual_seed(7)
    m = PGRU().eval()
    with torch.no_grad():                                   # larger weights: a harsher parity test
        for p in m.parameters():
            p.mul_(2.0)
    st = compute_norm_stats(split)
    ck = {"model": "gru", "seed": 0, "state_dict": m.state_dict(), "best_val_loss": 0.0, "best_epoch": 1,
          "n_params": 5633, "norm_stats": {"gru": st.gru_dict(), "mlp": st.mlp_dict()}}
    return m, checkpoint_to_arrays(ck), st


def _torch_p(m, seq, ln, ctx_std) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(m(torch.from_numpy(seq), torch.from_numpy(np.asarray(ln)),
                               torch.from_numpy(ctx_std))).numpy()


def _inp(ids, ctx_raw, t=0) -> PredInput:
    n = len(ids)
    x = np.zeros((n, 48), np.float32)
    x[:, 32:48] = ctx_raw
    return PredInput.from_x48(np.asarray(ids, np.int32), x, t)


def _obs(rng, n):
    o = rng.random((n, 8)).astype(np.float32) * 2
    o[:, 0] = rng.integers(0, 2, n)
    return o


def test_protocol_and_factory(setup, tmp_path) -> None:
    _, z, _ = setup
    g = PGRUNumpy(z)
    assert isinstance(g, Predictor) and g.name == "gru"
    np.savez(tmp_path / "gru_prod.npz", **z)
    assert isinstance(make_predictor("gru", tmp_path), PGRUNumpy)
    with pytest.raises(FileNotFoundError):
        make_predictor("mlp", tmp_path)
    with pytest.raises(ValueError):
        make_predictor("oracle", tmp_path)


def test_random_mixed_lengths_one_batch(setup) -> None:
    """One predict call with 0, short, 16 and more-than-16 events per client."""
    m, z, st = setup
    rng = np.random.default_rng(0)
    g = PGRUNumpy(z)
    counts = [0, 1, 3, 15, 16, 17, 40, 7, 0, 2]
    seqs = []
    for c, k in enumerate(counts):
        o = _obs(rng, k)
        for row in o:
            g.observe(c, int(row[0]), 0, row)
        seqs.append(o[-16:])
    ctx = rng.random((len(counts), 16)).astype(np.float32)
    p_np = g.predict(_inp(range(len(counts)), ctx))
    T = 16
    pad = np.zeros((len(counts), T, 8), np.float32)
    ln = np.array([len(s) for s in seqs])
    for r, s in enumerate(seqs):
        if len(s):
            pad[r, : len(s)] = st.obs(s)
    p_t = _torch_p(m, pad, ln, st.ctx(ctx))
    assert p_np.dtype == np.float32 and p_np.shape == (len(counts),)
    assert np.max(np.abs(p_np - p_t)) < TOL
    assert np.array_equal(g.hidden(0), np.zeros(32, np.float32))       # len 0 -> h = 0


def test_batched_equals_one_by_one(setup) -> None:
    _, z, _ = setup
    rng = np.random.default_rng(1)
    a, b = PGRUNumpy(z), PGRUNumpy(z)
    for c in range(12):
        for row in _obs(rng, int(rng.integers(0, 25))):
            a.observe(c, int(row[0]), 0, row)
            b.observe(c, int(row[0]), 0, row)
    ctx = rng.random((12, 16)).astype(np.float32)
    pa = a.predict(_inp(range(12), ctx))                          # one batched recompute
    pb = np.concatenate([b.predict(_inp([c], ctx[c : c + 1])) for c in range(12)])
    assert np.max(np.abs(pa - pb)) < 1e-6


def test_real_samples_replayed_incrementally(setup, split) -> None:
    """Replay the split's events through observe(), interleaved with predict() at each grant time
    (and reset_round calls): every p_hat must match PyTorch on the stored window."""
    m, z, st = setup
    gd = GRUData(split, st)
    ns = len(gd)
    seq, ln, ctx_std, _ = gd.batch(np.arange(ns))
    p_t = _torch_p(m, seq, ln, ctx_std)

    key_e = split["event_run"].astype(np.int64) * 65536 + split["event_client"]
    key_s = split["sample_run"].astype(np.int64) * 65536 + split["sample_client"]
    items = [(int(t), 0, k) for k, t in enumerate(split["sample_t_ns"])]       # samples first at ties
    items += [(int(t), 1, e) for e, t in enumerate(split["event_t_ns"])]
    items.sort()
    g = PGRUNumpy(z)
    p_np = np.full(ns, np.nan, np.float32)
    for i, (t, kind, k) in enumerate(items):
        if i % 997 == 0:
            g.reset_round(RoundCtx(round_id=1 + i // 997, D_ns=1, N=20, K=169))  # must not clear anything
        if kind == 0:
            p_np[k] = g.predict(_inp([key_s[k]], split["ctx"][k : k + 1], t))[0]
        else:
            o = split["events"][k]
            g.observe(int(key_e[k]), int(o[0]), t, o)
    assert not np.isnan(p_np).any()
    assert (ln == 0).any() and (ln < 16).any() and (ln == 16).any()     # all window cases occur
    assert np.max(np.abs(p_np - p_t)) < TOL


def test_incremental_equals_fresh_recompute(setup) -> None:
    _, z, _ = setup
    rng = np.random.default_rng(2)
    inc, events = PGRUNumpy(z), []
    ctx = rng.random((5, 16)).astype(np.float32)
    for rnd in range(1, 6):
        inc.reset_round(RoundCtx(rnd, 400_000_000, 5, 169))
        for _ in range(int(rng.integers(5, 30))):
            c = int(rng.integers(0, 5))
            o = _obs(rng, 1)[0]
            inc.observe(c, int(o[0]), 0, o)
            events.append((c, o))
            if rng.random() < 0.3:
                inc.predict(_inp(rng.choice(5, size=3, replace=False), ctx[:3]))
    fresh = PGRUNumpy(z)
    for c, o in events:
        fresh.observe(c, int(o[0]), 0, o)
    for c in range(5):
        np.testing.assert_array_equal(inc.window(c), fresh.window(c))
        assert np.max(np.abs(inc.hidden(c) - fresh.hidden(c))) < 1e-6
    pi, pf = inc.predict(_inp(range(5), ctx)), fresh.predict(_inp(range(5), ctx))
    assert np.max(np.abs(pi - pf)) < 1e-6


def test_cache_only_recomputes_stale(setup) -> None:
    _, z, _ = setup
    g = PGRUNumpy(z)
    o = np.array([1, 0.2, 0.1, 0.5, 0.1, 0.1, 0.3, 1], np.float32)
    g.observe(0, 1, 0, o)
    ctx = np.zeros((2, 16), np.float32)
    g.predict(_inp([0, 1], ctx))
    assert g.n_recomputed == 1                     # client 1 has no events: h = 0, nothing to compute
    g.predict(_inp([0, 1], ctx))
    assert g.n_recomputed == 1                     # head only
    g.observe(1, 1, 1, o)
    g.predict(_inp([0, 1, 1], np.zeros((3, 16), np.float32)))
    assert g.n_recomputed == 2                     # only client 1, once


def test_window_keeps_last_16_in_order(setup) -> None:
    _, z, _ = setup
    g = PGRUNumpy(z)
    rows = _obs(np.random.default_rng(3), 21)
    for r in rows:
        g.observe(9, int(r[0]), 0, r)
    np.testing.assert_array_equal(g.window(9), rows[-16:])


def test_observe_validation(setup) -> None:
    _, z, _ = setup
    g = PGRUNumpy(z)
    o = np.ones(8, np.float32)
    with pytest.raises(ValueError):
        g.observe(0, 0, 0, o)                      # obs[0] != y
    with pytest.raises(ValueError):
        g.observe(0, 1, 0, np.ones(7, np.float32))
    with pytest.raises(ValueError):
        g.observe(0, 2, 0, o)
    bad = o.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError):
        g.observe(0, 1, 0, bad)
    assert g.predict(_inp([], np.zeros((0, 16), np.float32))).shape == (0,)


def test_outputs_in_unit_interval_extreme_ctx(setup) -> None:
    _, z, _ = setup
    g = PGRUNumpy(z)
    p = g.predict(_inp([0, 1], np.array([[1e4] * 16, [-1e4] * 16], np.float32)))
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))