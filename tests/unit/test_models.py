"""Tests for fedgrant/ml/models.py and fedgrant/ml/sequences.py (Sections 4.2-4.4)."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fedgrant.ml.models import PGRU, PGRU_PARAMS, PMLP, PMLP_PARAMS, make_model, n_params  # noqa: E402
from fedgrant.ml.sequences import (  # noqa: E402
    GRUData,
    MLPData,
    NormStats,
    compute_norm_stats,
    load_norm_stats,
    save_norm_stats,
    window_index,
)


def fake_split(ns: int = 300, E: int = 400, seed: int = 0) -> dict[str, np.ndarray]:
    """Minimal arrays sequences.py reads; windows end anywhere in [0, E]."""
    rng = np.random.default_rng(seed)
    end = rng.integers(0, E + 1, ns).astype(np.int64)
    ln = np.minimum(rng.integers(0, 17, ns), end).astype(np.int8)
    x48 = rng.random((ns, 48)).astype(np.float32)
    ev = rng.random((E, 8)).astype(np.float32)
    ev[:, 0] = rng.integers(0, 2, E)
    return {
        "events": ev,
        "sample_end": end,
        "sample_len": ln,
        "ctx": x48[:, 32:48].copy(),
        "x48": x48,
        "y": rng.integers(0, 2, ns).astype(np.uint8),
    }


# --------------------------------------------------------------------------- parameter counts


def test_parameter_counts() -> None:
    assert n_params(PGRU()) == PGRU_PARAMS == 5633
    assert n_params(PMLP()) == PMLP_PARAMS == 7361
    assert isinstance(make_model("gru"), PGRU) and isinstance(make_model("mlp"), PMLP)
    with pytest.raises(ValueError):
        make_model("lstm")


def test_no_dropout_or_norm_layers() -> None:
    for m in (PGRU(), PMLP()):
        for mod in m.modules():
            assert not isinstance(mod, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.LayerNorm))
    g = PGRU()
    assert g.gru.input_size == 8 and g.gru.hidden_size == 32 and g.gru.num_layers == 1 and g.gru.batch_first


# --------------------------------------------------------------------------- P-GRU behaviour


def _inputs(B: int, seed: int = 0, lengths: list[int] | None = None):
    g = torch.Generator().manual_seed(seed)
    seq = torch.randn(B, 16, 8, generator=g)
    ln = torch.tensor(lengths if lengths is not None else torch.randint(0, 17, (B,), generator=g).tolist())
    for b in range(B):                                      # padding is zero, as GRUData produces
        seq[b, int(ln[b]):] = 0
    ctx = torch.randn(B, 16, generator=g)
    return seq, ln, ctx


def test_shapes() -> None:
    torch.manual_seed(0)
    g, m = PGRU(), PMLP()
    seq, ln, ctx = _inputs(7)
    assert g(seq, ln, ctx).shape == (7,)
    assert g.encode(seq, ln).shape == (7, 32)
    assert m(torch.randn(7, 48)).shape == (7,)
    assert g(seq[:0], ln[:0], ctx[:0]).shape == (0,)


def test_len0_bypasses_gru_and_uses_h0() -> None:
    torch.manual_seed(1)
    g = PGRU().eval()
    seq, ln, ctx = _inputs(5, lengths=[0, 3, 0, 16, 1])
    seq[0] = torch.randn(16, 8)                              # garbage in a len-0 row must be ignored
    with torch.no_grad():
        h = g.encode(seq, ln)
        out = g(seq, ln, ctx)
        ref0 = g.head(torch.cat([torch.zeros(1, 32), ctx[0:1]], 1)).squeeze(1)
    assert torch.equal(h[0], torch.zeros(32)) and torch.equal(h[2], torch.zeros(32))
    assert torch.allclose(out[0], ref0[0], atol=1e-6, rtol=0)
    all0 = g(seq, torch.zeros(5, dtype=torch.long), ctx)    # whole batch empty: no GRU call at all
    with torch.no_grad():
        assert torch.allclose(all0, g.head(torch.cat([torch.zeros(5, 32), ctx], 1)).squeeze(1))


def test_packed_hidden_equals_per_sample_unpacked() -> None:
    """Mixed, unsorted lengths in one batch == running the GRU on each true window alone."""
    torch.manual_seed(2)
    g = PGRU().eval()
    lens = [5, 16, 1, 0, 9, 16, 2]
    seq, ln, ctx = _inputs(len(lens), seed=3, lengths=lens)
    with torch.no_grad():
        h = g.encode(seq, ln)
        for b, n in enumerate(lens):
            if n == 0:
                ref = torch.zeros(32)
            else:
                _, hn = g.gru(seq[b : b + 1, :n])
                ref = hn[0, 0]
            assert torch.allclose(h[b], ref, atol=1e-6), b


def test_padding_values_do_not_matter() -> None:
    torch.manual_seed(4)
    g = PGRU().eval()
    seq, ln, ctx = _inputs(4, lengths=[3, 7, 16, 0])
    seq2 = seq.clone()
    for b, n in enumerate([3, 7, 16, 0]):
        seq2[b, n:] = 123.0
    with torch.no_grad():
        assert torch.allclose(g(seq, ln, ctx), g(seq2, ln, ctx))


def test_gradients_flow_through_both_paths() -> None:
    g = PGRU()
    seq, ln, ctx = _inputs(6, lengths=[0, 4, 16, 0, 2, 8])
    g(seq, ln, ctx).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in g.parameters())


@pytest.mark.parametrize("name", ["gru", "mlp"])
def test_tiny_overfit(name: str) -> None:
    """Loss on 512 samples must drop clearly when trained on them."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n = 512
    seq = torch.from_numpy(rng.standard_normal((n, 16, 8)).astype(np.float32))
    ln = torch.from_numpy(rng.integers(0, 17, n))
    for b in range(n):
        seq[b, int(ln[b]):] = 0
    ctx = torch.from_numpy(rng.standard_normal((n, 16)).astype(np.float32))
    x48 = torch.from_numpy(rng.standard_normal((n, 48)).astype(np.float32))
    y = torch.from_numpy((rng.random(n) < 0.5).astype(np.float32))
    model = make_model(name)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    lossf = torch.nn.BCEWithLogitsLoss()

    def f():
        return model(seq, ln, ctx) if name == "gru" else model(x48)

    with torch.no_grad():
        l0 = float(lossf(f(), y))
    for _ in range(300):
        loss = lossf(f(), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < 0.5 * l0, (l0, float(loss.detach()))


# --------------------------------------------------------------------------- sequences


def test_window_index_right_padded() -> None:
    w = window_index(np.array([5, 3, 20]), np.array([3, 0, 16]), pad=99)
    assert w.shape == (3, 16) and w.dtype == np.int64
    assert w[0, :3].tolist() == [2, 3, 4] and np.all(w[0, 3:] == 99)
    assert np.all(w[1] == 99)
    assert w[2].tolist() == list(range(4, 20))


def test_norm_stats_train_only_and_small_s() -> None:
    d = fake_split()
    d["x48"][:, 5] = 3.0                                    # constant feature -> s = 1
    d["ctx"] = d["x48"][:, 32:48].copy()
    st = compute_norm_stats(d)
    np.testing.assert_allclose(st.mu_obs, d["events"].astype(np.float64).mean(0))
    np.testing.assert_allclose(st.s_obs, d["events"].astype(np.float64).std(0))
    np.testing.assert_allclose(st.mu_ctx, d["ctx"].astype(np.float64).mean(0))
    assert st.s_x48[5] == 1.0 and st.mu_x48[5] == pytest.approx(3.0)
    z = st.x48(d["x48"])
    assert z.dtype == np.float32 and np.allclose(z[:, 5], 0.0)
    nz = [k for k in range(48) if k != 5]
    np.testing.assert_allclose(z[:, nz].mean(0), 0, atol=1e-5)
    np.testing.assert_allclose(z[:, nz].std(0), 1, atol=1e-4)


def test_norm_stats_json_round_trip(tmp_path) -> None:
    st = compute_norm_stats(fake_split())
    pg, pm = save_norm_stats(st, tmp_path)
    assert pg.name == "norm_stats_gru.json" and pm.name == "norm_stats_mlp.json"
    g, m = json.loads(pg.read_text()), json.loads(pm.read_text())
    assert set(g) >= {"mu_obs", "s_obs", "mu_ctx", "s_ctx"} and set(m) >= {"mu_x48", "s_x48"}
    assert len(g["mu_obs"]) == 8 and len(g["mu_ctx"]) == 16 and len(m["mu_x48"]) == 48
    st2 = load_norm_stats(tmp_path)
    for k in ("mu_obs", "s_obs", "mu_ctx", "s_ctx", "mu_x48", "s_x48"):
        assert np.array_equal(getattr(st, k), getattr(st2, k))


def test_norm_stats_rejects_bad() -> None:
    z8, z16, z48 = np.zeros(8), np.zeros(16), np.zeros(48)
    with pytest.raises(ValueError):
        NormStats(z8, np.zeros(8), z16, np.ones(16), z48, np.ones(48))     # s = 0
    with pytest.raises(ValueError):
        NormStats(z8, np.ones(7), z16, np.ones(16), z48, np.ones(48))      # wrong length


def test_gru_data_gather_matches_loop() -> None:
    d = fake_split()
    st = compute_norm_stats(d)
    gd = GRUData(d, st)
    idx = np.array([0, 5, 17, 299, 5, 120])
    seq, ln, ctx, y = gd.batch(idx)
    assert seq.shape == (6, 16, 8) and seq.dtype == np.float32
    assert ln.dtype == np.int64 and ctx.shape == (6, 16) and y.dtype == np.float32
    for r, k in enumerate(idx):
        e, n = int(d["sample_end"][k]), int(d["sample_len"][k])
        np.testing.assert_array_equal(seq[r, :n], st.obs(d["events"][e - n : e]))
        assert not seq[r, n:].any()                          # zero padding
        assert ln[r] == n and y[r] == d["y"][k]
        np.testing.assert_array_equal(ctx[r], st.ctx(d["ctx"][k]))


def test_mlp_data() -> None:
    d = fake_split()
    st = compute_norm_stats(d)
    md = MLPData(d, st)
    x, y = md.batch(np.array([3, 1]))
    np.testing.assert_array_equal(x, st.x48(d["x48"][[3, 1]]))
    assert len(md) == 300 and y.shape == (2,)