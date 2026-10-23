"""Tests for fedgrant/ml/export.py (Section 4.4): layout, production choice, and that a
plain NumPy forward pass on the exported arrays reproduces PyTorch."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fedgrant.ml.export import choose_production, export_model, load_prod, main  # noqa: E402
from fedgrant.ml.models import PGRU, PMLP  # noqa: E402
from fedgrant.ml.train import train_model  # noqa: E402

from tests.unit.test_models import fake_split  # noqa: E402

QUIET = lambda s: None  # noqa: E731


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    tr, va = fake_split(500, 700, seed=1), fake_split(200, 300, seed=2)
    for name in ("gru", "mlp"):
        for k in (0, 1, 2):
            train_model(name, k, tr, va, d, max_epochs=2, batch=128, threads=1, device="cpu", log=QUIET)
    return d


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _np_gru_logit(z, seq, ctx):
    """Reference NumPy P-GRU on ONE sample (seq [n, 8], already standardized)."""
    W_ih, W_hh, b_ih, b_hh = z["weight_ih_l0"], z["weight_hh_l0"], z["bias_ih_l0"], z["bias_hh_l0"]
    h = np.zeros(32, np.float32)
    for x in seq:
        gi, gh = W_ih @ x + b_ih, W_hh @ h + b_hh
        r = _sig(gi[:32] + gh[:32])
        u = _sig(gi[32:64] + gh[32:64])
        n = np.tanh(gi[64:] + r * gh[64:])
        h = (1 - u) * n + u * h
    a = np.maximum(z["head_w1"] @ np.concatenate([h, ctx]) + z["head_b1"], 0)
    return float((z["head_w2"] @ a + z["head_b2"])[0])


def test_production_seed_is_lowest_val_loss(models_dir) -> None:
    for name in ("gru", "mlp"):
        seed, path, losses = choose_production(models_dir, name)
        assert set(losses) == {0, 1, 2}
        assert losses[seed] == min(losses.values()) and path.name == f"{name}_seed{seed}.pt"


def test_gru_npz_layout_and_numpy_forward(models_dir) -> None:
    r = export_model(models_dir, "gru")
    z = load_prod(r["file"])
    assert z["weight_ih_l0"].shape == (96, 8) and z["weight_hh_l0"].shape == (96, 32)
    assert z["bias_ih_l0"].shape == (96,) and z["bias_hh_l0"].shape == (96,)
    assert z["head_w1"].shape == (32, 48) and z["head_w2"].shape == (1, 32)
    assert all(z[k].dtype == np.float32 for k in ("weight_ih_l0", "head_w1", "head_b2"))
    assert z["mu_obs"].shape == (8,) and z["s_ctx"].shape == (16,) and z["mu_obs"].dtype == np.float64
    assert str(z["gate_order"]) == "r,z,n" and int(z["seed"]) == r["production_seed"]
    assert int(z["n_params"]) == 5633

    ck = torch.load(models_dir / f"gru_seed{r['production_seed']}.pt", weights_only=True)
    m = PGRU()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    rng = np.random.default_rng(0)
    for n in (0, 1, 7, 16):
        seq = rng.standard_normal((n, 8)).astype(np.float32)
        ctx = rng.standard_normal(16).astype(np.float32)
        pad = np.zeros((16, 8), np.float32)
        pad[:n] = seq
        with torch.no_grad():
            ref = float(m(torch.from_numpy(pad)[None], torch.tensor([n]), torch.from_numpy(ctx)[None])[0])
        assert _np_gru_logit(z, seq, ctx) == pytest.approx(ref, abs=1e-5), n


def test_mlp_npz_layout_and_numpy_forward(models_dir) -> None:
    r = export_model(models_dir, "mlp")
    z = load_prod(r["file"])
    assert z["w1"].shape == (64, 48) and z["w2"].shape == (64, 64) and z["w3"].shape == (1, 64)
    assert z["mu_x48"].shape == (48,) and int(z["n_params"]) == 7361
    ck = torch.load(models_dir / f"mlp_seed{r['production_seed']}.pt", weights_only=True)
    m = PMLP()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    x = np.random.default_rng(1).standard_normal((5, 48)).astype(np.float32)
    a = np.maximum(x @ z["w1"].T + z["b1"], 0)
    a = np.maximum(a @ z["w2"].T + z["b2"], 0)
    out = (a @ z["w3"].T + z["b3"])[:, 0]
    with torch.no_grad():
        ref = m(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(out, ref, atol=1e-5)


def test_norm_stats_match_checkpoint(models_dir) -> None:
    r = export_model(models_dir, "gru")
    z = load_prod(r["file"])
    ck = torch.load(models_dir / f"gru_seed{r['production_seed']}.pt", weights_only=True)
    np.testing.assert_array_equal(z["mu_ctx"], np.array(ck["norm_stats"]["gru"]["mu_ctx"]))


def test_seed_override_and_cli(models_dir) -> None:
    r = export_model(models_dir, "mlp", seed=2)
    assert r["production_seed"] == 2 and int(load_prod(r["file"])["seed"]) == 2
    with pytest.raises(FileNotFoundError):
        export_model(models_dir, "mlp", seed=9)
    s = main(["--models", str(models_dir)])
    assert set(s) == {"gru", "mlp"}
    summ = json.loads((models_dir / "export_summary.json").read_text())
    assert summ["gru"]["production_seed"] == choose_production(models_dir, "gru")[0]


def test_no_checkpoints(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        choose_production(tmp_path, "gru")