"""PyTorch vs NumPy parity of P-MLP (Sections 4.3, 4.6): max |p_torch - p_numpy| < 1e-5."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fedgrant.ml import synthetic  # noqa: E402
from fedgrant.ml.export import checkpoint_to_arrays  # noqa: E402
from fedgrant.ml.models import PMLP  # noqa: E402
from fedgrant.ml.schema import load_split  # noqa: E402
from fedgrant.ml.sequences import MLPData, compute_norm_stats  # noqa: E402
from fedgrant.predictors.base import PredInput, Predictor, RoundCtx  # noqa: E402
from fedgrant.predictors.factory import make_predictor  # noqa: E402
from fedgrant.predictors.mlp_numpy import PMLPNumpy, stable_sigmoid  # noqa: E402

TOL = 1e-5


@pytest.fixture(scope="module")
def split(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    return load_split(synthetic.generate_split("train", 1, 1000, d, n_clients=20, n_rounds=2, verbose=False))


@pytest.fixture(scope="module")
def setup(split):
    torch.manual_seed(3)
    m = PMLP().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(2.0)
    st = compute_norm_stats(split)
    ck = {"model": "mlp", "seed": 4, "state_dict": m.state_dict(), "best_val_loss": 0.0, "best_epoch": 1,
          "n_params": 7361, "norm_stats": {"gru": st.gru_dict(), "mlp": st.mlp_dict()}}
    return m, checkpoint_to_arrays(ck), st


def _torch_p(m, x_std) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(m(torch.from_numpy(x_std))).numpy()


def test_protocol_factory_and_noop_observe(setup, tmp_path) -> None:
    _, z, _ = setup
    p = PMLPNumpy(z)
    assert isinstance(p, Predictor) and p.name == "mlp" and p.seed == 4
    p.reset_round(RoundCtx(1, 1, 1, 1))
    p.observe(0, 1, 0, np.ones(8, np.float32))
    np.savez(tmp_path / "mlp_prod.npz", **z)
    assert isinstance(make_predictor("mlp", tmp_path), PMLPNumpy)
    with pytest.raises(ValueError):
        PMLPNumpy({**z, "model": np.array("gru")})


def test_parity_on_real_samples(setup, split) -> None:
    m, z, st = setup
    p_np = PMLPNumpy(z).predict(PredInput.from_x48(np.arange(split["y"].size, dtype=np.int32) % 20,
                                                   split["x48"], 0))
    x, _ = MLPData(split, st).batch(np.arange(split["y"].size))
    assert p_np.dtype == np.float32
    assert np.max(np.abs(p_np - _torch_p(m, x))) < TOL


def test_parity_on_random_inputs(setup) -> None:
    m, z, st = setup
    x = (np.random.default_rng(0).standard_normal((257, 48)) * 3).astype(np.float32)
    p_np = PMLPNumpy(z).predict(PredInput.from_x48(np.arange(257, dtype=np.int32), x, 0))
    assert np.max(np.abs(p_np - _torch_p(m, st.x48(x)))) < TOL


def test_single_and_empty(setup, split) -> None:
    _, z, _ = setup
    p = PMLPNumpy(z)
    one = p.predict(PredInput.from_x48(np.array([3], np.int32), split["x48"][:1], 0))
    assert one.shape == (1,) and 0 <= one[0] <= 1
    assert p.predict(PredInput.from_x48(np.zeros(0, np.int32), np.zeros((0, 48), np.float32), 0)).shape == (0,)


def test_stable_sigmoid() -> None:
    x = np.array([-1e4, -50, 0, 50, 1e4], np.float32)
    with np.errstate(over="raise", invalid="raise", divide="raise"):      # underflow to 0 is fine
        s = stable_sigmoid(x)
    assert s.dtype == np.float32 and s[0] == 0.0 and s[2] == 0.5 and s[-1] == 1.0
    np.testing.assert_allclose(stable_sigmoid(np.array([0.3, -2.0], np.float32)),
                               1 / (1 + np.exp(-np.array([0.3, -2.0]))), rtol=1e-6)