"""Tests for fedgrant/ml/train.py and fedgrant/ml/train_all.py (Section 4.4)."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fedgrant.ml import synthetic  # noqa: E402
from fedgrant.ml.sequences import compute_norm_stats  # noqa: E402
from fedgrant.ml.train import (  # noqa: E402
    build_data,
    eval_loss,
    load_checkpoint,
    main,
    pick_device,
    predict_proba,
    train_model,
)
from fedgrant.ml.train_all import summarize  # noqa: E402

from tests.unit.test_models import fake_split  # noqa: E402

QUIET = lambda s: None  # noqa: E731


@pytest.mark.parametrize("name", ["gru", "mlp"])
def test_train_model_writes_outputs_and_restores_best(tmp_path, name: str) -> None:
    tr, va = fake_split(600, 800, seed=1), fake_split(200, 300, seed=2)
    cost = train_model(name, 0, tr, va, tmp_path, max_epochs=4, patience=5, batch=128, threads=1,
                       device="cpu", log=QUIET)
    for f in (f"{name}_seed0.pt", f"{name}_seed0_cost.json", "norm_stats_gru.json", "norm_stats_mlp.json"):
        assert (tmp_path / f).is_file(), f
    c = json.loads((tmp_path / f"{name}_seed0_cost.json").read_text())
    assert c["n_params"] == (5633 if name == "gru" else 7361)
    assert c["epochs_run"] == 4 and len(c["epoch_time_s"]) == 4 and c["peak_rss_mb"] > 0
    assert 1 <= c["best_epoch"] <= 4 and c == cost
    assert c["best_val_loss"] == min(h["val_loss"] for h in c["history"])

    model, ck = load_checkpoint(tmp_path / f"{name}_seed0.pt")       # weights_only=True
    assert ck["model"] == name and ck["seed"] == 0 and ck["best_epoch"] == c["best_epoch"]
    assert set(ck["norm_stats"]) == {"gru", "mlp"}
    st = compute_norm_stats(tr)
    # the saved weights are the best epoch: their val loss is the recorded best
    assert eval_loss(model, name, build_data(name, va, st), torch.device("cpu")) == pytest.approx(
        ck["best_val_loss"], rel=1e-6)
    p = predict_proba(model, name, build_data(name, va, st), torch.device("cpu"))
    assert p.shape == (200,) and np.all((p > 0) & (p < 1))


def test_early_stopping_patience(tmp_path) -> None:
    """With patience 1 and labels that are pure noise, training stops before max_epochs."""
    tr, va = fake_split(400, 500, seed=3), fake_split(400, 500, seed=4)
    c = train_model("mlp", 0, tr, va, tmp_path, max_epochs=30, patience=1, batch=32, threads=1,
                    device="cpu", log=QUIET)
    assert c["early_stopped"] and c["epochs_run"] == c["best_epoch"] + 1


def test_same_seed_is_deterministic(tmp_path) -> None:
    tr, va = fake_split(300, 400, seed=5), fake_split(100, 200, seed=6)
    a = train_model("gru", 3, tr, va, tmp_path / "a", max_epochs=2, batch=64, threads=1, device="cpu", log=QUIET)
    b = train_model("gru", 3, tr, va, tmp_path / "b", max_epochs=2, batch=64, threads=1, device="cpu", log=QUIET)
    assert a["best_val_loss"] == b["best_val_loss"]
    sa = torch.load(tmp_path / "a" / "gru_seed3.pt", weights_only=True)["state_dict"]
    sb = torch.load(tmp_path / "b" / "gru_seed3.pt", weights_only=True)["state_dict"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    c = train_model("gru", 4, tr, va, tmp_path / "c", max_epochs=2, batch=64, threads=1, device="cpu", log=QUIET)
    sc = torch.load(tmp_path / "c" / "gru_seed4.pt", weights_only=True)["state_dict"]
    assert not all(torch.equal(sa[k], sc[k]) for k in sa)


def test_cli_end_to_end_on_synthetic_schema(tmp_path) -> None:
    """Real schema files (SplitRecorder) -> CLI -> checkpoint."""
    data = tmp_path / "data"
    synthetic.generate_split("train", 1, 1000, data, n_clients=8, n_rounds=3, verbose=False)
    synthetic.generate_split("val", 1, 2000, data, n_clients=8, n_rounds=2, verbose=False)
    c = main(["--model", "gru", "--seed", "1", "--data", str(data), "--out", str(tmp_path / "m"),
              "--max-epochs", "2", "--threads", "1", "--device", "cpu"])
    assert c["epochs_run"] == 2 and (tmp_path / "m" / "gru_seed1.pt").is_file()


def test_summarize_picks_lowest_val_loss(tmp_path) -> None:
    def cost(m, k, v):
        (tmp_path / f"{m}_seed{k}_cost.json").write_text(json.dumps(
            dict(best_val_loss=v, best_epoch=3, epochs_run=8, train_time_s=1.0, peak_rss_mb=10.0)))

    for k, v in enumerate([0.30, 0.25, 0.25, 0.40]):        # tie 1 vs 2 -> lower seed
        cost("gru", k, v)
    cost("mlp", 2, 0.31)
    s = summarize(tmp_path)
    assert s["gru"]["production_seed"] == 1 and s["gru"]["missing"] == [4]
    assert s["mlp"]["production_seed"] == 2 and s["mlp"]["missing"] == [0, 1, 3, 4]
    assert json.loads((tmp_path / "train_summary.json").read_text()) == s


def test_pick_device() -> None:
    assert pick_device("cpu").type == "cpu"
    assert pick_device("auto").type in ("cpu", "cuda")