"""Tests for fedgrant/ml/evaluate.py (Section 4.7)."""

from __future__ import annotations

import csv
import itertools
import json
import math

import numpy as np
import pytest

from fedgrant.ml.evaluate import (
    bce,
    bin_index,
    brier,
    ece,
    evaluate_split,
    reliability_bins,
    roc_auc,
    t_summary,
)

# --------------------------------------------------------------------------- metrics


def test_brier_hand() -> None:
    assert brier(np.array([0.9, 0.2, 0.5]), np.array([1, 0, 1])) == pytest.approx((0.01 + 0.04 + 0.25) / 3)


def test_bins_equal_width_and_edges() -> None:
    p = np.array([0.0, 1 / 15 - 1e-12, 1 / 15, 0.5, 14 / 15, 1.0])
    assert bin_index(p).tolist() == [0, 0, 1, 7, 14, 14]
    r = reliability_bins(p, np.array([0, 0, 1, 1, 0, 1]))
    assert r["n"].sum() == 6 and len(r["n"]) == 15
    assert r["lo"][0] == 0 and r["hi"][-1] == 1
    assert np.isnan(r["mean_p"][3])                          # empty bin


def test_ece_hand() -> None:
    p = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9])            # bins 1 and 13
    y = np.array([0, 0, 0, 1, 1, 1])
    # bin1: n=4, pbar 0.1, ybar 0.25 -> 4/6 * 0.15 ; bin13: n=2, pbar 0.9, ybar 1 -> 2/6 * 0.1
    assert ece(p, y) == pytest.approx(4 / 6 * 0.15 + 2 / 6 * 0.1)
    assert ece(np.array([0.0, 1.0]), np.array([0, 1])) == 0.0


def _auc_brute(p, y) -> float:
    pos = [a for a, b in zip(p, y) if b == 1]
    neg = [a for a, b in zip(p, y) if b == 0]
    s = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in itertools.product(pos, neg))
    return s / (len(pos) * len(neg))


@pytest.mark.parametrize("seed", range(10))
def test_auc_matches_pairwise_with_ties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 80))
    p = rng.integers(0, 6, n) / 5.0                          # heavy ties
    y = rng.integers(0, 2, n)
    if y.min() == y.max():
        y[0] = 1 - y[0]
    assert roc_auc(p, y) == pytest.approx(_auc_brute(p, y), abs=1e-12)


def test_auc_edge_cases() -> None:
    assert roc_auc(np.array([0.2, 0.8]), np.array([0, 1])) == 1.0
    assert roc_auc(np.array([0.8, 0.2]), np.array([0, 1])) == 0.0
    assert roc_auc(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5
    assert math.isnan(roc_auc(np.array([0.3, 0.4]), np.array([1, 1])))


def test_bce_clipping() -> None:
    assert bce(np.array([1.0]), np.array([0])) == pytest.approx(-math.log(1e-6))
    assert bce(np.array([0.0]), np.array([0])) == pytest.approx(-math.log1p(-1e-6))
    assert bce(np.array([0.8, 0.3]), np.array([1, 0])) == pytest.approx(-(math.log(0.8) + math.log(0.7)) / 2)


def test_metric_input_validation() -> None:
    with pytest.raises(ValueError):
        brier(np.array([1.2]), np.array([1]))
    with pytest.raises(ValueError):
        brier(np.array([0.5]), np.array([2]))
    with pytest.raises(ValueError):
        brier(np.array([0.5, 0.5]), np.array([1]))
    with pytest.raises(ValueError):
        brier(np.array([]), np.array([]))


def test_t_summary_uses_t_0975_4() -> None:
    v = [1.0, 2.0, 3.0, 4.0, 5.0]
    s = t_summary(v)
    assert s["t_crit"] == pytest.approx(2.776, abs=5e-4)
    half = 2.7764451051977987 * np.std(v, ddof=1) / math.sqrt(5)
    assert (s["mean"], s["ci_lo"], s["ci_hi"]) == pytest.approx((3.0, 3.0 - half, 3.0 + half))
    one = t_summary([0.7])
    assert one["mean"] == 0.7 and math.isnan(one["ci_lo"])
    assert t_summary([1.0, float("nan"), 3.0])["n"] == 2


# --------------------------------------------------------------------------- tables


def _split(n: int = 400, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.8).astype(np.uint8)
    return {"y": y,
            "p_ewma": np.clip(0.8 + 0.1 * rng.standard_normal(n), 0, 1).astype(np.float32),
            "p_hmm": np.clip(0.8 + 0.1 * rng.standard_normal(n), 0, 1).astype(np.float32),
            "p_oracle": np.where(y == 1, 0.95, 0.05).astype(np.float32)}


def test_evaluate_split_tables_and_pairing() -> None:
    d = _split()
    rng = np.random.default_rng(1)
    preds = {("ewma", None): d["p_ewma"].astype(float), ("hmm", None): d["p_hmm"].astype(float),
             ("oracle", None): d["p_oracle"].astype(float)}
    for k in range(5):
        preds[("gru", k)] = np.clip(np.where(d["y"] == 1, 0.85, 0.3) + 0.05 * rng.standard_normal(400), 0, 1)
        if k != 3:                                            # mlp seed 3 missing
            preds[("mlp", k)] = np.clip(0.8 + 0.05 * rng.standard_normal(400), 0, 1)
    t = evaluate_split("test_id", d, preds)
    assert len(t["metrics"]) == (3 + 5 + 4) * 4
    assert len(t["reliability"]) == (3 + 5 + 4) * 15
    seeds = {r["seed"] for r in t["metrics"] if r["predictor"] == "ewma"}
    assert seeds == {""}
    paired = {(r["comparison"], r["metric"]): r for r in t["paired"]}
    assert paired[("gru-ewma", "brier")]["n_seeds"] == 5
    assert paired[("gru-mlp", "brier")]["n_seeds"] == 4               # seed 3 has no MLP pair
    # paired difference equals per-seed metric difference
    y = d["y"].astype(float)
    diffs = [brier(preds[("gru", k)], y) - brier(preds[("ewma", None)], y) for k in range(5)]
    assert paired[("gru-ewma", "brier")]["mean"] == pytest.approx(np.mean(diffs))
    assert paired[("gru-ewma", "brier")]["t_crit"] == pytest.approx(2.776, abs=5e-4)
    assert paired[("gru-ewma", "brier")]["mean"] < 0                 # the better predictor wins
    summ = {(r["predictor"], r["metric"]): r for r in t["summary"]}
    assert summ[("gru", "auc")]["n_seeds"] == 5 and summ[("oracle", "auc")]["n_seeds"] == 1


def test_evaluate_split_rejects_length_mismatch() -> None:
    d = _split()
    with pytest.raises(ValueError):
        evaluate_split("x", d, {("ewma", None): np.zeros(3)})


# --------------------------------------------------------------------------- end to end (needs torch)


def test_cli_end_to_end(tmp_path) -> None:
    pytest.importorskip("torch")
    from fedgrant.ml import synthetic
    from fedgrant.ml.evaluate import main
    from fedgrant.ml.schema import load_split
    from fedgrant.ml.train import train_model

    data = tmp_path / "data"
    for split, seed, rounds in (("train", 1000, 2), ("val", 2000, 2), ("test_id", 3000, 2), ("test_ood", 4000, 2)):
        synthetic.generate_split(split, 1, seed, data, n_clients=20, n_rounds=rounds, verbose=False)
    tr, va = load_split(data / "train.npz"), load_split(data / "val.npz")
    models = tmp_path / "models"
    for name in ("gru", "mlp"):
        for k in (0, 1):
            train_model(name, k, tr, va, models, max_epochs=1, batch=256, threads=1, device="cpu", log=lambda s: None)
    out = tmp_path / "summary"
    main(["--data", str(data), "--models", str(models), "--seeds", "0", "1", "--out", str(out)])
    for f in ("ml_metrics.csv", "ml_metrics_summary.csv", "ml_paired.csv", "ml_reliability.csv", "ml_eval_meta.json"):
        assert (out / f).is_file(), f
    rows = list(csv.DictReader((out / "ml_metrics.csv").open()))
    assert rows[0].keys() == {"split", "predictor", "seed", "metric", "value"}
    got = {(r["split"], r["predictor"], r["seed"]) for r in rows}
    assert ("test_ood", "gru", "1") in got and ("test_id", "oracle", "") in got
    paired = list(csv.DictReader((out / "ml_paired.csv").open()))
    assert {r["comparison"] for r in paired} == {"gru-ewma", "gru-hmm", "gru-mlp"}
    info = json.loads((out / "ml_eval_meta.json").read_text())
    assert info["splits"]["test_id"]["seeds_missing"]["gru"] == []