"""Tests for fedgrant/ml/plots_ml.py: figures are built from the summary CSVs alone."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from fedgrant.ml.evaluate import FIELDS, _write_csv, evaluate_split  # noqa: E402
from fedgrant.ml.plots_ml import pooled_bins, run  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import bench_predictors  # noqa: E402


def _summary_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    out = tmp_path / "summary"
    tables = {"metrics": [], "summary": [], "paired": [], "reliability": []}
    for split in ("test_id", "test_ood"):
        y = (rng.random(500) < 0.85).astype(np.uint8)
        preds = {(n, None): np.clip(0.85 + 0.1 * rng.standard_normal(500), 0, 1) for n in ("ewma", "hmm")}
        preds[("oracle", None)] = np.where(y == 1, 0.97, 0.03)
        for k in range(5):
            for n in ("mlp", "gru"):
                preds[(n, k)] = np.clip(np.where(y == 1, 0.9, 0.4) + 0.08 * rng.standard_normal(500), 0, 1)
        part = evaluate_split(split, {"y": y}, preds)
        for key in tables:
            tables[key].extend(part[key])
    names = {"metrics": "ml_metrics.csv", "summary": "ml_metrics_summary.csv", "paired": "ml_paired.csv",
             "reliability": "ml_reliability.csv"}
    for key, f in names.items():
        _write_csv(out / f, tables[key], FIELDS[f])
    bench_predictors.main(["--random", "--reps", "20", "--warmup", "2", "--out", str(out / "pred_latency.csv")])
    return out


def test_all_figures_from_csvs(tmp_path) -> None:
    sd = _summary_dir(tmp_path)
    for f in sd.glob("*.csv"):
        assert "np." not in f.read_text(), f.name            # plain numbers only
    made = run(sd, tmp_path / "figures")
    names = sorted(p.name for p in made)
    assert names == ["ml_latency.pdf", "ml_metrics.pdf", "ml_reliability.pdf"]
    for p in made:
        assert p.stat().st_size > 2000 and p.read_bytes()[:4] == b"%PDF"


def test_missing_inputs_are_skipped(tmp_path) -> None:
    sd = _summary_dir(tmp_path)
    (sd / "pred_latency.csv").unlink()
    (sd / "ml_reliability.csv").unlink()
    assert [p.name for p in run(sd, tmp_path / "f")] == ["ml_metrics.pdf"]
    assert run(tmp_path / "nothing", tmp_path / "g") == []


def test_pooled_bins_weighting(tmp_path) -> None:
    sd = _summary_dir(tmp_path)
    rows = list(csv.DictReader((sd / "ml_reliability.csv").open()))
    pooled = pooled_bins(rows)
    gru = pooled[("test_id", "gru")]
    assert len(gru) == 15
    seeds = [r for r in rows if r["split"] == "test_id" and r["predictor"] == "gru" and r["bin"] == "13"]
    n = sum(float(r["n"]) for r in seeds)
    assert gru[13]["n"] == n
    ws = [float(r["n"]) * float(r["mean_y"]) for r in seeds if float(r["n"]) > 0]
    assert gru[13]["mean_y"] == pytest.approx(sum(ws) / n)