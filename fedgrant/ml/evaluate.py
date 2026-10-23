"""ML evaluation of the five predictors on Test-ID and Test-OOD (Section 4.7).

    python -m fedgrant.ml.evaluate --data data/mldata_synth --models models/ --splits test_id test_ood

P-EWMA, P-HMM and P-ORACLE use the p_ewma / p_hmm / p_oracle columns stored in
each split (one value per sample). P-MLP and P-GRU are evaluated for every
training seed (default 0-4) from models/{mlp,gru}_seed{k}.pt, each with the
normalization statistics stored in its checkpoint. All predictors are scored on
the identical samples.

Metrics: Brier; ECE with 15 equal-width bins (sum_b n_b/n |ybar_b - pbar_b|);
ROC-AUC (rank statistic, ties get average ranks); BCE with p clipped to
[1e-6, 1 - 1e-6].

Writes (default results/summary/):
  ml_metrics.csv          long format: split, predictor, seed, metric, value
                          (seed is empty for the single-valued predictors)
  ml_metrics_summary.csv  split, predictor, metric, n_seeds, mean, sd, ci_lo, ci_hi
  ml_paired.csv           P-GRU minus {P-EWMA, P-HMM, P-MLP}, paired by seed on identical
                          samples: split, comparison, metric, n_seeds, mean, sd, ci_lo, ci_hi,
                          t_crit, seed_values
  ml_reliability.csv      split, predictor, seed, bin, lo, hi, n, mean_p, mean_y
  ml_eval_meta.json       seeds found / missing, sample counts, data meta

The metric functions are NumPy/SciPy only; PyTorch is imported only to run the
P-MLP / P-GRU checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import rankdata
from scipy.stats import t as student_t

from fedgrant.ml.schema import load_split, meta
from fedgrant.ml.sequences import resolve_split_path

N_BINS: int = 15
BCE_EPS: float = 1e-6
METRICS: tuple[str, ...] = ("brier", "ece", "auc", "bce")
STORED: dict[str, str] = {"ewma": "p_ewma", "hmm": "p_hmm", "oracle": "p_oracle"}
LEARNED: tuple[str, ...] = ("mlp", "gru")
BASELINES_FOR_PAIRING: tuple[str, ...] = ("ewma", "hmm", "mlp")
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------- metrics


def _check(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.ndim != 1 or p.shape != y.shape:
        raise ValueError(f"p and y must be 1-D of equal length, got {p.shape} and {y.shape}")
    if p.size == 0:
        raise ValueError("empty input")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be finite and in [0, 1]")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must be 0/1")
    return p, y


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """(1/n) sum (p - y)^2."""
    p, y = _check(p, y)
    return float(np.mean((p - y) ** 2))


def bin_index(p: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Equal-width bins on [0, 1]: [0, 1/B), ..., [(B-1)/B, 1] (p = 1 goes to the last bin)."""
    return np.minimum((np.asarray(p, np.float64) * n_bins).astype(np.int64), n_bins - 1)


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> dict[str, np.ndarray]:
    """Per bin: lo, hi, n, mean_p, mean_y (mean_* are NaN for empty bins)."""
    p, y = _check(p, y)
    b = bin_index(p, n_bins)
    n = np.bincount(b, minlength=n_bins).astype(np.int64)
    sp = np.bincount(b, weights=p, minlength=n_bins)
    sy = np.bincount(b, weights=y, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mp = np.where(n > 0, sp / n, np.nan)
        my = np.where(n > 0, sy / n, np.nan)
    edges = np.arange(n_bins + 1) / n_bins
    return {"lo": edges[:-1], "hi": edges[1:], "n": n, "mean_p": mp, "mean_y": my}


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    """ECE = sum_b (n_b / n) |ybar_b - pbar_b| over non-empty bins."""
    r = reliability_bins(p, y, n_bins)
    nz = r["n"] > 0
    return float(np.sum(r["n"][nz] / r["n"].sum() * np.abs(r["mean_y"][nz] - r["mean_p"][nz])))


def roc_auc(p: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC = Pr[p_pos > p_neg] + 0.5 Pr[tie] via the Mann-Whitney rank statistic.

    NaN if only one class is present.
    """
    p, y = _check(p, y)
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)                                   # average ranks for ties
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def bce(p: np.ndarray, y: np.ndarray, eps: float = BCE_EPS) -> float:
    """Binary cross-entropy with p clipped to [eps, 1 - eps]."""
    p, y = _check(p, y)
    q = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log1p(-q)))


METRIC_FN = {"brier": brier, "ece": ece, "auc": roc_auc, "bce": bce}


def all_metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {m: METRIC_FN[m](p, y) for m in METRICS}


def t_summary(values: Iterable[float], conf: float = 0.95) -> dict[str, float]:
    """Mean, sample sd and Student-t CI over seeds (t_{0.975,4} = 2.776 for 5 seeds).

    With one value the CI is NaN. NaN values are dropped first.
    """
    v = np.asarray([x for x in values if not math.isnan(x)], dtype=np.float64)
    n = int(v.size)
    if n == 0:
        return dict(n=0, mean=float("nan"), sd=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                    t_crit=float("nan"))
    mean = float(v.mean())
    if n == 1:
        return dict(n=1, mean=mean, sd=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                    t_crit=float("nan"))
    sd = float(v.std(ddof=1))
    tc = float(student_t.ppf(0.5 + conf / 2.0, n - 1))
    half = tc * sd / math.sqrt(n)
    return dict(n=n, mean=mean, sd=sd, ci_lo=mean - half, ci_hi=mean + half, t_crit=tc)


# --------------------------------------------------------------------------- predictions


def learned_predictions(model_name: str, ckpt_path: str | Path, split: dict[str, np.ndarray],
                        device: str = "cpu") -> np.ndarray:
    """p_hat of one P-MLP / P-GRU checkpoint on every sample of `split`, float32 [ns]."""
    import torch

    from fedgrant.ml.sequences import norm_stats_from_dicts
    from fedgrant.ml.train import build_data, load_checkpoint, predict_proba

    model, ck = load_checkpoint(ckpt_path, device=device)
    if ck["model"] != model_name:
        raise ValueError(f"{ckpt_path} holds {ck['model']!r}, expected {model_name!r}")
    stats = norm_stats_from_dicts(ck["norm_stats"]["gru"], ck["norm_stats"]["mlp"])
    return predict_proba(model, model_name, build_data(model_name, split, stats), torch.device(device))


def collect_predictions(split: dict[str, np.ndarray], models_dir: str | Path,
                        seeds: Iterable[int] = SEEDS, device: str = "cpu",
                        log=print) -> dict[tuple[str, int | None], np.ndarray]:
    """{(predictor, seed or None): p_hat} for every predictor available."""
    preds: dict[tuple[str, int | None], np.ndarray] = {}
    for name, col in STORED.items():
        preds[(name, None)] = np.asarray(split[col], dtype=np.float64)
    md = Path(models_dir)
    for name in LEARNED:
        for k in seeds:
            f = md / f"{name}_seed{k}.pt"
            if f.is_file():
                preds[(name, int(k))] = learned_predictions(name, f, split, device).astype(np.float64)
            else:
                log(f"warning: {f} not found; {name} seed {k} skipped")
    return preds


# --------------------------------------------------------------------------- evaluation


def evaluate_split(split_name: str, split: dict[str, np.ndarray],
                   preds: dict[tuple[str, int | None], np.ndarray]) -> dict[str, list[dict[str, Any]]]:
    """Rows for the four output tables of one split."""
    y = np.asarray(split["y"], dtype=np.float64)
    metric_rows: list[dict[str, Any]] = []
    rel_rows: list[dict[str, Any]] = []
    scores: dict[tuple[str, int | None], dict[str, float]] = {}
    for (name, seed), p in preds.items():
        if p.shape != y.shape:
            raise ValueError(f"{name} seed {seed}: {p.shape[0]} predictions for {y.shape[0]} samples")
        s = all_metrics(p, y)
        scores[(name, seed)] = s
        for m in METRICS:
            metric_rows.append(dict(split=split_name, predictor=name, seed="" if seed is None else seed,
                                    metric=m, value=s[m]))
        r = reliability_bins(p, y)
        for b in range(N_BINS):
            rel_rows.append(dict(split=split_name, predictor=name, seed="" if seed is None else seed, bin=b,
                                 lo=r["lo"][b], hi=r["hi"][b], n=int(r["n"][b]),
                                 mean_p=r["mean_p"][b], mean_y=r["mean_y"][b]))

    summary_rows: list[dict[str, Any]] = []
    for name in (*STORED, *LEARNED):
        seeds = sorted(k for (n, k) in scores if n == name and k is not None)
        for m in METRICS:
            if name in STORED:
                if (name, None) not in scores:
                    continue
                ts = t_summary([scores[(name, None)][m]])
            elif seeds:
                ts = t_summary([scores[(name, k)][m] for k in seeds])
            else:
                continue
            summary_rows.append(dict(split=split_name, predictor=name, metric=m, n_seeds=ts["n"],
                                     mean=ts["mean"], sd=ts["sd"], ci_lo=ts["ci_lo"], ci_hi=ts["ci_hi"]))

    paired_rows: list[dict[str, Any]] = []
    gru_seeds = sorted(k for (n, k) in scores if n == "gru" and k is not None)
    for base in BASELINES_FOR_PAIRING:
        for m in METRICS:
            diffs, used = [], []
            for k in gru_seeds:
                ref = scores.get((base, None)) if base in STORED else scores.get((base, k))
                if ref is None:
                    continue                          # e.g. P-MLP seed k missing: no pair for k
                diffs.append(scores[("gru", k)][m] - ref[m])
                used.append(k)
            if not diffs:
                continue
            ts = t_summary(diffs)
            paired_rows.append(dict(split=split_name, comparison=f"gru-{base}", metric=m, n_seeds=ts["n"],
                                    mean=ts["mean"], sd=ts["sd"], ci_lo=ts["ci_lo"], ci_hi=ts["ci_hi"],
                                    t_crit=ts["t_crit"],
                                    seed_values=";".join(f"{k}:{d:.10g}" for k, d in zip(used, diffs))))
    return dict(metrics=metric_rows, summary=summary_rows, paired=paired_rows, reliability=rel_rows)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})


FIELDS = {
    "ml_metrics.csv": ["split", "predictor", "seed", "metric", "value"],
    "ml_metrics_summary.csv": ["split", "predictor", "metric", "n_seeds", "mean", "sd", "ci_lo", "ci_hi"],
    "ml_paired.csv": ["split", "comparison", "metric", "n_seeds", "mean", "sd", "ci_lo", "ci_hi", "t_crit",
                      "seed_values"],
    "ml_reliability.csv": ["split", "predictor", "seed", "bin", "lo", "hi", "n", "mean_p", "mean_y"],
}


def run(data_dir: str | Path, models_dir: str | Path, splits: Iterable[str] = ("test_id", "test_ood"),
        out_dir: str | Path = "results/summary", seeds: Iterable[int] = SEEDS, device: str = "cpu",
        log=print) -> dict[str, list[dict[str, Any]]]:
    """Evaluate every split and write the four CSVs plus ml_eval_meta.json."""
    seeds = tuple(int(k) for k in seeds)
    tables: dict[str, list[dict[str, Any]]] = {"metrics": [], "summary": [], "paired": [], "reliability": []}
    info: dict[str, Any] = {"seeds_requested": list(seeds), "splits": {}}
    for s in splits:
        log(f"evaluating {s} ...")
        d = load_split(resolve_split_path(data_dir, s))
        preds = collect_predictions(d, models_dir, seeds, device, log)
        part = evaluate_split(s, d, preds)
        for k in tables:
            tables[k].extend(part[k])
        info["splits"][s] = dict(
            n_samples=int(d["y"].size), positive_rate=float(np.mean(d["y"])), data_meta=meta(d),
            seeds_found={n: sorted(k for (m, k) in preds if m == n) for n in LEARNED},
            seeds_missing={n: [k for k in seeds if (n, k) not in preds] for n in LEARNED},
        )
    out = Path(out_dir)
    names = {"metrics": "ml_metrics.csv", "summary": "ml_metrics_summary.csv", "paired": "ml_paired.csv",
             "reliability": "ml_reliability.csv"}
    for k, fname in names.items():
        _write_csv(out / fname, tables[k], FIELDS[fname])
    (out / "ml_eval_meta.json").write_text(json.dumps(info, indent=1))
    for r in tables["summary"]:
        if r["metric"] == "brier":
            log(f"  {r['split']:8s} {r['predictor']:6s} Brier {r['mean']:.5f} (n_seeds={r['n_seeds']})")
    log(f"wrote {out}/ml_metrics.csv, ml_metrics_summary.csv, ml_paired.csv, ml_reliability.csv")
    return tables


def main(argv: list[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    ap = argparse.ArgumentParser(description="Evaluate the five predictors (Section 4.7).")
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", default="models/")
    ap.add_argument("--splits", nargs="+", default=["test_id", "test_ood"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--out", default="results/summary")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    return run(a.data, a.models, a.splits, a.out, a.seeds, a.device, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()