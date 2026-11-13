"""Train P-GRU or P-MLP for one seed (Section 4.4).

    python -m fedgrant.ml.train --model gru --seed 0 --data data/mldata_synth --out models/

Loss BCEWithLogitsLoss (no class weighting); AdamW(lr 1e-3, betas (0.9, 0.999),
weight decay 1e-4), constant LR; batch 2048; at most 30 epochs; gradient-norm
clipping at 1.0; early stopping on validation loss with patience 5, restoring
the best epoch. Seeds PyTorch, NumPy and the data order.

Writes:
  <out>/norm_stats_gru.json, <out>/norm_stats_mlp.json   (train split only)
  <out>/{gru,mlp}_seed{k}.pt        best epoch: state_dict + norm stats + metadata
  <out>/{gru,mlp}_seed{k}_cost.json DL cost record (Section 4.7)

Load a checkpoint with torch.load(path, weights_only=True): it holds only
tensors and plain Python values.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from fedgrant.ml.models import make_model, n_params
from fedgrant.ml.schema import load_split, meta
from fedgrant.ml.sequences import (
    GRUData,
    MLPData,
    NormStats,
    compute_norm_stats,
    check_policy_revision,
    resolve_split_path,
    save_norm_stats,
)

LR: float = 1e-3
BETAS: tuple[float, float] = (0.9, 0.999)
WEIGHT_DECAY: float = 1e-4
BATCH: int = 2048
MAX_EPOCHS: int = 30
PATIENCE: int = 5
GRAD_CLIP: float = 1.0
THREADS: int = 8
EVAL_BATCH: int = 16384                 # evaluation only; does not affect training
CKPT_VERSION: int = 1


# --------------------------------------------------------------------------- helpers


def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (getrusage; psutil current RSS as fallback)."""
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (1024.0 * 1024.0) if sys.platform == "darwin" else r / 1024.0   # bytes on macOS, KB on Linux
    except (ImportError, OSError):
        import psutil

        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def pick_device(device: str | None) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def seed_everything(seed: int) -> np.random.Generator:
    """Seed PyTorch and NumPy; return the generator that fixes the data order."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)


def _to_t(a: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


def _forward(model: nn.Module, name: str, data: GRUData | MLPData, idx: np.ndarray,
             device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(logits, y) for the samples idx."""
    if name == "gru":
        seq, ln, ctx, y = data.batch(idx)                     # type: ignore[union-attr]
        logits = model(_to_t(seq, device), torch.from_numpy(ln), _to_t(ctx, device))
    else:
        x, y = data.batch(idx)                                # type: ignore[union-attr]
        logits = model(_to_t(x, device))
    return logits, _to_t(y, device)


@torch.no_grad()
def eval_loss(model: nn.Module, name: str, data: GRUData | MLPData, device: torch.device,
              batch: int = EVAL_BATCH) -> float:
    """Mean BCE over all samples of `data` (exact, sample-weighted)."""
    model.eval()
    lossf = nn.BCEWithLogitsLoss(reduction="sum")
    n = len(data)
    total = 0.0
    for lo in range(0, n, batch):
        logits, y = _forward(model, name, data, np.arange(lo, min(n, lo + batch)), device)
        total += float(lossf(logits, y))
    return total / n


@torch.no_grad()
def predict_proba(model: nn.Module, name: str, data: GRUData | MLPData, device: torch.device,
                  batch: int = EVAL_BATCH) -> np.ndarray:
    """sigmoid(logits) for every sample, float32 [n] (used by evaluate.py)."""
    model.eval()
    n = len(data)
    out = np.empty(n, np.float32)
    for lo in range(0, n, batch):
        hi = min(n, lo + batch)
        logits, _ = _forward(model, name, data, np.arange(lo, hi), device)
        out[lo:hi] = torch.sigmoid(logits).cpu().numpy()
    return out


def build_data(name: str, split: dict[str, np.ndarray], stats: NormStats) -> GRUData | MLPData:
    if name == "gru":
        return GRUData(split, stats)
    if name == "mlp":
        return MLPData(split, stats)
    raise ValueError(f"unknown model {name!r}")


# --------------------------------------------------------------------------- training


def train_model(
    name: str,
    seed: int,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    out_dir: str | Path,
    *,
    stats: NormStats | None = None,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch: int = BATCH,
    threads: int = THREADS,
    device: str | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train one model for one seed; write the checkpoint and cost record. Returns the cost record."""
    t_wall0 = time.perf_counter()
    if name not in ("gru", "mlp"):
        raise ValueError(f"unknown model {name!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(threads))
    dev = pick_device(device)
    rng = seed_everything(int(seed))

    if stats is None:
        stats = compute_norm_stats(train)                     # train split only
    save_norm_stats(stats, out)
    dtr, dva = build_data(name, train, stats), build_data(name, val, stats)
    n_tr = len(dtr)
    if n_tr == 0 or len(dva) == 0:
        raise ValueError("empty train or validation split")

    model = make_model(name).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    lossf = nn.BCEWithLogitsLoss()

    best_val, best_epoch, best_state = math.inf, 0, None
    history: list[dict[str, float]] = []
    bad = 0
    t_train0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = rng.permutation(n_tr)
        tot = 0.0
        for lo in range(0, n_tr, batch):
            idx = perm[lo : lo + batch]
            logits, y = _forward(model, name, dtr, idx, dev)
            loss = lossf(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tot += float(loss.detach()) * idx.size
        tr_loss = tot / n_tr
        va_loss = eval_loss(model, name, dva, dev)
        dt = time.perf_counter() - t0
        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss, time_s=dt))
        improved = va_loss < best_val
        if improved:
            best_val, best_epoch, bad = va_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        log(f"[{name} seed {seed}] epoch {epoch:2d}  train {tr_loss:.5f}  val {va_loss:.5f}  "
            f"{dt:6.1f}s{'  *' if improved else ''}")
        if bad >= patience:
            break
    train_time = time.perf_counter() - t_train0
    assert best_state is not None
    model.load_state_dict(best_state)                         # restore the best epoch

    epochs_run = len(history)
    data_meta = {k: (meta(v) if "meta_json" in v else {}) for k, v in (("train", train), ("val", val))}
    config = dict(lr=LR, betas=list(BETAS), weight_decay=WEIGHT_DECAY, batch=batch, max_epochs=max_epochs,
                  patience=patience, grad_clip=GRAD_CLIP, threads=int(threads), device=str(dev))
    ckpt = {
        "version": CKPT_VERSION,
        "model": name,
        "seed": int(seed),
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs_run": epochs_run,
        "n_params": n_params(model),
        "norm_stats": {"gru": stats.gru_dict(), "mlp": stats.mlp_dict()},
        "history": history,
        "config": config,
        "data": data_meta,
        "torch_version": str(torch.__version__),
    }
    ckpt_path = out / f"{name}_seed{seed}.pt"
    torch.save(ckpt, ckpt_path)

    times = [h["time_s"] for h in history]
    cost = {
        "model": name,
        "seed": int(seed),
        "n_params": n_params(model),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs_run": epochs_run,
        "early_stopped": epochs_run < max_epochs,
        "epoch_time_s": times,
        "mean_epoch_time_s": float(np.mean(times)),
        "train_time_s": train_time,
        "wall_time_s": time.perf_counter() - t_wall0,
        "peak_rss_mb": peak_rss_mb(),
        "n_train": n_tr,
        "n_val": len(dva),
        "device": str(dev),
        "threads": int(threads),
        "history": history,
        "checkpoint": str(ckpt_path),
    }
    (out / f"{name}_seed{seed}_cost.json").write_text(json.dumps(cost, indent=1))
    log(f"[{name} seed {seed}] best epoch {best_epoch}, val {best_val:.5f}; "
        f"{epochs_run} epochs in {train_time:.0f}s; peak RSS {cost['peak_rss_mb']:.0f} MB -> {ckpt_path}")
    return cost


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """(model with best weights, eval mode; checkpoint dict)."""
    ck = torch.load(path, map_location=device, weights_only=True)
    model = make_model(ck["model"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="Train P-GRU / P-MLP for one seed (Section 4.4).")
    ap.add_argument("--model", required=True, choices=("gru", "mlp"))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--data", required=True, help="directory with train.npz/val.npz (or train/, val/ shard dirs)")
    ap.add_argument("--out", default="models/")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--allow-stale-policy", action="store_true",
                    help="train on data from an older grant-policy revision (pipeline tests only)")
    a = ap.parse_args(argv)
    if a.seed < 0:
        ap.error("--seed must be >= 0")
    if not a.allow_stale_policy:
        check_policy_revision(a.data)
    print(f"loading {a.data} ...", flush=True)
    train = load_split(resolve_split_path(a.data, "train"))
    val = load_split(resolve_split_path(a.data, "val"))
    return train_model(a.model, a.seed, train, val, a.out, max_epochs=a.max_epochs, patience=a.patience,
                       batch=a.batch, threads=a.threads, device=a.device,
                       log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()