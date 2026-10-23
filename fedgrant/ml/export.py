"""Export the production P-GRU / P-MLP checkpoints to NumPy .npz files (Section 4.4).

    python -m fedgrant.ml.export --models models/

For each model the production seed is the one with the lowest best validation
loss among models/{gru,mlp}_seed{k}.pt (ties -> lower seed). The .npz holds
float32 weights plus the float64 standardization statistics and metadata, so
NumPy inference (predictors/gru_numpy.py, mlp_numpy.py) needs nothing else.
Load with np.load(path, allow_pickle=False).

models/gru_prod.npz
  weight_ih_l0 [96, 8], weight_hh_l0 [96, 32], bias_ih_l0 [96], bias_hh_l0 [96]
      PyTorch nn.GRU layout; rows are the gates in the order r, z, n (32 rows each):
      r = sig(W_ir x + b_ir + W_hr h + b_hr), z = sig(W_iz x + b_iz + W_hz h + b_hz),
      n = tanh(W_in x + b_in + r * (W_hn h + b_hn)), h' = (1 - z) * n + z * h
  head_w1 [32, 48], head_b1 [32], head_w2 [1, 32], head_b2 [1]
      logit = head_w2 @ relu(head_w1 @ [h; c~] + head_b1) + head_b2
  mu_obs, s_obs [8]; mu_ctx, s_ctx [16]          (float64)
models/mlp_prod.npz
  w1 [64, 48], b1 [64], w2 [64, 64], b2 [64], w3 [1, 64], b3 [1]
  mu_x48, s_x48 [48]                              (float64)
Both: format_version, model, seed, best_val_loss, best_epoch, n_params, gate_order (gru),
source_checkpoint.

A summary of the choice is written to models/export_summary.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION: int = 1
GATE_ORDER: str = "r,z,n"

GRU_TENSORS: dict[str, tuple[str, tuple[int, ...]]] = {   # npz key -> (state_dict key, shape)
    "weight_ih_l0": ("gru.weight_ih_l0", (96, 8)),
    "weight_hh_l0": ("gru.weight_hh_l0", (96, 32)),
    "bias_ih_l0": ("gru.bias_ih_l0", (96,)),
    "bias_hh_l0": ("gru.bias_hh_l0", (96,)),
    "head_w1": ("head.0.weight", (32, 48)),
    "head_b1": ("head.0.bias", (32,)),
    "head_w2": ("head.2.weight", (1, 32)),
    "head_b2": ("head.2.bias", (1,)),
}
MLP_TENSORS: dict[str, tuple[str, tuple[int, ...]]] = {
    "w1": ("net.0.weight", (64, 48)),
    "b1": ("net.0.bias", (64,)),
    "w2": ("net.2.weight", (64, 64)),
    "b2": ("net.2.bias", (64,)),
    "w3": ("net.4.weight", (1, 64)),
    "b3": ("net.4.bias", (1,)),
}
NORM_KEYS = {"gru": ("mu_obs", "s_obs", "mu_ctx", "s_ctx"), "mlp": ("mu_x48", "s_x48")}
_CKPT_RE = re.compile(r"^(gru|mlp)_seed(\d+)\.pt$")


def _torch_load(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def list_checkpoints(models_dir: str | Path, model: str) -> dict[int, Path]:
    """{seed: path} of models/<model>_seed<k>.pt."""
    out: dict[int, Path] = {}
    for f in Path(models_dir).glob(f"{model}_seed*.pt"):
        m = _CKPT_RE.match(f.name)
        if m and m.group(1) == model:
            out[int(m.group(2))] = f
    return dict(sorted(out.items()))


def choose_production(models_dir: str | Path, model: str) -> tuple[int, Path, dict[int, float]]:
    """(seed, path, {seed: best_val_loss}): lowest best validation loss, ties -> lower seed."""
    cks = list_checkpoints(models_dir, model)
    if not cks:
        raise FileNotFoundError(f"no {model}_seed*.pt in {models_dir}")
    losses = {k: float(_torch_load(p)["best_val_loss"]) for k, p in cks.items()}
    seed = min(losses, key=lambda k: (losses[k], k))
    return seed, cks[seed], losses


def checkpoint_to_arrays(ck: dict[str, Any]) -> dict[str, np.ndarray]:
    """npz arrays for one checkpoint (see the module docstring)."""
    model = ck["model"]
    spec = GRU_TENSORS if model == "gru" else MLP_TENSORS if model == "mlp" else None
    if spec is None:
        raise ValueError(f"unknown model {model!r}")
    sd = ck["state_dict"]
    expected = {k for k, _ in spec.values()}
    if set(sd) != expected:
        raise ValueError(f"unexpected state_dict keys: {sorted(set(sd) ^ expected)}")
    out: dict[str, np.ndarray] = {}
    for key, (sd_key, shape) in spec.items():
        a = sd[sd_key].detach().cpu().numpy().astype(np.float32)
        if a.shape != shape:
            raise ValueError(f"{sd_key}: shape {a.shape}, expected {shape}")
        out[key] = np.ascontiguousarray(a)
    ns = ck["norm_stats"][model]
    for k in NORM_KEYS[model]:
        out[k] = np.asarray(ns[k], dtype=np.float64)
    out["format_version"] = np.array(FORMAT_VERSION, dtype=np.int64)
    out["model"] = np.array(model)
    out["seed"] = np.array(int(ck["seed"]), dtype=np.int64)
    out["best_val_loss"] = np.array(float(ck["best_val_loss"]), dtype=np.float64)
    out["best_epoch"] = np.array(int(ck["best_epoch"]), dtype=np.int64)
    out["n_params"] = np.array(int(ck["n_params"]), dtype=np.int64)
    if model == "gru":
        out["gate_order"] = np.array(GATE_ORDER)
    return out


def load_prod(path: str | Path) -> dict[str, np.ndarray]:
    """Load a *_prod.npz into a dict (validates format and shapes)."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if int(d["format_version"]) != FORMAT_VERSION:
        raise ValueError(f"{path}: format_version {int(d['format_version'])}, expected {FORMAT_VERSION}")
    model = str(d["model"])
    spec = GRU_TENSORS if model == "gru" else MLP_TENSORS
    for key, (_, shape) in spec.items():
        if d[key].shape != shape or d[key].dtype != np.float32:
            raise ValueError(f"{path}: {key} must be float32 {shape}")
    return d


def export_model(models_dir: str | Path, model: str, seed: int | None = None) -> dict[str, Any]:
    """Write models/<model>_prod.npz from the production (or the given) seed."""
    md = Path(models_dir)
    cks = list_checkpoints(md, model)
    if seed is None:
        seed, path, losses = choose_production(md, model)
    else:
        if seed not in cks:
            raise FileNotFoundError(f"{model}_seed{seed}.pt not in {md}")
        path = cks[seed]
        losses = {k: float(_torch_load(p)["best_val_loss"]) for k, p in cks.items()}
    ck = _torch_load(path)
    arrs = checkpoint_to_arrays(ck)
    arrs["source_checkpoint"] = np.array(path.name)
    out = md / f"{model}_prod.npz"
    np.savez(out, **arrs)  # type: ignore[arg-type]
    load_prod(out)                                            # re-read and validate
    return {"model": model, "production_seed": int(seed), "best_val_loss": float(ck["best_val_loss"]),
            "seed_val_losses": {str(k): v for k, v in losses.items()}, "file": str(out),
            "source_checkpoint": path.name}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="Export production checkpoints to NumPy (Section 4.4).")
    ap.add_argument("--models", default="models/")
    ap.add_argument("--which", nargs="+", default=["gru", "mlp"], choices=("gru", "mlp"))
    ap.add_argument("--seed-gru", type=int, default=None, help="override the production seed")
    ap.add_argument("--seed-mlp", type=int, default=None, help="override the production seed")
    a = ap.parse_args(argv)
    summary = {}
    for m in a.which:
        r = export_model(a.models, m, a.seed_gru if m == "gru" else a.seed_mlp)
        summary[m] = r
        print(f"{m}: seed {r['production_seed']} (val loss {r['best_val_loss']:.5f}) -> {r['file']}", flush=True)
    (Path(a.models) / "export_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()