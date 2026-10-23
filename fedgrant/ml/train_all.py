"""Train all seeds of both models and pick the production seed per model (Section 4.4).

    python -m fedgrant.ml.train_all --data data/mldata_synth --out models/
    python -m fedgrant.ml.train_all --data data/mldata --out models/ --resume     # skip finished seeds

Each (model, seed) runs as its own `python -m fedgrant.ml.train` process, so
the recorded peak RSS and wall time belong to that run alone. Afterwards
<out>/train_summary.json lists every seed's best validation loss and the
production seed (lowest validation loss, ties -> lower seed); export.py reads it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MODELS: tuple[str, ...] = ("gru", "mlp")
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
SUMMARY_FILE = "train_summary.json"


def _done(out: Path, model: str, seed: int) -> bool:
    return (out / f"{model}_seed{seed}.pt").is_file() and (out / f"{model}_seed{seed}_cost.json").is_file()


def summarize(out_dir: str | Path, models: tuple[str, ...] = MODELS,
              seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Collect cost records and choose the production seed of each model; write train_summary.json."""
    out = Path(out_dir)
    summary: dict[str, Any] = {}
    for m in models:
        per_seed = {}
        for k in seeds:
            f = out / f"{m}_seed{k}_cost.json"
            if f.is_file():
                c = json.loads(f.read_text())
                per_seed[str(k)] = {"best_val_loss": c["best_val_loss"], "best_epoch": c["best_epoch"],
                                    "epochs_run": c["epochs_run"], "train_time_s": c["train_time_s"],
                                    "peak_rss_mb": c["peak_rss_mb"]}
        missing = [k for k in seeds if str(k) not in per_seed]
        prod = min(per_seed, key=lambda s: (per_seed[s]["best_val_loss"], int(s))) if per_seed else None
        summary[m] = {"seeds": per_seed, "missing": missing,
                      "production_seed": None if prod is None else int(prod)}
    (out / SUMMARY_FILE).write_text(json.dumps(summary, indent=1))
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-epochs", type=int, default=None, help="override (testing only)")
    ap.add_argument("--resume", action="store_true", help="skip (model, seed) pairs already trained")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    t0 = time.time()
    for m in a.models:
        for k in a.seeds:
            if a.resume and _done(out, m, k):
                print(f"== {m} seed {k}: already trained, skipping", flush=True)
                continue
            cmd = [sys.executable, "-m", "fedgrant.ml.train", "--model", m, "--seed", str(k),
                   "--data", a.data, "--out", str(out), "--threads", str(a.threads), "--device", a.device]
            if a.max_epochs is not None:
                cmd += ["--max-epochs", str(a.max_epochs)]
            print(f"== {m} seed {k}: {' '.join(cmd)}", flush=True)
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"!! {m} seed {k} failed with exit code {r.returncode}", flush=True)
                failed.append(f"{m}:{k}")
    s = summarize(out, tuple(a.models), tuple(a.seeds))
    print(f"done in {time.time() - t0:.0f}s; production seeds: "
          + ", ".join(f"{m}={s[m]['production_seed']}" for m in a.models), flush=True)
    if failed:
        print(f"failed runs: {failed}", flush=True)
        raise SystemExit(1)
    return s


if __name__ == "__main__":
    main()