"""Run full-participation FEMNIST training or simulated lossy FL experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fedgrant.encoder.ef import ErrorFeedback
from fedgrant.fl.bank import ValueBank
from fedgrant.fl.femnist import FEMNISTDataset, load_femnist
from fedgrant.fl.local import make_client_loaders
from fedgrant.fl.model import create_model
from fedgrant.fl.server import run_fl_simulated, run_full
from fedgrant.policy.constants import D_MODEL, K
from fedgrant.policy.types import PolicyConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_metadata(shard_root: str | Path, allow_unverified: bool) -> dict[str, Any]:
    root = Path(shard_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"FEMNIST client manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = manifest.get("dataset_provenance")
    recorded_hash = manifest.get("provenance_sha256")
    if not isinstance(provenance, dict) or not recorded_hash:
        if not allow_unverified:
            raise ValueError(
                f"dataset provenance is missing from {manifest_path}; reproduce FEMNIST "
                "with scripts/get_femnist.sh or pass --allow-unverified-data explicitly"
            )
        return {
            "source": "unverified",
            "preprocessing_version": "unknown",
            "provenance": None,
            "provenance_sha256": None,
            "client_manifest": str(manifest_path.resolve()),
            "client_manifest_sha256": _sha256(manifest_path),
        }

    provenance_path = root.parent / "provenance.json"
    if provenance_path.is_file() and _sha256(provenance_path) != recorded_hash:
        raise ValueError(f"provenance hash mismatch between {manifest_path} and {provenance_path}")
    if not provenance.get("source") or not provenance.get("preprocessing_version"):
        raise ValueError(f"incomplete FEMNIST provenance in {manifest_path}")
    return {
        "source": provenance["source"],
        "preprocessing_version": provenance["preprocessing_version"],
        "provenance": provenance,
        "provenance_sha256": recorded_hash,
        "provenance_path": str(provenance_path.resolve()),
        "client_manifest": str(manifest_path.resolve()),
        "client_manifest_sha256": _sha256(manifest_path),
        "client_files": manifest.get("files", []),
    }


def _test_clients(dataset: FEMNISTDataset):
    return [client.load_test() for client in dataset.clients]


def run_experiment(
    seed: int,
    rounds: int = 300,
    policy: str = "full",
    beta: float = 0.10,
    scenario: str = "L2",
    split: str = "test",
    trace_dir: str = "data/traces",
    no_ef: bool = False,
    data_root: str | Path = "data/femnist/reproduced/leaf/data",
    shard_root: str | Path = "data/femnist/reproduced/clients",
    bank_path: str | Path | None = None,
    device: str = "auto",
    log_every: int = 5,
    eval_every: int = 5,
    torch_threads: int = 1,
    client_workers: int = 10,
    allow_unverified_data: bool = False,
) -> list[dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_femnist(data_root, shard_root, 200, 10)
    dataset_metadata = _dataset_metadata(shard_root, allow_unverified_data)
    train_clients = [client.load_train() for client in dataset.clients]
    weights = dataset.client_weights
    loaders = make_client_loaders(train_clients, seed, batch_size=32)
    error_feedback = [ErrorFeedback(D_MODEL, enabled=not no_ef) for _ in dataset.clients]
    model = create_model(seed)

    pol_lower = policy.lower()
    run_dir = Path("data/runs")
    run_dir.mkdir(parents=True, exist_ok=True)

    if pol_lower == "full":
        bank = ValueBank(weights, K) if bank_path else None
        history = run_full(
            model=model,
            train_clients=train_clients,
            test_client_loader=lambda: _test_clients(dataset),
            client_weights=weights,
            loaders=loaders,
            error_feedback=error_feedback,
            num_rounds=rounds,
            seed=seed,
            device=device,
            eval_every=eval_every,
            bank=bank,
            log_every=log_every,
            client_workers=client_workers,
        )
        if bank is not None:
            bank.save(
                bank_path, seed=seed, num_rounds=rounds, dataset_metadata=dataset_metadata
            )
        metrics_path = run_dir / f"femnist_full_s{seed}.json"
    else:
        # Map policy name to PolicyConfig
        selector = "value"
        predictor_name = "ewma"
        if "rr" in pol_lower:
            selector = "rr"
        elif "srpt" in pol_lower:
            selector = "srpt"

        if "oracle" in pol_lower:
            predictor_name = "oracle"
        elif "hmm" in pol_lower:
            predictor_name = "hmm"
        elif "gru" in pol_lower:
            predictor_name = "gru"
        elif "mlp" in pol_lower:
            predictor_name = "mlp"

        cfg = PolicyConfig(selector=selector, predictor=predictor_name)

        history = run_fl_simulated(
            model=model,
            train_clients=train_clients,
            test_client_loader=lambda: _test_clients(dataset),
            client_weights=weights,
            loaders=loaders,
            error_feedback=error_feedback,
            num_rounds=rounds,
            seed=seed,
            beta=beta,
            scenario=scenario,
            split=split,
            trace_dir=trace_dir,
            policy_config=cfg,
            device=device,
            eval_every=eval_every,
            log_every=log_every,
            client_workers=client_workers,
        )
        metrics_path = run_dir / f"fl_{pol_lower}_b{int(beta*100):02d}_{scenario}_s{seed}.json"

    metrics_path.write_text(json.dumps({
        "seed": seed,
        "rounds": rounds,
        "policy": policy,
        "beta": beta,
        "scenario": scenario,
        "error_feedback": not no_ef,
        "device": device,
        "num_clients": len(dataset.clients),
        "num_train_samples": sum(c.num_train for c in dataset.clients),
        "num_test_samples": sum(c.num_test for c in dataset.clients),
        "dataset": dataset_metadata,
        "history": history,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metrics to {metrics_path}", flush=True)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--policy", default="full")
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--scenario", default="L2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--no-ef", action="store_true", help="Disable error feedback (A1)")
    parser.add_argument("--data-root", default="data/femnist/reproduced/leaf/data")
    parser.add_argument("--shard-root", default="data/femnist/reproduced/clients")
    parser.add_argument("--bank", default=None, help="optional NPZ path for adjusted-update energies")
    parser.add_argument(
        "--allow-unverified-data", action="store_true",
        help="permit training from shards without reproduction provenance",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--client-workers", type=int, default=10)
    args = parser.parse_args()
    run_experiment(
        seed=args.seed,
        rounds=args.rounds,
        policy=args.policy,
        beta=args.beta,
        scenario=args.scenario,
        split=args.split,
        no_ef=args.no_ef,
        data_root=args.data_root,
        shard_root=args.shard_root,
        bank_path=args.bank,
        device=args.device,
        log_every=args.log_every,
        eval_every=args.eval_every,
        torch_threads=args.torch_threads,
        client_workers=args.client_workers,
        allow_unverified_data=args.allow_unverified_data,
    )


if __name__ == "__main__":
    main()
