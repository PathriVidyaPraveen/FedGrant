"""LEAF FEMNIST loading and bounded-memory client sharding.

The LEAF preprocessing output lives under ``data/femnist/leaf/data/{train,test}``.
This module selects the first 200 sorted writer IDs and creates 20 client shards,
each containing 10 consecutive writers. Shards are NumPy ``.npy`` files so the
training process can memory-map one client at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ClientShard:
    client_id: int
    train_images_path: Path
    train_labels_path: Path
    test_images_path: Path
    test_labels_path: Path
    num_train: int
    num_test: int
    writers: tuple[str, ...]

    def load_train(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _load_arrays(self.train_images_path, self.train_labels_path)

    def load_test(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _load_arrays(self.test_images_path, self.test_labels_path)


@dataclass(frozen=True)
class FEMNISTDataset:
    clients: tuple[ClientShard, ...]
    writer_ids: tuple[str, ...]

    @property
    def client_weights(self) -> np.ndarray:
        counts = np.asarray([c.num_train for c in self.clients], dtype=np.float64)
        if not counts.sum():
            raise ValueError("selected FEMNIST clients have no training examples")
        return (counts / counts.sum()).astype(np.float32)


def _load_arrays(images_path: Path, labels_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    # npy memory maps avoid loading all client data into RAM. Indexing a batch
    # in CyclicBatchLoader materializes just that batch.
    images = np.load(images_path, mmap_mode="c")
    labels = np.load(labels_path, mmap_mode="c")
    return torch.from_numpy(images), torch.from_numpy(labels)


def _json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"LEAF data directory not found: {path}")
    files = sorted(path.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"no LEAF JSON shards found under {path}")
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _users_in_file(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return [str(user) for user in data.get("users", data.get("user_data", {}).keys())]


def _selected_users(train_paths: list[Path], count: int) -> list[str]:
    users: set[str] = set()
    for path in train_paths:
        users.update(_users_in_file(path))
    selected = sorted(users)[:count]
    if len(selected) < count:
        raise ValueError(f"LEAF provides {len(selected)} writers; {count} required")
    return selected


def _collect_group(paths: list[Path], group: tuple[str, ...]) -> dict[str, tuple[list, list]]:
    selected = set(group)
    result: dict[str, tuple[list, list]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        user_data = data.get("user_data", {})
        for writer in group:
            record = user_data.get(writer)
            if record is None:
                continue
            xs, ys = record.get("x", ()), record.get("y", ())
            if len(xs) != len(ys):
                raise ValueError(f"mismatched image/label counts for writer {writer} in {path}")
            old_x, old_y = result.setdefault(writer, ([], []))
            old_x.extend(xs)
            old_y.extend(ys)
    missing = selected.difference(result)
    if missing:
        raise ValueError(f"writers missing from LEAF shards: {sorted(missing)}")
    return result


def _arrays_for_group(records: dict[str, tuple[list, list]], group: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    image_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for writer in group:
        raw_images, raw_labels = records[writer]
        images = np.asarray(raw_images, dtype=np.float32)
        if images.size == 0:
            continue
        if images.ndim == 2 and images.shape[1] == 784:
            images = images.reshape(-1, 28, 28)
        elif images.ndim == 4 and images.shape[1:] == (1, 28, 28):
            images = images[:, 0]
        elif images.ndim != 3 or images.shape[1:] != (28, 28):
            raise ValueError(f"expected FEMNIST images as [n,784] or [n,28,28], got {images.shape}")
        if not np.isfinite(images).all() or images.min() < 0.0 or images.max() > 1.0:
            raise ValueError(
                f"LEAF pixels must already be finite and in [0,1]; observed range "
                f"[{images.min()}, {images.max()}]"
            )
        image_parts.append(images[:, None, :, :])
        labels = np.asarray(raw_labels, dtype=np.int64)
        if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
            raise ValueError(f"invalid labels for writer {writer}: {labels.shape}")
        if labels.size and (labels.min() < 0 or labels.max() >= 62):
            raise ValueError(f"FEMNIST labels must be in [0,61], got [{labels.min()}, {labels.max()}]")
        label_parts.append(labels)
    if not image_parts:
        return np.empty((0, 1, 28, 28), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(image_parts), np.concatenate(label_parts)


def prepare_femnist(
    leaf_root: str | Path = "data/femnist/leaf/data",
    output_dir: str | Path = "data/femnist/clients",
    num_writers: int = 200,
    writers_per_client: int = 10,
    provenance_file: str | Path | None = None,
) -> FEMNISTDataset:
    """Convert LEAF JSON shards into memory-mappable per-client ``.npy`` files."""
    if num_writers < 1 or writers_per_client < 1 or num_writers % writers_per_client:
        raise ValueError("num_writers must be positive and divisible by writers_per_client")
    leaf_root, output_dir = Path(leaf_root), Path(output_dir)
    train_paths = _json_files(leaf_root / "train")
    test_paths = _json_files(leaf_root / "test")
    writers = _selected_users(train_paths, num_writers)
    provenance: dict[str, object] | None = None
    provenance_sha256: str | None = None
    if provenance_file is not None:
        provenance_path = Path(provenance_file)
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not isinstance(provenance, dict) or not all(
            key in provenance for key in ("source", "preprocessing_version")
        ):
            raise ValueError(f"invalid FEMNIST provenance document: {provenance_path}")
        provenance_sha256 = _sha256(provenance_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards: list[ClientShard] = []
    shard_files: list[dict[str, str]] = []

    for client_id, start in enumerate(range(0, len(writers), writers_per_client)):
        group = tuple(writers[start : start + writers_per_client])
        train = _arrays_for_group(_collect_group(train_paths, group), group)
        test = _arrays_for_group(_collect_group(test_paths, group), group)
        prefix = output_dir / f"client_{client_id:02d}"
        paths = [
            prefix.with_name(prefix.name + "_train_images.npy"),
            prefix.with_name(prefix.name + "_train_labels.npy"),
            prefix.with_name(prefix.name + "_test_images.npy"),
            prefix.with_name(prefix.name + "_test_labels.npy"),
        ]
        for path, array in zip(paths, (*train, *test), strict=True):
            np.save(path, array, allow_pickle=False)
            shard_files.append({"path": path.name, "sha256": _sha256(path)})
        shards.append(ClientShard(client_id, *paths, len(train[1]), len(test[1]), group))

    manifest = {
        "num_writers": num_writers,
        "writers_per_client": writers_per_client,
        "writer_ids": writers,
        "clients": [
            {"num_train": c.num_train, "num_test": c.num_test} for c in shards
        ],
        "files": shard_files,
    }
    if provenance is not None:
        manifest["dataset_provenance"] = provenance
        manifest["provenance_sha256"] = provenance_sha256
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return FEMNISTDataset(tuple(shards), tuple(writers))


def load_femnist(
    leaf_root: str | Path = "data/femnist/leaf/data",
    output_dir: str | Path = "data/femnist/clients",
    num_writers: int = 200,
    writers_per_client: int = 10,
) -> FEMNISTDataset:
    """Load prepared client shards, preparing them from LEAF JSON when needed."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("num_writers") == num_writers
                and manifest.get("writers_per_client") == writers_per_client):
            writers = tuple(manifest["writer_ids"])
            shards: list[ClientShard] = []
            valid = True
            for client_id, start in enumerate(range(0, len(writers), writers_per_client)):
                prefix = output_dir / f"client_{client_id:02d}"
                paths = [prefix.with_name(prefix.name + suffix) for suffix in (
                    "_train_images.npy", "_train_labels.npy", "_test_images.npy", "_test_labels.npy"
                )]
                if not all(path.is_file() for path in paths):
                    valid = False
                    break
                counts = manifest["clients"][client_id]
                shards.append(ClientShard(
                    client_id, *paths, int(counts["num_train"]), int(counts["num_test"]),
                    tuple(writers[start : start + writers_per_client]),
                ))
            if valid:
                return FEMNISTDataset(tuple(shards), writers)
    return prepare_femnist(leaf_root, output_dir, num_writers, writers_per_client)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf-root", default="data/femnist/leaf/data")
    parser.add_argument("--output-dir", default="data/femnist/clients")
    parser.add_argument("--num-writers", type=int, default=200)
    parser.add_argument("--writers-per-client", type=int, default=10)
    parser.add_argument("--provenance-file", default=None)
    args = parser.parse_args()
    dataset = prepare_femnist(
        args.leaf_root, args.output_dir, args.num_writers, args.writers_per_client,
        args.provenance_file,
    )
    print(f"Prepared {len(dataset.clients)} clients from {len(dataset.writer_ids)} sorted writers")
    print("Training samples per client:", [c.num_train for c in dataset.clients])


if __name__ == "__main__":
    main()
