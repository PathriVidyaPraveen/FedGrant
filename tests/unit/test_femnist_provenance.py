from __future__ import annotations

import hashlib
import json

import pytest

from fedgrant.fl.run_fl import _dataset_metadata


def _write_manifest(path, provenance=None, provenance_sha256=None) -> None:
    path.mkdir(parents=True)
    manifest = {}
    if provenance is not None:
        manifest["dataset_provenance"] = provenance
        manifest["provenance_sha256"] = provenance_sha256
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_run_metadata_includes_verified_femnist_provenance(tmp_path) -> None:
    clients = tmp_path / "clients"
    provenance = {
        "source": "https://example.invalid/leaf",
        "preprocessing_version": "fedgrant-femnist-v1",
        "leaf_commit": "abc123",
    }
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    digest = hashlib.sha256(provenance_path.read_bytes()).hexdigest()
    _write_manifest(clients, provenance, digest)

    metadata = _dataset_metadata(clients, allow_unverified=False)

    assert metadata["source"] == provenance["source"]
    assert metadata["preprocessing_version"] == "fedgrant-femnist-v1"
    assert metadata["provenance_sha256"] == digest
    assert metadata["client_manifest_sha256"]


def test_unverified_dataset_requires_explicit_opt_in(tmp_path) -> None:
    clients = tmp_path / "clients"
    _write_manifest(clients)

    with pytest.raises(ValueError, match="dataset provenance is missing"):
        _dataset_metadata(clients, allow_unverified=False)
    assert _dataset_metadata(clients, allow_unverified=True)["source"] == "unverified"


def test_run_metadata_rejects_changed_provenance(tmp_path) -> None:
    clients = tmp_path / "clients"
    provenance = {"source": "leaf", "preprocessing_version": "v1"}
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    _write_manifest(clients, provenance, "incorrect-hash")

    with pytest.raises(ValueError, match="provenance hash mismatch"):
        _dataset_metadata(clients, allow_unverified=False)
