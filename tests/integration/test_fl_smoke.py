"""Integration test I3: Federated Learning over simulated network smoke test (Section 17)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fedgrant.encoder.ef import ErrorFeedback
from fedgrant.fl.femnist import load_femnist
from fedgrant.fl.local import make_client_loaders
from fedgrant.fl.model import create_model, get_flat_params
from fedgrant.fl.server import run_fl_simulated
from fedgrant.policy.constants import D_MODEL
from fedgrant.policy.types import PolicyConfig


def test_fl_simulated_smoke() -> None:
    """Run a 2-round smoke test of synchronous FedAvg over the simulated network.

    Verifies end-to-end integration:
    local training -> error feedback -> block encoding -> network simulator ->
    delivered bitmap -> error feedback update -> server model aggregation -> evaluation.
    """
    seed = 42
    dataset = load_femnist(
        "data/femnist/reproduced/leaf/data",
        "data/femnist/reproduced/clients",
        200,
        10,
    )
    train_clients = [client.load_train() for client in dataset.clients]
    weights = dataset.client_weights
    loaders = make_client_loaders(train_clients, master_seed=seed, batch_size=32)
    error_feedback = [ErrorFeedback(D_MODEL) for _ in dataset.clients]
    model = create_model(seed)

    start_params = get_flat_params(model).copy()

    cfg = PolicyConfig(selector="value", predictor="ewma")

    history = run_fl_simulated(
        model=model,
        train_clients=train_clients,
        test_client_loader=lambda: [c.load_test() for c in dataset.clients],
        client_weights=weights,
        loaders=loaders,
        error_feedback=error_feedback,
        num_rounds=2,
        seed=seed,
        beta=0.10,
        scenario="L2",
        policy_config=cfg,
        device="cpu",
        eval_every=1,
        log_every=1,
        client_workers=4,
    )

    # 1. Model parameters must have updated
    end_params = get_flat_params(model)
    assert not np.array_equal(start_params, end_params), "Model parameters must update after FL rounds"
    assert np.isfinite(end_params).all(), "Model parameters must remain finite"

    # 2. History must contain evaluations for each round
    assert len(history) == 2
    for record in history:
        assert "round" in record
        assert "test_loss" in record
        assert "test_accuracy" in record
        assert "eta" in record
        assert "delivered_blocks" in record
        assert 0.0 <= record["test_accuracy"] <= 1.0
        assert record["delivered_blocks"] > 0
        assert 0.0 < record["eta"] <= 1.0

    # 3. Client error feedback must have accumulated non-zero residual for undelivered blocks
    ef_residuals = [np.linalg.norm(ef.e) for ef in error_feedback]
    assert any(norm > 0.0 for norm in ef_residuals), "Partial delivery must leave residuals in error feedback"
