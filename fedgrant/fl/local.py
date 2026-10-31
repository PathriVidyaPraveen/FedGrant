"""Client local training logic (Section 5.1).

Local training specification:
- 20 SGD steps per client per round.
- Batch size: 32.
- Learning rate: 0.05.
- Loss: Cross-entropy loss.
- Per-client shuffled epoch-cyclic loader seeded once by (seed, client_id) and
  retained across rounds.
- Returns delta = local_params - round_start_params (fp32 [d], d = 117,694).
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn

from fedgrant.policy.constants import D_MODEL
from fedgrant.fl.model import FEMNISTCNN, get_flat_params


class CyclicBatchLoader:
    """Epoch-cyclic batch loader that continuously yields batches of size 32.
    
    When an epoch ends, data is reshuffled and cycling continues seamlessly,
    ensuring exactly `n_steps` batches of full batch size 32.
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 32,
        seed: int = 0,
    ) -> None:
        self.images = images
        self.labels = labels
        self.n_samples = images.shape[0]
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.indices = self.rng.permutation(self.n_samples)
        self.cursor = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch the next batch of size `batch_size`, cycling across epochs if needed."""
        batch_indices = []
        while len(batch_indices) < self.batch_size:
            remaining_needed = self.batch_size - len(batch_indices)
            available = self.n_samples - self.cursor
            take = min(remaining_needed, available)
            batch_indices.extend(self.indices[self.cursor : self.cursor + take])
            self.cursor += take
            if self.cursor >= self.n_samples:
                # Epoch finished, reshuffle and reset cursor
                self.indices = self.rng.permutation(self.n_samples)
                self.cursor = 0

        idx = torch.as_tensor(batch_indices, dtype=torch.long)
        return self.images[idx], self.labels[idx]


def client_loader_seed(master_seed: int, client_id: int) -> int:
    """Stable non-negative seed derived only from the experiment and client."""
    return int((int(master_seed) * 1_000_003 + int(client_id) * 10_007) % (2**32))


def make_client_loaders(
    clients: list[tuple[torch.Tensor, torch.Tensor]],
    master_seed: int,
    batch_size: int = 32,
) -> list[CyclicBatchLoader]:
    """Create the persistent, once-seeded batch stream for each client."""
    return [
        CyclicBatchLoader(images, labels, batch_size=batch_size,
                          seed=client_loader_seed(master_seed, client_id))
        for client_id, (images, labels) in enumerate(clients)
    ]


def train_client(
    client_id: int,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    start_model: FEMNISTCNN,
    round_idx: int = 0,
    master_seed: int = 0,
    lr: float = 0.05,
    n_steps: int = 20,
    batch_size: int = 32,
    device: torch.device | str = "cpu",
    loader: CyclicBatchLoader | None = None,
) -> np.ndarray:
    """Perform local training on one client.
    
    1. Makes a fresh copy of start_model.
    2. Trains for exactly `n_steps` SGD steps with lr=0.05 and batch_size=32.
    3. Computes and returns delta = theta_local - theta_start as float32 [D_MODEL].
    """
    dev = torch.device(device)
    local_model = copy.deepcopy(start_model).to(dev)
    local_model.train()

    theta_start = get_flat_params(start_model)

    # The caller owns and reuses this loader over rounds. Constructing one here
    # remains convenient for standalone use and is deterministic across rounds.
    if loader is None:
        loader = CyclicBatchLoader(
            train_images, train_labels, batch_size=batch_size,
            seed=client_loader_seed(master_seed, client_id),
        )

    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(n_steps):
        x_batch, y_batch = loader.next_batch()
        x_batch = x_batch.to(dev)
        y_batch = y_batch.to(dev)

        optimizer.zero_grad()
        logits = local_model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

    theta_local = get_flat_params(local_model)
    delta = (theta_local - theta_start).astype(np.float32)
    assert delta.shape == (D_MODEL,), f"Expected shape ({D_MODEL},), got {delta.shape}"
    return delta


__all__ = ["CyclicBatchLoader", "client_loader_seed", "make_client_loaders", "train_client"]
