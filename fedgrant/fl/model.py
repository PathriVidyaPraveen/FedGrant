"""FEMNIST CNN model architecture and parameter flattening utilities (Section 5.1).

Model specification:
Conv(1 -> 16, 5 x 5, pad 2) -> ReLU -> MaxPool 2;
Conv(16 -> 32, 5 x 5, pad 2) -> ReLU -> MaxPool 2;
Flatten(1568);
FC(1568 -> 64) -> ReLU;
FC(64 -> 62).
Total trainable parameters: d = 117,694 (matches constants.D_MODEL).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from fedgrant.policy.constants import D_MODEL

PARAM_NAMES = (
    "conv1.weight",
    "conv1.bias",
    "conv2.weight",
    "conv2.bias",
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
)


class FEMNISTCNN(nn.Module):
    """Convolutional neural network for 62-class FEMNIST character classification."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(in_features=1568, out_features=64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=62)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input shape: (B, 1, 28, 28), Output shape: (B, 62)."""
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def create_model(seed: int | None = None) -> FEMNISTCNN:
    """Create a new FEMNISTCNN model, optionally seeded for deterministic initialization."""
    if seed is not None:
        torch.manual_seed(seed)
    model = FEMNISTCNN()
    assert count_parameters(model) == D_MODEL, (
        f"Parameter count mismatch: got {count_parameters(model)}, expected {D_MODEL}"
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Return total count of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_flat_params(model: nn.Module) -> np.ndarray:
    """Extract model parameters into a contiguous 1D float32 numpy array.
    
    Order is guaranteed deterministic based on PARAM_NAMES.
    """
    state = model.state_dict()
    parts = []
    for name in PARAM_NAMES:
        tensor = state[name].detach().cpu().numpy()
        parts.append(tensor.reshape(-1).astype(np.float32))
    flat = np.concatenate(parts, axis=0)
    assert flat.shape == (D_MODEL,), f"Expected shape ({D_MODEL},), got {flat.shape}"
    return flat


def set_flat_params(model: nn.Module, flat: np.ndarray) -> None:
    """Load flat float32 array into model parameters in-place."""
    if not isinstance(flat, np.ndarray) or flat.shape != (D_MODEL,):
        raise ValueError(f"flat must be ndarray with shape ({D_MODEL},), got {getattr(flat, 'shape', None)}")
    
    state = model.state_dict()
    offset = 0
    for name in PARAM_NAMES:
        param = state[name]
        n_elem = param.numel()
        chunk = flat[offset : offset + n_elem].reshape(param.shape)
        state[name] = torch.from_numpy(chunk).to(dtype=param.dtype, device=param.device)
        offset += n_elem
    assert offset == D_MODEL, f"Offset {offset} != D_MODEL {D_MODEL}"
    model.load_state_dict(state)


__all__ = [
    "FEMNISTCNN",
    "create_model",
    "count_parameters",
    "get_flat_params",
    "set_flat_params",
    "PARAM_NAMES",
]
