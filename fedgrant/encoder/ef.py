"""Client-side error feedback and delivered-block bitmaps (Sections 2.1, 3.3, 8, 9).

u = Delta + e; u_hat keeps fp32(fp16(z_j)) for delivered blocks and 0 elsewhere
(Eq. 3); after the round e <- u - u_hat (Eq. 6), which includes the fp16
rounding error of delivered blocks. No rescaling or renormalisation.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from fedgrant.encoder.blocks import n_blocks, quantize_fp16, split_blocks
from fedgrant.policy.constants import L as L_DEFAULT

Delivered = bytes | bytearray | memoryview | np.ndarray | Iterable[int]


# --------------------------------------------------------------------------- bitmaps (Section 8)


def bitmap_len(K: int) -> int:
    """ceil(K / 8) bytes."""
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    return -(-K // 8)


def bitmap_from_indices(idx: Iterable[int], K: int) -> bytes:
    """ROUND_END bitmap: bit j <-> block_idx j, least significant bit first (Section 8)."""
    mask = np.zeros(bitmap_len(K) * 8, dtype=bool)
    mask[_check_indices(idx, K)] = True
    return np.packbits(mask, bitorder="little").tobytes()


def indices_from_bitmap(bitmap: bytes | bytearray | memoryview, K: int) -> np.ndarray:
    """Inverse of bitmap_from_indices: sorted 0-based block indices (int64).

    Raises ValueError if the length is not ceil(K/8) or a padding bit (j >= K) is set.
    """
    raw = np.frombuffer(bytes(bitmap), dtype=np.uint8)
    if raw.shape[0] != bitmap_len(K):
        raise ValueError(f"bitmap must be {bitmap_len(K)} bytes for K={K}, got {raw.shape[0]}")
    bits = np.unpackbits(raw, bitorder="little").astype(bool)
    if bits[K:].any():
        raise ValueError("bitmap has padding bits set beyond K")
    return np.flatnonzero(bits[:K]).astype(np.int64)


def _check_indices(idx: Iterable[int], K: int) -> np.ndarray:
    arr = np.asarray(list(idx) if not isinstance(idx, np.ndarray) else idx)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    if arr.ndim != 1 or arr.dtype == bool or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("block indices must be a 1-D sequence of ints")
    if arr.min() < 0 or arr.max() >= K:
        raise ValueError(f"block indices must be in [0, {K})")
    return arr.astype(np.int64)


def delivered_mask(delivered: Delivered, K: int) -> np.ndarray:
    """Normalise `delivered` to a bool [K] mask over block indices.

    Accepted forms: a ROUND_END bitmap (bytes/bytearray/memoryview); a bool
    ndarray of shape (K,); or an iterable / integer ndarray of 0-based block indices.
    """
    if isinstance(delivered, (bytes, bytearray, memoryview)):
        mask = np.zeros(K, dtype=bool)
        mask[indices_from_bitmap(delivered, K)] = True
        return mask
    if isinstance(delivered, np.ndarray) and delivered.dtype == bool:
        if delivered.shape != (K,):
            raise ValueError(f"bool mask must have shape ({K},), got {delivered.shape}")
        return delivered.copy()
    mask = np.zeros(K, dtype=bool)
    mask[_check_indices(delivered, K)] = True
    return mask


# --------------------------------------------------------------------------- error feedback


def reconstruct(u: np.ndarray, delivered: Delivered, L: int = L_DEFAULT) -> np.ndarray:
    """Eq. (3): u_hat, float32 [d]. Delivered blocks are fp32(fp16(z_j)); others are 0."""
    blocks = split_blocks(u, L)
    K = blocks.shape[0]
    mask = delivered_mask(delivered, K)
    u_hat = np.zeros_like(blocks)
    if mask.any():
        u_hat[mask] = quantize_fp16(blocks[mask]).astype(np.float32)
    return u_hat.reshape(-1)[: u.shape[0]].copy()


class ErrorFeedback:
    """Per-client residual e (float32 [d], init 0, persistent across rounds; Section 2.1).

    enabled=False is ablation A1: the residual is discarded (e stays 0, u = Delta).
    """

    def __init__(self, d: int, L: int = L_DEFAULT, enabled: bool = True) -> None:
        if isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d < 1:
            raise ValueError(f"d must be a positive int, got {d!r}")
        self.d = int(d)
        self.L = int(L)
        self.K = n_blocks(self.d, self.L)
        self.enabled = bool(enabled)
        self.e = np.zeros(self.d, dtype=np.float32)

    def _check(self, x: np.ndarray, name: str) -> None:
        if not isinstance(x, np.ndarray) or x.dtype != np.float32 or x.shape != (self.d,):
            raise TypeError(f"{name} must be a float32 ndarray of shape ({self.d},)")

    def prepare(self, delta: np.ndarray) -> np.ndarray:
        """u = Delta + e in fp32 (Section 3.1). Returns a new array."""
        self._check(delta, "delta")
        if not self.enabled:
            return delta.copy()
        return delta + self.e

    def reconstruct(self, u: np.ndarray, delivered: Delivered) -> np.ndarray:
        """Eq. (3) for this client's u (see module-level reconstruct)."""
        self._check(u, "u")
        return reconstruct(u, delivered, self.L)

    def finish_round(self, u: np.ndarray, delivered: Delivered) -> np.ndarray:
        """Eq. (6): e <- u - u_hat, from the ROUND_END bitmap. Returns u_hat."""
        u_hat = self.reconstruct(u, delivered)
        self.e = (u - u_hat) if self.enabled else np.zeros(self.d, dtype=np.float32)
        return u_hat

    def on_round_end_timeout(self, u: np.ndarray) -> np.ndarray:
        """ROUND_END missing 200 ms after D: every block undelivered, so e <- u (Section 9)."""
        return self.finish_round(u, ())