"""Energy-ordered block encoder (Sections 2.1, 3.1, 3.2).

u (float32 [d]) is zero-padded to K*L and split into K contiguous blocks
z_j = u[jL : jL+L]; eps_j = ||z_j||_2^2 in fp32; the send order pi sorts the
blocks by energy, descending, stable (Eq. 1). Block indices j are 0-based;
send positions s are 1-based, with block pi[s-1] at position s (Section 8).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedgrant.policy.constants import L as L_DEFAULT

FP16_MAX: float = float(np.finfo(np.float16).max)   # 65504


def n_blocks(d: int, L: int = L_DEFAULT) -> int:
    """K = ceil(d / L) (Section 3.1)."""
    if d < 1 or L < 1:
        raise ValueError(f"need d >= 1 and L >= 1, got d={d}, L={L}")
    return -(-d // L)


def _as_update(u: np.ndarray) -> np.ndarray:
    if not isinstance(u, np.ndarray) or u.dtype != np.float32 or u.ndim != 1:
        raise TypeError("update must be a 1-D float32 ndarray")
    if u.shape[0] < 1:
        raise ValueError("update must be non-empty")
    if not np.all(np.isfinite(u)):
        raise ValueError("update contains non-finite values")
    return u


def split_blocks(u: np.ndarray, L: int = L_DEFAULT) -> np.ndarray:
    """Zero-pad u to K*L and reshape to float32 [K, L] (a new array; Section 2.1)."""
    u = _as_update(u)
    d = u.shape[0]
    K = n_blocks(d, L)
    out = np.zeros(K * L, dtype=np.float32)
    out[:d] = u
    return out.reshape(K, L)


def block_coords(d: int, L: int = L_DEFAULT) -> np.ndarray:
    """True coordinate count of each block j (int64 [K]); the last block has d - (K-1)L."""
    K = n_blocks(d, L)
    n = np.full(K, L, dtype=np.int64)
    n[-1] = d - (K - 1) * L
    return n


def block_energies(blocks: np.ndarray) -> np.ndarray:
    """eps_j = ||z_j||_2^2 computed in fp32, float32 [K] (Section 3.1)."""
    if not isinstance(blocks, np.ndarray) or blocks.dtype != np.float32 or blocks.ndim != 2:
        raise TypeError("blocks must be a 2-D float32 ndarray")
    b = np.ascontiguousarray(blocks)
    return np.asarray(np.sum(b * b, axis=1, dtype=np.float32), dtype=np.float32)


def send_order(energies: np.ndarray) -> np.ndarray:
    """pi = stable descending argsort of energies; ties broken by lower j (Eq. 1). int64 [K]."""
    e = np.asarray(energies)
    if e.ndim != 1 or e.shape[0] < 1:
        raise ValueError("energies must be a non-empty 1-D array")
    if not np.all(np.isfinite(e)):
        raise ValueError("energies must be finite")
    return np.argsort(-e, kind="stable")


@dataclass(frozen=True, slots=True)
class EncodedUpdate:
    """Encoder output for one client and round (Section 2.1).

    order: pi, int64 [K], 0-based block indices in send order (position s -> order[s-1]).
    energies_sorted: float32 [K], eps_{pi(1)} >= ... >= eps_{pi(K)}; sent in READY.
    blocks: float32 [K, L], zero-padded blocks z_j indexed by block index j.
    n_coords: int64 [K], true coordinate count of block j (indexed by block index).
    """

    order: np.ndarray
    energies_sorted: np.ndarray
    blocks: np.ndarray
    n_coords: np.ndarray

    @property
    def K(self) -> int:
        return int(self.blocks.shape[0])

    @property
    def L(self) -> int:
        return int(self.blocks.shape[1])

    @property
    def d(self) -> int:
        return int(self.n_coords.sum())

    def block_at(self, s: int) -> int:
        """Block index j at 1-based send position s."""
        if not 1 <= s <= self.K:
            raise ValueError(f"send position must be in [1, {self.K}], got {s}")
        return int(self.order[s - 1])


def encode(u: np.ndarray, L: int = L_DEFAULT) -> EncodedUpdate:
    """Section 2.1 encoder: split, compute energies, sort by energy descending (Eq. 1)."""
    blocks = split_blocks(u, L)
    quantize_fp16(blocks)          # fail before READY, not mid-round, if u overflows binary16
    eps = block_energies(blocks)
    order = send_order(eps)
    return EncodedUpdate(
        order=order,
        energies_sorted=eps[order],
        blocks=blocks,
        n_coords=block_coords(u.shape[0], L),
    )


def quantize_fp16(z: np.ndarray) -> np.ndarray:
    """q(.) of Section 3.3: IEEE-754 binary16 round-to-nearest-even.

    SPEC-NOTE: the spec does not say what happens when |z| exceeds the binary16
    range; such a coordinate would become inf and corrupt the aggregate, so this
    raises OverflowError instead (values in (65504, 65520) still round to 65504).
    """
    with np.errstate(over="ignore"):
        q = z.astype(np.float16)
    if not np.all(np.isfinite(q)):
        raise OverflowError(f"value outside binary16 range (|x| >= 65520; max finite {FP16_MAX})")
    return q


def block_payload_f16(enc: EncodedUpdate, j: int) -> tuple[np.ndarray, int]:
    """DATA payload of block index j: (float16 [L], zero-padded; n_coords) (Sections 2.1, 8).

    Byte order is applied by the wire layer, not here.
    """
    if isinstance(j, bool) or not isinstance(j, (int, np.integer)) or not 0 <= int(j) < enc.K:
        raise ValueError(f"block index must be an int in [0, {enc.K}), got {j!r}")
    j = int(j)
    return quantize_fp16(enc.blocks[j]), int(enc.n_coords[j])


def values(energies_sorted: np.ndarray, w: float) -> np.ndarray:
    """Eq. (2): v_i(s) = w_i^2 * eps_{i,pi(s)}, float32 [K].

    Formed in float64 and rounded once to float32 (Table 2 `val` is float32).
    """
    w = float(w)
    if not np.isfinite(w) or w < 0:
        raise ValueError(f"w must be finite and >= 0, got {w!r}")
    e = np.asarray(energies_sorted, dtype=np.float64)
    return (w * w * e).astype(np.float32)