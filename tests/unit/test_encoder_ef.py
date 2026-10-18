"""Tests for fedgrant/encoder (Sections 2.1, 3.1-3.3, 8, 9)."""

from __future__ import annotations

import numpy as np
import pytest

from fedgrant.core.state import values_from_energies
from fedgrant.encoder.blocks import (
    block_energies,
    block_payload_f16,
    encode,
    n_blocks,
    send_order,
    split_blocks,
    values,
)
from fedgrant.encoder.ef import (
    ErrorFeedback,
    bitmap_from_indices,
    delivered_mask,
    indices_from_bitmap,
    reconstruct,
)
from fedgrant.policy.constants import D_MODEL, K, L

RNG = np.random.default_rng(0)


def _update(d: int = D_MODEL, scale: float = 1e-2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(d) * scale).astype(np.float32)


def _bits(x: np.ndarray) -> np.ndarray:
    return x.view(np.uint32)


# --------------------------------------------------------------------------- blocks


def test_dimensions_section_3_1() -> None:
    assert n_blocks(D_MODEL) == K == 169
    enc = encode(_update())
    assert enc.K == 169 and enc.L == 700 and enc.d == D_MODEL
    assert enc.blocks.shape == (169, 700) and enc.blocks.dtype == np.float32
    assert enc.n_coords[-1] == 94 and np.all(enc.n_coords[:-1] == 700)
    assert int(enc.n_coords.sum()) == D_MODEL


def test_split_blocks_contiguous_and_zero_padded() -> None:
    u = _update()
    b = split_blocks(u)
    np.testing.assert_array_equal(b.reshape(-1)[:D_MODEL], u)
    assert not b[-1, 94:].any()
    for j in (0, 1, 100, 168):
        np.testing.assert_array_equal(b[j, : min(L, D_MODEL - j * L)], u[j * L : j * L + L])
    assert not np.shares_memory(b, u)


def test_split_blocks_exact_multiple_has_no_padding() -> None:
    u = _update(d=3 * L)
    assert split_blocks(u).shape == (3, L)


def test_block_energies_fp32() -> None:
    b = split_blocks(_update())
    eps = block_energies(b)
    assert eps.dtype == np.float32 and eps.shape == (K,)
    np.testing.assert_array_equal(eps, np.sum(b * b, axis=1, dtype=np.float32))
    ref = np.sum(b.astype(np.float64) ** 2, axis=1)
    np.testing.assert_allclose(eps, ref, rtol=1e-5)


def test_send_order_ties_lower_index_first() -> None:
    e = np.array([1.0, 3.0, 3.0, 0.0, 1.0, 3.0, 0.0], dtype=np.float32)
    assert send_order(e).tolist() == [1, 2, 5, 0, 4, 3, 6]
    assert send_order(np.zeros(5, np.float32)).tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        send_order(np.array([1.0, np.nan], np.float32))


@pytest.mark.parametrize("seed", range(5))
def test_send_order_many_ties_matches_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    e = rng.integers(0, 4, size=K).astype(np.float32)   # K = 169 values, heavy ties
    ref = sorted(range(K), key=lambda j: (-float(e[j]), j))
    assert send_order(e).tolist() == ref


def test_encode_order_is_permutation_and_non_increasing() -> None:
    enc = encode(_update(seed=3))
    assert sorted(enc.order.tolist()) == list(range(K))
    assert np.all(np.diff(enc.energies_sorted) <= 0)
    assert enc.energies_sorted.dtype == np.float32
    np.testing.assert_array_equal(enc.energies_sorted, block_energies(enc.blocks)[enc.order])
    assert enc.block_at(1) == int(enc.order[0]) and enc.block_at(K) == int(enc.order[-1])


def test_encode_ties_in_real_update() -> None:
    u = np.zeros(5 * L, np.float32)
    u[2 * L] = 2.0                  # block 2 strongest
    u[4 * L] = 1.0                  # blocks 4 and 1 tie
    u[1 * L] = -1.0
    enc = encode(u)
    assert enc.order.tolist() == [2, 1, 4, 0, 3]
    np.testing.assert_array_equal(enc.energies_sorted, [4, 1, 1, 0, 0])


def test_encode_rejects_bad_input() -> None:
    with pytest.raises(TypeError):
        encode(np.zeros(10, np.float64))
    with pytest.raises(ValueError):
        encode(np.array([1.0, np.inf], np.float32))
    with pytest.raises(OverflowError):
        encode(np.array([1.0, 70000.0], np.float32))
    encode(np.array([65519.0], np.float32))          # rounds to 65504, still finite


def test_block_payload_f16() -> None:
    u = _update()
    enc = encode(u)
    p, n = block_payload_f16(enc, 168)
    assert p.dtype == np.float16 and p.shape == (L,) and n == 94
    np.testing.assert_array_equal(p[:94], u[168 * L :].astype(np.float16))
    assert not p[94:].any()
    p0, n0 = block_payload_f16(enc, 0)
    assert n0 == 700
    np.testing.assert_array_equal(p0, u[:L].astype(np.float16))
    for bad in (-1, K, 1.0, True):
        with pytest.raises(ValueError):
            block_payload_f16(enc, bad)  # type: ignore[arg-type]


def test_values_eq2_and_state_identical() -> None:
    enc = encode(_update())
    w = 0.0371
    v = values(enc.energies_sorted, w)
    assert v.dtype == np.float32 and np.all(np.diff(v) <= 0)
    np.testing.assert_array_equal(v, (w * w * enc.energies_sorted.astype(np.float64)).astype(np.float32))
    np.testing.assert_array_equal(v, values_from_energies(enc.energies_sorted, w))
    with pytest.raises(ValueError):
        values(enc.energies_sorted, -0.1)


# --------------------------------------------------------------------------- bitmaps


@pytest.mark.parametrize("K_", [1, 7, 8, 9, 169])
def test_bitmap_round_trip(K_: int) -> None:
    rng = np.random.default_rng(K_)
    for _ in range(20):
        idx = np.flatnonzero(rng.random(K_) < 0.4)
        bm = bitmap_from_indices(idx, K_)
        assert isinstance(bm, bytes) and len(bm) == -(-K_ // 8)
        np.testing.assert_array_equal(indices_from_bitmap(bm, K_), idx)


def test_bitmap_layout_lsb_first() -> None:
    assert bitmap_from_indices([0], 169)[0] == 0x01
    assert bitmap_from_indices([7], 169)[0] == 0x80
    bm = bitmap_from_indices([8, 168], 169)
    assert len(bm) == 22 and bm[1] == 0x01 and bm[21] == 0x01
    assert bitmap_from_indices([], 169) == bytes(22)
    assert bitmap_from_indices(range(169), 169)[-1] == 0x01   # bits 169..175 stay 0


def test_bitmap_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        indices_from_bitmap(bytes(21), 169)                  # wrong length
    bad = bytearray(22)
    bad[21] = 0x02                                           # bit 169 = padding
    with pytest.raises(ValueError):
        indices_from_bitmap(bytes(bad), 169)
    with pytest.raises(ValueError):
        bitmap_from_indices([169], 169)
    with pytest.raises(ValueError):
        bitmap_from_indices([-1], 169)
    with pytest.raises(TypeError):
        bitmap_from_indices([0.5], 169)


def test_delivered_mask_forms_agree() -> None:
    idx = [0, 5, 168]
    m = delivered_mask(idx, K)
    np.testing.assert_array_equal(m, delivered_mask(bitmap_from_indices(idx, K), K))
    np.testing.assert_array_equal(m, delivered_mask(np.array(idx), K))
    np.testing.assert_array_equal(m, delivered_mask(m, K))
    np.testing.assert_array_equal(m, delivered_mask({168, 0, 5, 5}, K))
    assert m.sum() == 3


# --------------------------------------------------------------------------- error feedback


def test_reconstruct_eq3() -> None:
    u = _update()
    idx = [0, 7, 168]
    u_hat = reconstruct(u, idx)
    assert u_hat.dtype == np.float32 and u_hat.shape == (D_MODEL,)
    mask = np.zeros(D_MODEL, bool)
    for j in idx:
        mask[j * L : min((j + 1) * L, D_MODEL)] = True
    np.testing.assert_array_equal(u_hat[mask], u[mask].astype(np.float16).astype(np.float32))
    assert not u_hat[~mask].any()


def test_u_minus_uhat_equals_e_bitwise() -> None:
    ef = ErrorFeedback(D_MODEL)
    delta = _update(seed=1)
    u = ef.prepare(delta)
    np.testing.assert_array_equal(u, delta)                   # e starts at 0
    delivered = np.flatnonzero(RNG.random(K) < 0.3)
    u_hat = ef.finish_round(u, bitmap_from_indices(delivered, K))
    np.testing.assert_array_equal(_bits(u - u_hat), _bits(ef.e))
    # q(u) - u is exact in fp32 (Sterbenz), so u_hat + e reproduces u exactly
    np.testing.assert_array_equal(u_hat + ef.e, u)


def test_all_delivered_leaves_only_fp16_rounding_error() -> None:
    ef = ErrorFeedback(D_MODEL)
    u = ef.prepare(_update(seed=2))
    ef.finish_round(u, range(K))
    expected = u - u.astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(_bits(ef.e), _bits(expected))
    assert np.any(ef.e != 0)
    assert np.max(np.abs(ef.e)) <= np.max(np.abs(u)) * 2.0 ** -11   # half an fp16 ulp


def test_none_delivered_residual_is_u() -> None:
    ef = ErrorFeedback(D_MODEL)
    u = ef.prepare(_update(seed=4))
    ef.finish_round(u, bitmap_from_indices([], K))
    np.testing.assert_array_equal(_bits(ef.e), _bits(u))


def test_round_end_timeout_treats_all_undelivered() -> None:
    ef = ErrorFeedback(D_MODEL)
    u = ef.prepare(_update(seed=5))
    u_hat = ef.on_round_end_timeout(u)
    assert not u_hat.any()
    np.testing.assert_array_equal(_bits(ef.e), _bits(u))


def test_residual_persists_across_rounds() -> None:
    ef = ErrorFeedback(D_MODEL)
    d1, d2 = _update(seed=6), _update(seed=7)
    u1 = ef.prepare(d1)
    ef.finish_round(u1, [3, 4])
    e1 = ef.e.copy()
    u2 = ef.prepare(d2)
    np.testing.assert_array_equal(_bits(u2), _bits(d2 + e1))
    assert not np.shares_memory(u2, ef.e)


def test_ablation_a1_discards_residual() -> None:
    ef = ErrorFeedback(D_MODEL, enabled=False)
    d1 = _update(seed=8)
    u1 = ef.prepare(d1)
    np.testing.assert_array_equal(u1, d1)
    ef.finish_round(u1, [0])
    assert not ef.e.any()
    d2 = _update(seed=9)
    np.testing.assert_array_equal(ef.prepare(d2), d2)


def test_error_feedback_input_checks() -> None:
    ef = ErrorFeedback(D_MODEL)
    with pytest.raises(TypeError):
        ef.prepare(np.zeros(D_MODEL, np.float64))
    with pytest.raises(TypeError):
        ef.prepare(np.zeros(D_MODEL - 1, np.float32))
    with pytest.raises(ValueError):
        ErrorFeedback(0)


def test_small_d_generic() -> None:
    ef = ErrorFeedback(10, L=4)                 # K = 3, last block 2 coords
    assert ef.K == 3
    u = ef.prepare(np.arange(1, 11, dtype=np.float32) / 3)
    u_hat = ef.finish_round(u, [2])
    np.testing.assert_array_equal(u_hat[:8], 0)
    np.testing.assert_array_equal(u_hat[8:], u[8:].astype(np.float16).astype(np.float32))
    np.testing.assert_array_equal(u_hat + ef.e, u)