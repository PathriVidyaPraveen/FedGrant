"""Socket-independent tests for the FedGrant client core."""

from __future__ import annotations

import numpy as np
import pytest

from fedgrant.common.constants import TransferMode
from fedgrant.core.client_core import ClientCore, ClientCoreError
from fedgrant.encoder.ef import bitmap_from_indices


def make_core() -> ClientCore:
    return ClientCore(3, d=5, L=2)


def start(core: ClientCore, *, round_id: int = 1, delta=None) -> tuple[float, ...]:
    if delta is None:
        delta = np.asarray([1, 2, 3, 4, 5], dtype=np.float32)
    return core.start_round(
        round_id=round_id,
        start_ns=100,
        deadline_ns=1000,
        K=3,
        L=2,
        mode=TransferMode.GRANT,
        delta=delta,
    )


def test_ready_energy_and_data_payload_follow_stable_send_order() -> None:
    core = make_core()

    assert start(core) == (25.0, 25.0, 5.0)

    first = core.data_for_grant(1, 1, 1)
    assert first["block_idx"] == 1
    assert first["n_coords"] == 2
    np.testing.assert_array_equal(first["payload"], np.asarray([3, 4], dtype=np.float16))

    partial = core.data_for_grant(1, 2, 2)
    assert partial["block_idx"] == 2
    assert partial["n_coords"] == 1
    np.testing.assert_array_equal(partial["payload"], np.asarray([5, 0], dtype=np.float16))


def test_retransmission_requires_new_grant_id_and_uses_granted_position() -> None:
    core = make_core()
    start(core)

    first = core.data_for_grant(1, 7, 2)
    retry = core.data_for_grant(1, 8, 2)
    assert first["block_idx"] == retry["block_idx"] == 2
    assert first["send_pos"] == retry["send_pos"] == 2
    with pytest.raises(ClientCoreError, match="duplicate grant_id"):
        core.data_for_grant(1, 7, 1)
    with pytest.raises(ClientCoreError, match="outside this round"):
        core.data_for_grant(1, 9, 4)


def test_round_end_applies_bitmap_once_and_residual_persists() -> None:
    core = make_core()
    delta = np.asarray([1, 2, 3, 4, 5], dtype=np.float32)
    start(core, delta=delta)
    u = core.round.u.copy()

    reconstructed = core.finish_round(1, bitmap_from_indices([1], 3))
    expected = np.zeros(5, dtype=np.float32)
    expected[2:4] = u[2:4].astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(reconstructed, expected)
    np.testing.assert_array_equal(core.error_feedback.e, u - expected)
    with pytest.raises(ClientCoreError, match="active round"):
        core.finish_round(1, bitmap_from_indices([1], 3))

    next_delta = np.ones(5, dtype=np.float32)
    next_energies = start(core, round_id=2, delta=next_delta)
    next_u = next_delta + (u - expected)
    blocks = np.zeros((3, 2), dtype=np.float32)
    blocks.reshape(-1)[:5] = next_u
    energies = np.sum(blocks * blocks, axis=1, dtype=np.float32)
    np.testing.assert_array_equal(next_energies, energies[np.argsort(-energies, kind="stable")])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"round_id": 0, "start_ns": 100, "deadline_ns": 1000, "K": 3, "L": 2},
        {"round_id": 1, "start_ns": 1000, "deadline_ns": 1000, "K": 3, "L": 2},
        {"round_id": 1, "start_ns": 100, "deadline_ns": 1000, "K": 2, "L": 2},
        {"round_id": 1, "start_ns": 100, "deadline_ns": 1000, "K": 3, "L": 3},
    ],
)
def test_invalid_round_start_does_not_create_active_state(kwargs: dict) -> None:
    core = make_core()
    with pytest.raises(ClientCoreError):
        core.start_round(
            **kwargs,
            mode=TransferMode.GRANT,
            delta=np.ones(5, dtype=np.float32),
        )
    assert core.round is None
    assert not core.error_feedback.e.any()


def test_invalid_mode_update_and_bitmap_are_rejected_without_corrupting_round() -> None:
    core = make_core()
    with pytest.raises(ClientCoreError, match="GRANT mode"):
        core.start_round(
            round_id=1,
            start_ns=100,
            deadline_ns=1000,
            K=3,
            L=2,
            mode=TransferMode.PUSH,
            delta=np.ones(5, dtype=np.float32),
        )
    with pytest.raises(TypeError):
        start(core, delta=np.ones(5, dtype=np.float64))

    start(core)
    with pytest.raises(ClientCoreError, match="bitmap"):
        core.finish_round(1, b"\xff")
    assert core.round is not None and core.last_completed_round_id is None


def test_round_end_timeout_marks_every_block_undelivered() -> None:
    core = make_core()
    start(core)
    u = core.round.u.copy()

    reconstructed = core.finish_round_timeout(1)

    assert not reconstructed.any()
    np.testing.assert_array_equal(core.error_feedback.e, u)
    assert core.round is None and core.last_completed_round_id == 1
