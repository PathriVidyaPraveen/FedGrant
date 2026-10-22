from __future__ import annotations

import numpy as np
import pytest

from fedgrant.common.model import ArrayPayloadError, deserialize_array, serialize_array


@pytest.mark.parametrize(
    "array",
    [
        np.asarray([1.25, -2.5], dtype=np.float32),
        np.asarray([[1, 2], [3, 4]], dtype=np.int16),
    ],
)
def test_array_payload_round_trip_preserves_shape_dtype_and_values(array: np.ndarray) -> None:
    actual = deserialize_array(serialize_array(array))
    assert actual.shape == array.shape
    assert actual.dtype == array.dtype
    np.testing.assert_array_equal(actual, array)


@pytest.mark.parametrize(
    "payload",
    [b"", b"\x93NUMPY\x01\x00", b"not an npy payload"],
)
def test_invalid_or_truncated_array_payload_is_rejected(payload: bytes) -> None:
    with pytest.raises(ArrayPayloadError):
        deserialize_array(payload)


def test_array_payload_rejects_trailing_bytes_and_object_arrays() -> None:
    payload = serialize_array(np.asarray([1.0], dtype=np.float32))
    with pytest.raises(ArrayPayloadError, match="trailing"):
        deserialize_array(payload + b"extra")
    with pytest.raises(ArrayPayloadError, match="numeric"):
        serialize_array(np.asarray([object()], dtype=object))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_array_payload_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ArrayPayloadError, match="finite"):
        serialize_array(np.asarray([value], dtype=np.float32))
