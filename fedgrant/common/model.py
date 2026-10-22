"""Safe NumPy array serialization for FL model and aggregate payloads."""

from __future__ import annotations

from io import BytesIO

import numpy as np


class ArrayPayloadError(ValueError):
    """Raised when a serialized FL array is invalid or incompatible."""


def serialize_array(array: np.ndarray) -> bytes:
    """Serialize a finite numeric ndarray without losing dtype or shape."""
    if not isinstance(array, np.ndarray):
        raise ArrayPayloadError("array payload must be a NumPy ndarray")
    if array.dtype.kind not in "fiu" or array.dtype.hasobject or array.size == 0:
        raise ArrayPayloadError("array payload must be a non-empty real numeric ndarray")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ArrayPayloadError("array payload must contain only finite values")
    stream = BytesIO()
    try:
        np.save(stream, array, allow_pickle=False)
    except (TypeError, ValueError, OSError) as exc:
        raise ArrayPayloadError(f"could not serialize array payload: {exc}") from exc
    return stream.getvalue()


def deserialize_array(payload: bytes | bytearray | memoryview) -> np.ndarray:
    """Load one complete NPY payload; reject truncation, trailing data, and objects."""
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ArrayPayloadError("array payload must be bytes-like")
    raw = bytes(payload)
    stream = BytesIO(raw)
    try:
        array = np.load(stream, allow_pickle=False)
    except (EOFError, OSError, TypeError, ValueError) as exc:
        raise ArrayPayloadError(f"invalid or truncated NPY payload: {exc}") from exc
    if not isinstance(array, np.ndarray):
        raise ArrayPayloadError("NPY payload did not contain an ndarray")
    if stream.tell() != len(raw):
        raise ArrayPayloadError("NPY payload contains trailing bytes")
    if array.dtype.kind not in "fiu" or array.dtype.hasobject or array.size == 0:
        raise ArrayPayloadError("array payload must be a non-empty real numeric ndarray")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ArrayPayloadError("array payload must contain only finite values")
    return array


__all__ = ["ArrayPayloadError", "deserialize_array", "serialize_array"]
