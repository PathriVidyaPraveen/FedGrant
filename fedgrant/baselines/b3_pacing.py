"""Shared integer-nanosecond pacing calculations for B3 PLAN transmissions."""

from __future__ import annotations

_NANOSECONDS_PER_SECOND = 1_000_000_000
_DATA_FRAME_FIXED_BYTES = 14 + 20 + 8 + 32
_MAX_TIMESTAMP_NS = 0xFFFFFFFFFFFFFFFF


def data_frame_size_bytes(block_length: int) -> int:
    """Return Ethernet + IPv4 + UDP + FedGrant DATA header and FP16 payload size."""
    if isinstance(block_length, bool) or not isinstance(block_length, int) or block_length < 1:
        raise ValueError("block_length must be a positive integer")
    return _DATA_FRAME_FIXED_BYTES + 2 * block_length


def _validate_plan(k: int, rate_bps: int, block_size_bytes: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    if isinstance(rate_bps, bool) or not isinstance(rate_bps, int) or rate_bps < 0:
        raise ValueError("rate_bps must be a non-negative integer")
    if isinstance(block_size_bytes, bool) or not isinstance(block_size_bytes, int) or block_size_bytes < 1:
        raise ValueError("block_size_bytes must be a positive integer")
    if k > 0 and rate_bps == 0:
        raise ValueError("rate_bps must be positive when k is positive")


def paced_blocks_sent(
    time_ns: int,
    *,
    t_start_ns: int,
    k: int,
    rate_bps: int,
    block_size_bytes: int,
) -> int:
    """Count PLAN blocks due by ``time_ns`` using the PLAN rate and frame size.

    Integer arithmetic keeps exact send boundaries stable and matches the
    client's nanosecond pacing schedule.
    """
    _validate_plan(k, rate_bps, block_size_bytes)
    if isinstance(time_ns, bool) or not isinstance(time_ns, int) or not 0 <= time_ns <= _MAX_TIMESTAMP_NS:
        raise ValueError("time_ns must be a u64 nanosecond timestamp")
    if (
        isinstance(t_start_ns, bool)
        or not isinstance(t_start_ns, int)
        or not 0 <= t_start_ns <= _MAX_TIMESTAMP_NS
    ):
        raise ValueError("t_start_ns must be a u64 nanosecond timestamp")
    if k == 0 or time_ns < t_start_ns:
        return 0
    elapsed_ns = time_ns - t_start_ns
    bits_per_block_scaled = 8 * block_size_bytes * _NANOSECONDS_PER_SECOND
    return min(k, (elapsed_ns * rate_bps) // bits_per_block_scaled + 1)


def pacing_send_time_ns(
    send_position: int,
    *,
    t_start_ns: int,
    k: int,
    rate_bps: int,
    block_size_bytes: int,
) -> int:
    """Return the integer-nanosecond due time for a 1-based send position."""
    _validate_plan(k, rate_bps, block_size_bytes)
    if isinstance(send_position, bool) or not isinstance(send_position, int) or not 1 <= send_position <= k:
        raise ValueError("send_position must be in [1, k]")
    if (
        isinstance(t_start_ns, bool)
        or not isinstance(t_start_ns, int)
        or not 0 <= t_start_ns <= _MAX_TIMESTAMP_NS
    ):
        raise ValueError("t_start_ns must be a u64 nanosecond timestamp")
    if rate_bps == 0:
        raise ValueError("rate_bps must be positive when k is positive")
    offset_num = (send_position - 1) * 8 * block_size_bytes * _NANOSECONDS_PER_SECOND
    target_ns = t_start_ns + (offset_num + rate_bps - 1) // rate_bps
    if target_ns > _MAX_TIMESTAMP_NS:
        raise ValueError("paced send timestamp exceeds u64 range")
    return target_ns


__all__ = ["data_frame_size_bytes", "paced_blocks_sent", "pacing_send_time_ns"]
