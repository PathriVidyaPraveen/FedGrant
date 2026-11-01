"""Pure, validated tc command builders for the FedGrant Mininet topology."""

from __future__ import annotations

import math
import re

from fedgrant.policy.constants import C, Q, S

ACCESS_QUEUE_BYTES = 32 * S
BOTTLENECK_QUEUE_BYTES = Q * S
_IFACE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,14}$")


def _iface(value: str) -> str:
    if not isinstance(value, str) or not _IFACE.fullmatch(value):
        raise ValueError(f"invalid Linux interface name {value!r}")
    return value


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _rate(value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError("rate_bps must be a positive integer")
    return int(value)


def _size(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _percent(value: float, name: str) -> str:
    value = _finite_nonnegative(value, name)
    if value > 1:
        raise ValueError(f"{name} must be a probability in [0, 1]")
    return f"{value * 100:.4g}%"


def client_egress(
    iface: str, delay_ms: float, p: float, r: float, rate_bps: int,
    *, burst: int = 3000, limit: int = ACCESS_QUEUE_BYTES,
) -> list[str]:
    iface = _iface(iface)
    delay_ms = _finite_nonnegative(delay_ms, "delay_ms")
    p_text, r_text = _percent(p, "p"), _percent(r, "r")
    rate, burst, limit = _rate(rate_bps), _size(burst, "burst"), _size(limit, "limit")
    return [
        f"tc qdisc add dev {iface} root handle 1: netem delay {delay_ms:g}ms loss gemodel {p_text} {r_text} 100% 0%",
        f"tc qdisc add dev {iface} parent 1:1 handle 10: tbf rate {rate}bit burst {burst} limit {limit}",
    ]


def reverse_access(iface: str, delay_ms: float) -> list[str]:
    iface = _iface(iface)
    delay_ms = _finite_nonnegative(delay_ms, "delay_ms")
    return [f"tc qdisc add dev {iface} root netem delay {delay_ms:g}ms"]


def bottleneck(
    iface: str, *, delay_ms: float = 5, rate_bps: int = C, burst: int = 3000,
    limit: int = BOTTLENECK_QUEUE_BYTES,
) -> list[str]:
    iface = _iface(iface)
    delay_ms = _finite_nonnegative(delay_ms, "delay_ms")
    rate, burst, limit = _rate(rate_bps), _size(burst, "burst"), _size(limit, "limit")
    return [
        f"tc qdisc add dev {iface} root handle 1: netem delay {delay_ms:g}ms",
        f"tc qdisc add dev {iface} parent 1:1 handle 10: tbf rate {rate}bit burst {burst} limit {limit}",
    ]


def gateway_egress(iface: str, delay_ms: float = 5) -> list[str]:
    iface = _iface(iface)
    delay_ms = _finite_nonnegative(delay_ms, "delay_ms")
    return [f"tc qdisc add dev {iface} root netem delay {delay_ms:g}ms"]


def tbf_change(iface: str, rate_bps: int, *, burst: int = 3000, limit: int = ACCESS_QUEUE_BYTES) -> str:
    iface = _iface(iface)
    rate, burst, limit = _rate(rate_bps), _size(burst, "burst"), _size(limit, "limit")
    return f"tc qdisc change dev {iface} parent 1:1 handle 10: tbf rate {rate}bit burst {burst} limit {limit}"


__all__ = ["ACCESS_QUEUE_BYTES", "BOTTLENECK_QUEUE_BYTES", "client_egress", "reverse_access", "bottleneck", "gateway_egress", "tbf_change"]
