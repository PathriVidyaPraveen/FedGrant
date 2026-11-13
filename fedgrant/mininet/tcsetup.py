"""Pure, validated tc command builders for the FedGrant Mininet topology."""

from __future__ import annotations

import math
import re

from fedgrant.policy.constants import C, Q, S

ACCESS_QUEUE_BYTES = 32 * S
BOTTLENECK_QUEUE_BYTES = Q * S
GATEWAY_UDP_PORT = 9000
READY_TYPE = 2      # FedGrant MessageType.READY (Section 8)
FEDGRANT_MAGIC = 0x4647
# Client TBF parent: under the lossy netem (2:1) in the READY-lossless layout
# (CHANGELOG amendment 2026-11-13), directly under the root netem (1:1) in the
# original Section 7 layout.
CLIENT_TBF_PARENT = "2:1"
LEGACY_CLIENT_TBF_PARENT = "1:1"
# Expected `tc filter show` keys: IPv4 protocol UDP, destination port 9000, and
# the kernel's merged key for magic 0x4647 (bytes 28-29) + type 2 (byte 31).
READY_FILTER_MATCHES = ("00110000/00ff0000 at 8", "00002328/0000ffff at 20", "46470002/ffff00ff at 28")
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
    *, burst: int = 3000, limit: int = ACCESS_QUEUE_BYTES, ready_lossless: bool = True,
    gateway_port: int = GATEWAY_UDP_PORT,
) -> list[str]:
    """Client uplink qdiscs.

    ``ready_lossless`` (default, CHANGELOG amendment 2026-11-13): a two-band
    prio root sends READY frames (UDP to the gateway port, FedGrant magic,
    type 2) through a delay-only netem, as the simulator does (Section 6);
    every other frame takes the unchanged Section 7 chain, GE-loss netem then
    TBF. READY frames do not advance the GE chain, again as in the simulator.
    ``ready_lossless=False`` builds the original Section 7 layout.
    """
    iface = _iface(iface)
    delay_ms = _finite_nonnegative(delay_ms, "delay_ms")
    p_text, r_text = _percent(p, "p"), _percent(r, "r")
    rate, burst, limit = _rate(rate_bps), _size(burst, "burst"), _size(limit, "limit")
    if ready_lossless:
        port = _size(gateway_port, "gateway_port")
        return [
            f"tc qdisc add dev {iface} root handle 1: prio bands 2 priomap {' '.join(['1'] * 16)}",
            f"tc qdisc add dev {iface} parent 1:1 handle 30: netem delay {delay_ms:g}ms",
            f"tc qdisc add dev {iface} parent 1:2 handle 2: netem delay {delay_ms:g}ms loss gemodel {p_text} {r_text} 100% 0%",
            f"tc qdisc add dev {iface} parent {CLIENT_TBF_PARENT} handle 10: tbf rate {rate}bit burst {burst} limit {limit}",
            f"tc filter add dev {iface} parent 1: protocol ip prio 1 u32 match ip protocol 17 0xff "
            f"match ip dport {port} 0xffff match u16 0x{FEDGRANT_MAGIC:04x} 0xffff at 28 "
            f"match u8 0x{READY_TYPE:02x} 0xff at 31 flowid 1:1",
        ]
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


def tbf_change(iface: str, rate_bps: int, *, burst: int = 3000, limit: int = ACCESS_QUEUE_BYTES,
               parent: str = CLIENT_TBF_PARENT) -> str:
    iface = _iface(iface)
    rate, burst, limit = _rate(rate_bps), _size(burst, "burst"), _size(limit, "limit")
    if parent not in (CLIENT_TBF_PARENT, LEGACY_CLIENT_TBF_PARENT):
        raise ValueError(f"unexpected client TBF parent {parent!r}")
    return f"tc qdisc change dev {iface} parent {parent} handle 10: tbf rate {rate}bit burst {burst} limit {limit}"


__all__ = ["ACCESS_QUEUE_BYTES", "BOTTLENECK_QUEUE_BYTES", "CLIENT_TBF_PARENT", "LEGACY_CLIENT_TBF_PARENT",
           "READY_FILTER_MATCHES", "client_egress", "reverse_access", "bottleneck", "gateway_egress", "tbf_change"]
