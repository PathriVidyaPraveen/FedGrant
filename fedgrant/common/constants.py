"""Constants for the FedGrant v3.1 wire protocol (Section 8)."""

from __future__ import annotations

from enum import IntEnum

from fedgrant.policy.constants import ETH_HDR, FG_DATA_HDR, IP_HDR, L, PAYLOAD_BYTES, S, UDP_HDR


class MessageType(IntEnum):
    """Message identifiers from the common-header ``type`` field."""

    ROUND_START = 1
    READY = 2
    GRANT = 3
    DATA = 4
    ROUND_END = 5
    PLAN = 6


class TransferMode(IntEnum):
    """ROUND_START mode identifiers (Section 8)."""

    GRANT = 0
    PUSH = 1
    PLAN = 2
    TCP = 3


MAGIC: int = 0x4647  # ASCII "FG"
PROTOCOL_VERSION: int = 1
COMMON_HEADER_FORMAT: str = "!HBBIHHQ"
COMMON_HEADER_BYTES: int = 20
BYTE_ORDER: str = "big"
INTEGER_FIELDS_UNSIGNED: bool = True

DATA_GRANT_ID: int = 0xFFFFFFFF  # Push/plan DATA has no grant.
FP16_BYTES: int = 2
FP32_BYTES: int = 4

# Section 3.1 frame accounting includes Ethernet, IPv4, UDP, FedGrant header,
# and a full FP16 DATA payload. The existing policy constants remain the source
# for L and FG_DATA_HDR; the protocol layer does not redefine them.

assert COMMON_HEADER_BYTES == 20
assert FG_DATA_HDR == COMMON_HEADER_BYTES + 12
assert PAYLOAD_BYTES == FP16_BYTES * L
assert S == ETH_HDR + IP_HDR + UDP_HDR + FG_DATA_HDR + PAYLOAD_BYTES
