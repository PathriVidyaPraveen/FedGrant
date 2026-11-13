# policy/constants.py
from __future__ import annotations

# Encoder/blocks
L: int = 700                      # coordinates per block (fp16)
D_MODEL: int = 117_694            # model dimension d (Section 5.1)
K: int = -(-D_MODEL // L)         # blocks per update, ceil(d / L) = 169
LAST_BLOCK_COORDS: int = D_MODEL - (K - 1) * L   # = 94

# --- Wire frame size S (Section 3.1) -----------------------------------------
ETH_HDR: int = 14
IP_HDR: int = 20
UDP_HDR: int = 8
FG_DATA_HDR: int = 32             # 20-byte common header + 12-byte DATA body
PAYLOAD_BYTES: int = 2 * L        # 1400
S: int = ETH_HDR + IP_HDR + UDP_HDR + FG_DATA_HDR + PAYLOAD_BYTES   # 1474

# --- Bottleneck (Section 3.1) ------------------------------------------------
C: int = 10_000_000               # bottleneck rate, bit/s
Q: int = 64                       # bottleneck buffer, frames

# --- Policy (Sections 3.8, 3.9) ----------------------------------------------
THETA: float = 0.05               # eligibility threshold, rule (v)
TAU_Q_NS: int = 5_000_000         # tau_q = 5 ms, Eq. (12)

# Revision of the grant policy's behaviour (pacing, window, eligibility). Any
# change that alters which grants are issued changes the ML data distribution;
# ml/datagen.py writes this into every manifest and ml/train.py refuses data
# generated under another revision. Bump it with every such CHANGELOG amendment.
#   2026-11-13: rule (v) bounded probe (<= 1 grant per round for a client failing only (v)).
POLICY_REVISION: str = "2026-11-13-probe"

# --- Estimator initial values (Table 2) --------------------------------------
SRTT0_NS: int = 100_000_000       # 100 ms
RTTVAR0_NS: int = 50_000_000      # 50 ms
BHAT0_BPS: float = 4_000_000.0    # 4 Mbit/s
RFC6298_ALPHA: float = 1.0 / 8.0  # Section 3.7
RFC6298_BETA: float = 1.0 / 4.0   # Section 3.7 (beta_rtt)
EWMA_FAST_W: float = 0.3
EWMA_SLOW_W: float = 0.03
EWMA_INIT: float = 0.9

# --- Ring-buffer lengths (Table 2, Section 4.2) ------------------------------
HIST_LEN: int = 32
RTT_LAST_LEN: int = 8
T_EVBUF: int = 16                 # GRU window length T
GRU_HIDDEN: int = 32

# --- Predictor input widths (Sections 4.2, 4.3) ------------------------------
N_OBS: int = 8                    # per-event observation o in R^8
N_X48: int = 48                   # P-MLP feature vector
N_CTX: int = 16                   # context c = features 33..48
CTX_SLICE: slice = slice(32, 48)  # 0-based slice of x48 holding the context

# --- Per-client window, Eq. (11) ---------------------------------------------
CAP_MIN: int = 4
CAP_HEADROOM: int = 2
# Eq. (11) at the initial estimator values, in exact integer arithmetic:
# ceil(1.25 * bhat * srtt / (8 S)) with srtt in ns = ceil(5*bhat*srtt_ns / (32*S*1e9)).
CAP0: int = max(
    CAP_MIN,
    -(-(5 * int(BHAT0_BPS) * SRTT0_NS) // (32 * S * 1_000_000_000)) + CAP_HEADROOM,
)                                 # = 45

# --- Import-time consistency checks ------------------------------------------
assert K == 169, K
assert LAST_BLOCK_COORDS == 94, LAST_BLOCK_COORDS
assert S == 1474, S
assert N_X48 == 32 + N_CTX
assert CTX_SLICE.stop - CTX_SLICE.start == N_CTX
assert CAP0 == 45, CAP0