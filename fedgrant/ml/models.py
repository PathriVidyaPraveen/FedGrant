"""P-GRU and P-MLP (Sections 4.2, 4.3). Both output logits; sigmoid gives p_hat.

P-GRU: GRU(8 -> 32, 1 layer) over the standardized event window, final hidden
state h; head Linear(32 + 16 -> 32) - ReLU - Linear(32 -> 1) on [h; c~].
Samples with n = 0 bypass the GRU and use h = 0 (Section 4.4).
P-MLP: 48 -> 64 -> 64 -> 1 with ReLU.
PyTorch default initialization; no dropout, no normalization layers.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from fedgrant.policy.constants import GRU_HIDDEN, N_CTX, N_OBS, N_X48

PGRU_PARAMS: int = 5633
PMLP_PARAMS: int = 7361
HEAD_HIDDEN: int = 32
MLP_HIDDEN: int = 64


def n_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PGRU(nn.Module):
    """P-GRU (proposed), 5,633 parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(N_OBS, GRU_HIDDEN, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(GRU_HIDDEN + N_CTX, HEAD_HIDDEN),
            nn.ReLU(),
            nn.Linear(HEAD_HIDDEN, 1),
        )
        n = n_params(self)
        assert n == PGRU_PARAMS, f"P-GRU has {n} parameters, expected {PGRU_PARAMS}"

    def encode(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Final hidden state h [B, 32]; rows with length 0 get h = 0.

        seq: float32 [B, T, 8], right-padded. lengths: int [B] in [0, T].
        """
        B = seq.shape[0]
        lengths = lengths.to(torch.int64)
        h = seq.new_zeros(B, GRU_HIDDEN)
        nz = torch.nonzero(lengths > 0, as_tuple=True)[0]
        if nz.numel() == 0:
            return h
        packed = pack_padded_sequence(seq[nz], lengths[nz].cpu(), batch_first=True, enforce_sorted=False)
        _, hn = self.gru(packed)                        # hn: [1, B_nz, 32], in the original order
        return h.index_put((nz,), hn[0])                # out of place, so autograd is clean

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Logits [B]."""
        h = self.encode(seq, lengths)
        return self.head(torch.cat([h, ctx], dim=1)).squeeze(1)


class PMLP(nn.Module):
    """P-MLP (non-sequential neural baseline), 7,361 parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_X48, MLP_HIDDEN),
            nn.ReLU(),
            nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
            nn.ReLU(),
            nn.Linear(MLP_HIDDEN, 1),
        )
        n = n_params(self)
        assert n == PMLP_PARAMS, f"P-MLP has {n} parameters, expected {PMLP_PARAMS}"

    def forward(self, x48: torch.Tensor) -> torch.Tensor:
        """Logits [B]."""
        return self.net(x48).squeeze(1)


def make_model(name: str) -> nn.Module:
    """'gru' -> PGRU, 'mlp' -> PMLP."""
    if name == "gru":
        return PGRU()
    if name == "mlp":
        return PMLP()
    raise ValueError(f"unknown model {name!r}; expected 'gru' or 'mlp'")