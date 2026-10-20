"""Selector interface and policy configuration (Section 6.1)."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

SELECTORS: frozenset[str] = frozenset({"value", "rr", "srpt"})
PREDICTORS: frozenset[str] = frozenset({"ewma", "hmm", "mlp", "gru", "oracle"})


@dataclass(slots=True)
class CandidateView:
    """Eligible clients only (rules (i)-(v), Section 3.9).

    client_id: int32 [n].
    next_value: float64 [n], v~_i of Eq. (13): v_i(s_next_i), or +inf for starved clients.
    undelivered: int32 [n], K - m_i - outstanding_i.
    rr_pointer: cyclic round-robin pointer over client ids (selector "rr", B4).
    """

    client_id: np.ndarray
    next_value: np.ndarray
    undelivered: np.ndarray
    rr_pointer: int

    def __post_init__(self) -> None:
        cid, val, und = self.client_id, self.next_value, self.undelivered
        if not isinstance(cid, np.ndarray) or cid.dtype != np.int32 or cid.ndim != 1:
            raise TypeError("client_id must be a 1-D int32 ndarray")
        n = cid.shape[0]
        if not isinstance(val, np.ndarray) or val.dtype != np.float64 or val.shape != (n,):
            raise TypeError(f"next_value must be a float64 ndarray of shape ({n},)")
        if not isinstance(und, np.ndarray) or und.dtype != np.int32 or und.shape != (n,):
            raise TypeError(f"undelivered must be an int32 ndarray of shape ({n},)")
        rr = self.rr_pointer
        if isinstance(rr, (bool, np.bool_)) or not isinstance(rr, (int, np.integer)):
            raise TypeError("rr_pointer must be an int")
        self.rr_pointer = int(rr)

    def __len__(self) -> int:
        return int(self.client_id.shape[0])


@runtime_checkable
class Selector(Protocol):
    """Chooses one client among the eligible candidates (Section 6.1)."""

    def select(self, cand: CandidateView) -> int:
        """Return the selected client id; it must be one of cand.client_id."""
        ...


@dataclass(slots=True)
class PolicyConfig:
    """Policy configuration, fields and defaults exactly as Section 6.1.

    pred_in_pacing=False is ablation A3 (pacing uses p_hat = 1);
    pred_in_eligibility=False is ablation A4 (rule (v) disabled, theta = 0);
    retransmit=False is ablation A2; explore_eps=0.1 only in ML data generation (Section 9).
    """

    selector: str                     # "value" | "rr" | "srpt"
    predictor: str                    # "ewma" | "hmm" | "mlp" | "gru" | "oracle"
    pred_in_pacing: bool = True
    pred_in_eligibility: bool = True
    retransmit: bool = True
    explore_eps: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ValueError/TypeError on any value outside Section 6.1."""
        if not isinstance(self.selector, str) or self.selector not in SELECTORS:
            raise ValueError(f"selector must be one of {sorted(SELECTORS)}, got {self.selector!r}")
        if not isinstance(self.predictor, str) or self.predictor not in PREDICTORS:
            raise ValueError(f"predictor must be one of {sorted(PREDICTORS)}, got {self.predictor!r}")
        for name in ("pred_in_pacing", "pred_in_eligibility", "retransmit"):
            v = getattr(self, name)
            if not isinstance(v, (bool, np.bool_)):
                raise TypeError(f"{name} must be a bool, got {v!r}")
            setattr(self, name, bool(v))
        eps = self.explore_eps
        if isinstance(eps, (bool, np.bool_)) or not isinstance(eps, (int, float, np.integer, np.floating)):
            raise TypeError(f"explore_eps must be a number, got {eps!r}")
        eps = float(eps)
        if not math.isfinite(eps) or not 0.0 <= eps <= 1.0:
            raise ValueError(f"explore_eps must be in [0, 1], got {eps!r}")
        self.explore_eps = eps

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PolicyConfig":
        """Build from a mapping (e.g. a parsed configs/policies/*.yaml); unknown keys are rejected."""
        if not isinstance(d, Mapping):
            raise TypeError(f"expected a mapping, got {type(d).__name__}")
        allowed = {f.name for f in fields(cls)}
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(f"unknown PolicyConfig keys: {sorted(map(str, unknown))}")
        missing = {"selector", "predictor"} - set(d)
        if missing:
            raise ValueError(f"missing required PolicyConfig keys: {sorted(missing)}")
        return cls(**dict(d))