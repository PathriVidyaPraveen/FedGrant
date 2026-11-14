"""Rule (v) bounded probe (CHANGELOG, PROPOSED 2026-11-13) and the ML policy-revision stamp."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from fedgrant.core.state import ClientState
from fedgrant.policy.constants import POLICY_REVISION
from fedgrant.policy.eligibility import Reason, is_eligible
from fedgrant.policy.types import PolicyConfig

MS = 1_000_000
D = 398 * MS
CFG = PolicyConfig("value", "ewma")


def _ready(K: int = 5) -> ClientState:
    st = ClientState(0, 0.5, K=K)
    st.set_ready(np.linspace(1.0, 0.1, K, dtype=np.float32))
    return st


def test_probe_only_when_allowance_available() -> None:
    st = _ready()
    assert is_eligible(st, 0, D, 0.01, CFG) == (False, Reason.LOW_PREDICTION)          # default: unamended
    assert is_eligible(st, 0, D, 0.01, CFG, probe_available=False) == (False, Reason.LOW_PREDICTION)
    assert is_eligible(st, 0, D, 0.01, CFG, probe_available=True) == (True, Reason.PROBE)
    assert is_eligible(st, 0, D, math.nan, CFG, probe_available=True) == (True, Reason.PROBE)


def test_probe_flag_irrelevant_when_rule_v_passes() -> None:
    st = _ready()
    assert is_eligible(st, 0, D, 0.5, CFG, probe_available=True) == (True, Reason.ELIGIBLE)
    assert is_eligible(st, 0, D, 0.5, CFG, probe_available=False) == (True, Reason.ELIGIBLE)


def test_probe_never_bypasses_rules_i_to_iv() -> None:
    st = ClientState(0, 0.5, K=5)
    assert is_eligible(st, 0, D, 0.0, CFG, probe_available=True) == (False, Reason.NOT_READY)
    st = _ready(K=1)
    st.take_position(1)
    assert is_eligible(st, 0, D, 0.0, CFG, probe_available=True) == (False, Reason.NO_PENDING)
    st = _ready()
    st.cap = 0
    assert is_eligible(st, 0, D, 0.0, CFG, probe_available=True) == (False, Reason.CAP_REACHED)
    st = _ready()
    assert is_eligible(st, D, D, 0.0, CFG, probe_available=True) == (False, Reason.TOO_LATE)


def test_a4_has_no_probes() -> None:
    a4 = PolicyConfig("value", "ewma", pred_in_eligibility=False)
    assert is_eligible(_ready(), 0, D, 0.0, a4, probe_available=True) == (True, Reason.ELIGIBLE)


def test_synthetic_generator_probe_allowance(monkeypatch) -> None:
    """ml/synthetic.py: each client gets at most one PROBE grant per round, consumed at issue."""
    import fedgrant.ml.synthetic as syn
    from fedgrant.ml.schema import SplitRecorder

    real = syn.is_eligible
    log: list[tuple[int, int, bool, bool]] = []      # (round_start, client, probe_available, is_probe)

    def spy(state, t, D_ns, p_hat, cfg, probe_available=False):
        out = real(state, t, D_ns, p_hat, cfg, probe_available=probe_available)
        log.append((state.round_start_ns, state.client_id, probe_available, out[1] is Reason.PROBE))
        return out

    monkeypatch.setattr(syn, "is_eligible", spy)
    syn.simulate_run(SplitRecorder("train", generator="test"), 1000, "train", n_clients=20, n_rounds=8)
    # Once a client's allowance is gone it stays gone for the rest of that round.
    seen_used: set[tuple[int, int]] = set()
    for rnd, cid, avail, _ in log:
        if (rnd, cid) in seen_used:
            assert not avail
        elif not avail:
            seen_used.add((rnd, cid))
    # Every round starts with the allowance available again.
    first = {}
    for rnd, cid, avail, _ in log:
        first.setdefault((rnd, cid), avail)
    assert all(first.values())


def test_check_policy_revision(tmp_path) -> None:
    from fedgrant.ml.sequences import check_policy_revision

    for split in ("train", "val"):
        (tmp_path / split).mkdir()
        (tmp_path / split / "manifest.json").write_text(json.dumps({"policy": {"selector": "value"}}))
    with pytest.raises(ValueError, match="policy revision None"):
        check_policy_revision(tmp_path)
    for split in ("train", "val"):
        (tmp_path / split / "manifest.json").write_text(json.dumps({"policy": {"revision": POLICY_REVISION}}))
    check_policy_revision(tmp_path)
