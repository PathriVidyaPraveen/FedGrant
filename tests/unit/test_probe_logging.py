"""Probe-grant logging columns (CHANGELOG 2026-11-14 eligibility amendment)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fedgrant.common.logschema import GRANT_COLUMNS, ROUND_METRIC_COLUMNS
from fedgrant.mininet.gateway_proc import _round_probe_grants
from fedgrant.mininet.logging_tap import _probe_flag


def test_schema_appends_probe_columns_last():
    assert GRANT_COLUMNS[-1] == "probe"
    assert ROUND_METRIC_COLUMNS[-1] == "probe_grants"


def test_probe_flag_uses_core_and_is_blank_without_probe_support():
    core = SimpleNamespace(is_probe_grant=lambda cid, gid: (cid, gid) == (3, 7))
    assert _probe_flag(core, 3, 7) == 1
    assert _probe_flag(core, 3, 8) == 0
    assert _probe_flag(SimpleNamespace(), 3, 7) == ""


def test_round_probe_grants_accepts_attribute_or_method():
    assert _round_probe_grants(SimpleNamespace(round_probe_grants=4)) == 4
    assert _round_probe_grants(SimpleNamespace(round_probe_grants=lambda: 2)) == 2
    assert _round_probe_grants(SimpleNamespace()) == ""


def test_simulator_round_metric_counts_core_probe_grants():
    """run_network_round's probe_grants (read by run_grid) equals the probe grants it decided."""
    from fedgrant.mininet.contract import POLICY_CONFIGS
    from fedgrant.sim.run_net import run_network_round

    from fedgrant.policy.constants import K

    n, k = 20, K
    weights = np.full(n, 1 / n, np.float32)
    energies = np.tile(np.linspace(1.0, 0.01, k, dtype=np.float32), (n, 1))
    gateway, seen = None, 0
    for round_id in range(1, 9):
        decisions = []
        res = run_network_round(round_id=round_id, num_clients=n, client_weights=weights, client_energies=energies,
                                beta=0.10, scenario="L4", policy_config=POLICY_CONFIGS["B6"], seed=round_id,
                                gateway=gateway, decision_hook=decisions.append)
        gateway = res["gateway"]
        probes = sum(d.get("probe", 0) for d in decisions if d.get("outcome") == "grant_issued")
        assert res["metrics"]["probe_grants"] == probes
        seen += probes
    assert seen > 0, "scenario must exercise the probe"
