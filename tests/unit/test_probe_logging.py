"""Probe-grant logging columns (CHANGELOG 2026-11-14 eligibility amendment)."""

from __future__ import annotations

from types import SimpleNamespace

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
