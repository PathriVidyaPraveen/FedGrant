"""M1 reproduction fingerprints (scripts/m1_fingerprint.py, docs/DOCKER_M1_REPRODUCTION.md)."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("m1_fingerprint", ROOT / "scripts" / "m1_fingerprint.py")
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)

REFERENCE_DIR = ROOT / "results" / "summary" / "m1_reference"


def _fingerprint(**metrics):
    base = {"eta": 0.40, "delivered_blocks": 290.0, "clients_represented": 19.9, "goodput_bps": 8.2e6,
            "utilization": 0.85, "queue_drops_fg": 1000.0, "late_after_deadline": 85.0,
            "queueing_delay_p99_ms": 75.5, "gw_cpu_pct": 26.0}
    base.update(metrics)
    return {"format": fp.FORMAT, "run": "L2-B2-s0",
            "identity": {"scenario": "L2", "policy": "B2", "seed": 0, "value_bank_sha256": "ab",
                         "effective_queue_settings": {"tc_configured": {"c0-eth0": ["tc qdisc add ..."]}}},
            "metrics": base}


def test_identical_run_passes_and_machine_metrics_are_not_gated():
    ref = _fingerprint()
    ok, lines = fp.compare(_fingerprint(gw_cpu_pct=90.0), ref)
    assert ok
    assert any(line.startswith("info gw_cpu_pct") for line in lines)


@pytest.mark.parametrize("metric,value,ok", [
    ("eta", 0.429, True), ("eta", 0.431, False),
    ("goodput_bps", 8.2e6 * 1.049, True), ("goodput_bps", 8.2e6 * 0.94, False),
    ("queue_drops_fg", 1149.0, True), ("queue_drops_fg", 1151.0, False),
    ("eta", None, False),
])
def test_metric_tolerances(metric, value, ok):
    assert fp.compare(_fingerprint(**{metric: value}), _fingerprint())[0] is ok


def test_identity_mismatch_fails_even_with_equal_metrics():
    cand = _fingerprint()
    cand["identity"]["value_bank_sha256"] = "cd"
    ok, lines = fp.compare(cand, _fingerprint())
    assert not ok and any("value_bank_sha256" in line for line in lines)
    cand = copy.deepcopy(_fingerprint())
    cand["identity"]["effective_queue_settings"]["tc_configured"]["c0-eth0"] = ["tc qdisc add ... loss 9%"]
    assert not fp.compare(cand, _fingerprint())[0]


def test_tc_readback_text_is_excluded_from_identity():
    meta = {"effective_queue_settings": str({
        "bottleneck_rate_bps": 10_000_000,
        "tc_readback": [{"interface": "c0-eth0", "host": "c0", "configured_commands": ["tc qdisc add x"],
                         "effective": "qdisc netem 2: ... seed 7627627686075921235"}]})}
    settings = fp._effective_queue(meta)
    assert settings == {"bottleneck_rate_bps": 10_000_000, "tc_configured": {"c0-eth0": ["tc qdisc add x"]}}


@pytest.mark.parametrize("path", sorted(REFERENCE_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_committed_references_are_complete(path):
    ref = json.loads(path.read_text(encoding="utf-8"))
    assert ref["format"] == fp.FORMAT and ref["run"] == path.stem
    scenario, policy, seed = path.stem.split("-")
    ident = ref["identity"]
    assert (ident["scenario"], ident["policy"], f"s{ident['seed']}") == (scenario, policy, seed)
    assert ident["experiment_id"] == "M1" and ident["N"] == 20 and ident["n_rounds"] == 100
    assert len(ident["value_bank_sha256"]) == 64 and ident["trace_sha256"]
    assert len(ident["effective_queue_settings"]["tc_configured"]) > 0
    assert all(ref["metrics"][key] is not None for key in fp.TOLERANCES)
    assert fp.compare(ref, ref)[0]
