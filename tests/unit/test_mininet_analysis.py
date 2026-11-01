from __future__ import annotations

import csv
import json

import pytest

from fedgrant.analysis.calibrate import compare, holm_adjust
from fedgrant.analysis.raw_metrics import run_row
from fedgrant.analysis.runs_schema import make_row, parse_row
from fedgrant.common.logschema import ROUND_METRIC_COLUMNS


def test_holm_adjustment_is_monotone_and_restores_input_order():
    assert holm_adjust([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])


def test_paired_simulator_comparison_reports_difference_and_missing_pair():
    m1 = parse_row(make_row("M1", "L2", 0.1, 20, "B2", 0, 100, eta=0.8))
    sim = parse_row(make_row("M-sim", "L2", 0.1, 20, "B2", 0, 100, eta=0.6))
    m1_only = parse_row(make_row("M1", "L4", 0.1, 20, "B2", 1, 100, eta=0.7))
    rows, missing = compare([m1, sim, m1_only], metrics=("eta",))
    assert rows[0]["mean_mininet_minus_sim"] == pytest.approx(0.2)
    assert rows[0]["n_pairs"] == 1
    assert missing == [{"config_id": "L4-b0.10-N20-c0", "policy": "B2", "seed": 1, "missing_from": "M-sim metrics"}]


def test_run_row_uses_warmup_and_sample_level_percentiles(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({
        "experiment_id": "M1", "scenario": "L2", "beta": 0.1, "N": 20,
        "policy": "B2", "seed": 0, "n_rounds": 22, "cross_mbps": 0,
    }))
    with (tmp_path / "round_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROUND_METRIC_COLUMNS)
        writer.writeheader()
        writer.writerow({"round_id": 1, "eta": 0.1, "delivered_blocks": 1})
        writer.writerow({"round_id": 21, "eta": 0.8, "delivered_blocks": 10})
        writer.writerow({"round_id": 22, "eta": 1.0, "delivered_blocks": 20})
    (tmp_path / "rtt_samples.csv").write_text("t_mono_ns,round_id,rtt_ms\n1,21,1\n2,21,10\n3,22,100\n")
    row = run_row(tmp_path)
    assert row["eta"] == pytest.approx(0.9)
    assert row["delivered_blocks"] == pytest.approx(15)
    assert row["rtt_p50_ms"] == pytest.approx(10)
    assert row["rtt_p99_ms"] == pytest.approx(98.2)
