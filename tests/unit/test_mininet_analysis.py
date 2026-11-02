from __future__ import annotations

import csv
import json

import pytest

from fedgrant.analysis.calibrate import compare, holm_adjust
from fedgrant.analysis.raw_metrics import run_row
from fedgrant.analysis.runs_schema import COLUMNS, make_row, parse_row, read_runs_csv
from fedgrant.common.logschema import ROUND_METRIC_COLUMNS
from fedgrant.mininet.round_summary import enrich_round_metrics


def test_holm_adjustment_is_monotone_and_restores_input_order():
    assert holm_adjust([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])


def test_paired_simulator_comparison_reports_difference_and_missing_pair():
    m1 = parse_row(make_row("M1", "L2", 0.1, 20, "B2", 0, 100, eta=0.8))
    sim = parse_row(make_row("M-sim", "L2", 0.1, 20, "B2", 0, 100, eta=0.6))
    m1_only = parse_row(make_row("M1", "L4", 0.1, 20, "B2", 1, 100, eta=0.7))
    rows, missing = compare([m1, sim, m1_only], metrics=("eta",), include_expected=False)
    assert rows[0]["mean_mininet_minus_sim"] == pytest.approx(0.2)
    assert rows[0]["n_pairs"] == 1
    assert missing == [{"config_id": "L4-b0.10-N20-c0", "policy": "B2", "seed": 1, "missing_from": "M-sim metrics"}]


def test_calibration_keeps_configuration_pairs_separate_and_omits_expected_b1_sim_twin():
    m1_a = parse_row(make_row("M1", "L2", 0.1, 20, "B2", 0, 100, eta=0.7))
    sim_a = parse_row(make_row("M-sim", "L2", 0.1, 20, "B2", 0, 100, eta=0.6))
    m1_b = parse_row(make_row("M1", "L4", 0.1, 20, "B2", 0, 100, eta=0.9))
    sim_b = parse_row(make_row("M-sim", "L4", 0.1, 20, "B2", 0, 100, eta=0.5))
    tcp = parse_row(make_row("M1", "L2", 0.1, 20, "B1", 0, 100, eta=0.4))
    rows, missing = compare([m1_a, sim_a, m1_b, sim_b, tcp], metrics=("eta",), include_expected=False)
    assert [(r["config_id"], r["mean_mininet_minus_sim"]) for r in rows] == [
        ("L2-b0.10-N20-c0", pytest.approx(0.1)), ("L4-b0.10-N20-c0", pytest.approx(0.4))
    ]
    assert missing == []  # B1 has no simulator twin by design; B2 has both pairs


def test_calibration_reports_expected_rows_absent_from_both_inputs():
    rows, missing = compare([], metrics=("eta",))
    assert rows == []
    assert len(missing) == 66  # all 36 M1 rows plus 30 required M-sim twins
    assert not any(row["policy"] == "B1" and row["missing_from"] == "M-sim metrics" for row in missing)


def test_runs_reader_accepts_v1_csv_without_new_queue_delay_columns(tmp_path):
    row = make_row("M-sim", "L2", 0.1, 20, "B4", 0, 100, eta=0.5)
    path = tmp_path / "runs-v1.csv"
    old_columns = [c for c in COLUMNS if c not in {"queueing_delay_mean_ms", "queueing_delay_p99_ms"}]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=old_columns)
        writer.writeheader()
        writer.writerow({k: v for k, v in row.items() if k in old_columns})
    loaded = read_runs_csv(path)
    assert len(loaded) == 1
    assert loaded[0].value("queueing_delay_mean_ms") != loaded[0].value("queueing_delay_mean_ms")


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


def test_run_row_derives_queue_and_protocol_metrics_from_round_window_samples(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({
        "experiment_id": "M1", "scenario": "L2", "beta": 0.1, "N": 20,
        "policy": "B2", "seed": 0, "n_rounds": 22, "cross_mbps": 0,
    }))
    with (tmp_path / "round_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROUND_METRIC_COLUMNS)
        writer.writeheader()
        writer.writerow({"round_id": 21, "round_start_mono_ns": 100, "deadline_mono_ns": 1_000_000_100,
                         "eta": 0.8, "goodput_bps": 800})
        writer.writerow({"round_id": 22, "round_start_mono_ns": 1_000_000_200,
                         "deadline_mono_ns": 2_000_000_200, "eta": 0.8, "goodput_bps": 800})
    (tmp_path / "queues.csv").write_text(
        "t_mono_ns,round_id,interface,qdisc,backlog_bytes,bytes,packets,drops,overlimits,queueing_delay_ms\n"
        "90,0,s1-gw,tbf,0,0,0,0,0,0\n"
        "500000100,21,s1-gw,tbf,100,500,10,1,0,0.08\n"
        "1000000100,21,s1-gw,tbf,200,1000,20,2,0,0.16\n"
        "1500000200,22,s1-gw,tbf,100,1500,30,3,0,0.08\n"
        "2000000200,22,s1-gw,tbf,0,2000,40,4,0,0\n"
    )
    (tmp_path / "packets.csv").write_text(
        "t_mono_ns,round_id,direction,client_id,packet_seq,grant_id,send_pos,block_idx,packet_type,bytes,payload_bytes,result\n"
        "150,21,sent,0,,,,,READY,100,0,sent\n"
        "160,21,sent,0,,1,1,,GRANT,28,0,sent\n"
        "170,21,sent,0,,1,1,0,DATA,36,4,sent\n"
        "180,21,sent,0,,,,,ROUND_END,25,0,sent\n"
        "200,22,sent,0,,,,,READY,100,0,sent\n"
        "210,22,sent,0,,2,1,,GRANT,28,0,sent\n"
        "220,22,sent,0,,2,1,0,DATA,36,4,sent\n"
        "230,22,sent,0,,,,,ROUND_END,25,0,sent\n"
    )
    row = run_row(tmp_path)
    assert row["utilization"] == pytest.approx(20 * 1474 * 8 / 10_000_000)
    assert row["queue_drops_fg"] == pytest.approx(2.0)
    assert row["queue_drops_cross"] == 0.0
    assert row["queueing_delay_mean_ms"] == pytest.approx(0.08)
    assert row["queueing_delay_p99_ms"] == pytest.approx(0.1576)
    assert row["protocol_overhead"] == pytest.approx(1.85)


def test_mininet_round_csv_is_enriched_with_sampled_metrics(tmp_path):
    round_row = {column: "" for column in ROUND_METRIC_COLUMNS}
    round_row.update(round_id=21, round_start_mono_ns=100, deadline_mono_ns=1_000_000_100,
                     goodput_bps=800)
    with (tmp_path / "round_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROUND_METRIC_COLUMNS)
        writer.writeheader()
        writer.writerow(round_row)
    (tmp_path / "queues.csv").write_text(
        "t_mono_ns,round_id,interface,qdisc,backlog_bytes,bytes,packets,drops,overlimits,queueing_delay_ms\n"
        "90,0,s1-gw,tbf,0,0,0,0,0,0\n"
        "500000100,21,s1-gw,tbf,100,500,10,1,0,0.08\n"
        "1000000100,21,s1-gw,tbf,200,1000,20,2,0,0.16\n"
    )
    (tmp_path / "packets.csv").write_text(
        "t_mono_ns,round_id,direction,client_id,packet_seq,grant_id,send_pos,block_idx,packet_type,bytes,payload_bytes,result\n"
        "150,21,sent,0,,,,,READY,100,0,sent\n"
        "160,21,sent,0,,1,1,,GRANT,28,0,sent\n"
        "170,21,sent,0,,1,1,0,DATA,1432,1400,sent\n"
        "180,21,sent,0,,,,,ROUND_END,25,0,sent\n"
    )
    enrich_round_metrics(tmp_path, cross_mbps=0)
    with (tmp_path / "round_metrics.csv").open(newline="") as stream:
        row, = csv.DictReader(stream)
    assert float(row["utilization"]) == pytest.approx(20 * 1474 * 8 / 10_000_000)
    assert float(row["queue_drops_fg"]) == 2
    assert float(row["queueing_delay_mean_ms"]) == pytest.approx(0.12)
    assert float(row["protocol_overhead"]) == pytest.approx(1.85)
