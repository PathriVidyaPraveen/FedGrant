import csv
import json
import os

import pytest

from fedgrant.common.constants import MessageType
from fedgrant.common.logschema import PROCESS_COLUMNS
from fedgrant.common.wire import encode_message
from fedgrant.core.gateway import SendGrant
from fedgrant.mininet import logging_tap
from fedgrant.mininet.logging_tap import LoggingUdpGateway
from fedgrant.mininet.stub_policy import create_gateway
from fedgrant.mininet.telemetry import (
    cpu_usage_percent,
    sample_process,
    summarize_gateway_profile,
)


def test_cpu_usage_is_process_cpu_time_over_monotonic_wall_time():
    assert cpu_usage_percent(2.0, 3.0, 2.0) == 50.0
    with pytest.raises(ValueError, match="positive"):
        cpu_usage_percent(1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="monotonic"):
        cpu_usage_percent(2.0, 1.0, 1.0)


def test_process_sample_serializes_cpu_wall_and_rss_units():
    row = sample_process(os.getpid(), "gateway", clock_ns=lambda: 123)
    assert row["t_mono_ns"] == 123
    assert row["cpu_time_s"] >= 0
    assert row["rss_bytes"] > 0
    assert row["wall_interval_s"] == ""
    assert row["cpu_pct"] == ""
    assert set(row) == set(PROCESS_COLUMNS)

    import io

    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=PROCESS_COLUMNS)
    writer.writeheader()
    writer.writerow(row)
    stream.seek(0)
    serialized = next(csv.DictReader(stream))
    assert int(serialized["rss_bytes"]) == row["rss_bytes"]
    assert float(serialized["cpu_time_s"]) >= 0


def test_profile_summary_excludes_warmup_and_reports_supported_percentiles(tmp_path):
    (tmp_path / "gateway_summary.json").write_text(json.dumps([
        {"round_id": 1, "start_mono_ns": 1_000_000_000},
        {"round_id": 2, "start_mono_ns": 1_200_000_000},
    ]))
    process_rows = [
        {"t_mono_ns": 1_150_000_000, "process": "gateway", "cpu_pct": "10", "wall_interval_s": "0.05",
         "rss_bytes": "1000"},
        {"t_mono_ns": 1_300_000_000, "process": "gateway", "cpu_pct": "30", "wall_interval_s": "0.05",
         "rss_bytes": "3000"},
        {"t_mono_ns": 1_400_000_000, "process": "gateway", "cpu_pct": "50", "wall_interval_s": "0.05",
         "rss_bytes": "5000"},
        {"t_mono_ns": 1_400_000_000, "process": "client", "cpu_pct": "99", "wall_interval_s": "0.05",
         "rss_bytes": "9999"},
    ]
    with (tmp_path / "processes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROCESS_COLUMNS)
        writer.writeheader()
        writer.writerows(process_rows)
    with (tmp_path / "scheduler_samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("t_mono_ns", "round_id", "trigger", "latency_us"))
        writer.writeheader()
        writer.writerows([
            {"t_mono_ns": 1_150_000_000, "round_id": 1, "trigger": "datagram", "latency_us": "100"},
            {"t_mono_ns": 1_300_000_000, "round_id": 2, "trigger": "datagram", "latency_us": "20"},
            {"t_mono_ns": 1_400_000_000, "round_id": 3, "trigger": "scheduler_tick", "latency_us": "40"},
        ])

    result = summarize_gateway_profile(tmp_path, warmup_rounds=1)
    assert result["process_sample_boundary_mono_ns"] == 1_200_000_000
    assert result["cpu_usage"]["sample_count"] == 2
    assert result["cpu_usage"]["p50"] == 40.0
    assert result["cpu_usage"]["p95"] is None
    assert result["rss"]["sample_count"] == 2
    assert result["rss"]["unit"] == "bytes"
    assert result["rss"]["p50"] == 4000
    assert result["scheduler_latency"]["sample_count"] == 2
    assert result["scheduler_latency"]["p50"] == 30
    assert result["scheduler_latency"]["unit"] == "microseconds"
    json.dumps(result, allow_nan=False)


def test_empty_profile_samples_are_null_not_zero(tmp_path):
    result = summarize_gateway_profile(tmp_path)
    assert result["cpu_usage"]["sample_count"] == 0
    assert result["cpu_usage"]["p50"] is None
    assert result["rss"]["sample_count"] == 0
    assert result["rss"]["p50"] is None
    assert result["scheduler_latency"]["sample_count"] == 0
    assert result["scheduler_latency"]["p50"] is None


def test_percentiles_are_reported_when_sample_count_supports_them(tmp_path):
    with (tmp_path / "scheduler_samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("t_mono_ns", "round_id", "trigger", "latency_us"),
        )
        writer.writeheader()
        writer.writerows(
            {
                "t_mono_ns": index,
                "round_id": 21,
                "trigger": "scheduler_tick",
                "latency_us": index,
            }
            for index in range(100)
        )

    summary = summarize_gateway_profile(tmp_path)
    stats = summary["scheduler_latency"]
    assert stats["sample_count"] == 100
    assert stats["p50"] == 49.5
    assert stats["p95"] == pytest.approx(94.05)
    assert stats["p99"] == pytest.approx(98.01)


def test_scheduler_latency_measures_wall_time_to_grant_action(tmp_path, monkeypatch):
    core = create_gateway(1)
    core.on_round_start(1, round_id=1, duration_ns=1_000_000_000, K=2)
    ready = encode_message(
        MessageType.READY,
        round_id=1,
        client_id=0,
        t_send_ns=2,
        body={"K": 2, "energy": [2.0, 1.0]},
    )

    perf_times = iter((1_000_000, 1_004_000, 1_010_000, 1_014_000))
    monkeypatch.setattr(logging_tap.time, "monotonic_ns", lambda: 2_000_000)
    monkeypatch.setattr(logging_tap.time, "perf_counter_ns", lambda: next(perf_times))

    class FakeTransport:
        def sendto(self, packet, endpoint):
            assert packet
            assert endpoint == ("127.0.0.1", 9001)

    gateway = LoggingUdpGateway(
        core,
        log_dir=tmp_path,
        profile_scheduler=True,
        clock_ns=lambda: 2,
        endpoints={0: ("127.0.0.1", 9001)},
    )
    gateway.transport = FakeTransport()
    gateway.datagram_received(ready, ("127.0.0.1", 9001))
    assert 0 in core.energies
    grant_id, grant = next(iter(core.clients[0].outstanding.items()))
    tick_action = SendGrant(0, grant_id, grant.pos)
    core.on_tick = lambda _: [tick_action]
    gateway._run_tick()
    for stream in (gateway._packet_file, gateway._rtt_file, gateway._scheduler_file, gateway._grant_file):
        stream.close()

    with (tmp_path / "scheduler_samples.csv").open(newline="") as stream:
        samples = list(csv.DictReader(stream))
    sample = samples[0]
    assert sample["t_mono_ns"] == "2000000"
    assert float(sample["latency_us"]) == 4.0
    assert sample["trigger"] == "datagram"
    assert samples[1]["trigger"] == "scheduler_tick"
    assert float(samples[1]["latency_us"]) == 4.0


def test_scheduler_measurement_is_disabled_by_default(tmp_path):
    core = create_gateway(1)
    core.on_round_start(1, round_id=1, duration_ns=1_000_000_000, K=2)
    ready = encode_message(
        MessageType.READY,
        round_id=1,
        client_id=0,
        t_send_ns=2,
        body={"K": 2, "energy": [2.0, 1.0]},
    )
    gateway = LoggingUdpGateway(core, log_dir=tmp_path, clock_ns=lambda: 2)
    gateway.datagram_received(ready, ("127.0.0.1", 9001))
    for stream in (gateway._packet_file, gateway._rtt_file, gateway._scheduler_file, gateway._grant_file):
        stream.close()

    with (tmp_path / "scheduler_samples.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == []
