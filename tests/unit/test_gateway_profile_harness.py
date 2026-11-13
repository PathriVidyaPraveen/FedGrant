"""Deterministic tests for the gateway profiling harness (fedgrant.profiling.gateway)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from fedgrant.common.constants import MessageType
from fedgrant.common.wire import encode_message
from fedgrant.core.gateway import SendGrant
from fedgrant.profiling import gateway as prof
from tests.unit.test_gateway import FakeIntegrity, make_gateway


# -- statistics ------------------------------------------------------------------


def test_latency_summary_percentiles_thresholds_and_units() -> None:
    values_ns = [1_000 * v for v in range(1, 101)]  # 1..100 us
    s = prof.latency_summary(values_ns)
    assert s["unit"] == "us" and s["sample_count"] == 100 and s["invalid_count"] == 0
    assert s["p50"] == pytest.approx(np.percentile(np.arange(1, 101), 50))
    assert s["p90"] == pytest.approx(np.percentile(np.arange(1, 101), 90))
    assert s["p99"] == pytest.approx(np.percentile(np.arange(1, 101), 99))
    assert (s["max"], s["min"], s["mean"]) == (100.0, 1.0, pytest.approx(50.5))

    few = prof.latency_summary([5_000] * 9)
    assert few["p50"] == 5.0 and few["p90"] is None and few["p95"] is None and few["p99"] is None
    assert prof.latency_summary([5_000] * 19)["p95"] is None
    assert prof.latency_summary([5_000] * 20)["p95"] == 5.0


def test_latency_summary_counts_missing_and_invalid_samples() -> None:
    s = prof.latency_summary([2_000, None, "", "x", float("nan"), float("inf"), -1, 4_000])
    assert s["sample_count"] == 2 and s["invalid_count"] == 6
    assert s["p50"] == 3.0 and s["max"] == 4.0
    empty = prof.latency_summary([])
    assert empty["sample_count"] == 0 and all(empty[k] is None for k in ("p50", "p99", "max", "mean"))
    # Scheduling delay is signed (a wake-up can precede the grid by clock resolution).
    signed = prof.latency_summary([-1_000, 1_000], allow_negative=True)
    assert signed["sample_count"] == 2 and signed["min"] == -1.0


def _process_rows(points, name="gateway"):
    return [{"t_mono_ns": str(t), "process": name, "cpu_time_s": str(c), "rss_bytes": str(r)}
            for t, c, r in points]


def test_cpu_summary_uses_cpu_time_deltas_and_drops_boundary_intervals() -> None:
    s = 1_000_000_000
    rows = _process_rows([
        (0, 0.0, 1), (1 * s, 0.5, 1),  # before the window
        (2 * s, 1.0, 1), (3 * s, 1.25, 1), (4 * s, 2.25, 1),  # inside: 25% then 100%
        (5 * s, 3.25, 1),  # crosses the window end -> dropped
    ])
    cpu = prof.cpu_summary(rows, window=(2 * s, int(4.5 * s)), active_windows=[(3 * s, 4 * s)])
    assert cpu["scope"] == "process" and cpu["unit"] == "percent_of_one_logical_cpu"
    assert cpu["interval_sample_count"] == 2 and cpu["interval_wall_s"] == 2.0
    assert cpu["interval_mean_pct"] == pytest.approx(62.5)  # 1.25 s CPU over 2 s
    assert cpu["active_sample_count"] == 1 and cpu["active_mean_pct"] == pytest.approx(100.0)
    assert cpu["per_sample_max_pct"] == pytest.approx(100.0)
    # cumulative CPU time is never reported as a percentage directly
    assert cpu["per_sample_p50_pct"] == pytest.approx(62.5)


def test_cpu_summary_rejects_non_monotonic_pairs_and_reports_empty_windows() -> None:
    s = 1_000_000_000
    rows = _process_rows([(0, 1.0, 1), (s, 0.5, 1), (2 * s, 1.5, 1)])  # CPU time went backwards once
    cpu = prof.cpu_summary(rows, window=(0, 2 * s), active_windows=[])
    assert cpu["dropped_non_monotonic_intervals"] == 1 and cpu["interval_sample_count"] == 1
    assert cpu["active_mean_pct"] is None and cpu["active_sample_count"] == 0
    none = prof.cpu_summary([], window=(0, s), active_windows=[(0, s)])
    assert none["interval_mean_pct"] is None and none["interval_sample_count"] == 0


def test_rss_summary_baseline_peak_growth_in_bytes_and_mib() -> None:
    mib = 1024 * 1024
    rows = _process_rows([(10, 0, 40 * mib), (20, 0, 42 * mib), (30, 0, 50 * mib), (40, 0, 60 * mib),
                          (99, 0, 999 * mib)])
    rss = prof.rss_summary(rows, baseline=(10, 30), window=(30, 40))
    assert rss["baseline_bytes"] == 41 * mib and rss["baseline_sample_count"] == 2
    assert rss["peak_bytes"] == 60 * mib and rss["measured_sample_count"] == 2
    assert rss["growth_bytes"] == 19 * mib and rss["peak_mib"] == 60.0
    missing = prof.rss_summary(rows, baseline=None, window=(30, 40))
    assert missing["baseline_bytes"] is None and missing["growth_bytes"] is None


def test_skipped_tick_count_from_scheduled_grid() -> None:
    assert prof.count_skipped_ticks([0, 1_000, 2_000], 1_000) == 0
    assert prof.count_skipped_ticks([0, 1_000, 4_000, 5_000], 1_000) == 2


def test_data_outcome_classification() -> None:
    base = dict.fromkeys(prof._OUTCOME_COUNTERS, 0)
    assert prof.classify_data_outcome(base, {**base, "accepted_data": 1}) == "accepted"
    assert prof.classify_data_outcome(base, {**base, "stale_round_data": 1, "rejected_data": 1}) == "stale_round"
    assert prof.classify_data_outcome(base, {**base, "late_after_deadline": 1}) == "late_after_deadline"
    assert prof.classify_data_outcome(base, {**base, "rejected_data": 1}) == "rejected"
    assert prof.classify_data_outcome(base, base) == "other"


# -- configuration and run identity ---------------------------------------------------


def test_config_defaults_match_m1_timing_and_validate_inputs() -> None:
    cfg = prof.ProfileConfig()
    assert cfg.client_counts == (5, 10, 20, 40, 80)
    assert (cfg.compute_window_s, cfg.late_drain_s, cfg.block_length) == (1.0, 0.1, 700)
    assert cfg.blocks == 169 and cfg.deadline_ns(20) == 398_000_000
    for bad in ({"client_counts": ()}, {"client_counts": (5, 5)}, {"client_counts": (0,)},
                {"rounds": 0}, {"rounds": 3, "warmup_rounds": 3}, {"policy": "B2"}, {"beta": 0.0},
                {"late_drain_s": -1.0}, {"tick_interval_ns": 0}, {"sample_interval_s": 0.0},
                {"delay_model": "mininet"}, {"block_length": 0}):
        with pytest.raises(ValueError):
            prof.ProfileConfig(**bad)


def test_spec_uplink_delay_is_the_access_rtt() -> None:
    assert [prof.uplink_delay_ns(i, "spec_rtt") // 1_000_000 for i in range(5)] == [30, 50, 70, 90, 30]
    assert prof.uplink_delay_ns(3, "none") == 0


def test_run_ids_are_unique_and_run_dirs_are_never_reused(tmp_path) -> None:
    now = datetime(2026, 11, 13, 12, 0, 0, tzinfo=timezone.utc)
    a = prof.make_run_id(now, "0123456789abcdef", "a1b2c3")
    assert a == "20261113T120000Z-01234567-a1b2c3"
    assert prof.make_run_id(now, None, "a1b2c3").split("-")[1] == "norev"
    assert a != prof.make_run_id(now, "0123456789abcdef", "d4e5f6")
    assert a != prof.make_run_id(now + timedelta(seconds=1), "0123456789abcdef", "a1b2c3")
    with pytest.raises(ValueError):
        prof.make_run_id(datetime(2026, 1, 1), "x", "a1")  # naive timestamp
    first = prof.create_run_dir(tmp_path, a)
    assert first.is_dir()
    with pytest.raises(FileExistsError):
        prof.create_run_dir(tmp_path, a)


# -- instrumented gateway: separation of the measured intervals -----------------------


@dataclass
class PerfClock:
    now: int = 1_000_000

    def __call__(self) -> int:
        return self.now


@dataclass
class MonoClock:
    now: int = 0

    def __call__(self) -> int:
        return self.now


@dataclass
class CostlySelector:
    """Policy decision that costs 5 us of (fake) perf time."""

    perf: PerfClock

    def select(self, cand):
        self.perf.now += 5_000
        return int(cand.client_id[0])


@dataclass
class CostlyIntegrity(FakeIntegrity):
    """DATA validation that costs 3 us."""

    perf: PerfClock = field(default_factory=PerfClock)

    def valid(self, client_id, block_idx, payload, reported_energies, n_coords):
        self.perf.now += 3_000
        return super().valid(client_id, block_idx, payload, reported_energies, n_coords)


@dataclass
class CostlyTransport:
    """sendto costs 2 us."""

    perf: PerfClock
    sent: list = field(default_factory=list)

    def sendto(self, data, addr):
        self.perf.now += 2_000
        self.sent.append((data, addr))


def _profiled(perf: PerfClock, mono: MonoClock):
    core, state = make_gateway(integrity=CostlyIntegrity(perf=perf))
    core.selector = CostlySelector(perf)
    recorder = prof.SampleRecorder()
    gateway = prof.ProfilingUdpGateway(core, recorder=recorder, perf_ns=perf, clock_ns=mono,
                                       endpoints={1: ("127.0.0.1", 9001)})
    gateway.transport = CostlyTransport(perf)
    return gateway, core, state, recorder


def _data(grant_id: int, send_pos: int, round_id: int = 7) -> bytes:
    return encode_message(MessageType.DATA, round_id=round_id, client_id=1, t_send_ns=0, block_length=2,
                          body={"grant_id": grant_id, "send_pos": send_pos, "block_idx": 0, "n_coords": 2,
                                "payload": [1.0, 0.0]})


def test_tick_scheduling_delay_execution_and_decision_are_separate() -> None:
    perf, mono = PerfClock(), MonoClock(200)
    gateway, core, state, recorder = _profiled(perf, mono)
    gateway.dispatch_actions(core.on_ready(1, [4.0, 2.0, 0.0], 200))  # grant 1, timeout 210
    recorder.pumps.clear()
    recorder.grants.clear()

    gateway.scheduled_tick_ns = 205  # grid point the loop was waiting for
    mono.now = 211  # loop woke 6 ns late; grant 1 has expired -> retransmit as grant 2
    gateway._run_tick()

    (tick,) = recorder.ticks
    round_id, active, scheduled, actual, exec_ns, core_ns, pump_ns, grants = tick
    assert (round_id, active, scheduled, actual) == (7, 1, 205, 211)
    assert actual - scheduled == 6  # scheduling delay (monotonic grid clock)
    assert core_ns == 5_000 and pump_ns == 5_000 and grants == 1  # core.on_tick: the policy decision
    assert exec_ns == 7_000  # decision 5 us + GRANT send 2 us
    assert recorder.pumps == [(7, "tick", 5_000, 1)]
    assert recorder.grants == [(7, 1, 2, 2_000, 2_000)]  # sendto 2 us; decision end -> sent 2 us


def test_data_processing_validation_and_dispatch_are_separate() -> None:
    perf, mono = PerfClock(), MonoClock(200)
    gateway, core, state, recorder = _profiled(perf, mono)
    gateway.dispatch_actions(core.on_ready(1, [4.0, 2.0, 0.0], 200))
    recorder.pumps.clear()
    recorder.grants.clear()

    mono.now = 205  # before grant 1's timeout
    gateway.datagram_received(_data(1, 1), ("127.0.0.1", 9001))

    (sample,) = recorder.data
    round_id, rx_to_core_done, core_ns, pump_ns, total, outcome = sample
    assert outcome == "accepted" and state.m == 1
    assert pump_ns == 5_000  # next-grant decision triggered by this DATA
    assert core_ns == 8_000 and core_ns - pump_ns == 3_000  # validation alone
    assert rx_to_core_done == 8_000  # no transport cost before the core in this fake
    assert total == 10_000  # + GRANT send 2 us, outside DATA processing
    assert recorder.pumps == [(7, "datagram", 5_000, 1)]
    assert [g[3:] for g in recorder.grants] == [(2_000, 2_000)]


def test_stale_and_dropped_datagrams_are_labelled_or_not_sampled() -> None:
    perf, mono = PerfClock(), MonoClock(200)
    gateway, core, state, recorder = _profiled(perf, mono)
    gateway.datagram_received(_data(1, 1, round_id=6), ("127.0.0.1", 9001))
    assert recorder.data[-1][-1] == "stale_round" and core.stats.stale_round_data == 1
    gateway.datagram_received(b"garbage", ("127.0.0.1", 9001))  # dropped by the transport
    assert len(recorder.data) == 1 and gateway.stats.malformed == 1


def test_recorder_csv_round_trips_and_summary_matches_memory(tmp_path) -> None:
    recorder = prof.SampleRecorder(
        ticks=[(3, 1, 1_000, 1_050, 7_000, 5_000, 5_000, 1), (3, 0, 2_000, 2_010, 100, None, 0, 0)],
        pumps=[(3, "tick", 5_000, 1), (3, "datagram", 900, 0)],
        grants=[(3, 1, 9, 2_000, None)],
        data=[(3, 8_000, 8_000, 5_000, 10_000, "accepted")],
    )
    assert recorder.write(tmp_path) == {"ticks": 2, "pumps": 2, "grants": 1, "data": 1}
    with (tmp_path / "tick_samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(rows[0]) == prof.TICK_COLUMNS and rows[1]["core_ns"] == ""
    with (tmp_path / "grant_samples.csv").open() as stream:
        (grant,) = list(csv.DictReader(stream))
    assert grant["decision_to_sent_ns"] == ""  # missing, not zero
    from_file = prof.latency_summary(r["exec_ns"] for r in rows if r["active"] == "1")
    assert from_file == prof.latency_summary([7_000])


def _write_csv(path, columns, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def test_summarize_n_filters_warmup_and_keeps_metrics_separate(tmp_path) -> None:
    cfg = prof.ProfileConfig(client_counts=(20,), rounds=2, warmup_rounds=1, late_drain_s=0.1)
    s = 1_000_000_000
    (tmp_path / "gateway_summary.json").write_text(json.dumps([
        {"round_id": 1, "start_mono_ns": 10 * s, "deadline_mono_ns": 11 * s},
        {"round_id": 2, "start_mono_ns": 20 * s, "deadline_mono_ns": 21 * s},
    ]))
    (tmp_path / "gateway_ready.json").write_text(json.dumps({"t_mono_ns": 5 * s}))
    mib = 1024 * 1024
    proc_rows = [(t, "gateway", c, r) for t, c, r in [
        (6 * s, 0.0, 30 * mib), (7 * s, 0.1, 30 * mib),  # baseline
        (20 * s, 1.0, 40 * mib), (21 * s, 1.5, 44 * mib), (21 * s + s // 10, 1.55, 41 * mib)]]
    proc_rows += [(20 * s, "clients", 0.0, mib), (21 * s, "clients", 0.2, mib)]
    _write_csv(tmp_path / "processes.csv", ("t_mono_ns", "process", "cpu_time_s", "rss_bytes"), proc_rows)
    _write_csv(tmp_path / "tick_samples.csv", prof.TICK_COLUMNS, [
        (1, 1, 100, 999_100, 999, 999, 0, 0),  # warm-up: excluded
        (2, 1, 1_000, 3_000, 7_000, 5_000, 5_000, 1), (2, 0, 2_000, 2_000, 50, "", 0, 0)])
    _write_csv(tmp_path / "pump_samples.csv", prof.PUMP_COLUMNS, [(2, "tick", 5_000, 1), (2, "tick", 400, 0)])
    _write_csv(tmp_path / "grant_samples.csv", prof.GRANT_COLUMNS, [(2, 0, 1, 2_000, 2_500)])
    _write_csv(tmp_path / "data_samples.csv", prof.DATA_COLUMNS, [(2, 8_000, 8_000, 5_000, 10_000, "accepted"),
                                                                   (1, 1, 1, 0, 1, "accepted")])
    _write_csv(tmp_path / "round_metrics.csv", ("round_id", "delivered_blocks"), [(1, 3), (2, 4)])
    (tmp_path / "clients.json").write_text(json.dumps([
        {"client_id": c, "completed_rounds": [1, 2], "round_end_timeouts": 0} for c in range(2)]))

    out = prof.summarize_n(tmp_path, cfg, 20)

    assert out["status"] == "ok", out["issues"]
    assert out["measured_rounds"] == [2]
    assert out["measurement_window_mono_ns"] == [20 * s, 21 * s + s // 10]
    assert out["gateway_cpu"]["interval_mean_pct"] == pytest.approx(100 * 0.55 / 1.1)
    assert out["gateway_cpu"]["active_mean_pct"] == pytest.approx(50.0)
    assert out["gateway_rss"]["baseline_bytes"] == 30 * mib and out["gateway_rss"]["peak_bytes"] == 44 * mib
    lat = out["latency"]
    assert lat["tick_scheduling_delay"]["sample_count"] == 1 and lat["tick_scheduling_delay"]["p50"] == 2.0
    assert lat["tick_execution"]["p50"] == 7.0 and lat["tick_core"]["p50"] == 5.0
    assert lat["grant_decision_pump"]["sample_count"] == 1 and lat["pump_without_grant"]["p50"] == 0.4
    assert lat["grant_sendto"]["p50"] == 2.0 and lat["grant_decision_to_sent"]["p50"] == 2.5
    assert lat["data_core"]["p50"] == 8.0 and lat["data_validation"]["p50"] == 3.0
    assert lat["data_core"]["sample_count"] == 1  # the warm-up DATA is excluded
    assert out["counts"]["active_ticks"] == 1 and out["counts"]["idle_ticks"] == 1
    assert out["counts"]["data_outcomes"] == {"accepted": 1}

    prof.write_summaries(tmp_path, [out])
    with (tmp_path / "summary_latency.csv").open() as stream:
        latency_rows = {(r["num_clients"], r["metric"]): r for r in csv.DictReader(stream)}
    assert float(latency_rows[("20", "data_core")]["p50"]) == lat["data_core"]["p50"]
    assert json.loads((tmp_path / "summary.json").read_text())[0]["latency"] == lat
    assert "localhost_udp_synthetic_clients (not Mininet)" in (tmp_path / "summary.md").read_text()


def test_summarize_n_reports_failures_instead_of_inventing_results(tmp_path) -> None:
    cfg = prof.ProfileConfig(client_counts=(2,), rounds=3, warmup_rounds=1)
    (tmp_path / "gateway_summary.json").write_text(json.dumps(
        [{"round_id": 1, "start_mono_ns": 1, "deadline_mono_ns": 2}]))
    out = prof.summarize_n(tmp_path, cfg, 2)
    assert out["status"] == "failed" and "latency" not in out
    assert any("1 of 3 rounds" in issue for issue in out["issues"])


def test_profiling_gateway_only_wraps_its_own_core_instance() -> None:
    core_a, _ = make_gateway()
    core_b, _ = make_gateway()
    prof.ProfilingUdpGateway(core_a, recorder=prof.SampleRecorder())
    assert "_pump" in vars(core_a) and "_pump" not in vars(core_b)
    assert not any(isinstance(a, SendGrant) for a in core_b.on_tick(150))


def test_summary_explains_when_spec_deadline_blocks_every_grant(tmp_path) -> None:
    # N = 5, beta = 0.10: D = 100 ms < srtt_0 + 2 ms, so rule (iv) never admits a grant.
    cfg = prof.ProfileConfig(client_counts=(5,), rounds=2, warmup_rounds=1)
    assert cfg.deadline_ns(5) == 100_000_000
    (tmp_path / "gateway_summary.json").write_text(json.dumps([
        {"round_id": r, "start_mono_ns": r * 10**9, "deadline_mono_ns": r * 10**9 + 10**8} for r in (1, 2)]))
    out = prof.summarize_n(tmp_path, cfg, 5)
    assert any("eligibility rule (iv)" in issue for issue in out["issues"])
    assert out["latency"]["grant_decision_pump"]["sample_count"] == 0
    assert out["latency"]["grant_decision_pump"]["p50"] is None


def test_udp_gateway_records_the_grid_time_it_waited_for_without_changing_cadence() -> None:
    # Real UdpGateway._scheduler_loop with a fake clock: the first tick overruns
    # 2.5 intervals, so the loop skips to the 4 ms grid point (same cadence as
    # tests/unit/test_udp_gateway.py::test_scheduler_uses_stable_deadlines_without_overlapping_ticks).
    import asyncio

    from fedgrant.net.udp_gateway import UdpGateway
    from tests.unit.test_udp_gateway import Clock, TickEstimator, make_core

    class Recording(UdpGateway):
        scheduled: list[tuple[int | None, int]]

        def _run_tick(self) -> None:
            self.scheduled.append((self.scheduled_tick_ns, self.clock_ns()))
            super()._run_tick()

    async def scenario():
        clock = Clock(0)
        estimator = TickEstimator(clock=clock)
        core = make_core(estimator=estimator)
        core.on_round_start(0, round_id=1, deadline_ns=20_000_000, K=3, L=2)

        async def fake_sleep(seconds: float) -> None:
            clock.now += round(seconds * 1_000_000_000)

        daemon = Recording(core, clock_ns=clock, sleep=fake_sleep)
        daemon.scheduled = []
        assert daemon.scheduled_tick_ns is None
        estimator.daemon = daemon
        await daemon._scheduler_loop()
        return daemon, estimator

    daemon, estimator = asyncio.run(scenario())
    assert estimator.ticks == [1_000_000, 4_000_000, 5_000_000]  # cadence unchanged
    assert daemon.scheduled == [(1_000_000, 1_000_000), (4_000_000, 4_000_000), (5_000_000, 5_000_000)]
