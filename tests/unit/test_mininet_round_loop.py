"""Shared Mininet round loop and round-state drain phase (no root, simulated clock)."""

from __future__ import annotations

import asyncio
import csv
import json

import pytest

from fedgrant.mininet.round_loop import RoundTiming, RoundWindow, run_rounds
from fedgrant.mininet.round_state import RoundStateFile


class FakeClock:
    def __init__(self):
        self.now = 1_000_000_000

    def __call__(self) -> int:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += int(round(seconds * 1e9))


class RecordingProtocol:
    def __init__(self, state: RoundStateFile, clock: FakeClock):
        self.state, self.clock, self.calls = state, clock, []

    def begin_round(self, round_id, now_ns, duration_ns):
        self.calls.append(("begin", round_id, self.state.read()["active"]))
        return RoundWindow(round_id, now_ns, now_ns + duration_ns)

    def send_round_start(self, window):
        st = self.state.read()
        self.calls.append(("send", window.round_id, st["active"], st["start_ns"] == window.start_ns))

    def at_deadline(self, window):
        self.calls.append(("deadline", window.round_id, self.clock.now == window.deadline_ns))

    def finish_round(self, window):
        phase = self.state.phase_at(self.clock.now)
        self.calls.append(("finish", window.round_id, self.clock.now - window.deadline_ns, phase))
        return {"eta": 0.5, "delivered_blocks": window.round_id}, {"accepted_data": 3}


def test_round_loop_order_timing_drain_and_outputs(tmp_path):
    clock = FakeClock()
    state = RoundStateFile(tmp_path / "rs.json")
    protocol = RecordingProtocol(state, clock)
    timing = RoundTiming(rounds=2, duration_ns=398_000_000, compute_window_s=1.0, late_drain_s=0.1)
    records = asyncio.run(run_rounds(protocol, timing, round_state=state, log_dir=tmp_path,
                                     clock=clock, sleep=clock.sleep))
    assert protocol.calls[:4] == [
        ("begin", 1, False),
        ("send", 1, True, True),          # round window published before ROUND_START
        ("deadline", 1, True),            # at_deadline exactly at D
        ("finish", 1, 100_000_000, (1, "drain")),  # after the 100 ms drain, still labelled round 1
    ]
    assert [c[0] for c in protocol.calls[4:]] == ["begin", "send", "deadline", "finish"]
    rows = list(csv.DictReader((tmp_path / "round_metrics.csv").open()))
    assert [r["round_id"] for r in rows] == ["1", "2"] and rows[1]["delivered_blocks"] == "2"
    assert int(rows[0]["deadline_mono_ns"]) - int(rows[0]["round_start_mono_ns"]) == 398_000_000
    summary = json.loads((tmp_path / "gateway_summary.json").read_text())
    assert summary == records and summary[0]["accepted_data"] == 3
    assert state.read()["active"] is False and state.phase_at(clock.now)[1] == "idle"


def test_round_state_phases_keep_metric_window_at_deadline(tmp_path):
    state = RoundStateFile(tmp_path / "rs.json")
    state.write(round_id=4, start_ns=100, deadline_ns=200, active=True)
    assert state.round_at(150) == 4 and state.phase_at(150) == (4, "active")
    state.write(round_id=4, start_ns=100, deadline_ns=200, active=False, drain_until_ns=300)
    assert state.round_at(250) == 0           # metrics stay on [0, D]
    assert state.phase_at(250) == (4, "drain")
    assert state.phase_at(301) == (0, "idle")
    with pytest.raises(ValueError):
        state.write(round_id=4, start_ns=100, deadline_ns=200, active=False, drain_until_ns=150)


def test_round_timing_and_protocol_window_validation(tmp_path):
    with pytest.raises(ValueError):
        RoundTiming(rounds=0, duration_ns=1)
    with pytest.raises(ValueError):
        RoundTiming(rounds=1, duration_ns=1, late_drain_s=-1)

    class BadProtocol(RecordingProtocol):
        def begin_round(self, round_id, now_ns, duration_ns):
            return RoundWindow(round_id + 1, now_ns, now_ns + duration_ns)

    clock = FakeClock()
    state = RoundStateFile(tmp_path / "rs.json")
    with pytest.raises(ValueError, match="invalid window"):
        asyncio.run(run_rounds(BadProtocol(state, clock), RoundTiming(rounds=1, duration_ns=10),
                               round_state=state, log_dir=tmp_path, clock=clock, sleep=clock.sleep))
