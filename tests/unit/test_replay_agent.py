"""Per-client trace replay agent and its runner wiring (no root; fake tc and fake Mininet)."""

from __future__ import annotations

import csv
import subprocess
import threading
import time

import pytest

import fedgrant.mininet.run_mininet as rm
from fedgrant.mininet.replay_agent import REPLAY_COLUMNS, run_agent, tc_applier
from fedgrant.mininet.round_state import RoundStateFile
from tests.unit.test_mininet_teardown import FakeNet, FakeProc, FakeSampler


def test_tc_applier_runs_validated_command_and_raises_on_failure():
    calls = []

    def ok(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    tc_applier("c3-eth0", run=ok)(2_000_000)
    assert calls == [["tc", "qdisc", "change", "dev", "c3-eth0", "parent", "1:1", "handle", "10:", "tbf",
                      "rate", "2000000bit", "burst", "3000", "limit", "47168"]]

    def bad(command, **kwargs):
        return subprocess.CompletedProcess(command, 2, "", "RTNETLINK answers: Invalid argument")

    with pytest.raises(RuntimeError, match="RTNETLINK"):
        tc_applier("c3-eth0", run=bad)(2_000_000)


def test_agent_replays_each_round_from_bin_zero_and_logs_lag(tmp_path):
    state = RoundStateFile(tmp_path / "rs.json")
    state.write(round_id=0, active=False)
    applied, stop = [], threading.Event()
    log = tmp_path / "trace_replay_client_0.csv"
    worker = threading.Thread(target=run_agent, args=([1e6, 2e6, 3e6], "c0-eth0", str(state.path), log, stop),
                              kwargs={"apply_rate": applied.append})
    worker.start()
    for rid in (1, 2):
        now = time.monotonic_ns()
        state.write(round_id=rid, start_ns=now, deadline_ns=now + 250_000_000, active=True)
        time.sleep(0.3)
        state.write(round_id=rid, active=False)
        time.sleep(0.05)
    stop.set()
    worker.join(5)
    rows = list(csv.DictReader(log.open()))
    assert tuple(rows[0]) == REPLAY_COLUMNS
    for rid in ("1", "2"):
        bins = [r for r in rows if r["round_id"] == rid]
        assert [r["bin_index"] for r in bins] == ["0", "1", "2"]          # 250 ms round, 100 ms bins
        assert [r["rate_bps"] for r in bins] == ["1000000", "2000000", "3000000"]
        assert all(0 <= int(r["lag_ns"]) < 50_000_000 for r in bins)
    assert applied == [1_000_000, 2_000_000, 3_000_000] * 2


def test_agent_surfaces_tc_failure(tmp_path):
    state = RoundStateFile(tmp_path / "rs.json")
    now = time.monotonic_ns()
    state.write(round_id=1, start_ns=now, deadline_ns=now + 200_000_000, active=True)

    def broken(rate):
        raise RuntimeError("tc failed")

    with pytest.raises(RuntimeError, match="replay failed"):
        run_agent([1e6], "c0-eth0", str(state.path), tmp_path / "log.csv", threading.Event(), apply_rate=broken)


def test_replay_agent_command_only_for_trace_scenarios(tmp_path):
    kwargs = dict(python="py", client_id=4, seed=2, trace_dir="data/traces",
                  round_state_path=tmp_path / "rs.json", out=tmp_path)
    assert rm.replay_agent_command(scenario="L2", **kwargs) is None
    cmd = rm.replay_agent_command(scenario="L5", **kwargs)
    assert cmd[:3] == ["py", "-m", "fedgrant.mininet.replay_agent"]
    assert cmd[cmd.index("--split") + 1] == "ood" and cmd[cmd.index("--iface") + 1] == "c4-eth0"
    assert rm.replay_agent_command(scenario="L4", **kwargs)[cmd.index("--split") + 1] == "test"


def test_stop_replay_agents_reports_failures():
    ok, crashed, stubborn = FakeProc(None), FakeProc(3), FakeProc(None, ignores_terminate=True)
    crashed.wait()
    failures = rm.stop_replay_agents([ok, crashed, stubborn])
    assert ok.terminated and stubborn.killed
    assert len(failures) == 2 and "exited 3" in failures[0] and "exited -9" in failures[1]


def test_runner_starts_agents_before_gateway_and_stops_them_on_failure(tmp_path, monkeypatch):
    FakeSampler.instances = []
    monkeypatch.setattr(rm, "cpu_roles", lambda: {"gateway": None, "telemetry": None, "clients": [],
                                                  "pinning_enabled": False})
    monkeypatch.setattr(rm, "ProcessSampler", FakeSampler)
    monkeypatch.setattr(rm, "_QueueSampler", FakeSampler)
    monkeypatch.setattr(rm, "verify_tc", lambda net, expected: None)
    order = []

    def factory(host, cmd):
        order.append((host, cmd[cmd.index("-m") + 1]))
        if host == "gw":
            raise OSError("stop at gateway spawn")
        return FakeProc(None)

    net = FakeNet(factory)
    monkeypatch.setattr(rm, "build", lambda **kwargs: net)
    monkeypatch.setattr(rm, "wait_for_replay_agents", lambda agents, logs: order.append(("wait", len(agents))))
    with pytest.raises(OSError):
        rm.run_experiment(n=3, scenario="L4", seed=0, rounds=1, beta=0.1, trace_dir=tmp_path,
                          output_dir=tmp_path / "run")
    assert order == [("c0", "fedgrant.mininet.replay_agent"), ("c1", "fedgrant.mininet.replay_agent"),
                     ("c2", "fedgrant.mininet.replay_agent"), ("wait", 3), ("gw", "fedgrant.mininet.gateway_proc")]
    assert all(proc.terminated for proc in net.procs)


class FakeIPRoute:
    def __init__(self, links=("c2-eth0",)):
        self.links, self.calls = links, []

    def link_lookup(self, ifname):
        return [7] if ifname in self.links else []

    def tc(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_netlink_applier_sets_same_parameters_as_tc():
    from fedgrant.mininet.replay_agent import netlink_applier
    from fedgrant.mininet.tcsetup import ACCESS_QUEUE_BYTES
    ipr = FakeIPRoute()
    netlink_applier("c2-eth0", ipr)(4_123_456)
    args, kwargs = ipr.calls[0]
    assert args == ("change", "tbf", 7)
    assert kwargs == {"handle": 0x00100000, "parent": 0x00010001, "rate": "4123456bit", "burst": 3000,
                      "limit": ACCESS_QUEUE_BYTES}
    with pytest.raises(RuntimeError, match="not found"):
        netlink_applier("c9-eth0", FakeIPRoute())
    with pytest.raises(ValueError):
        netlink_applier("c2-eth0", FakeIPRoute())(0)


def test_wait_for_replay_agents_ready_exit_and_timeout(tmp_path):
    from fedgrant.mininet.replay_agent import READY_MARKER
    logs = [tmp_path / f"a{i}.log" for i in range(2)]
    logs[0].write_text(f"x\n{READY_MARKER} c0-eth0\n")
    logs[1].write_text(f"{READY_MARKER} c1-eth0\n")
    rm.wait_for_replay_agents([FakeProc(None), FakeProc(None)], logs, timeout_s=1)
    logs[1].write_text("starting\n")
    crashed = FakeProc(2)
    crashed.wait()
    with pytest.raises(RuntimeError, match="exited 2 before ready"):
        rm.wait_for_replay_agents([FakeProc(None), crashed], logs, timeout_s=1)
    with pytest.raises(RuntimeError, match="not ready after"):
        rm.wait_for_replay_agents([FakeProc(None), FakeProc(None)], logs, timeout_s=0.2)
