"""Injected-failure teardown and --config handling for run_mininet (no root needed).

Mininet, tc and the child processes are replaced by fakes, so these tests
check the runner's control flow: every failure path stops children and the
network exactly once and leaves run_failure.json instead of meta.json.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import fedgrant.mininet.run_mininet as rm
from fedgrant.mininet.teardown import stop_processes, teardown

REPO = Path(__file__).resolve().parents[2]
PROD = "tests.fake:create_gateway"


class FakeProc:
    _next_pid = 40000

    def __init__(self, exit_code: int | None = 0, *, ignores_terminate: bool = False):
        FakeProc._next_pid += 1
        self.pid = FakeProc._next_pid
        self.exit_code = exit_code
        self.returncode = None
        self.ignores_terminate = ignores_terminate
        self.terminated = self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            if self.exit_code is None:
                raise subprocess.TimeoutExpired("fake", timeout)
            self.returncode = self.exit_code
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.ignores_terminate:
            self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


class FakeHost:
    def __init__(self, net, name):
        self.net, self.name = net, name

    def popen(self, cmd, **kwargs):
        proc = self.net.proc_factory(self.name, cmd)
        self.net.procs.append(proc)
        return proc

    def cmd(self, command):
        return ""


class FakeNet:
    def __init__(self, proc_factory=None):
        self.stop_calls = 0
        self.procs: list[FakeProc] = []
        self.proc_factory = proc_factory or (lambda host, cmd: FakeProc(0))
        self._fedgrant_expected_tc = {("s1", "s1-gw"): ["tc qdisc add dev s1-gw root netem delay 5ms"]}

    def get(self, name):
        return FakeHost(self, name)

    def stop(self):
        self.stop_calls += 1


class FakeSampler:
    instances: list["FakeSampler"] = []

    def __init__(self, *args, **kwargs):
        self.stopped = 0
        FakeSampler.instances.append(self)

    def start(self):
        return self

    def stop(self):
        self.stopped += 1


@pytest.fixture
def fake_runtime(monkeypatch):
    FakeSampler.instances = []
    monkeypatch.setattr(rm, "cpu_roles", lambda: {"gateway": None, "telemetry": None, "clients": [],
                                                  "pinning_enabled": False})
    monkeypatch.setattr(rm, "ProcessSampler", FakeSampler)
    monkeypatch.setattr(rm, "_QueueSampler", FakeSampler)
    monkeypatch.setattr(rm, "verify_tc", lambda net, expected: None)
    return monkeypatch


def _run(tmp_path, **kwargs):
    defaults = dict(n=2, scenario="L2", seed=0, rounds=1, beta=0.1, trace_dir=tmp_path / "traces",
                    output_dir=tmp_path / "run")
    defaults.update(kwargs)
    return rm.run_experiment(**defaults)


def _assert_failed_run(out: Path, net: FakeNet | None, error_type: str):
    assert not (out / "meta.json").exists(), "a failed run must not look complete to raw_metrics"
    failure = json.loads((out / "run_failure.json").read_text())
    assert failure["status"] == "failed" and failure["error_type"] == error_type
    assert failure["experiment_id"] == rm.STUB_EXPERIMENT_ID
    if net is not None:
        assert net.stop_calls == 1
        assert all(proc.poll() is not None for proc in net.procs)
    for sampler in FakeSampler.instances:
        assert sampler.stopped >= 1


def test_build_failure_writes_failure_record(tmp_path, fake_runtime):
    def broken_build(**kwargs):
        raise RuntimeError("RTNETLINK answers: No such file or directory")

    fake_runtime.setattr(rm, "build", broken_build)
    with pytest.raises(RuntimeError, match="RTNETLINK"):
        _run(tmp_path)
    _assert_failed_run(tmp_path / "run", None, "RuntimeError")


def test_verify_tc_failure_stops_network_before_any_process(tmp_path, fake_runtime):
    net = FakeNet()
    fake_runtime.setattr(rm, "build", lambda **kwargs: net)

    def bad_qdisc(net_, expected):
        raise RuntimeError("tbf limit mismatch on s1-gw")

    fake_runtime.setattr(rm, "verify_tc", bad_qdisc)
    with pytest.raises(RuntimeError, match="tbf limit"):
        _run(tmp_path)
    assert net.procs == []
    _assert_failed_run(tmp_path / "run", net, "RuntimeError")


def test_client_crash_terminates_every_child_and_stops_network(tmp_path, fake_runtime):
    # The gateway hangs (never exits by itself); client 1 crashes.
    def factory(host, cmd):
        if host == "gw":
            return FakeProc(None)
        return FakeProc(3 if host == "c1" else 0)

    net = FakeNet(factory)
    fake_runtime.setattr(rm, "build", lambda **kwargs: net)
    with pytest.raises(RuntimeError, match="exited 3"):
        _run(tmp_path)
    gateway = net.procs[0]
    assert gateway.terminated and gateway.poll() is not None
    _assert_failed_run(tmp_path / "run", net, "RuntimeError")


def test_round_state_is_initialised_before_gateway_starts(tmp_path, fake_runtime):
    seen = []

    def factory(host, cmd):
        if host == "gw":
            seen.append((tmp_path / "run" / ".round_state.json").exists())
            raise OSError("injected spawn failure")
        return FakeProc(0)

    net = FakeNet(factory)
    fake_runtime.setattr(rm, "build", lambda **kwargs: net)
    with pytest.raises(OSError, match="injected"):
        _run(tmp_path)
    assert seen == [True]
    _assert_failed_run(tmp_path / "run", net, "OSError")


def test_gateway_receives_actual_compute_window(tmp_path, fake_runtime):
    commands = []

    def factory(host, cmd):
        commands.append((host, cmd))
        raise OSError("stop after first spawn")

    fake_runtime.setattr(rm, "build", lambda **kwargs: FakeNet(factory))
    with pytest.raises(OSError):
        _run(tmp_path, compute_window_s=0.25)
    host, cmd = commands[0]
    assert host == "gw"
    assert cmd[cmd.index("--compute-window-s") + 1] == "0.25"


def test_teardown_is_idempotent_and_kills_stubborn_children():
    net = FakeNet()
    stubborn = FakeProc(None, ignores_terminate=True)
    polite = FakeProc(None)
    teardown(net, [stubborn, polite])
    teardown(net, [stubborn, polite])
    assert net.stop_calls == 1
    assert polite.terminated and not polite.killed
    assert stubborn.terminated and stubborn.killed
    stop_processes([None, FakeProc(0)])  # finished/absent processes are ignored


# --- identity and --config -------------------------------------------------

def test_stub_cannot_be_labelled_as_m1():
    with pytest.raises(ValueError, match="stub"):
        rm.run_identity("M1", "B6", rm.DEFAULT_STUB_FACTORY)
    with pytest.raises(ValueError, match="experiment-id"):
        rm.run_identity(None, None, PROD)
    assert rm.run_identity(None, None, rm.DEFAULT_STUB_FACTORY) == (rm.STUB_EXPERIMENT_ID, rm.STUB_POLICY)
    assert rm.run_identity("M1", "FG-GRU", PROD) == ("M1", "FG-GRU")


def test_m1_grid_config_resolves_one_run():
    path = REPO / "configs/grids/mn_m1.yaml"
    with pytest.raises(ValueError, match="several values"):
        rm.load_run_config(path, {"seed": 1, "policy": "B6"})
    s = rm.load_run_config(path, {"seed": 1, "policy": "B6", "scenario": "L4"})
    assert (s["experiment_id"], s["scenario"], s["policy"], s["seed"]) == ("M1", "L4", "B6", 1)
    assert (s["clients"], s["rounds"], s["beta"], s["cross_mbps"]) == (20, 100, 0.10, 0)
    assert len(s["config_sha256"]) == 64
    with pytest.raises(ValueError, match="not in the config"):
        rm.load_run_config(path, {"seed": 7, "policy": "B6", "scenario": "L2"})


def test_scenario_configs_match_code_loss_table():
    for name in ("L1", "L2", "L3", "L4", "L5"):
        s = rm.load_run_config(REPO / f"configs/scenarios/{name}.yaml")
        assert s["scenario"] == name
    x2 = rm.load_run_config(REPO / "configs/scenarios/X2_cross.yaml")
    assert x2["scenario"] == "L2" and x2["cross_mbps"] == 3


def test_simulator_grid_and_bad_keys_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="not a Mininet config"):
        rm.load_run_config(REPO / "configs/grids/mn_sim.yaml", {"seed": 0, "policy": "B6", "scenario": "L2"})
    bad = tmp_path / "bad.yaml"
    bad.write_text("scenario: L2\nloss_probability: 0.07\n")
    with pytest.raises(ValueError, match="loss_probability"):
        rm.load_run_config(bad)
    bad.write_text("scenario: L2\nlink_rate: 5\n")
    with pytest.raises(ValueError, match="unknown keys"):
        rm.load_run_config(bad)


def test_cli_config_plus_seed_with_defaults(tmp_path):
    s = rm.resolve_cli(["--config", str(REPO / "configs/scenarios/L3.yaml"), "--seed", "2",
                        "--output", str(tmp_path / "out")])
    assert (s["scenario"], s["seed"], s["clients"], s["gateway_factory"]) == ("L3", 2, 3, rm.DEFAULT_STUB_FACTORY)
    s = rm.resolve_cli(["--config", str(REPO / "configs/grids/mn_m1.yaml"), "--seed", "0",
                        "--scenario", "L2", "--policy", "FG-GRU", "--gateway-factory", PROD])
    assert s["output"].startswith("results/mininet/M1-L2-FG-GRU-s0-")
