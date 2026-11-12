"""Run a Mininet UDP gateway/client smoke experiment with guaranteed teardown."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from fedgrant.analysis.runs_schema import make_config_id
from fedgrant.common.logschema import PACKET_COLUMNS, QUEUE_COLUMNS
from fedgrant.mininet.contract import (
    BASELINE_MODES, DEFAULT_MODELS_DIR, DEFAULT_VALUE_BANK, GRANT_POLICIES, POLICY_MODEL_FILES, STUB_POLICY,
    check_entry_points_available, entry_points, is_baseline,
)
from fedgrant.mininet.crosstraffic import CrossTraffic, udp_payload_bps
from fedgrant.mininet.profile_update import profile_bank
from fedgrant.mininet.replay import RateReplay
from fedgrant.mininet.teardown import teardown
from fedgrant.mininet.telemetry import ProcessSampler, sample_qdisc
from fedgrant.mininet.round_state import RoundStateFile
from fedgrant.mininet.round_summary import enrich_round_metrics
from fedgrant.mininet.topo import SCENARIO_LOSS, build, client_parameters, verify_tc
from fedgrant.mininet.telemetry import summarize_gateway_profile
from fedgrant.mininet.tcsetup import tbf_change
from fedgrant.traces.client_trace import client_trace
from fedgrant.sim.run_net import compute_deadline_ns
from fedgrant.policy.constants import C, Q, S

DEFAULT_STUB_FACTORY = "fedgrant.mininet.stub_policy:create_gateway"
STUB_EXPERIMENT_ID = "smoke-stub"
MININET_EXPERIMENTS = frozenset({"M1", "M2"})
SCENARIOS = ("L1", "L2", "L3", "L4", "L5")

# Run settings a --config YAML may provide. Scenario files use
# cross_traffic_mbps; grid files use clients/policies/seeds/scenario lists.
_CONFIG_ALIASES = {
    "clients": "clients", "N": "clients", "n_clients": "clients",
    "scenario": "scenario", "seed": "seed", "seeds": "seed",
    "rounds": "rounds", "n_rounds": "rounds", "beta": "beta",
    "cross_traffic_mbps": "cross_mbps", "cross_mbps": "cross_mbps",
    "trace_dir": "trace_dir", "output": "output",
    "gateway_factory": "gateway_factory", "policy": "policy", "policies": "policy",
    "experiment_id": "experiment_id", "compute_window_s": "compute_window_s",
    "value_bank": "value_bank", "models_dir": "models_dir",
}
# Scenario-file keys that describe the scenario itself; they are checked
# against the code's scenario table rather than used as run settings.
_SCENARIO_DESCRIPTIVE_KEYS = frozenset({
    "loss_probability", "mean_loss_burst_packets", "access_rate_profile", "trace_split", "platform",
})


def _pick(name: str, value, override):
    """Resolve a config value (scalar or grid list) against an optional CLI override."""
    if isinstance(value, list):
        if override is None:
            if len(value) != 1:
                raise ValueError(f"config gives several values for {name} {value}; select one with --{name.replace('_', '-')}")
            return value[0]
        if override not in value and str(override) not in [str(v) for v in value]:
            raise ValueError(f"--{name.replace('_', '-')} {override!r} is not in the config's {name} list {value}")
        return override
    return value if override is None else override


def load_run_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Load a scenario or grid YAML and resolve it to one run's settings.

    Values given on the command line (``overrides`` entries that are not
    None) win over the file; for grid files a command-line value must be one
    of the listed values. Returns a dict with keys from ``_CONFIG_ALIASES``.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - pyyaml is in the testbed extra
        raise RuntimeError("--config needs PyYAML (pip install pyyaml)") from exc
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{config_path}: top level must be a mapping")
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    unknown = sorted(set(data) - set(_CONFIG_ALIASES) - _SCENARIO_DESCRIPTIVE_KEYS)
    if unknown:
        raise ValueError(f"{config_path}: unknown keys {unknown}")
    if data.get("platform") not in (None, "mininet"):
        raise ValueError(f"{config_path}: platform {data['platform']!r} is not a Mininet config")
    resolved: dict = {}
    for key, value in data.items():
        if key in _SCENARIO_DESCRIPTIVE_KEYS:
            continue
        name = _CONFIG_ALIASES[key]
        resolved[name] = _pick(name, value, overrides.get(name))
    for name, value in overrides.items():
        resolved.setdefault(name, value)
    if "scenario" in resolved:
        resolved["scenario"] = str(resolved["scenario"]).upper()
        if resolved["scenario"] == "X2":
            # The X2 scenario file is L2 loss/capacity plus 3 Mbit/s cross traffic.
            resolved["scenario"] = "L2"
        _check_scenario_loss(config_path, data, resolved["scenario"])
    resolved["config_path"] = str(config_path)
    resolved["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return resolved


def _check_scenario_loss(config_path: Path, data: dict, scenario: str) -> None:
    if scenario not in SCENARIO_LOSS:
        raise ValueError(f"{config_path}: unknown scenario {scenario!r}")
    loss, burst = SCENARIO_LOSS[scenario]
    if "loss_probability" in data and abs(float(data["loss_probability"]) - loss) > 1e-12:
        raise ValueError(f"{config_path}: loss_probability {data['loss_probability']} != {loss} for {scenario}")
    if "mean_loss_burst_packets" in data and abs(float(data["mean_loss_burst_packets"]) - burst) > 1e-9:
        raise ValueError(f"{config_path}: mean_loss_burst_packets {data['mean_loss_burst_packets']} != {burst} for {scenario}")


def run_identity(experiment_id: str | None, policy: str | None, gateway_factory: str) -> tuple[str, str]:
    """Return (experiment_id, policy) for meta.json, refusing to label stub output as M1/M2.

    B1/B2/B3 run their own gateway/client processes, so --gateway-factory
    does not apply to them.
    """
    if policy in BASELINE_MODES:
        return str(experiment_id or STUB_EXPERIMENT_ID), str(policy)
    is_stub = gateway_factory == DEFAULT_STUB_FACTORY
    experiment_id = experiment_id or (STUB_EXPERIMENT_ID if is_stub else None)
    policy = policy or (STUB_POLICY if is_stub else None)
    if experiment_id is None or policy is None:
        raise ValueError("a non-stub gateway factory needs --experiment-id and --policy")
    if is_stub and (experiment_id in MININET_EXPERIMENTS or policy != STUB_POLICY):
        raise ValueError(
            f"the first-eligible stub gateway cannot produce {experiment_id}/{policy} results; "
            "pass the production --gateway-factory for M1/M2 runs"
        )
    return str(experiment_id), str(policy)


class _QueueSampler:
    def __init__(self, net, path: Path, interfaces: list[tuple[str, str]], interval_s: float = 0.01,
                 cpu_id: int | None = None, round_state_path: Path | None = None):
        self.net, self.path, self.interfaces, self.interval_s = net, path, interfaces, interval_s
        self.cpu_id = cpu_id
        self.round_state = RoundStateFile(round_state_path) if round_state_path else None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="fedgrant-queue-sampler", daemon=True)
        self.error: BaseException | None = None

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        try:
            if self.cpu_id is not None and hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(threading.get_native_id(), {self.cpu_id})
            with self.path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=QUEUE_COLUMNS)
                writer.writeheader()
                interval_ns = max(1, int(self.interval_s * 1e9))
                next_sample_ns = time.monotonic_ns()
                while not self.stop_event.is_set():
                    for host_name, iface in self.interfaces:
                        host = self.net.get(host_name)
                        for row in sample_qdisc(host, iface):
                            row["round_id"] = (self.round_state.round_at(int(row["t_mono_ns"]))
                                               if self.round_state else 0)
                            writer.writerow(row)
                    stream.flush()
                    next_sample_ns += interval_ns
                    now_ns = time.monotonic_ns()
                    while next_sample_ns <= now_ns:
                        next_sample_ns += interval_ns
                    self.stop_event.wait((next_sample_ns - now_ns) / 1e9)
        except BaseException as exc:
            self.error = exc
            self.stop_event.set()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise TimeoutError("queue sampler did not stop")
        if self.error is not None:
            raise RuntimeError("queue sampler failed") from self.error


def cpu_roles(allowed_cpus: set[int] | None = None,
              sysfs_cpu_root: str | Path = "/sys/devices/system/cpu") -> dict[str, object]:
    """Choose a fastest logical CPU for the gateway and lower-frequency CPUs for clients."""
    if allowed_cpus is None:
        allowed_cpus = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    allowed = sorted(allowed_cpus)
    if not allowed:
        return {"gateway": None, "telemetry": None, "clients": [], "pinning_enabled": False}
    root = Path(sysfs_cpu_root)
    frequencies: dict[int, int] = {}
    for cpu in allowed:
        for filename in ("cpuinfo_max_freq", "scaling_max_freq"):
            try:
                frequencies[cpu] = int((root / f"cpu{cpu}" / "cpufreq" / filename).read_text().strip())
                break
            except (OSError, ValueError):
                continue
    if frequencies:
        fastest = max(frequencies.values())
        performance = [cpu for cpu in allowed if frequencies.get(cpu) == fastest]
        efficiency = [cpu for cpu in allowed if cpu not in performance]
    else:
        performance, efficiency = allowed[:1], allowed[1:]
    client_cpus = efficiency or [cpu for cpu in allowed if cpu != performance[0]]
    return {"gateway": performance[0], "telemetry": performance[1] if len(performance) > 1 else None,
            "clients": client_cpus, "pinning_enabled": shutil.which("taskset") is not None}


def smoke_duration_ns(beta: float, n: int, gateway_factory: str) -> int:
    """Keep a small-client stub smoke round long enough for a grant/data exchange."""
    duration_ns = compute_deadline_ns(beta, n)
    if gateway_factory == DEFAULT_STUB_FACTORY:
        # N=3 gives only ~60 ms at beta=0.1, shorter than the core's 100 ms
        # initial SRTT. Use the N=20 reference deadline for the test-only stub.
        duration_ns = max(duration_ns, compute_deadline_ns(beta, 20))
    return duration_ns


def process_commands(*, python: str, policy: str, n: int, rounds: int, seed: int, duration_ns: int,
                     compute_window_s: float, out: Path, round_state_path: Path,
                     gateway_factory: str, value_bank: str | None, models_dir: str,
                     profile_gateway: bool, late_drain_s: float = 0.1,
                     round_timeout_s: float = 5.0) -> tuple[list[str], list[list[str]]]:
    """Return the gateway command and one command per client (contract Section 2)."""
    gw_prefix, client_prefix = entry_points(policy)
    gw_cmd = [python, *gw_prefix, "--clients", str(n), "--rounds", str(rounds), "--seed", str(seed),
              "--duration-ns", str(duration_ns), "--compute-window-s", repr(float(compute_window_s)),
              "--late-drain-s", repr(float(late_drain_s)),
              "--log-dir", str(out), "--round-state", str(round_state_path)]
    if not is_baseline(policy):
        gw_cmd += ["--gateway-factory", gateway_factory, "--policy", policy, "--models-dir", str(models_dir)]
    if value_bank:
        gw_cmd += ["--value-bank", str(value_bank)]
    if profile_gateway:
        gw_cmd.append("--profile-gateway")
    client_cmds = []
    for i in range(n):
        cmd = [python, *client_prefix, "--client-id", str(i), "--rounds", str(rounds), "--seed", str(seed),
               "--round-timeout-s", repr(float(round_timeout_s)), "--log-file", str(out / f"client_{i}.json")]
        if value_bank:
            cmd += ["--clients", str(n), "--value-bank", str(value_bank)]
        client_cmds.append(cmd)
    return gw_cmd, client_cmds


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def run_experiment(*, n: int, scenario: str, seed: int, rounds: int, beta: float,
                   trace_dir: str | Path, output_dir: str | Path, cross_mbps: float = 0.0,
                   gateway_factory: str = DEFAULT_STUB_FACTORY,
                   profile_gateway: bool = False, experiment_id: str | None = None,
                   policy: str | None = None, compute_window_s: float = 1.0,
                   config: dict | None = None, value_bank: str | None = None,
                   models_dir: str = DEFAULT_MODELS_DIR) -> Path:
    """Run Mininet UDP rounds; the default stub factory is a protocol smoke test only.

    Every exit path stops samplers, replay threads, cross traffic, child
    processes and the Mininet network. A failed run leaves ``run_failure.json``
    (not ``meta.json``), so the analysis never mistakes it for a complete run.
    """
    if not isinstance(profile_gateway, bool):
        raise TypeError("profile_gateway must be a bool")
    scenario = str(scenario).upper()
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    if not compute_window_s >= 0:
        raise ValueError("compute_window_s must be non-negative")
    experiment_id, policy = run_identity(experiment_id, policy, gateway_factory)
    stub_run = policy == STUB_POLICY
    if not stub_run and policy not in GRANT_POLICIES and policy not in BASELINE_MODES:
        raise ValueError(f"policy {policy!r} has no Mininet implementation; "
                         f"expected one of {sorted(GRANT_POLICIES | set(BASELINE_MODES))}")
    if experiment_id in MININET_EXPERIMENTS:
        # Spec Section 5.2/12: real runs use the FULL value bank and always
        # record gateway CPU/RSS (sampling is otherwise opt-in).
        value_bank = value_bank or DEFAULT_VALUE_BANK
        profile_gateway = True
    if value_bank is not None and not Path(value_bank).is_file():
        raise FileNotFoundError(f"value bank not found: {value_bank} (get it from the FULL seed-0 run)")
    if not stub_run:
        check_entry_points_available(policy)
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    identity = {"experiment_id": experiment_id, "policy": policy, "scenario": scenario,
                "seed": int(seed), "N": int(n), "beta": float(beta), "cross_mbps": float(cross_mbps),
                "n_rounds": int(rounds), "platform": "mininet"}
    started = time.monotonic_ns()
    roles = cpu_roles()
    net = None
    processes = []
    log_handles = []
    replayers = []
    cross = None
    cross_result = None
    sampler = None
    process_sampler = None
    try:
        net = build(n=n, scenario=scenario, seed=seed)
        verify_tc(net, net._fedgrant_expected_tc)
        round_state_path = out / ".round_state.json"
        RoundStateFile(round_state_path).write(round_id=0, active=False)
        env = os.environ.copy()
        project_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        py = sys.executable
        duration_ns = smoke_duration_ns(beta, n, DEFAULT_STUB_FACTORY if stub_run else policy)
        gw_cmd, client_cmds = process_commands(
            python=py, policy=policy, n=n, rounds=rounds, seed=seed, duration_ns=duration_ns,
            compute_window_s=compute_window_s, out=out, round_state_path=round_state_path,
            gateway_factory=gateway_factory, value_bank=value_bank, models_dir=models_dir,
            profile_gateway=profile_gateway)
        gateway_log = (out / "gateway.log").open("w", encoding="utf-8")
        log_handles.append(gateway_log)
        if roles["pinning_enabled"] and roles["gateway"] is not None:
            gw_cmd = ["taskset", "-c", str(roles["gateway"]), *gw_cmd]
        processes.append(net.get("gw").popen(gw_cmd, stdout=gateway_log, stderr=subprocess.STDOUT, env=env))
        for i in range(n):
            client_log = (out / f"client_{i}.log").open("w", encoding="utf-8")
            log_handles.append(client_log)
            cmd = client_cmds[i]
            if roles["pinning_enabled"] and roles["clients"]:
                cmd = ["taskset", "-c", str(roles["clients"][i % len(roles["clients"])]), *cmd]
            processes.append(net.get(f"c{i}").popen(cmd, stdout=client_log, stderr=subprocess.STDOUT, env=env))
        if profile_gateway:
            process_sampler = ProcessSampler([(processes[0], "gateway")], out / "processes.csv",
                                             cpu_id=roles["telemetry"],
                                             round_state_path=round_state_path).start()

        if scenario.upper() in {"L4", "L5"}:
            split = "ood" if scenario.upper() == "L5" else "test"
            for i in range(n):
                rates = client_trace(i, split, seed, data_dir=trace_dir)
                host, iface = net.get(f"c{i}"), f"c{i}-eth0"
                replay_log = out / f"trace_replay_client_{i}.csv"

                def log_rate_change(actual_ns: int, replay_round: int, bin_index: int,
                                    rate_bps: int, scheduled_ns: int, *, path=replay_log) -> None:
                    exists = path.exists()
                    with path.open("a", newline="", encoding="utf-8") as stream:
                        writer = csv.writer(stream)
                        if not exists:
                            writer.writerow(("t_mono_ns", "round_id", "bin_index", "rate_bps",
                                             "scheduled_mono_ns", "lag_ns"))
                        writer.writerow((actual_ns, replay_round, bin_index, rate_bps,
                                         scheduled_ns, actual_ns - scheduled_ns))

                replayer = RateReplay(rates, lambda rate, h=host, dev=iface: h.cmd(tbf_change(dev, rate)),
                                      repeat=True, round_state_path=round_state_path,
                                      on_change=log_rate_change)
                replayers.append(replayer.start())
        if cross_mbps > 0:
            cross = CrossTraffic(net.get("gw"), net.get("xt"))
            cross.start(cross_mbps, duration_s=max(2, int(rounds * 2)))

        # The specification requires tc -s sampling every 10 ms at the shared
        # bottleneck. Access-link queues are not part of the reported bottleneck
        # queueing-delay or drop metrics.
        queue_ifaces = [("s1", "s1-gw")]
        sampler = _QueueSampler(net, out / "queues.csv", queue_ifaces, cpu_id=roles["telemetry"],
                                round_state_path=round_state_path).start()
        deadline = time.monotonic() + max(10.0, rounds * 3.0)
        failures = []
        for proc in processes:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                code = proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                failures.append(f"process pid={proc.pid} timed out")
                continue
            if code != 0:
                failures.append(f"process pid={proc.pid} exited {code}")
        if failures:
            raise RuntimeError("Mininet run failed: " + "; ".join(failures))
        # Combine sender-side client packets (which include DATA lost on the
        # access link) with gateway-side observations into the canonical log.
        packet_path = out / "packets.csv"
        packet_rows = []
        with packet_path.open(newline="", encoding="utf-8") as stream:
            packet_rows.extend(csv.DictReader(stream))
        for client_path in sorted(out.glob("client_*_packets.csv")):
            with client_path.open(newline="", encoding="utf-8") as stream:
                packet_rows.extend(csv.DictReader(stream))
        packet_rows.sort(key=lambda row: (int(row["t_mono_ns"]), row["direction"], row["client_id"]))
        with packet_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=PACKET_COLUMNS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(packet_rows)
        if cross is not None:
            cross_result = cross.wait(timeout=max(5.0, rounds * 2.0 + 3.0))
        if sampler is not None:
            sampler.stop()
            sampler = None
        if process_sampler is not None:
            process_sampler.stop()
            process_sampler = None
        enrich_round_metrics(out, cross_mbps=cross_mbps)
        if stub_run:
            summary_path = out / "gateway_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            accepted = sum(int(row.get("accepted_data", 0)) for row in summary)
            invalid = sum(int(row.get("invalid_events", 0)) for row in summary)
            if accepted < 1 or invalid:
                raise RuntimeError(
                    f"stub smoke did not complete a valid grant/data exchange: "
                    f"accepted_data={accepted}, invalid_events={invalid}; see {summary_path}"
                )
        trace_root = Path(trace_dir)
        trace_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(trace_root.glob("*.csv"))
        } if trace_root.exists() else {}
        tc_readback = []
        for (host_name, iface), commands in sorted(net._fedgrant_expected_tc.items()):
            host = net.get(host_name)
            tc_readback.append({"host": host_name, "interface": iface, "configured_commands": commands,
                                "effective": host.cmd(f"tc -d qdisc show dev {iface}").strip()})
        loss_target, burst_target = SCENARIO_LOSS[scenario.upper()]
        meta = {
            **identity, "status": "ok", "rounds": int(rounds),
            "config_id": (make_config_id(scenario, beta, n, cross_mbps)
                          if experiment_id in MININET_EXPERIMENTS else None),
            "config": ({k: config[k] for k in ("config_path", "config_sha256") if k in config}
                       if config else None),
            "compute_window_s": float(compute_window_s), "duration_ns": int(duration_ns),
            "started_mono_ns": started, "ended_mono_ns": time.monotonic_ns(),
            "gateway_factory": None if is_baseline(policy) else gateway_factory,
            "gateway_entry_point": " ".join(entry_points(policy)[0][1:]),
            "client_entry_point": " ".join(entry_points(policy)[1][1:]),
            "value_bank": ({"path": str(value_bank), "sha256": _sha256(Path(value_bank))}
                           if value_bank else None),
            "profile_source": "value_bank" if value_bank else "synthetic",
            "models_dir": str(models_dir),
            "model_sha256": {name: _sha256(Path(models_dir) / name)
                             for name in POLICY_MODEL_FILES.get(policy, ())},
            "trace_dir": str(trace_dir),
            "trace_files": sorted(trace_hashes), "trace_sha256": trace_hashes,
            "trace_source": (trace_root.joinpath("SOURCE").read_text(encoding="utf-8")
                             if trace_root.joinpath("SOURCE").is_file() else None),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
            "cpu_roles": roles,
            "queue_interfaces": [list(item) for item in queue_ifaces],
            "effective_queue_settings": {
                "bottleneck_rate_bps": C, "bottleneck_limit_frames": Q,
                "frame_size_bytes": S, "bottleneck_limit_bytes": Q * S,
                "sampling_interval_ms": 10, "bottleneck_interface": "s1-gw",
                "per_client_access": [{"client_id": (p := client_parameters(i)).client_id,
                                       "delay_ms": p.delay_ms, "rate_bps": p.rate_bps,
                                       "rtt_ms": p.rtt_ms} for i in range(n)],
                "ge_target_loss_rate": loss_target, "ge_target_mean_burst": burst_target,
                "tc_readback": tc_readback,
            },
            "cross_traffic": ({"iperf3": cross_result, "receiver_payload_bps": udp_payload_bps(cross_result)}
                              if cross_result is not None else None),
            "gateway_profiling": {
                "enabled": profile_gateway,
                "profiled_process": "gateway" if profile_gateway else None,
                "warmup_rounds_excluded": 20 if profile_gateway else None,
            },
        }
        if profile_gateway:
            profile = summarize_gateway_profile(out)
            (out / "gateway_profile.json").write_text(
                json.dumps(profile, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            meta["gateway_profiling"]["summary_file"] = "gateway_profile.json"
        (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out
    except BaseException as exc:
        try:
            failure = {**identity, "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                       "started_mono_ns": started, "failed_mono_ns": time.monotonic_ns(),
                       "gateway_factory": gateway_factory}
            (out / "run_failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n",
                                                  encoding="utf-8")
        except OSError:
            pass
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup_errors = []
        for resource in ([sampler] if sampler is not None else []) + ([process_sampler] if process_sampler is not None else []):
            try:
                resource.stop()
            except BaseException as exc:
                cleanup_errors.append(exc)
        for replayer in replayers:
            try:
                replayer.stop()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cross is not None:
            try:
                cross.stop()
            except BaseException as exc:
                cleanup_errors.append(exc)
        try:
            teardown(net, processes)
        except BaseException as exc:
            cleanup_errors.append(exc)
        finally:
            for handle in log_handles:
                handle.close()
        if cleanup_errors and not active_error:
            raise RuntimeError("one or more Mininet cleanup steps failed") from cleanup_errors[0]


_DEFAULTS = {"clients": 3, "scenario": "L2", "seed": 0, "rounds": 1, "beta": 0.10, "cross_mbps": 0.0,
             "trace_dir": "data/traces", "gateway_factory": DEFAULT_STUB_FACTORY, "compute_window_s": 1.0,
             "experiment_id": None, "policy": None, "output": None, "profile_gateway": False,
             "value_bank": None, "models_dir": DEFAULT_MODELS_DIR}


def resolve_cli(argv: list[str] | None = None) -> dict:
    """Parse the command line; ``--config <yaml> --seed <s>`` is the specified form (Section 7)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="scenario or grid YAML (e.g. configs/grids/mn_m1.yaml)")
    parser.add_argument("--clients", type=int)
    parser.add_argument("--scenario", type=str.upper, choices=SCENARIOS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--cross-mbps", dest="cross_mbps", type=float)
    parser.add_argument("--trace-dir", dest="trace_dir")
    parser.add_argument("--output", help="new, empty run directory (default: results/mininet/<run id>)")
    parser.add_argument("--gateway-factory", dest="gateway_factory")
    parser.add_argument("--experiment-id", dest="experiment_id")
    parser.add_argument("--policy")
    parser.add_argument("--compute-window-s", dest="compute_window_s", type=float)
    parser.add_argument("--value-bank", dest="value_bank",
                        help=f"FULL value bank .npz (default for M1/M2: {DEFAULT_VALUE_BANK})")
    parser.add_argument("--models-dir", dest="models_dir")
    parser.add_argument("--profile-gateway", dest="profile_gateway", action="store_true", default=None,
                        help="collect gateway process CPU/RSS and scheduler-latency samples")
    args = vars(parser.parse_args(argv))
    config_path = args.pop("config")
    overrides = {k: v for k, v in args.items() if v is not None}
    try:
        settings = load_run_config(config_path, overrides) if config_path else dict(overrides)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for key, value in _DEFAULTS.items():
        settings.setdefault(key, value)
    if settings["output"] is None:
        label = "-".join(str(part) for part in (
            settings["experiment_id"] or STUB_EXPERIMENT_ID, settings["scenario"],
            settings["policy"] or "stub", f"s{settings['seed']}", time.strftime("%Y%m%d-%H%M%S")))
        settings["output"] = str(Path("results/mininet") / label)
    return settings


def main(argv: list[str] | None = None) -> None:
    s = resolve_cli(argv)
    path = run_experiment(n=int(s["clients"]), scenario=s["scenario"], seed=int(s["seed"]),
                          rounds=int(s["rounds"]), beta=float(s["beta"]), trace_dir=s["trace_dir"],
                          output_dir=s["output"], cross_mbps=float(s["cross_mbps"]),
                          gateway_factory=s["gateway_factory"], profile_gateway=bool(s["profile_gateway"]),
                          experiment_id=s["experiment_id"],
                          policy=s["policy"], compute_window_s=float(s["compute_window_s"]),
                          config=s if "config_path" in s else None, value_bank=s["value_bank"],
                          models_dir=s["models_dir"])
    print(path)


if __name__ == "__main__":
    main()
