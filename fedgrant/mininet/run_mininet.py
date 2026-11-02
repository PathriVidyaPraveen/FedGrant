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

from fedgrant.common.logschema import PACKET_COLUMNS, QUEUE_COLUMNS
from fedgrant.mininet.crosstraffic import CrossTraffic, udp_payload_bps
from fedgrant.mininet.profile_update import profile_bank
from fedgrant.mininet.replay import RateReplay
from fedgrant.mininet.teardown import teardown
from fedgrant.mininet.telemetry import ProcessSampler, sample_qdisc
from fedgrant.mininet.round_state import RoundStateFile
from fedgrant.mininet.round_summary import enrich_round_metrics
from fedgrant.mininet.topo import SCENARIO_LOSS, build, client_parameters, verify_tc
from fedgrant.mininet.tcsetup import tbf_change
from fedgrant.traces.client_trace import client_trace
from fedgrant.sim.run_net import compute_deadline_ns
from fedgrant.policy.constants import C, Q, S

DEFAULT_STUB_FACTORY = "fedgrant.mininet.stub_policy:create_gateway"


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


def run_experiment(*, n: int, scenario: str, seed: int, rounds: int, beta: float,
                   trace_dir: str | Path, output_dir: str | Path, cross_mbps: float = 0.0,
                   gateway_factory: str = DEFAULT_STUB_FACTORY) -> Path:
    """Run a real UDP smoke round with local stub policy providers."""
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
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
        env = os.environ.copy()
        project_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        py = sys.executable
        duration_ns = smoke_duration_ns(beta, n, gateway_factory)
        gateway_log = (out / "gateway.log").open("w", encoding="utf-8")
        log_handles.append(gateway_log)
        gw_cmd = [py, "-m", "fedgrant.mininet.gateway_proc", "--clients", str(n), "--rounds", str(rounds),
                  "--duration-ns", str(duration_ns),
                  "--gateway-factory", gateway_factory, "--log-dir", str(out),
                  "--round-state", str(out / ".round_state.json")]
        if roles["pinning_enabled"] and roles["gateway"] is not None:
            gw_cmd = ["taskset", "-c", str(roles["gateway"]), *gw_cmd]
        processes.append(net.get("gw").popen(gw_cmd, stdout=gateway_log, stderr=subprocess.STDOUT, env=env))
        for i in range(n):
            client_log = (out / f"client_{i}.log").open("w", encoding="utf-8")
            log_handles.append(client_log)
            cmd = [py, "-m", "fedgrant.mininet.client_proc", "--client-id", str(i), "--rounds", str(rounds),
                   "--seed", str(seed), "--log-file", str(out / f"client_{i}.json")]
            if roles["pinning_enabled"] and roles["clients"]:
                cmd = ["taskset", "-c", str(roles["clients"][i % len(roles["clients"])]), *cmd]
            processes.append(net.get(f"c{i}").popen(cmd, stdout=client_log, stderr=subprocess.STDOUT, env=env))
        round_state_path = out / ".round_state.json"
        RoundStateFile(round_state_path).write(round_id=0, active=False)
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
        if gateway_factory == DEFAULT_STUB_FACTORY:
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
            "experiment_id": "smoke-stub", "platform": "mininet", "scenario": scenario.upper(),
            "seed": int(seed), "N": int(n), "beta": float(beta), "cross_mbps": float(cross_mbps),
            "rounds": int(rounds), "compute_window_s": 1.0, "duration_ns": int(duration_ns),
            "started_mono_ns": started, "ended_mono_ns": time.monotonic_ns(),
            "gateway_factory": gateway_factory, "trace_dir": str(trace_dir),
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
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--scenario", choices=("L1", "L2", "L3", "L4", "L5"), default="L2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--cross-mbps", type=float, default=0.0)
    parser.add_argument("--trace-dir", default="data/traces")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gateway-factory", default=DEFAULT_STUB_FACTORY)
    args = parser.parse_args()
    path = run_experiment(n=args.clients, scenario=args.scenario, seed=args.seed, rounds=args.rounds,
                          beta=args.beta, trace_dir=args.trace_dir, output_dir=args.output,
                          cross_mbps=args.cross_mbps, gateway_factory=args.gateway_factory)
    print(path)


if __name__ == "__main__":
    main()
