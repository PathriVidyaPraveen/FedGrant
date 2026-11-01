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

from fedgrant.common.logschema import QUEUE_COLUMNS
from fedgrant.mininet.crosstraffic import CrossTraffic, udp_payload_bps
from fedgrant.mininet.profile_update import profile_bank
from fedgrant.mininet.replay import RateReplay
from fedgrant.mininet.teardown import teardown
from fedgrant.mininet.telemetry import ProcessSampler
from fedgrant.mininet.topo import build, verify_tc
from fedgrant.mininet.tcsetup import tbf_change
from fedgrant.traces.client_trace import client_trace
from fedgrant.sim.run_net import compute_deadline_ns

DEFAULT_STUB_FACTORY = "fedgrant.mininet.stub_policy:create_gateway"


class _QueueSampler:
    def __init__(self, net, path: Path, interfaces: list[tuple[str, str]], interval_s: float = 0.1,
                 cpu_id: int | None = None):
        self.net, self.path, self.interfaces, self.interval_s = net, path, interfaces, interval_s
        self.cpu_id = cpu_id
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
                while not self.stop_event.is_set():
                    now = time.monotonic_ns()
                    for host_name, iface in self.interfaces:
                        host = self.net.get(host_name)
                        raw = host.cmd(f"tc -s -j qdisc show dev {iface}")
                        try:
                            qdiscs = json.loads(raw)
                        except json.JSONDecodeError as exc:
                            raise RuntimeError(f"bad tc JSON on {host_name}:{iface}: {raw[:120]!r}") from exc
                        for qdisc in qdiscs:
                            stats = qdisc.get("stats", {})
                            writer.writerow({"t_mono_ns": now, "interface": iface,
                                             "qdisc": qdisc.get("kind", "unknown"),
                                             "backlog_bytes": stats.get("backlog", 0),
                                             "bytes": stats.get("bytes", 0),
                                             "drops": stats.get("drops", 0),
                                             "overlimits": stats.get("overlimits", 0)})
                    stream.flush()
                    self.stop_event.wait(self.interval_s)
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
                  "--gateway-factory", gateway_factory, "--log-dir", str(out)]
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
        process_sampler = ProcessSampler([(processes[0], "gateway")], out / "processes.csv",
                                         cpu_id=roles["telemetry"]).start()

        if scenario.upper() in {"L4", "L5"}:
            split = "ood" if scenario.upper() == "L5" else "test"
            for i in range(n):
                rates = client_trace(i, split, seed, data_dir=trace_dir)
                host, iface = net.get(f"c{i}"), f"c{i}-eth0"
                replayer = RateReplay(rates, lambda rate, h=host, dev=iface: h.cmd(tbf_change(dev, rate)),
                                      repeat=True)
                replayers.append(replayer.start())
        if cross_mbps > 0:
            cross = CrossTraffic(net.get("gw"), net.get("xt"))
            cross.start(cross_mbps, duration_s=max(2, int(rounds * 2)))

        queue_ifaces = [("s1", "s1-gw"), *(('s1', f"s1-c{i}") for i in range(n))]
        sampler = _QueueSampler(net, out / "queues.csv", queue_ifaces, cpu_id=roles["telemetry"]).start()
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
        if cross is not None:
            cross_result = cross.wait(timeout=max(5.0, rounds * 2.0 + 3.0))
        trace_root = Path(trace_dir)
        trace_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(trace_root.glob("*.csv"))
        } if trace_root.exists() else {}
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
