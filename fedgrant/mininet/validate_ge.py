"""Sequence-numbered UDP loss/burst measurement for netem GE validation."""

from __future__ import annotations

import argparse
import json
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from fedgrant.mininet.ge import analyze_loss, bootstrap_loss_intervals

_PACKET = struct.Struct("!IQ")


def receive(bind_host: str, port: int, expected: int, timeout_s: float) -> np.ndarray:
    if expected <= 0 or timeout_s <= 0:
        raise ValueError("expected and timeout_s must be positive")
    seen = np.zeros(expected, dtype=np.bool_)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, min(64 * 1024 * 1024, expected * 128))
    sock.bind((bind_host, port))
    sock.settimeout(0.1)
    deadline = time.monotonic() + timeout_s
    last_packet: float | None = None
    idle_limit = max(0.5, min(2.0, timeout_s * 0.1))
    try:
        while time.monotonic() < deadline:
            if last_packet is not None and time.monotonic() - last_packet >= idle_limit:
                break
            try:
                packet, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            last_packet = time.monotonic()
            if len(packet) < _PACKET.size:
                continue
            sequence, _sent_ns = _PACKET.unpack_from(packet)
            if sequence < expected:
                seen[sequence] = True
            # Keep listening to the end of the sender's window so trailing loss
            # is not mistaken for a short experiment.
    finally:
        sock.close()
    return seen


def send(destination: str, port: int, count: int, interval_s: float) -> None:
    if count <= 0 or interval_s < 0:
        raise ValueError("count must be positive and interval_s non-negative")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        start = time.monotonic_ns()
        for sequence in range(count):
            target = start + int(sequence * interval_s * 1e9)
            while (remaining := target - time.monotonic_ns()) > 0:
                time.sleep(min(remaining / 1e9, 0.001))
            sock.sendto(_PACKET.pack(sequence, time.monotonic_ns()), (destination, port))
    finally:
        sock.close()


def run_scenario(scenario: str, *, count: int = 100_000, interval_s: float = 0.00025,
                 output: str | Path = "results/mininet/ge") -> dict[str, float | int | str]:
    """Run one measured GE trial through the real client uplink in a 1-host topology."""
    if scenario not in {"L1", "L2", "L3"}:
        raise ValueError("scenario must be L1, L2, or L3")
    from fedgrant.mininet.teardown import teardown
    from fedgrant.mininet.topo import build

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    net = build(n=1, scenario=scenario)
    receiver = sender = None
    try:
        receiver = net.get("gw").popen(
            [sys.executable, "-m", "fedgrant.mininet.validate_ge", "receive", "--bind", "10.0.0.254",
             "--port", "6000", "--count", str(count), "--timeout-s", "45", "--output", str(out)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        time.sleep(0.2)
        sender = net.get("c0").popen(
            [sys.executable, "-m", "fedgrant.mininet.validate_ge", "send", "--destination", "10.0.0.254",
             "--port", "6000", "--count", str(count), "--interval-s", str(interval_s)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        _, sender_err = sender.communicate(timeout=max(30.0, count * interval_s + 10.0))
        if sender.returncode != 0:
            raise RuntimeError(f"GE sender failed: {sender_err.decode(errors='replace')}")
        _, receiver_err = receiver.communicate(timeout=50)
        if receiver.returncode != 0:
            raise RuntimeError(f"GE receiver failed: {receiver_err.decode(errors='replace')}")
        stats = json.loads(out.read_text(encoding="utf-8"))
        stats["scenario"] = scenario
        stats["target_loss_rate"] = 0.05
        stats["target_mean_burst"] = {"L1": 1 / 0.95, "L2": 8.0, "L3": 32.0}[scenario]
        out.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return stats
    finally:
        teardown(net, [p for p in (sender, receiver) if p is not None])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="role", required=True)
    recv = sub.add_parser("receive")
    recv.add_argument("--bind", default="0.0.0.0")
    recv.add_argument("--port", type=int, default=6000)
    recv.add_argument("--count", type=int, default=100_000)
    recv.add_argument("--timeout-s", type=float, default=30.0)
    recv.add_argument("--output", required=True)
    snd = sub.add_parser("send")
    snd.add_argument("--destination", required=True)
    snd.add_argument("--port", type=int, default=6000)
    snd.add_argument("--count", type=int, default=100_000)
    snd.add_argument("--interval-s", type=float, default=0.0001)
    measure = sub.add_parser("measure", help="run a real topology trial for one GE scenario")
    measure.add_argument("--scenario", choices=("L1", "L2", "L3"), required=True)
    measure.add_argument("--count", type=int, default=100_000)
    measure.add_argument("--interval-s", type=float, default=0.00025)
    measure.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.role == "send":
        send(args.destination, args.port, args.count, args.interval_s)
    elif args.role == "receive":
        received = receive(args.bind, args.port, args.count, args.timeout_s)
        stats = analyze_loss(received)
        output = {"n": stats.n, "lost": stats.lost, "loss_rate": stats.loss_rate,
                  "mean_burst": stats.mean_burst, "n_bursts": stats.n_bursts,
                  "complete_bursts": stats.complete_bursts,
                  **bootstrap_loss_intervals(received)}
        Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(output, sort_keys=True))
    else:
        print(json.dumps(run_scenario(args.scenario, count=args.count, interval_s=args.interval_s,
                                      output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["receive", "send"]
