"""iperf3 cross-traffic process management for the Mininet namespace hosts."""

from __future__ import annotations

import json
import math
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


class CrossTraffic:
    def __init__(self, server_host: Any, client_host: Any, *, port: int = 9999):
        self.server_host = server_host
        self.client_host = client_host
        self.port = int(port)
        self.server: subprocess.Popen | None = None
        self.client: subprocess.Popen | None = None

    def start(self, rate_mbps: float, *, duration_s: int = 30, payload_bytes: int = 1432) -> None:
        if rate_mbps <= 0 or duration_s <= 0 or payload_bytes <= 0:
            raise ValueError("rate_mbps, duration_s, and payload_bytes must be positive")
        if self.server is not None:
            raise RuntimeError("cross traffic already started")
        self.server = self.server_host.popen(["iperf3", "-s", "-p", str(self.port)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        time.sleep(0.15)
        if self.server.poll() is not None:
            err = self.server.stderr.read().decode(errors="replace") if self.server.stderr else ""
            self.stop()
            raise RuntimeError(f"iperf3 server failed to start: {err}")
        self.client = self.client_host.popen(
            ["iperf3", "-c", "10.0.0.254", "-p", str(self.port), "-u", "-b", f"{rate_mbps:g}M",
             "-l", str(payload_bytes), "-t", str(duration_s), "-J"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def wait(self, timeout: float | None = None) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("cross traffic has not been started")
        stdout, stderr = self.client.communicate(timeout=timeout)
        if self.client.returncode:
            raise RuntimeError(f"iperf3 client failed ({self.client.returncode}): {stderr.decode(errors='replace')}")
        try:
            return json.loads(stdout)
        finally:
            self.client = None

    def stop(self) -> None:
        for proc in (self.client, self.server):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        self.client = self.server = None

    def __enter__(self) -> "CrossTraffic":
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


class CrossCapture:
    """Capture cross-traffic datagrams at the gateway for per-flow drop attribution.

    ``tcpdump`` writes only the first 96 bytes of each iperf3 datagram (UDP
    destination port ``port``), enough for the sequence/timestamp header.
    ``realtime_minus_mono_ns`` converts its CLOCK_REALTIME stamps to the
    monotonic clock used by every other log.
    """

    def __init__(self, host: Any, iface: str, path: str | Path, *, port: int = 9999):
        self.host, self.iface, self.path, self.port = host, iface, Path(path), int(port)
        self.proc: subprocess.Popen | None = None
        self.realtime_minus_mono_ns: int | None = None

    def start(self) -> "CrossCapture":
        if shutil.which("tcpdump") is None:
            raise RuntimeError("tcpdump is required to attribute cross-traffic drops (dnf install tcpdump)")
        self.proc = self.host.popen(
            ["tcpdump", "-i", self.iface, "-n", "-U", "-s", "96", "--time-stamp-precision=nano",
             "-w", str(self.path), "udp", "and", "dst", "port", str(self.port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        self.realtime_minus_mono_ns = time.time_ns() - time.monotonic_ns()
        time.sleep(0.3)  # let tcpdump open the interface before traffic starts
        if self.proc.poll() is not None:
            err = self.proc.stderr.read().decode(errors="replace") if self.proc.stderr else ""
            self.proc = None
            raise RuntimeError(f"tcpdump failed to start: {err}")
        return self

    def stop(self) -> None:
        """Stop with SIGINT so tcpdump flushes the capture file; idempotent."""
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)


def cross_duration_s(rounds: int, duration_ns: int, compute_window_s: float, *,
                     start_delay_s: float = 0.4, late_drain_s: float = 0.1, margin_s: float = 3.0) -> int:
    """iperf3 duration covering every round of a run (whole seconds)."""
    per_round = compute_window_s + duration_ns / 1e9 + late_drain_s
    return int(math.ceil(start_delay_s + rounds * per_round + margin_s))


def udp_payload_bps(result: dict[str, Any]) -> float:
    """Return receiver application bit/s from iperf3 JSON output."""
    try:
        return float(result["end"]["sum"]["bits_per_second"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("iperf3 JSON has no end.sum.bits_per_second") from exc


__all__ = ["CrossCapture", "CrossTraffic", "cross_duration_s", "udp_payload_bps"]
