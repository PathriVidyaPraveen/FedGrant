"""iperf3 cross-traffic process management for the Mininet namespace hosts."""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any


class CrossTraffic:
    def __init__(self, server_host: Any, client_host: Any, *, port: int = 5201):
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


def udp_payload_bps(result: dict[str, Any]) -> float:
    """Return receiver application bit/s from iperf3 JSON output."""
    try:
        return float(result["end"]["sum"]["bits_per_second"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("iperf3 JSON has no end.sum.bits_per_second") from exc


__all__ = ["CrossTraffic", "udp_payload_bps"]
