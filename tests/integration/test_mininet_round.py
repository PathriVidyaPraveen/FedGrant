"""Root-required Mininet acceptance test; skipped when host prerequisites are absent."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time

import pytest

from fedgrant.mininet.topo import build

pytestmark = pytest.mark.skipif(
    os.geteuid() != 0 or importlib.util.find_spec("mininet") is None,
    reason="requires root and the host Mininet package",
)


def _udp_goodput(server, client, *, offered_mbps: int, duration_s: int = 3, port: int = 39173) -> float:
    """Measure received UDP payload without iperf's in-band TCP control flow."""
    receiver_code = """import json, socket, sys, time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((sys.argv[1], int(sys.argv[2])))
s.settimeout(.1)
duration = float(sys.argv[3])
deadline = None
total = 0
first = None
last = None
while True:
    try:
        data, _ = s.recvfrom(65535)
        now = time.monotonic()
        total += len(data)
        first = now if first is None else first
        last = now
        if deadline is None:
            deadline = now + duration + .25
    except socket.timeout:
        if deadline is not None and time.monotonic() >= deadline:
            break
print(json.dumps({'bytes': total, 'first': first, 'last': last}))
"""
    sender_code = """import socket, sys, time
dst = (sys.argv[1], int(sys.argv[2]))
rate = float(sys.argv[3]) * 1e6
duration = float(sys.argv[4])
payload = b'x' * 1432
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
start = time.monotonic()
deadline = start + duration
next_send = start
interval = len(payload) * 8 / rate
while time.monotonic() < deadline:
    now = time.monotonic()
    if now < next_send:
        time.sleep(min(next_send - now, .001))
        continue
    s.sendto(payload, dst)
    next_send = max(next_send + interval, now)
"""
    receiver = server.popen(
        [sys.executable, "-c", receiver_code, "10.0.0.254", str(port), str(duration_s)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        time.sleep(0.2)
        sender = client.popen(
            [sys.executable, "-c", sender_code, "10.0.0.254", str(port), str(offered_mbps), str(duration_s)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        _, stderr = sender.communicate(timeout=duration_s + 5)
        assert sender.returncode == 0, stderr.decode(errors="replace")
        stdout, stderr = receiver.communicate(timeout=duration_s + 5)
        assert receiver.returncode == 0, stderr.decode(errors="replace")
        result = json.loads(stdout)
        measured_s = result["last"] - result["first"]
        assert measured_s > 0, f"receiver measured no UDP interval: {result}"
        return result["bytes"] * 8 / measured_s
    finally:
        for proc in (locals().get("sender"), receiver):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)


def test_small_topology_rtt_access_cap_and_shared_bottleneck():
    net = build(n=3, scenario="L2")
    try:
        ping = net.get("c0").cmd("ping -n -c 1 -W 3 10.0.0.254")
        match = re.search(r"time=([0-9.]+)\s*ms", ping)
        assert match, ping
        assert abs(float(match.group(1)) - 30.0) <= 3.0

        access_bps = _udp_goodput(net.get("gw"), net.get("c0"), offered_mbps=20)
        expected_access_payload_bps = 2_000_000 * (1432 / 1474) * 0.95
        assert abs(access_bps - expected_access_payload_bps) / expected_access_payload_bps <= 0.08

        bottleneck_bps = _udp_goodput(net.get("gw"), net.get("xt"), offered_mbps=20, port=39174)
        assert 9_600_000 <= bottleneck_bps <= 9_800_000
    finally:
        net.stop()
