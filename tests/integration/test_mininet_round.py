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
    """Measure delivered payload for datagrams sent inside the offered interval."""
    receiver_code = """import json, socket, sys, time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((sys.argv[1], int(sys.argv[2])))
s.settimeout(.1)
duration = float(sys.argv[3])
deadline = None
events = []
while True:
    try:
        data, _ = s.recvfrom(65535)
        events.append((time.monotonic_ns(), len(data)))
        if deadline is None:
            deadline = time.monotonic() + duration + .5
    except socket.timeout:
        if deadline is not None and time.monotonic() >= deadline:
            break
print(json.dumps({'events': events}))
"""
    sender_code = """import json, socket, sys, time
dst = (sys.argv[1], int(sys.argv[2]))
rate = float(sys.argv[3]) * 1e6
duration = float(sys.argv[4])
payload_bytes = 1432
payload = b'x' * payload_bytes
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
start_ns = time.monotonic_ns()
deadline_ns = start_ns + int(duration * 1e9)
next_send = time.monotonic()
interval = payload_bytes * 8 / rate
while time.monotonic_ns() < deadline_ns:
    now = time.monotonic()
    if now < next_send:
        time.sleep(min(next_send - now, .001))
        continue
    s.sendto(payload, dst)
    next_send = max(next_send + interval, now)
end_ns = time.monotonic_ns()
print(json.dumps({'start_ns': start_ns, 'end_ns': end_ns}))
"""
    receiver = server.popen(
        [sys.executable, "-c", receiver_code, "10.0.0.254", str(port), str(duration_s)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        time.sleep(0.2)
        sender = client.popen(
            [sys.executable, "-c", sender_code, "10.0.0.254", str(port),
             str(offered_mbps), str(duration_s)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        sender_stdout, stderr = sender.communicate(timeout=duration_s + 5)
        assert sender.returncode == 0, stderr.decode(errors="replace")
        stdout, stderr = receiver.communicate(timeout=duration_s + 5)
        assert receiver.returncode == 0, stderr.decode(errors="replace")
        received = json.loads(stdout)
        sent = json.loads(sender_stdout)
        start_ns, end_ns = int(sent["start_ns"]), int(sent["end_ns"])
        # Use arrival time, so packets that drain after the offered flow stops
        # do not inflate the measured receiver goodput.
        delivered_bytes = sum(
            int(size) for arrived_ns, size in received["events"]
            if start_ns <= int(arrived_ns) <= end_ns
        )
        measured_s = (end_ns - start_ns) / 1e9
        assert delivered_bytes > 0, f"receiver measured no UDP packets: {received}"
        return delivered_bytes * 8 / measured_s
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
        # At 20 Mbit/s offered load the surviving packets still saturate the
        # 2 Mbit/s TBF, so GE loss is not an extra multiplier on the shaped
        # egress goodput. Convert the shaped Ethernet frame rate to UDP payload.
        expected_access_payload_bps = 2_000_000 * (1432 / 1474)
        assert abs(access_bps - expected_access_payload_bps) / expected_access_payload_bps <= 0.03

        bottleneck_bps = _udp_goodput(net.get("gw"), net.get("xt"), offered_mbps=20, port=39174)
        assert 9_600_000 <= bottleneck_bps <= 9_800_000
    finally:
        net.stop()
