"""FedGrant Mininet topology with explicit interface names and tc verification.

Mininet is imported only by :func:`build`, so trace and configuration utilities
remain importable on machines without Mininet or root access.
"""

from __future__ import annotations

import re
import shlex
import math
from dataclasses import dataclass
from typing import Any

from fedgrant.mininet.ge import ge_params
from fedgrant.mininet.tcsetup import bottleneck, client_egress, gateway_egress, reverse_access
from fedgrant.policy.constants import C, Q, S
from fedgrant.sim.links import access_delay_ns, mean_access_rate_bps

SCENARIO_LOSS = {
    "L1": (0.05, 1 / 0.95),
    "L2": (0.05, 8.0),
    "L3": (0.05, 32.0),
    "L4": (0.05, 8.0),
    "L5": (0.05, 8.0),
    "X2": (0.05, 8.0),
}


@dataclass(frozen=True)
class ClientLink:
    client_id: int
    delay_ms: int
    rate_bps: int
    rtt_ms: int


def client_parameters(client_id: int) -> ClientLink:
    if isinstance(client_id, bool) or not isinstance(client_id, int) or client_id < 0:
        raise ValueError("client_id must be a non-negative integer")
    delay_ms = access_delay_ns(client_id) // 1_000_000
    return ClientLink(client_id, delay_ms, int(mean_access_rate_bps(client_id)), 2 * (delay_ms + 5))


def _parse_rate(text: str) -> float | None:
    match = re.search(r"\brate\s+([0-9.]+)\s*([kmg]?bit)", text, re.IGNORECASE)
    if not match:
        return None
    multiplier = {"bit": 1, "kbit": 1e3, "mbit": 1e6, "gbit": 1e9}[match.group(2).lower()]
    return float(match.group(1)) * multiplier


def _verify_iface(host: Any, iface: str, commands: list[str]) -> None:
    output = host.cmd(f"tc -d qdisc show dev {shlex.quote(iface)}")
    if not output.strip():
        raise RuntimeError(f"tc returned no qdiscs for {iface}")
    lines = output.lower().splitlines()
    expected = "\n".join(commands).lower()
    if "netem" in expected:
        netem_lines = [line for line in lines if "netem" in line]
        if not netem_lines:
            raise RuntimeError(f"{iface}: expected netem qdisc; got {output.strip()!r}")
        for delay in re.findall(r"delay\s+([0-9.]+)ms", expected):
            if not any(re.search(rf"\bdelay\s+{re.escape(delay)}(?:\.0+)?ms\b", line) for line in netem_lines):
                raise RuntimeError(f"{iface}: expected netem delay {delay}ms; got {output.strip()!r}")
        gemodel = re.search(r"loss gemodel\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", expected)
        if gemodel and not any("gemodel" in line and all(value in line for value in gemodel.groups()) for line in lines):
            raise RuntimeError(f"{iface}: configured Gilbert–Elliott loss does not match tc readback: {output.strip()!r}")
    if "tbf" in expected:
        tbf_lines = [line for line in lines if "tbf" in line]
        if not tbf_lines:
            raise RuntimeError(f"{iface}: expected tbf qdisc; got {output.strip()!r}")
        rate_match = re.search(r"rate\s+(\d+)bit", expected)
        if rate_match:
            actual = next((_parse_rate(line) for line in tbf_lines if _parse_rate(line) is not None), None)
            if actual is None or abs(actual - int(rate_match.group(1))) > 1:
                raise RuntimeError(f"{iface}: expected rate {rate_match.group(1)}bit; got {output.strip()!r}")
        limit_match = re.search(r"limit\s+(\d+)", expected)
        if limit_match:
            expected_limit = int(limit_match.group(1))
            # iproute2 commonly prints TBF's byte limit as derived latency,
            # not as `limit N`; kernels differ in which form they expose.
            direct_limit = any(
                re.search(rf"\blimit\s+{expected_limit}(?:\s|b|bytes|$)", line)
                for line in tbf_lines
            )
            rate_bps = int(rate_match.group(1)) if rate_match else None
            burst_match = re.search(r"\bburst\s+(\d+)", expected)
            burst_bytes = int(burst_match.group(1)) if burst_match else 0
            if not direct_limit and rate_bps:
                expected_latency_ms = math.ceil(max(0, expected_limit - burst_bytes) * 8_000 / rate_bps)
                observed_latency = any(
                    (match := re.search(r"\blat\s+([0-9.]+)\s*ms\b", line))
                    and abs(float(match.group(1)) - expected_latency_ms) <= 1
                    for line in tbf_lines
                )
                if not observed_latency:
                    raise RuntimeError(
                        f"{iface}: expected TBF queue limit {expected_limit} bytes "
                        f"(about {expected_latency_ms}ms at {rate_bps}bit/s); got {output.strip()!r}"
                    )


def verify_tc(net: Any, expected: dict[tuple[str, str], list[str]]) -> None:
    """Read every expected qdisc back and fail closed on missing/mismatched state."""
    for (host_name, iface), commands in expected.items():
        host = net.get(host_name)
        _verify_iface(host, iface, commands)


def build(n: int = 20, scenario: str = "L2", *, seed: int = 0, controller: Any = None, verify: bool = True):
    """Build and start the network; caller owns ``net.stop()`` in a finally block."""
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 50:
        raise ValueError("n must be an integer in [1, 50]")
    scenario = scenario.upper()
    if scenario not in SCENARIO_LOSS:
        raise ValueError(f"unknown scenario {scenario!r}")
    try:
        from mininet.link import Link
        from mininet.net import Mininet
        from mininet.node import OVSBridge
        from mininet.topo import Topo
    except ImportError as exc:
        raise RuntimeError("Mininet is required to build the testbed; install the Fedora Mininet package first") from exc

    class FedGrantTopo(Topo):
        def build(self):
            switch = self.addSwitch("s1")
            for i in range(int(n)):
                client = self.addHost(f"c{i}")
                self.addLink(client, switch, intfName1=f"c{i}-eth0", intfName2=f"s1-c{i}")
            gateway = self.addHost("gw")
            cross = self.addHost("xt")
            server = self.addHost("srv")
            self.addLink(gateway, switch, intfName1="gw-eth0", intfName2="s1-gw")
            self.addLink(cross, switch, intfName1="xt-eth0", intfName2="s1-xt")
            self.addLink(gateway, server, intfName1="gw-eth1", intfName2="srv-eth0")

    net = Mininet(topo=FedGrantTopo(), switch=OVSBridge, controller=controller, link=Link,
                  autoSetMacs=True, autoStaticArp=True, build=True)
    expected: dict[tuple[str, str], list[str]] = {}
    try:
        net.start()
        loss, burst_len = SCENARIO_LOSS[scenario]
        p, r = ge_params(loss, burst_len)
        for i in range(int(n)):
            params = client_parameters(i)
            client_name, iface = f"c{i}", f"c{i}-eth0"
            client = net.get(client_name)
            client.setIP(f"10.0.0.{i + 1}/24", intf=iface)
            commands = client_egress(iface, params.delay_ms, p, r, params.rate_bps)
            for command in commands:
                result = client.cmd(command)
                if "Error" in result or "RTNETLINK" in result:
                    raise RuntimeError(f"failed tc setup on {client_name}: {command}: {result}")
            expected[(client_name, iface)] = commands

            switch_commands = reverse_access(f"s1-c{i}", params.delay_ms)
            for command in switch_commands:
                result = net.get("s1").cmd(command)
                if "Error" in result or "RTNETLINK" in result:
                    raise RuntimeError(f"failed tc setup on switch: {command}: {result}")
            expected[("s1", f"s1-c{i}")] = switch_commands

        gw = net.get("gw")
        gw.setIP("10.0.0.254/24", intf="gw-eth0")
        gw.setIP("10.0.1.1/24", intf="gw-eth1")
        net.get("srv").setIP("10.0.1.2/24", intf="srv-eth0")
        net.get("xt").setIP("10.0.0.201/24", intf="xt-eth0")
        # Mininet's autoStaticArp runs during startup, before these explicit
        # addresses are assigned. Refresh it now so first-packet ARP discovery
        # does not contaminate the RTT acceptance measurement.
        net.staticArp()
        bn_commands = bottleneck("s1-gw", rate_bps=C)
        for command in bn_commands:
            result = net.get("s1").cmd(command)
            if "Error" in result or "RTNETLINK" in result:
                raise RuntimeError(f"failed bottleneck setup: {command}: {result}")
        expected[("s1", "s1-gw")] = bn_commands
        gw_commands = gateway_egress("gw-eth0", delay_ms=5)
        for command in gw_commands:
            result = gw.cmd(command)
            if "Error" in result or "RTNETLINK" in result:
                raise RuntimeError(f"failed gateway setup: {command}: {result}")
        expected[("gw", "gw-eth0")] = gw_commands

        # Disable segmentation/coalescing on every veth exposed by the topology.
        for host in [net.get(f"c{i}") for i in range(int(n))] + [gw, net.get("xt"), net.get("srv")]:
            for intf in host.intfList():
                if intf.name != "lo":
                    host.cmd(f"ethtool -K {shlex.quote(intf.name)} tso off gso off gro off lro off 2>/dev/null || true")
        switch = net.get("s1")
        for intf in switch.intfList():
            if intf.name != "lo":
                switch.cmd(f"ethtool -K {shlex.quote(intf.name)} tso off gso off gro off lro off 2>/dev/null || true")
        if verify:
            verify_tc(net, expected)
        net._fedgrant_expected_tc = expected
        net._fedgrant_scenario = scenario
        net._fedgrant_seed = int(seed)
        return net
    except BaseException:
        net.stop()
        raise


__all__ = ["ClientLink", "client_parameters", "SCENARIO_LOSS", "build", "verify_tc"]
