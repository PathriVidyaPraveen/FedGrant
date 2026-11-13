from __future__ import annotations

import pytest

from fedgrant.mininet.topo import _verify_iface, client_parameters


@pytest.mark.parametrize("client_id,delay,rate,rtt", [
    (0, 10, 2_000_000, 30), (1, 20, 4_000_000, 50), (2, 30, 6_000_000, 70),
    (3, 40, 2_000_000, 90), (11, 40, 6_000_000, 90),
])
def test_client_link_parameters(client_id, delay, rate, rtt):
    got = client_parameters(client_id)
    assert (got.delay_ms, got.rate_bps, got.rtt_ms) == (delay, rate, rtt)


def test_client_parameters_reject_negative_id():
    with pytest.raises(ValueError):
        client_parameters(-1)


class _TcHost:
    def __init__(self, output):
        self.output = output

    def cmd(self, _command):
        return self.output


def test_tc_verifier_accepts_kernel_tbf_latency_readback():
    # On this iproute2/kernel combination TBF's byte limit is exposed as
    # derived latency; the root netem's `limit 1000` is a different setting.
    readback = (
        "qdisc netem 1: root refcnt 13 limit 1000 delay 10ms "
        "loss gemodel p 0.6579% r 12.5% 1-h 100% 1-k 0%\n"
        "qdisc tbf 10: parent 1:1 rate 2Mbit burst 3000b/1 mpu 0b "
        "lat 177ms linklayer ethernet\n"
    )
    commands = [
        "tc qdisc add dev c0-eth0 root handle 1: netem delay 10ms "
        "loss gemodel 0.6579% 12.5% 100% 0%",
        "tc qdisc add dev c0-eth0 parent 1:1 handle 10: tbf rate 2000000bit "
        "burst 3000 limit 47168",
    ]
    _verify_iface(_TcHost(readback), "c0-eth0", commands)


def test_tc_verifier_rejects_mismatched_tbf_latency():
    readback = "qdisc tbf 10: parent 1:1 rate 2Mbit burst 3000b lat 120ms\n"
    commands = ["tc qdisc add dev c0-eth0 root tbf rate 2000000bit burst 3000 limit 47168"]
    with pytest.raises(RuntimeError, match="queue limit 47168"):
        _verify_iface(_TcHost(readback), "c0-eth0", commands)


class _TcHostWithFilters:
    def __init__(self, qdiscs, filters):
        self.qdiscs, self.filters = qdiscs, filters

    def cmd(self, command):
        return self.filters if " filter " in command else self.qdiscs


_LOSSLESS_QDISCS = (
    "qdisc prio 1: root refcnt 2 bands 2 priomap 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n"
    "qdisc netem 2: parent 1:2 limit 1000 delay 10ms loss gemodel p 0.6579% r 12.5% 1-h 100% 1-k 0%\n"
    "qdisc netem 30: parent 1:1 limit 1000 delay 10ms\n"
    "qdisc tbf 10: parent 2:1 rate 2Mbit burst 3000b/1 mpu 0b lat 177ms linklayer ethernet\n"
)
_READY_FILTER = (
    "filter parent 1: protocol ip pref 1 u32 chain 0 fh 800::800 order 2048 key ht 800 bkt 0 flowid 1:1\n"
    "  match 00110000/00ff0000 at 8\n  match 00002328/0000ffff at 20\n  match 46470002/ffff00ff at 28\n"
)


def test_tc_verifier_accepts_ready_lossless_layout_and_rejects_bad_filter():
    from fedgrant.mininet.tcsetup import client_egress
    commands = client_egress("c0-eth0", 10, 0.006579, 0.125, 2_000_000)
    _verify_iface(_TcHostWithFilters(_LOSSLESS_QDISCS, _READY_FILTER), "c0-eth0", commands)
    wrong_port = _READY_FILTER.replace("00002328/0000ffff at 20", "0000238c/0000ffff at 20")
    with pytest.raises(RuntimeError, match="READY filter"):
        _verify_iface(_TcHostWithFilters(_LOSSLESS_QDISCS, wrong_port), "c0-eth0", commands)
    lossy_ready = _LOSSLESS_QDISCS.replace("netem 30: parent 1:1 limit 1000 delay 10ms",
                                           "netem 30: parent 1:1 limit 1000 delay 10ms loss 5%")
    with pytest.raises(RuntimeError, match="loss-free netem"):
        _verify_iface(_TcHostWithFilters(lossy_ready, _READY_FILTER), "c0-eth0", commands)
    legacy = client_egress("c0-eth0", 10, 0.006579, 0.125, 2_000_000, ready_lossless=False)
    with pytest.raises(RuntimeError, match="unexpected prio root"):
        _verify_iface(_TcHostWithFilters(_LOSSLESS_QDISCS, _READY_FILTER), "c0-eth0", legacy)
