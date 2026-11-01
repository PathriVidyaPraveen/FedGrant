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
