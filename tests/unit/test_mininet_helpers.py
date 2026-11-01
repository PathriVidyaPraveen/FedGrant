from __future__ import annotations

import pytest

from fedgrant.common.logschema import SCHEMAS
from fedgrant.mininet.ge import analyze_loss, bootstrap_loss_intervals, ge_params
from fedgrant.mininet.tcsetup import (
    ACCESS_QUEUE_BYTES, BOTTLENECK_QUEUE_BYTES, bottleneck, client_egress,
    gateway_egress, reverse_access, tbf_change,
)
from fedgrant.policy.constants import Q, S


@pytest.mark.parametrize("burst,expected_p,expected_r", [
    (1 / 0.95, 0.05, 0.95), (8, 0.006578947368421053, 0.125), (32, 0.0016447368421052632, 0.03125),
])
def test_ge_parameters(burst, expected_p, expected_r):
    p, r = ge_params(0.05, burst)
    assert p == pytest.approx(expected_p)
    assert r == pytest.approx(expected_r)
    assert p / (p + r) == pytest.approx(0.05)


def test_loss_stats_count_runs_and_received_true_means_success():
    stats = analyze_loss([True, False, False, True, False, True])
    assert (stats.n, stats.lost, stats.n_bursts) == (6, 3, 2)
    assert stats.loss_rate == 0.5
    assert stats.mean_burst == 1.5
    censored = analyze_loss([False, False, True, False, False, True])
    assert censored.n_bursts == 2
    assert censored.complete_bursts == 1
    assert censored.mean_burst == 2.0


def test_loss_block_bootstrap_is_deterministic_and_contains_estimate():
    received = [True] * 20 + [False, False, True] * 10 + [True] * 20
    first = bootstrap_loss_intervals(received, replicates=100, block_size=8, seed=17)
    second = bootstrap_loss_intervals(received, replicates=100, block_size=8, seed=17)
    assert first == second
    stats = analyze_loss(received)
    assert first["loss_rate_ci_low"] <= stats.loss_rate <= first["loss_rate_ci_high"]
    assert first["mean_burst_ci_low"] <= stats.mean_burst <= first["mean_burst_ci_high"]
    assert first["bootstrap_replicates"] == 100


@pytest.mark.parametrize("kwargs", [
    {"replicates": 99}, {"block_size": 0}, {"confidence": 1.0}, {"seed": 1.5},
])
def test_loss_block_bootstrap_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        bootstrap_loss_intervals([True, False], **kwargs)


def test_tc_builders_exact_strings_and_frame_queue_sizes():
    assert S == 1474 and ACCESS_QUEUE_BYTES == 32 * S == 47168
    assert BOTTLENECK_QUEUE_BYTES == Q * S == 94336
    assert client_egress("c3-eth0", 40, 0.006579, 0.125, 2_000_000) == [
        "tc qdisc add dev c3-eth0 root handle 1: netem delay 40ms loss gemodel 0.6579% 12.5% 100% 0%",
        "tc qdisc add dev c3-eth0 parent 1:1 handle 10: tbf rate 2000000bit burst 3000 limit 47168",
    ]
    assert reverse_access("s1-c3", 40) == ["tc qdisc add dev s1-c3 root netem delay 40ms"]
    assert bottleneck("s1-gw") == [
        "tc qdisc add dev s1-gw root handle 1: netem delay 5ms",
        "tc qdisc add dev s1-gw parent 1:1 handle 10: tbf rate 10000000bit burst 3000 limit 94336",
    ]
    assert gateway_egress("gw-eth0") == ["tc qdisc add dev gw-eth0 root netem delay 5ms"]
    assert tbf_change("c3-eth0", 3_456_000).endswith("rate 3456000bit burst 3000 limit 47168")


@pytest.mark.parametrize("args", [
    ("interface-name-too-long", 5, 0.1, 0.2, 1000),
    ("c0-eth0", -1, 0.1, 0.2, 1000),
    ("c0-eth0", 5, -0.1, 0.2, 1000),
    ("c0-eth0", 5, 0.1, 1.2, 1000),
    ("c0-eth0", 5, 0.1, 0.2, 0),
])
def test_tc_builders_reject_invalid_inputs(args):
    with pytest.raises(ValueError):
        client_egress(*args)


def test_logs_schema_names_are_unique_and_nonempty():
    for name, columns in SCHEMAS.items():
        assert columns and len(columns) == len(set(columns)), name
