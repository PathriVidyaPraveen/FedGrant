"""Unit tests for cellular trace manager (Section 5.3)."""

import numpy as np
import pytest

from fedgrant.sim.links import mean_access_rate_bps
from fedgrant.sim.traces import ClientTrace, TraceManager, TraceSegment


def test_trace_segment() -> None:
    t_ms = np.array([0, 100, 200, 300], dtype=np.int64)
    rates = np.array([1_000_000.0, 2_000_000.0, 3_000_000.0, 4_000_000.0])
    seg = TraceSegment(t_ms, rates)

    assert seg.bin_ms == 100
    assert seg.duration_ms == 400
    assert seg.mean_rate_bps == 2_500_000.0

    # Query offset within segment
    assert seg.rate_at(50) == 1_000_000.0   # bin 0 covers [0, 100)
    assert seg.rate_at(150) == 2_000_000.0  # bin 1 covers [100, 200)
    # Wrapping at 400 ms — offset 400 wraps to 0 -> bin 0
    assert seg.rate_at(400) == 1_000_000.0


def test_client_trace_scaling_and_floor() -> None:
    t_ms = np.array([0, 100], dtype=np.int64)
    rates = np.array([1_000_000.0, 2_000_000.0])
    seg = TraceSegment(t_ms, rates) # mean 1.5 Mbps

    # Client 0: mean rate is 2 Mbps
    ct = ClientTrace(client_id=0, segment=seg, offset_ms=0, floor_bps=100_000.0)
    assert ct.target_mean_bps == 2_000_000.0
    scale = 2_000_000.0 / 1_500_000.0
    assert pytest.approx(ct.scale) == scale

    rate0 = ct.get_rate_bps(0)
    assert pytest.approx(rate0) == 1_000_000.0 * scale

    # Floor: scale drives rate to something; floor only kicks in if scaled < floor
    # With scale = 4/3 and rate=10.0, scaled = 13.3 bps which is < floor=100_000
    low_seg = TraceSegment(np.array([0]), np.array([10.0]))
    # client_id=0, target 2 Mbps; scale = 2e6 / 10 = 200000; scaled = 10 * 200000 = 2e6 > floor
    # Use floor_bps large enough to trigger
    ct_floor = ClientTrace(client_id=0, segment=low_seg, floor_bps=3_000_000.0)
    assert ct_floor.get_rate_bps(0) == 3_000_000.0


def test_trace_manager_synthetic_fallback() -> None:
    tm = TraceManager(num_clients=20, scenario="L2", split="test", trace_dir="/nonexistent")
    # For L2 (constant rates), each client rate is constant mean access rate
    for cid in range(20):
        expected_rate = mean_access_rate_bps(cid)
        assert tm.get_client_rate(cid, 0) == expected_rate
        assert tm.get_client_rate(cid, 100_000_000) == expected_rate

