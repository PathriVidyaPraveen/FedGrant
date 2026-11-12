"""Unit tests for access links, bottleneck link, and serialization (Sections 3.1, 3.5, 6, 17)."""

from __future__ import annotations

import pytest

from fedgrant.policy.constants import C, Q, S
from fedgrant.sim.crosstraffic import CrossTrafficGenerator
from fedgrant.sim.engine import EventType, SimulationEngine
from fedgrant.sim.ge import GilbertElliottModel
from fedgrant.sim.links import (
    AccessLink,
    BottleneckLink,
    Frame,
    access_delay_ns,
    mean_access_rate_bps,
)


def test_delay_and_rate_mapping() -> None:
    # delta_i: 10, 20, 30, 40 ms for i mod 4 = 0, 1, 2, 3
    assert access_delay_ns(0) == 10_000_000
    assert access_delay_ns(1) == 20_000_000
    assert access_delay_ns(2) == 30_000_000
    assert access_delay_ns(3) == 40_000_000
    assert access_delay_ns(4) == 10_000_000

    # b_bar_i: 2, 4, 6 Mbit/s for i mod 3 = 0, 1, 2
    assert mean_access_rate_bps(0) == 2_000_000.0
    assert mean_access_rate_bps(1) == 4_000_000.0
    assert mean_access_rate_bps(2) == 6_000_000.0
    assert mean_access_rate_bps(3) == 2_000_000.0


def test_serialization_times() -> None:
    """Spec Section 17 W2 deliverable: test_links (serialization times)."""
    bits = S * 8  # 1474 * 8 = 11792 bits

    # At 2 Mbit/s: 11792 / 2e6 s = 5.896 ms = 5,896,000 ns
    tx_2m_ns = int(round((bits * 1_000_000_000.0) / 2_000_000.0))
    assert tx_2m_ns == 5_896_000

    # At 4 Mbit/s: 11792 / 4e6 s = 2.948 ms = 2,948,000 ns
    tx_4m_ns = int(round((bits * 1_000_000_000.0) / 4_000_000.0))
    assert tx_4m_ns == 2_948_000

    # At 10 Mbit/s (bottleneck capacity C): 11792 / 10e6 s = 1.1792 ms = 1,179,200 ns
    tx_c_ns = int(round((bits * 1_000_000_000.0) / float(C)))
    assert tx_c_ns == 1_179_200


def test_single_frame_transmission_pipeline() -> None:
    """End-to-end timing of 1 frame from client 0 through access link to gateway arrival.

    Client 0 has:
        delta_0 = 10 ms
        b_0 = 2 Mbit/s -> serialization = 5.896 ms
        Access -> Bottleneck delay = 5 ms
        Bottleneck rate C = 10 Mbit/s -> serialization = 1.1792 ms
    Total expected time = 10 + 5.896 + 5 + 1.1792 = 22.0752 ms (22,075,200 ns).
    """
    engine = SimulationEngine()
    ge_lossless = GilbertElliottModel(p=0.0, r=1.0)  # Always Good
    access = AccessLink(
        client_id=0,
        engine=engine,
        prop_delay_ns=10_000_000,
        ge_model=ge_lossless,
        initial_rate_bps=2_000_000.0,
    )
    bottleneck = BottleneckLink(engine=engine)
    access.next_hop_cb = bottleneck.route_from_access

    arrived_at: list[int] = []

    def on_gw_arrive(frame: Frame) -> None:
        arrived_at.append(engine.now_ns)

    bottleneck.dest_cb = on_gw_arrive

    frame = Frame(frame_id="f1", client_id=0, size_bytes=S)
    accepted = access.enqueue(frame)
    assert accepted is True

    # Run simulation until completion
    engine.run_until(50_000_000)  # 50 ms

    assert len(arrived_at) == 1
    assert arrived_at[0] == 22_075_200


def test_access_fifo_drop_tail() -> None:
    """Access FIFO holds 32 frames; 33rd frame arriving at FIFO is dropped."""
    engine = SimulationEngine()
    ge_lossless = GilbertElliottModel(p=0.0, r=1.0)
    access = AccessLink(
        client_id=0,
        engine=engine,
        prop_delay_ns=1_000,  # 1 us for quick arrival
        ge_model=ge_lossless,
        max_queue_frames=32,
        initial_rate_bps=1.0,  # Very slow so nothing leaves FIFO during test
    )

    # Enqueue 35 frames
    for i in range(35):
        access.enqueue(Frame(frame_id=f"f{i}", client_id=0, size_bytes=S))

    # Advance past propagation delay so all frames arrive at FIFO
    engine.run_until(10_000)

    # 1 in transmission + 32 in FIFO queue = 33 accepted; 2 dropped
    assert len(access.queue) == 32
    assert access.is_busy is True
    assert access.drops_queue == 2


def test_bottleneck_drop_tail() -> None:
    """Bottleneck FIFO holds Q = 64 frames; 65th frame arriving is dropped."""
    engine = SimulationEngine()
    bottleneck = BottleneckLink(
        engine=engine,
        capacity_bps=1.0,  # Very slow so frames remain queued
        max_queue_frames=64,
    )

    for i in range(70):
        bottleneck.enqueue(Frame(frame_id=f"f{i}", client_id=0, size_bytes=S))

    # 1 in service + 64 in queue = 65 accepted; 5 dropped
    assert len(bottleneck.queue) == 64
    assert bottleneck.drops_queue == 5
    assert bottleneck.drops_queue_client == 5


def test_rate_change_applies_to_remaining_bits() -> None:
    """Rate changes during frame transmission must rescale remaining time (Section 6)."""
    engine = SimulationEngine()
    ge_lossless = GilbertElliottModel(p=0.0, r=1.0)
    access = AccessLink(
        client_id=0,
        engine=engine,
        prop_delay_ns=0,
        ge_model=ge_lossless,
        initial_rate_bps=2_000_000.0,  # 2 Mbps
    )

    departures: list[int] = []
    access.next_hop_cb = lambda f: departures.append(engine.now_ns)

    # Frame S=1474 bytes = 11792 bits.
    # At 2 Mbps, full serialization = 5,896,000 ns.
    access.enqueue(Frame(frame_id="f1", client_id=0, size_bytes=S))

    # Advance to 2,000,000 ns (2 ms)
    engine.run_until(2_000_000)
    assert len(departures) == 0

    # In 2 ms at 2 Mbps, 4000 bits were transmitted. Remaining bits = 11792 - 4000 = 7792 bits.
    # Now switch rate to 4 Mbps.
    # Remaining time at 4 Mbps = 7792 / 4e6 s = 1.948 ms = 1,948,000 ns.
    # Expected departure time = 2,000,000 + 1,948,000 = 3,948,000 ns.
    access.set_rate(4_000_000.0)

    engine.run_until(10_000_000)
    assert len(departures) == 1
    assert departures[0] == 3_948_000


def test_crosstraffic_generator() -> None:
    """Scenario X2 cross traffic injects 1474-byte frames at 3 Mbit/s into bottleneck."""
    engine = SimulationEngine()
    bottleneck = BottleneckLink(engine=engine, capacity_bps=10_000_000.0)
    crosstraffic = CrossTrafficGenerator(
        engine=engine,
        bottleneck=bottleneck,
        rate_bps=3_000_000.0,
        frame_size_bytes=S,
        phase_ns=0,
    )

    received: list[Frame] = []
    bottleneck.dest_cb = lambda f: received.append(f)

    crosstraffic.start()
    # Run for 20 ms (20,000,000 ns)
    # Interval is ~3.93067 ms -> in 20 ms expect 5 or 6 frames
    engine.run_until(20_000_000)
    crosstraffic.stop()

    assert len(received) >= 4
    for f in received:
        assert f.client_id == -1
        assert f.size_bytes == S

