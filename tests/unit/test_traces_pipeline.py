from __future__ import annotations

import numpy as np
import pytest
import json
from pathlib import Path

from fedgrant.traces.binning import bin_opportunities, split_rates, process_directory, TRACE_FILES
from fedgrant.traces.client_trace import client_trace, offset_fraction
from fedgrant.sim.traces import TraceManager


def test_opportunities_bin_to_exact_100ms_rates_and_drop_partial_tail():
    timestamps = np.array([0, 10, 99.9, 100, 150, 199.9, 200.1])
    t_ms, rates = bin_opportunities(timestamps)
    assert t_ms.tolist() == [0, 100]
    assert rates.tolist() == [360_000.0, 360_000.0]


def test_nonmonotonic_timestamps_rejected():
    with pytest.raises(ValueError, match="monotonically"):
        bin_opportunities([0, 120, 80, 250])


def test_split_is_contiguous_and_floors_train_and_validation():
    t = np.arange(100, dtype=np.int64) * 100
    r = np.arange(100, dtype=np.float64)
    parts = split_rates(t, r)
    assert [len(parts[k][1]) for k in ("train", "val", "test")] == [70, 15, 15]
    assert [parts[k][0][0] for k in ("train", "val", "test")] == [0, 0, 0]
    np.testing.assert_array_equal(np.concatenate([parts[k][1] for k in ("train", "val", "test")]), r)


def test_process_directory_outputs_id_splits_and_whole_ood(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    content = "\n".join(str(i) for i in range(1001))
    for name in TRACE_FILES:
        (src / f"{name}.up").write_text(content)
    summary = process_directory(src, out)
    assert len(summary) == 11
    assert (out / "Verizon-LTE-driving_train.csv").exists()
    assert (out / "TMobile-UMTS-driving_ood.csv").exists()
    assert not (out / "TMobile-UMTS-driving_test.csv").exists()


def test_client_trace_determinism_scaling_and_floor(tmp_path):
    (tmp_path / "Verizon-LTE-driving_test.csv").write_text(
        "t_ms,rate_bps\n0,1000000\n100,2000000\n200,3000000\n300,4000000\n"
    )
    a = client_trace(0, "test", 7, data_dir=tmp_path, floor_bps=0)
    b = client_trace(0, "test", 7, data_dir=tmp_path, floor_bps=0)
    np.testing.assert_array_equal(a, b)
    assert np.isclose(a.mean(), 2_000_000)
    assert offset_fraction(7, 0) != offset_fraction(8, 0)
    floor = client_trace(0, "test", 7, data_dir=tmp_path, floor_bps=100_000)
    assert np.all(floor >= 100_000)


@pytest.mark.parametrize("seed", [0, 1])
def test_checked_in_golden_offsets_and_rates(tmp_path, seed):
    golden_path = Path(__file__).parents[1] / "data" / "golden_trace.json"
    golden = json.loads(golden_path.read_text())
    rates = np.asarray(golden["input_rates_bps"], dtype=np.float64)
    for name in ("Verizon-LTE-driving", "Verizon-LTE-short", "ATT-LTE-driving"):
        (tmp_path / f"{name}_test.csv").write_text(
            "t_ms,rate_bps\n" + "".join(f"{i * 100},{rate:g}\n" for i, rate in enumerate(rates))
        )
    for client_id in range(5):
        fraction = golden["offset_fraction"][str(seed)][client_id]
        assert offset_fraction(seed, client_id) == pytest.approx(fraction, abs=1e-15)
        expected = np.roll(rates, -int(fraction * len(rates)))
        expected *= golden["target_mean_bps"][client_id] / rates.mean()
        actual = client_trace(client_id, "test", seed, data_dir=tmp_path)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)


def test_simulator_and_replay_library_use_identical_first_client_trace(tmp_path):
    rates = np.array([200_000.0, 500_000.0, 900_000.0, 1_400_000.0, 2_000_000.0])
    names = ("Verizon-LTE-driving", "Verizon-LTE-short", "ATT-LTE-driving")
    for name in names:
        (tmp_path / f"{name}_test.csv").write_text(
            "t_ms,rate_bps\n" + "".join(f"{i * 100},{rate:g}\n" for i, rate in enumerate(rates))
        )
    seed = 19
    library_rates = client_trace(0, "test", seed, data_dir=tmp_path)
    trace_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(6)[0])
    sim = TraceManager(num_clients=1, scenario="L4", split="test", trace_dir=tmp_path, rng=trace_rng)
    simulator_rates = np.array([sim.get_client_rate(0, i * 100_000_000) for i in range(len(rates))])
    np.testing.assert_allclose(simulator_rates, library_rates, rtol=0, atol=1e-8)


def test_invalid_trace_header_rejected(tmp_path):
    (tmp_path / "Verizon-LTE-driving_test.csv").write_text("x,y\n0,1\n")
    with pytest.raises(ValueError, match="t_ms,rate_bps"):
        client_trace(0, "test", 0, data_dir=tmp_path)
