"""Unit tests for fedgrant.analysis.metrics (Section 12)."""

import numpy as np
import pytest

from fedgrant.analysis.metrics import (
    compute_accuracy_vs_wallclock,
    compute_bce,
    compute_brier,
    compute_ece,
    compute_eta,
    compute_final_accuracy,
    compute_goodput_bps,
    compute_queueing_delay_percentiles,
    compute_roc_auc,
    compute_rtt_percentiles,
    compute_target_accuracy,
    compute_time_to_target,
    compute_utilization,
    compute_wasted_grants_ratio,
    summarize_run_history,
)


def test_compute_eta() -> None:
    N, K = 3, 4
    weights = [0.2, 0.3, 0.5]
    energies = np.ones((N, K), dtype=np.float32)
    # Total energy: sum_i w_i^2 * 4
    # Client 0: w_0^2 * 4 = 0.04 * 4 = 0.16
    # Client 1: w_1^2 * 4 = 0.09 * 4 = 0.36
    # Client 2: w_2^2 * 4 = 0.25 * 4 = 1.00
    # Total = 1.52

    # Delivered: client 0 got blocks 0, 1 (2 blocks = 0.08)
    # client 1 got block 0 (1 block = 0.09)
    # client 2 got blocks 0, 1, 2, 3 (4 blocks = 1.00)
    # Delivered = 1.17
    deliv = [[0, 1], [0], [0, 1, 2, 3]]
    eta = compute_eta(weights, energies, deliv, K=K)
    assert pytest.approx(eta, 1e-4) == 1.17 / 1.52

    # Full delivery gives 1.0
    full_deliv = [list(range(K)) for _ in range(N)]
    assert pytest.approx(compute_eta(weights, energies, full_deliv, K=K), 1e-5) == 1.0

    # Empty delivery gives 0.0
    empty_deliv = [[] for _ in range(N)]
    assert compute_eta(weights, energies, empty_deliv, K=K) == 0.0


def test_networking_metrics() -> None:
    # 100 blocks delivered in D = 0.398 s
    gp = compute_goodput_bps(100, 0.398, payload_bytes_per_block=1400)
    expected_gp = (100 * 1400 * 8) / 0.398
    assert pytest.approx(gp) == expected_gp

    # Utilization
    util = compute_utilization(bottleneck_frames_sent=150, D_s=0.398, capacity_bps=10_000_000, frame_size_bytes=1474)
    expected_util = (150 * 1474 * 8) / (10_000_000 * 0.398)
    assert pytest.approx(util) == expected_util

    # Wasted grants
    assert compute_wasted_grants_ratio(100, 15) == 0.15
    assert compute_wasted_grants_ratio(0, 0) == 0.0


def test_percentiles() -> None:
    rtts = [10_000_000, 20_000_000, 30_000_000] # in ns: 10ms, 20ms, 30ms
    med, p99 = compute_rtt_percentiles(rtts)
    assert pytest.approx(med) == 20.0
    assert p99 > 20.0

    backlogs = [0, 10_000_000] # bits, capacity 10 Mbps -> 0s, 1s (0ms, 1000ms)
    mean_d, p99_d = compute_queueing_delay_percentiles(backlogs, capacity_bps=10_000_000)
    assert pytest.approx(mean_d) == 500.0


def test_fl_metrics() -> None:
    accs = [0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.88, 0.89, 0.90]
    final_acc = compute_final_accuracy(accs, last_n_evals=5)
    assert pytest.approx(final_acc) == np.mean(accs[-5:])

    # Target accuracy
    full_seeds = [0.865, 0.866, 0.864]
    a_star = compute_target_accuracy(full_seeds)
    assert pytest.approx(a_star) == 0.9 * 0.865

    # TTA with moving average of window 3
    records = [
        {"round": 5, "test_accuracy": 0.50},
        {"round": 10, "test_accuracy": 0.60},
        {"round": 15, "test_accuracy": 0.70},
        {"round": 20, "test_accuracy": 0.80},
    ]
    # Window 3 at round 15: mean(0.5, 0.6, 0.7) = 0.60
    # Window 3 at round 20: mean(0.6, 0.7, 0.8) = 0.70
    r_target, wall_clock = compute_time_to_target(records, target_acc=0.65, window=3, t_c_s=1.8, d_s=0.2)
    assert r_target == 20
    assert pytest.approx(wall_clock) == 20 * 2.0

    # Unreached target
    r_unreached, _ = compute_time_to_target(records, target_acc=0.99, window=3)
    assert r_unreached is None


def test_ml_metrics() -> None:
    p = np.array([0.9, 0.8, 0.2, 0.1])
    y = np.array([1, 1, 0, 0])

    brier = compute_brier(p, y)
    assert brier < 0.05

    bce = compute_bce(p, y)
    assert bce > 0.0

    ece = compute_ece(p, y, n_bins=5)
    assert 0.0 <= ece <= 1.0

    auc = compute_roc_auc(p, y)
    assert auc == 1.0


def test_roc_auc_gives_half_credit_for_tied_positive_negative_scores() -> None:
    assert compute_roc_auc(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5
    # One positive-negative tie among four cross-class pairs.
    assert compute_roc_auc(
        np.array([0.1, 0.5, 0.5, 0.9]),
        np.array([0, 1, 0, 1]),
    ) == pytest.approx(0.875)


def test_summarize_run_history() -> None:
    history = []
    for r in range(1, 31):
        history.append({
            "round": r,
            "test_accuracy": 0.5 + 0.01 * r,
            "eta": 0.5 + 0.01 * r,
            "metrics": {
                "goodput_bps": 1000.0 * r,
                "wasted_grants_ratio": 0.1,
            },
        })

    summary = summarize_run_history(history, warmup_rounds=20)
    assert "mean_eta" in summary
    assert "mean_goodput_bps" in summary
    assert "mean_wasted_grants_ratio" in summary
    assert "final_accuracy" in summary

    # Rounds 21 to 30: r in [21, 30], mean eta is 0.5 + 0.01 * 25.5 = 0.755
    assert pytest.approx(summary["mean_eta"]) == 0.755

