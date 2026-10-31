"""Evaluation metrics for FedGrant experiments (Section 12).

Implements all metrics specified in Section 12:
1. FL: Final accuracy (rounds 255-300), target accuracy a*, time-to-target (TTA),
   and accuracy vs wall-clock.
2. Allocation: Delivered energy eta_r (Eq. 7), delivered blocks, clients represented,
   deadline completion.
3. Networking: Goodput, utilization, bottleneck queue drops, wasted grants,
   late packets, RTT percentiles, queueing delay percentiles.
4. ML: Brier, ECE, ROC-AUC, BCE.
5. Summary: Per-run scalars averaged over rounds 21..last (first 20 excluded as warm-up).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from fedgrant.analysis.runs_schema import (
    Run,
    RunsSchemaError,
    make_config_id,
    make_row,
    read_runs_csv,
    write_runs_csv,
)
from fedgrant.policy.constants import C, S


# ============================================================================
# Allocation Metrics
# ============================================================================


def compute_eta(
    client_weights: Sequence[float] | np.ndarray,
    client_energies: np.ndarray,
    delivered_indices_per_client: Sequence[Sequence[int]],
    K: int = 169,
) -> float:
    """Delivered energy ratio eta_r (Section 3.1, Eq. 7).

    eta_r = (sum_{i=1}^N w_i^2 * sum_{j in D_i} eps_{i,j}) /
            (sum_{i=1}^N w_i^2 * sum_{j=1}^K eps_{i,j})
    """
    weights = np.asarray(client_weights, dtype=np.float64)
    energies = np.asarray(client_energies, dtype=np.float64)
    N = len(weights)
    if energies.shape != (N, K):
        raise ValueError(f"client_energies shape {energies.shape} != ({N}, {K})")

    total_energy = float(np.sum((weights[:, None] ** 2) * energies))
    if total_energy <= 0.0:
        return 0.0

    delivered_energy = 0.0
    for cid in range(N):
        w_sq = weights[cid] ** 2
        for idx in delivered_indices_per_client[cid]:
            if 0 <= idx < K:
                delivered_energy += w_sq * energies[cid, idx]

    return float(np.clip(delivered_energy / total_energy, 0.0, 1.0))


# ============================================================================
# Networking Metrics
# ============================================================================


def compute_goodput_bps(
    delivered_blocks: int,
    D_s: float,
    payload_bytes_per_block: int = 1400,
) -> float:
    """Delivered payload bits in [0, D] divided by D (Section 12)."""
    if D_s <= 0.0:
        return 0.0
    return float(delivered_blocks * payload_bytes_per_block * 8) / float(D_s)


def compute_utilization(
    bottleneck_frames_sent: int,
    D_s: float,
    capacity_bps: float = float(C),
    frame_size_bytes: int = S,
) -> float:
    """Bits leaving bottleneck in [0, D] divided by C * D (Section 12)."""
    if D_s <= 0.0 or capacity_bps <= 0.0:
        return 0.0
    return float(bottleneck_frames_sent * frame_size_bytes * 8) / (capacity_bps * D_s)


def compute_wasted_grants_ratio(grants_issued: int, grants_timed_out: int) -> float:
    """Grants with y = 0 divided by grants issued (Section 12)."""
    if grants_issued <= 0:
        return 0.0
    return float(grants_timed_out) / float(grants_issued)


def compute_rtt_percentiles(rtt_samples_ns: Sequence[int]) -> tuple[float, float]:
    """Median and 99th percentile of grant RTT samples, returned in milliseconds."""
    if not rtt_samples_ns:
        return 0.0, 0.0
    arr = np.asarray(rtt_samples_ns, dtype=np.float64) / 1e6
    return float(np.median(arr)), float(np.percentile(arr, 99))


def compute_queueing_delay_percentiles(
    backlog_bits_samples: Sequence[int],
    capacity_bps: float = float(C),
) -> tuple[float, float]:
    """Mean and 99th percentile of bottleneck queueing delay, returned in milliseconds."""
    if not backlog_bits_samples or capacity_bps <= 0.0:
        return 0.0, 0.0
    delays_ms = (np.asarray(backlog_bits_samples, dtype=np.float64) / capacity_bps) * 1000.0
    return float(np.mean(delays_ms)), float(np.percentile(delays_ms, 99))


# ============================================================================
# FL Metrics
# ============================================================================


def compute_final_accuracy(accuracies: Sequence[float], last_n_evals: int = 10) -> float:
    """Mean test accuracy over the last 10 evaluations (rounds 255..300) (Section 12)."""
    if not accuracies:
        return 0.0
    recent = accuracies[-last_n_evals:]
    return float(np.mean(recent))


def compute_target_accuracy(full_final_accuracies_seeds_0_to_2: Sequence[float]) -> float:
    """Target accuracy a* = 0.9 * mean final accuracy of FULL over seeds 0-2 (Section 12)."""
    if not full_final_accuracies_seeds_0_to_2:
        raise ValueError("Need FULL final accuracies to compute target accuracy a*")
    return float(0.9 * np.mean(full_final_accuracies_seeds_0_to_2))


def compute_time_to_target(
    eval_records: Sequence[dict[str, Any]],
    target_acc: float,
    max_round: int = 300,
    window: int = 3,
    t_c_s: float = 1.0,
    d_s: float = 0.398,
) -> tuple[int | None, float | None]:
    """First round r at which 3-eval moving average of test accuracy >= a* (Section 12).

    Returns:
        (round_r, wall_clock_seconds) or (None, None) if censored at max_round.
    """
    if len(eval_records) < window:
        return None, None

    # Sort by round
    sorted_recs = sorted(eval_records, key=lambda r: int(r.get("round", 0)))
    accs = [float(r["test_accuracy"]) for r in sorted_recs]
    rounds = [int(r["round"]) for r in sorted_recs]

    for i in range(len(accs) - window + 1):
        ma3 = float(np.mean(accs[i : i + window]))
        if ma3 >= target_acc:
            r_target = rounds[i + window - 1]
            if r_target <= max_round:
                wall_clock_s = r_target * (t_c_s + d_s)
                return r_target, wall_clock_s

    return None, None


def compute_accuracy_vs_wallclock(
    eval_records: Sequence[dict[str, Any]],
    t_c_s: float = 1.0,
    d_s: float = 0.398,
) -> list[tuple[float, float]]:
    """Test accuracy against r * (Tc + D) in seconds (Section 12)."""
    res: list[tuple[float, float]] = []
    for r in sorted(eval_records, key=lambda x: int(x.get("round", 0))):
        rnd = int(r["round"])
        acc = float(r["test_accuracy"])
        wall_clock_s = rnd * (t_c_s + d_s)
        res.append((wall_clock_s, acc))
    return res


# ============================================================================
# ML Metrics (Section 4.6 & Section 12)
# ============================================================================


def compute_brier(p_hat: np.ndarray, y: np.ndarray) -> float:
    """Brier score = mean((p_hat - y)^2)."""
    p = np.asarray(p_hat, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    if p.shape != labels.shape:
        raise ValueError(f"Shape mismatch: {p.shape} vs {labels.shape}")
    return float(np.mean((p - labels) ** 2))


def compute_bce(p_hat: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    """Binary Cross-Entropy loss = -mean(y * log(p) + (1-y) * log(1-p))."""
    p = np.clip(np.asarray(p_hat, dtype=np.float64), eps, 1.0 - eps)
    labels = np.asarray(y, dtype=np.float64)
    loss = -(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p))
    return float(np.mean(loss))


def compute_ece(p_hat: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) with equal-width probability bins."""
    p = np.asarray(p_hat, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    n = len(p)
    if n == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (p >= low) & (p < high) if i < n_bins - 1 else (p >= low) & (p <= high)
        bin_count = np.count_nonzero(mask)
        if bin_count > 0:
            bin_conf = float(np.mean(p[mask]))
            bin_acc = float(np.mean(labels[mask]))
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_roc_auc(p_hat: np.ndarray, y: np.ndarray) -> float:
    """Receiver Operating Characteristic Area Under Curve (ROC-AUC)."""
    p = np.asarray(p_hat, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int32)
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)
    n_pos = int(np.count_nonzero(pos_mask))
    n_neg = int(np.count_nonzero(neg_mask))

    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined single-class edge case

    # Mann–Whitney rank-sum calculation. Average ranks give tied positive and
    # negative pairs half credit, as required by ROC-AUC.
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    sorted_ranks = np.arange(1, len(p) + 1, dtype=np.float64)
    start = 0
    while start < len(p):
        end = start + 1
        while end < len(p) and sorted_p[end] == sorted_p[start]:
            end += 1
        if end - start > 1:
            sorted_ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    rank = np.empty_like(sorted_ranks)
    rank[order] = sorted_ranks
    pos_ranks_sum = float(np.sum(rank[pos_mask]))
    u_pos = pos_ranks_sum - (n_pos * (n_pos + 1)) / 2.0
    return float(u_pos / (n_pos * n_neg))


# ============================================================================
# Summary Aggregation (Section 12)
# ============================================================================


def summarize_run_history(
    history: Sequence[dict[str, Any]],
    warmup_rounds: int = 20,
) -> dict[str, float]:
    """Average per-run scalars over rounds (warmup_rounds + 1)..last (Section 12).

    Excludes the first 20 rounds as estimator warm-up for all policies.
    """
    valid_rounds = [r for r in history if int(r.get("round", 0)) > warmup_rounds]
    if not valid_rounds:
        return {}

    keys_to_average = [
        "eta",
        "delivered_blocks",
        "goodput_bps",
        "utilization",
        "wasted_grants_ratio",
        "clients_represented",
        "deadline_completion",
        "rtt_median_ms",
        "rtt_p99_ms",
        "queueing_delay_mean_ms",
        "queueing_delay_p99_ms",
        "drops_ge",
        "drops_queue_access",
        "drops_queue_bn_client",
        "drops_queue_bn_cross",
        "late_after_timeout",
        "late_after_deadline",
        "duplicate_packets",
    ]

    summary: dict[str, float] = {}

    for k in keys_to_average:
        vals = []
        for r in valid_rounds:
            if k in r:
                vals.append(float(r[k]))
            elif "metrics" in r and k in r["metrics"]:
                vals.append(float(r["metrics"][k]))
            elif "late_metrics" in r and k in r["late_metrics"]:
                vals.append(float(r["late_metrics"][k]))
        if vals:
            summary[f"mean_{k}"] = float(np.mean(vals))

    accs = [float(r["test_accuracy"]) for r in history if "test_accuracy" in r]
    if accs:
        summary["final_accuracy"] = compute_final_accuracy(accs)

    return summary


__all__ = [
    "compute_eta",
    "compute_goodput_bps",
    "compute_utilization",
    "compute_wasted_grants_ratio",
    "compute_rtt_percentiles",
    "compute_queueing_delay_percentiles",
    "compute_final_accuracy",
    "compute_target_accuracy",
    "compute_time_to_target",
    "compute_accuracy_vs_wallclock",
    "compute_brier",
    "compute_bce",
    "compute_ece",
    "compute_roc_auc",
    "summarize_run_history",
    "make_row",
    "write_runs_csv",
    "read_runs_csv",
    "make_config_id",
    "Run",
    "RunsSchemaError",
]

