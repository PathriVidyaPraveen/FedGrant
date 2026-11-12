"""Metrics package for FedGrant experiment analysis (Section 12)."""

from fedgrant.analysis.metrics import (
    compute_accuracy_vs_wallclock,
    compute_bce,
    compute_brier,
    compute_ece,
    compute_eta,
    compute_final_accuracy,
    compute_goodput_bps,
    compute_roc_auc,
    compute_target_accuracy,
    compute_time_to_target,
    compute_utilization,
    compute_wasted_grants_ratio,
    summarize_run_history,
)

__all__ = [
    "compute_eta",
    "compute_goodput_bps",
    "compute_utilization",
    "compute_wasted_grants_ratio",
    "compute_final_accuracy",
    "compute_target_accuracy",
    "compute_time_to_target",
    "compute_accuracy_vs_wallclock",
    "compute_brier",
    "compute_ece",
    "compute_roc_auc",
    "compute_bce",
    "summarize_run_history",
]

