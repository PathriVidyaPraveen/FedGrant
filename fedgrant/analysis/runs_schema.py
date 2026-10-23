"""Per-run summary file results/summary/runs.csv (Sections 11, 12, 14).

The executable form of docs/SUMMARY_CSV_SCHEMA.md. analysis/metrics.py should
build each row with make_row() (or at least make_config_id()) and write the file
with write_runs_csv(), which validates every row; analysis/stats.py reads it with
read_runs_csv(), which applies the same validation.

NumPy and the standard library only.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION: int = 1

EXPERIMENTS: tuple[str, ...] = ("N-main", "N-X1", "N-X2", "N-abl", "F-main", "F-ref", "F-abl", "M1", "M2", "M-sim")
PLATFORM_OF: dict[str, str] = {e: ("mininet" if e in ("M1", "M2") else "sim") for e in EXPERIMENTS}
FL_EXPERIMENTS: frozenset[str] = frozenset({"F-main", "F-ref", "F-abl", "M2"})
POLICIES: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "FG-GRU", "FG-MLP", "ORACLE", "FULL",
                             "FG-GRU-A1", "FG-GRU-A2", "FG-GRU-A3", "FG-GRU-A4")
SCENARIOS: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5", "none")      # "none" = FULL (no network)
STATUSES: tuple[str, ...] = ("ok", "rerun", "missing")
TTA_CENSOR_ROUNDS: int = 300                                              # Section 12

ID_COLUMNS: tuple[str, ...] = ("experiment_id", "platform", "config_id", "scenario", "beta", "N", "cross_mbps",
                               "policy", "seed", "n_rounds")
NET_METRICS: tuple[str, ...] = (
    "eta", "delivered_blocks", "clients_represented", "deadline_completion", "goodput_bps", "utilization",
    "queue_drops_fg", "queue_drops_cross", "wasted_grants", "late_after_timeout", "late_after_deadline",
    "rtt_p50_ms", "rtt_p99_ms", "gw_cpu_pct", "gw_rss_mb", "sched_lat_p50_us", "sched_lat_p99_us",
    "protocol_overhead",
)
FL_METRICS: tuple[str, ...] = ("final_acc", "tta_rounds", "tta_censored", "tta_seconds")
COLUMNS: tuple[str, ...] = (*ID_COLUMNS, *NET_METRICS, *FL_METRICS, "status")
FLOAT_METRICS: tuple[str, ...] = (*NET_METRICS, "final_acc", "tta_rounds", "tta_seconds")


class RunsSchemaError(ValueError):
    """runs.csv does not follow docs/SUMMARY_CSV_SCHEMA.md."""


def make_config_id(scenario: str, beta: float, N: int, cross_mbps: float = 0.0) -> str:
    """Stable configuration id, e.g. 'L2-b0.10-N20-c0' (cross traffic in Mbit/s, 0 or 3)."""
    if scenario not in SCENARIOS:
        raise RunsSchemaError(f"unknown scenario {scenario!r}")
    c = float(cross_mbps)
    cs = f"{int(c)}" if c.is_integer() else f"{c:g}"
    return f"{scenario}-b{float(beta):.2f}-N{int(N)}-c{cs}"


@dataclass(frozen=True)
class Run:
    """One validated row. Metrics are floats (NaN when not measured); tta_censored is bool or None."""

    experiment_id: str
    platform: str
    config_id: str
    scenario: str
    beta: float
    N: int
    cross_mbps: float
    policy: str
    seed: int
    n_rounds: int
    metrics: dict[str, float]
    tta_censored: bool | None
    status: str

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.experiment_id, self.config_id, self.policy, self.seed)

    def value(self, metric: str) -> float:
        """Metric value; tta_rounds of a censored run is entered as 300 (Section 14)."""
        if metric == "tta_rounds" and self.tta_censored:
            return float(TTA_CENSOR_ROUNDS)
        return self.metrics.get(metric, math.nan)


# --------------------------------------------------------------------------- parsing


def _float(v: Any, name: str) -> float:
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return math.nan
    try:
        x = float(v)
    except (TypeError, ValueError) as e:
        raise RunsSchemaError(f"{name}: not a number: {v!r}") from e
    if math.isinf(x):
        raise RunsSchemaError(f"{name}: infinite value")
    return x


def _int(v: Any, name: str) -> int:
    x = _float(v, name)
    if math.isnan(x) or x != int(x):
        raise RunsSchemaError(f"{name}: expected an integer, got {v!r}")
    return int(x)


def _bool(v: Any, name: str) -> bool | None:
    if isinstance(v, bool):
        return v
    s = "" if v is None else str(v).strip().lower()
    if s == "":
        return None
    if s in ("true", "1"):
        return True
    if s in ("false", "0"):
        return False
    raise RunsSchemaError(f"{name}: expected true/false, got {v!r}")


def parse_row(row: Mapping[str, Any], where: str = "") -> Run:
    """Validate one CSV row (strings or Python values) and return a Run."""
    at = f"{where}: " if where else ""
    missing = [c for c in COLUMNS if c not in row]
    if missing:
        raise RunsSchemaError(f"{at}missing columns {missing}")
    exp, plat, pol = str(row["experiment_id"]), str(row["platform"]), str(row["policy"])
    scen, status = str(row["scenario"]), str(row["status"])
    if exp not in EXPERIMENTS:
        raise RunsSchemaError(f"{at}unknown experiment_id {exp!r}")
    if plat != PLATFORM_OF[exp]:
        raise RunsSchemaError(f"{at}{exp} must have platform {PLATFORM_OF[exp]!r}, got {plat!r}")
    if pol not in POLICIES:
        raise RunsSchemaError(f"{at}unknown policy {pol!r}")
    if status not in STATUSES:
        raise RunsSchemaError(f"{at}unknown status {status!r}")
    beta, N = _float(row["beta"], "beta"), _int(row["N"], "N")
    cross = _float(row["cross_mbps"], "cross_mbps")
    cross = 0.0 if math.isnan(cross) else cross
    cid = make_config_id(scen, beta, N, cross)
    if str(row["config_id"]) != cid:
        raise RunsSchemaError(f"{at}config_id {row['config_id']!r} != make_config_id(...) = {cid!r}")
    seed, n_rounds = _int(row["seed"], "seed"), _int(row["n_rounds"], "n_rounds")
    if seed < 0 or n_rounds < 0:
        raise RunsSchemaError(f"{at}seed and n_rounds must be >= 0")
    metrics = {m: _float(row[m], f"{at}{m}") for m in FLOAT_METRICS}
    cens = _bool(row["tta_censored"], f"{at}tta_censored")
    if status != "missing":
        if exp in FL_EXPERIMENTS:
            if math.isnan(metrics["final_acc"]):
                raise RunsSchemaError(f"{at}FL run needs final_acc")
            if cens is None:
                raise RunsSchemaError(f"{at}FL run needs tta_censored")
            if not cens and math.isnan(metrics["tta_rounds"]):
                raise RunsSchemaError(f"{at}uncensored FL run needs tta_rounds")
        elif pol != "FULL" and math.isnan(metrics["eta"]):
            raise RunsSchemaError(f"{at}network run needs eta")
    return Run(exp, plat, cid, scen, beta, N, cross, pol, seed, n_rounds, metrics, cens, status)


def read_runs_csv(path: str | Path) -> list[Run]:
    """Read and validate runs.csv; duplicate (experiment, config, policy, seed) is an error."""
    runs: list[Run] = []
    seen: set[tuple[str, str, str, int]] = set()
    with Path(path).open(newline="") as f:
        r = csv.DictReader(f)
        missing = [c for c in COLUMNS if c not in (r.fieldnames or [])]
        if missing:
            raise RunsSchemaError(f"{path}: missing columns {missing}")
        for i, row in enumerate(r, start=2):
            run = parse_row(row, f"{path}:{i}")
            if run.key in seen:
                raise RunsSchemaError(f"{path}:{i}: duplicate run {run.key}")
            seen.add(run.key)
            runs.append(run)
    return runs


def make_row(experiment_id: str, scenario: str, beta: float, N: int, policy: str, seed: int, n_rounds: int,
             *, cross_mbps: float = 0.0, status: str = "ok", tta_censored: bool | None = None,
             **metrics: float) -> dict[str, Any]:
    """A complete runs.csv row (unknown metric names are rejected; absent metrics are blank)."""
    unknown = set(metrics) - set(FLOAT_METRICS)
    if unknown:
        raise RunsSchemaError(f"unknown metrics {sorted(unknown)}")
    row: dict[str, Any] = {c: "" for c in COLUMNS}
    row.update(experiment_id=experiment_id, platform=PLATFORM_OF.get(experiment_id, ""),
               config_id=make_config_id(scenario, beta, N, cross_mbps), scenario=scenario, beta=beta, N=N,
               cross_mbps=cross_mbps, policy=policy, seed=seed, n_rounds=n_rounds, status=status,
               tta_censored="" if tta_censored is None else str(bool(tta_censored)).lower())
    for k, v in metrics.items():
        row[k] = "" if v is None or (isinstance(v, float) and math.isnan(v)) else v
    parse_row(row)                                         # validate now, not at analysis time
    return row


def write_runs_csv(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Validate every row and write runs.csv with the schema's column order."""
    p = Path(path)
    rows = list(rows)
    seen = set()
    for i, r in enumerate(rows):
        run = parse_row(r, f"row {i}")
        if run.key in seen:
            raise RunsSchemaError(f"row {i}: duplicate run {run.key}")
        seen.add(run.key)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in COLUMNS})
    return p


# --------------------------------------------------------------------------- the design (Section 11)

L_ALL = ("L1", "L2", "L3", "L4", "L5")
L24 = ("L2", "L4")
P8 = ("B2", "B3", "B4", "B5", "B6", "B7", "FG-GRU", "ORACLE")
P9 = (*P8, "FG-MLP")


def expected_runs() -> list[tuple[str, str, str, int]]:
    """Every (experiment, config_id, policy, seed) of the experimental matrix; 1,074 runs in total."""
    out: list[tuple[str, str, str, int]] = []

    def add(exp, scens, betas, policies, seeds, N=20, cross=0.0):
        for s in scens:
            for b in betas:
                for p in policies:
                    for k in seeds:
                        out.append((exp, make_config_id(s, b, N, cross), p, k))

    s5, s3 = range(5), range(3)
    add("N-main", L_ALL, (0.05, 0.10, 0.20), P9, s5)
    add("N-X1", L24, (0.10,), P8, s5, N=50)
    add("N-X2", L24, (0.10,), P8, s5, cross=3.0)
    add("N-abl", L24, (0.05, 0.10), ("FG-GRU-A2", "FG-GRU-A3", "FG-GRU-A4"), s5)
    add("F-main", L24, (0.05, 0.10), P8, s3)
    add("F-ref", ("none",), (0.0,), ("FULL",), s3)
    add("F-abl", L24, (0.05, 0.10), ("FG-GRU-A1",), s3)
    add("M1", L24, (0.10,), ("B1", "B2", "B3", "B4", "B6", "FG-GRU"), s3)
    add("M2", ("L2",), (0.10,), ("B3", "FG-GRU"), (0,))
    add("M-sim", L24, (0.10,), ("B2", "B3", "B4", "B6", "FG-GRU"), s3)
    return out