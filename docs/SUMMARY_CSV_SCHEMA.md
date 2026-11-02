# Per-run summary schema (contract for `analysis/metrics.py`)

Spec: Section 11 (experimental matrix), 12 (metrics), 14 (statistics).
Code: `fedgrant/analysis/runs_schema.py` is the executable form of this document.
**Build every row with `make_row()` and write the file with `write_runs_csv()`**; both validate,
so a bad row fails when it is written, not when the statistics run.
`fedgrant/analysis/stats.py` reads the file with `read_runs_csv()`.

Schema version: **2**.

## 1. File

`results/summary/runs.csv`: one CSV with a header, one row per run, columns in the order of §2.
Every run of the matrix gets a row, including crashed ones (`status = missing`).
Re-running an analysis must rewrite the whole file from the raw logs (`make plots` works from raw results).

A run is identified by **(experiment_id, config_id, policy, seed)**. Duplicates are an error.

## 2. Columns

Empty cell = not measured (read as NaN). Booleans are `true` / `false`.

### Identification

| Column | Type | Values |
|---|---|---|
| `experiment_id` | str | `N-main`, `N-X1`, `N-X2`, `N-abl`, `F-main`, `F-ref`, `F-abl`, `M1`, `M2`, `M-sim` |
| `platform` | str | `mininet` for `M1`, `M2`; `sim` for every other experiment |
| `config_id` | str | `make_config_id(scenario, beta, N, cross_mbps)`, e.g. `L2-b0.10-N20-c0`, `L4-b0.10-N20-c3`, `L2-b0.10-N50-c0` |
| `scenario` | str | `L1`–`L5`; `none` for FULL (F-ref) |
| `beta` | float | 0.05, 0.10, 0.20 (0 for FULL) |
| `N` | int | 20, or 50 in N-X1 |
| `cross_mbps` | float | 0, or 3 in N-X2 |
| `policy` | str | `B1`–`B7`, `FG-GRU`, `FG-MLP`, `ORACLE`, `FULL`, ablations `FG-GRU-A1` (F-abl), `FG-GRU-A2`, `FG-GRU-A3`, `FG-GRU-A4` (N-abl) |
| `seed` | int | 0–4 network-only; 0–2 FL, M1, M-sim; 0 for M2 |
| `n_rounds` | int | rounds actually run (200 network-only, 300 FL, 100 M1/M-sim) |

`config_id` must equal `make_config_id(...)` of the row's own fields; the validator checks it.
The same `config_id` is used across experiments, so an M1 run and its M-sim twin share it.

### Network metrics (Section 12), averaged over rounds 21..last

| Column | Unit / definition |
|---|---|
| `eta` | delivered energy η_r, Eq. (7). **Required** for every network run (all policies except FULL) |
| `delivered_blocks` | Σ_i m_i per round |
| `clients_represented` | #{i : m_i > 0} per round |
| `deadline_completion` | fraction of rounds in which every grant resolved by D (0–1) |
| `goodput_bps` | delivered payload bits in [0, D] / D, bit/s |
| `utilization` | bits leaving the bottleneck in [0, D] / (C D), 0–1 |
| `queue_drops_fg` | bottleneck drop-tail drops of FedGrant frames per round |
| `queue_drops_cross` | the same for cross-traffic frames (0 without cross traffic) |
| `wasted_grants` | grants with y = 0 / grants issued, 0–1 (blank for push / plan / TCP modes) |
| `late_after_timeout` | DATA arriving after its timeout but before D, per round |
| `late_after_deadline` | DATA arriving after D, per round |
| `rtt_p50_ms`, `rtt_p99_ms` | median / 99th percentile of grant RTT samples, ms |
| `gw_cpu_pct` | mean psutil CPU% of the gateway process during [0, D] |
| `gw_rss_mb` | peak RSS of the gateway process, MB |
| `sched_lat_p50_us`, `sched_lat_p99_us` | event receipt to GRANT send, µs |
| `queueing_delay_mean_ms`, `queueing_delay_p99_ms` | mean and 99th percentile of bottleneck backlog/C samples, sampled every 10 ms |
| `protocol_overhead` | non-payload bytes / delivered payload bytes |

Per-round metrics are means over rounds 21..last. Percentiles (RTT, scheduler latency) are over all
samples of rounds 21..last. Gateway metrics (CPU, RSS, scheduler latency) may be blank in the simulator.

### FL metrics (F-main, F-ref, F-abl, M2)

| Column | Definition |
|---|---|
| `final_acc` | mean test accuracy over the last 10 evaluations (rounds 255–300). **Required** |
| `tta_rounds` | first round whose 3-evaluation moving average ≥ a*. **Blank when censored and for F-ref** |
| `tta_censored` | `true` if a* was not reached in 300 rounds. **Required except for F-ref** |
| `tta_seconds` | `tta_rounds · (T_c + D)`; blank when censored and for F-ref |

FL runs also fill the network columns they measure (`eta`, ...). Censored TTA enters the statistics
as 300 and is flagged in every output table (Section 14).

### Status

| Column | Values |
|---|---|
| `status` | `ok`; `rerun` (crashed once, re-run with the same seed, crash logged; counted as valid); `missing` (crashed twice; metric columns may be blank) |

## 3. Expected rows

`runs_schema.expected_runs()` lists the 1,074 runs of the matrix: N-main 675, N-X1 80, N-X2 80,
N-abl 60, F-main 96, F-ref 3, F-abl 12, M1 36, M2 2, M-sim 30. `stats.py` reports every expected run
that is absent from the file, every `missing` run, and every row that is not in the matrix
(`results/summary/stats/stats_missing.csv`). Nothing is dropped silently, and no run is excluded as an
outlier.

## 4. Writing it (`analysis/metrics.py`)

```python
from fedgrant.analysis.runs_schema import make_row, write_runs_csv

rows = []
for run in finished_runs:                         # parsed from results/raw/...
    rows.append(make_row(
        run.experiment_id, run.scenario, run.beta, run.N, run.policy, run.seed, run.n_rounds,
        cross_mbps=run.cross_mbps, status=run.status,
        tta_censored=run.tta_censored,            # FL only; None otherwise
        eta=..., utilization=..., wasted_grants=..., final_acc=..., tta_rounds=...,   # metrics measured
    ))
write_runs_csv(rows, "results/summary/runs.csv")
```

Then: `python -m fedgrant.analysis.stats --runs results/summary/runs.csv`.
