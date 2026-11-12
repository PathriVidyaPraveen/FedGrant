# Mininet Repository Audit

Audit of the repository at commit `5bda3ff` (branch `main`) plus the uncommitted working-tree changes present before this work, which are preserved untouched:

- modified `fedgrant/baselines/sim_adapter.py`, `fedgrant/fl/server.py`, `fedgrant/sim/run_net.py`, `tests/integration/test_sim_gateway_parity.py`
- untracked `tests/integration/test_b3_rtt_fl_path.py`, `tests/unit/test_b2_b3_accounting.py`

Requirement numbers (e.g. 2.9) refer to `docs/MININET_SPECIFICATION_AUDIT.md`.

Classification: **COMPLETE**, **PARTIAL**, **MISSING**, **BLOCKED** (needs an external dependency), **NEEDS_VERIFICATION** (exists, correctness not established).

## 1. Environment

| Item | Finding |
|---|---|
| Mininet / Open vSwitch / iperf3 | Not installed on the audit host; the host is non-root. No real Mininet run is possible here. `tests/integration/test_mininet_round.py` is skipped by its root/host gate. |
| `data/` (traces, FEMNIST, bank, runs) | Absent. `data/bank/femnist_full_s0.npz` does not exist in this checkout. |
| `models/` | Tracked: `gru_prod.npz`, `mlp_prod.npz`, `gru_seed{0..4}.pt`, `mlp_seed{0..4}.pt`, `norm_stats_{gru,mlp}.json`, `export_summary.json` (production GRU = seed 3, lowest validation loss 0.1909). |
| Configuration | `configs/grids/mn_m1.yaml`, `mn_sim.yaml`, `configs/scenarios/L1..L5,X2_cross.yaml`. `configs/policies/` and `configs/grids/mn_m2.yaml` (§15) are absent. |

## 2. Gateway and protocol

| Component | Path | Status | Evidence / notes |
|---|---|---|---|
| Wire protocol, six messages | `fedgrant/common/wire.py`, `constants.py` | COMPLETE | `tests/unit/test_wire.py` |
| GatewayCore (Algorithm 1) | `fedgrant/core/gateway.py` | COMPLETE for the simulator; **defect for the daemon** | Batched 5 ms prediction refresh fires only when `(t − round_start) % 5 ms == 0` (`on_tick`). The UDP daemon ticks every 1 ms from an arbitrary monotonic origin, so the refresh (req. 2.9) effectively never fires in Mininet. The simulator ticks at exact millisecond multiples and is unaffected. |
| Provider assembly (eligibility, selector, pacing, timeout, estimator hooks) | `fedgrant/sim/run_net.py` (`GatewaySim`, `_Core*`) | COMPLETE (simulator); not reachable by the daemon | Providers translate absolute time using `GatewaySim.round_start_abs_ns` and `D_ns`, which are set only by `GatewaySim.on_round_start`. The daemon calls `GatewayCore.on_round_start` directly, so those fields would be stale. Simulator–replay parity is proven by `tests/integration/test_sim_gateway_parity.py`. |
| Predictor factory | `fedgrant/predictors/factory.py` (`make_predictor`) | COMPLETE | `ewma`, `hmm`, `gru`, `mlp` load; missing checkpoint raises `FileNotFoundError`; `oracle` is rejected (simulator-only). |
| P-GRU NumPy inference | `fedgrant/predictors/gru_numpy.py` | COMPLETE | `tests/unit/test_gru_parity.py`, `test_fallback.py`: 27 passed. |
| Fallback (2 ms / exception / non-finite) | `fedgrant/predictors/fallback.py` | COMPLETE | Auto-wrapped by GatewayCore. |
| Async UDP gateway, 1 ms tick | `fedgrant/net/udp_gateway.py` | COMPLETE | Routes actions; supports per-client `rho_us` through `GatewayCore.on_round_start`. |
| Async UDP client | `fedgrant/net/udp_client.py` | COMPLETE | Waits `rho_us`, grant-directed DATA, ROUND_END with 200 ms fallback. |
| Production gateway factory for Mininet | — | MISSING | Only `fedgrant/mininet/stub_policy.py:create_gateway` (first-eligible selector, fixed p̂ = 0.95, fixed 100 ms timeout, allow-all pacing). It is not B4, B6 or FG-GRU. |

## 3. Baselines

| Component | Path | Status | Notes |
|---|---|---|---|
| B1 TCP | `fedgrant/baselines/b1_tcp.py` | PARTIAL for Mininet | Real TCP client/gateway; the gateway closes at D. Single-round object; no ROUND_START/ROUND_END control; error feedback is applied out of band in tests. |
| B2 UDP push | `fedgrant/baselines/b2_udp.py` | PARTIAL for Mininet | Real paced UDP client; `B2UdpGateway` binds per round on port 9101 (spec: 9000) and has no control path. `B2RoundAccumulator` validates DATA. |
| B3 PLAN-then-send | `fedgrant/baselines/b3_plan.py` | PARTIAL for Mininet | `B3UdpGateway` is multi-round with persistent RFC 6298/rate/delivery-EWMA state, but sends `rho_us = 0`, sends no ROUND_END, defaults to port 9102. `B3UdpClient` sends READY immediately (ignores ρ) and rejects ROUND_END, so a second round needs an out-of-band `finish_round`. |
| Simulator adapters | `fedgrant/baselines/sim_adapter.py`, `fedgrant/sim/run_baselines.py` | COMPLETE (simulator only) | Must not be used for Mininet (spec req. 1.8). |

## 4. Mininet package (`fedgrant/mininet/`, Tejaswi-owned except where noted)

| Module | Status | Notes |
|---|---|---|
| `topo.py`, `tcsetup.py`, `teardown.py` | COMPLETE (per testbed report) / NEEDS_VERIFICATION here | Fedora N=3 acceptance reported in `docs/TESTBED_TRACES_REPORT.md`; not re-run (no Mininet on this host). |
| `replay.py`, `crosstraffic.py`, `ge.py`, `validate_ge.py` | COMPLETE / NEEDS_VERIFICATION here | Testbed concerns. |
| `telemetry.py`, `logging_tap.py` | COMPLETE for the grant path | `LoggingUdpGateway` writes packets, grants, RTT (`grant_resolution`), scheduler samples. |
| `round_state.py` | COMPLETE | Atomic round window file for samplers. |
| `round_summary.py` | COMPLETE | — |
| `run_mininet.py` | PARTIAL | Launches `gateway_proc` and `client_proc` only (grant protocol). No baseline process selection; no `--config` YAML (spec req. 1.9); metadata labels the stub. |
| `gateway_proc.py` | PARTIAL | Round loop is inline (compute window → ROUND_START → D → deadline → late drain 0.1 s → metrics). Calls `on_round_start` without `rho_us` (ρi = 0, spec req. 6.4). Factory contract: `factory(num_clients) -> GatewayCore`. |
| `client_proc.py` | PARTIAL | Uses `profile_update.profile_bank` (synthetic profiles), not the FEMNIST bank (spec req. 5.1). |
| `stub_policy.py` | COMPLETE (as a stub) | Must not be used for M1/M2. |
| `profile_update.py` | COMPLETE | `synthetic_update(energies, seed)` builds a payload with prescribed per-block energies; reused by this work. |

## 5. FL and value profiles

| Component | Path | Status | Notes |
|---|---|---|---|
| Value-bank loader | `fedgrant/sim/value_bank.py` | COMPLETE (code) / BLOCKED (data) | `load_value_bank(path, N)`, `energies_for_run_round(seed, r)`, `client_weights`, provenance-hash verification. The bank file is absent (J). |
| Bank writer | `fedgrant/fl/bank.py` | COMPLETE (code) | Written by the FULL seed-0 run (J). |
| FL training/server | `fedgrant/fl/` | COMPLETE for the simulator path | No Mininet FL orchestration; `fedgrant/net/fl_tcp.py` provides model/aggregate TCP transport only and is unused by `fedgrant/fl/`. M2 is MISSING and BLOCKED (J, FEMNIST data). |

## 6. Experiment runner and logging

| Component | Status | Notes |
|---|---|---|
| `scripts/run_grid.py` | COMPLETE for simulator/FL; M1/M2 BLOCKED | Blocks M1/M2 with reasons in `blocked_jobs.csv`. |
| `fedgrant/common/logschema.py` | COMPLETE | `ROUND_METRIC_COLUMNS`, `PACKET_COLUMNS`, `RTT_COLUMNS` (`rtt_type` distinguishes `plan_data`), `ROUND_COLUMNS` (free-form `detail`). No `send_pos` in RTT rows. |
| `fedgrant/analysis/raw_metrics.py` | COMPLETE | Reads `meta.json`, `round_metrics.csv`, `rtt_samples.csv` (typed), `packets.csv` (protocol overhead), `processes.csv`. |

## 7. Tests run during the audit

| Command | Result |
|---|---|
| `python -m pytest -q tests/unit/test_mininet_helpers.py tests/unit/test_mininet_runtime.py tests/unit/test_mininet_topology_contract.py tests/unit/test_mininet_analysis.py tests/unit/test_gateway_profiling.py tests/integration/test_mininet_round.py tests/unit/test_udp_gateway.py tests/unit/test_udp_client.py tests/integration/test_udp_client_round.py tests/unit/test_b1_tcp.py tests/integration/test_b1_tcp_loopback.py tests/unit/test_b2_udp.py tests/integration/test_b2_udp_loopback.py tests/unit/test_b3_plan.py tests/integration/test_b3_plan_loopback.py tests/unit/test_fl_tcp.py tests/integration/test_fl_round_transport.py` | 136 passed, 1 skipped (`test_mininet_round.py`: requires root and the host Mininet package) |
| `python -m pytest -q tests/unit/test_gru_parity.py tests/unit/test_fallback.py` | 27 passed |

## 8. Reuse plan

| Need | Reuse |
|---|---|
| Daemon policy providers | `GatewaySim` provider assembly (`fedgrant/sim/run_net.py`) bound to a live GatewayCore clock |
| Predictors | `make_predictor` |
| B1/B2/B3 protocol logic | `B1TcpGateway`, `B1TcpClient`, `B1RoundAccumulator`, `B2UdpClient`, `B2RoundAccumulator`, `B3UdpGateway`, `B3UdpClient`, `B3GatewayCore` |
| Payload with a prescribed profile | `profile_update.synthetic_update` |
| Bank | `value_bank.load_value_bank` |
| Readiness draws | Simulator convention: `SeedSequence(network_round_seed(seed, r)).spawn(6)[2]`, `integers(0, ⌊0.1·D⌋ + 1)` per client in id order (`run_net.py`, `sim_adapter.py`, `run_baselines.py`) |
| Logs | `logschema` column sets and the file names used by `gateway_proc` / `client_proc` |

## 9. Status after the independent integration work

Changes are listed in `docs/MININET_DESIGN_DECISIONS.md` ("Implementation status"). Updated classifications:

| Component | Before | After |
|---|---|---|
| Production gateway factory (B4, B6, FG-GRU) | MISSING | COMPLETE in code (`fedgrant/mininet/production.py`); runtime BLOCKED until the real bank supplies weights (J). Not run in Mininet. |
| GatewayCore 5 ms refresh in the daemon | Defect | Fixed (bucket crossing); simulator behaviour unchanged |
| B1/B2/B3 Mininet processes | PARTIAL | COMPLETE in code (`fedgrant/mininet/baseline_proc.py`); verified over localhost sockets only; NEEDS_VERIFICATION in Mininet |
| Seeded ρi shared with the simulator | MISSING | COMPLETE for baseline processes; `gateway_proc.py` adoption pending (T) |
| Bank-backed client updates | MISSING | COMPLETE in code (`fedgrant/mininet/value_profiles.py`); BLOCKED on the bank file (J) |
| B1/B2 η in Mininet | MISSING | PARTIAL: join implemented (`baseline_metrics.py`), not wired into `raw_metrics`/runner (T) |
| `B1TcpGateway.wait()` after D | Defect | Fixed |
| M2 | MISSING | MISSING / BLOCKED (J) |
