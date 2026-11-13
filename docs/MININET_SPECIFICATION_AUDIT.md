# Mininet Specification Audit

Source: `docs/Networks Final Project Specification.pdf` (FedGrant Final Execution Specification v3.1, 25 pages), read in full. Section and page numbers below refer to that document. "Binding" means the specification states it as a parameter or rule ("Every parameter in it is binding", p. 1); "suggestion" means descriptive or an estimate.

Dependency column: **P** = Praveen (policy/predictors/ML), **J** = Jayadev (FEMNIST, bank, FL, simulator), **T** = Tejaswi (testbed/traces), **—** = none / Prajin-owned.

Status abbreviations used in "Existing": see `docs/MININET_REPOSITORY_AUDIT.md` for evidence.

## 1. Mininet integration and experiments

| # | Exact requirement (spec wording, abridged only where marked …) | § / page | Implementation meaning | Existing | To implement | Dep. |
|---|---|---|---|---|---|---|
| 1.1 | "Ubuntu 22.04, Mininet 2.3, Open vSwitch in standalone bridge mode (OVSBridge, no controller); plain Link with all shaping applied by explicit tc commands." | §7 p.14 | Testbed host requirements; no controller | `fedgrant/mininet/topo.py` (Fedora-validated per `docs/TESTBED_TRACES_REPORT.md`) | Nothing in this task | T |
| 1.2 | Hosts "c0..c{N-1} (one client each), gw (gateway daemon), srv (FL server), xt (cross-traffic source). One switch s1." Links "ci–s1, xt–s1, s1–gw, gw–srv." | §7 p.14 | Process placement: gateway daemon on `gw`, one client process per `ci`, FL server on `srv` | `topo.py` | Baseline processes must run on the same hosts | T |
| 1.3 | Client egress `netem delay δi ms loss gemodel p r 100% 0%` + child `tbf rate b burst 3000 limit 47168`; reverse access `netem delay δi ms`; bottleneck `root netem delay 5ms; child tbf rate 10mbit burst 3000 limit 94336`; gateway egress `netem delay 5ms`; gw–srv unshaped | §7 p.14 | Shaping is fixed by spec; process code must not shape | `tcsetup.py` | None | T |
| 1.4 | "`fedgrant/mininet/replay.py`: one thread per client issues `tc qdisc change … tbf rate R …` at each 100 ms bin boundary … and logs the actual change time." | §7 p.14 | Trace replay is a runner concern, relative to round origin | `replay.py`, `run_mininet.py` | None | T |
| 1.5 | "UDP: gateway port 9000, clients port 9001. TCP (baseline B1): gateway port 9100." | §7 p.14 | All UDP baselines (grant, B2, B3) must use 9000/9001 in Mininet; B1 TCP 9100 | Grant path uses 9000/9001; B2 default 9101, B3 default 9102 (constructor defaults) | Baseline processes bind spec ports | — |
| 1.6 | "Cross traffic: `iperf3 -u -b 3M -l 1432` from xt to a sink on gw port 9999." | §7 p.14 | Port 9999 must not collide with gateway sockets | `crosstraffic.py` | None | T |
| 1.7 | "All timestamps use `time.monotonic_ns()`. All hosts share one kernel, hence one clock." | §7 p.14 | Absolute monotonic deadlines in ROUND_START are valid across hosts | Used throughout transports | Keep monotonic clock in new processes | — |
| 1.8 | "Experiments run in Mininet. Exactly M1 and M2 of Section 11. All other experiments run in the simulator." | §7 p.14 | No simulator substitution for M1/M2 and no other Mininet experiments | `scripts/run_grid.py` blocks M1/M2 | Keep simulator adapters out of Mininet | — |
| 1.9 | "`sudo python -m fedgrant.mininet.run_mininet --config <yaml> --seed <s>`. One run per machine at a time." | §7 p.14 | Runner CLI contract | Runner exists with flag-based CLI (`--clients/--scenario/...`), no `--config` yet | Not in this task (T-owned); noted as gap | T |
| 1.10 | "The netem GE random generator cannot be seeded. Mininet runs are not bit-identical; all other randomness is seeded." | §7 p.14 | All process-level randomness (ρi, payload synthesis, exploration) must be seeded | Grant runs use ρ=0 (no randomness); synthetic payload seeded | Seeded ρi derivation | — |
| 1.11 | I2 integration point: "one Mininet round, N = 3, B6"; W2 tests include `test_mininet_round` | §17 p.22 | Smoke acceptance; N=3 | Stub-policy smoke exists | Production B6 factory enables real I2 | — |
| 1.12 | Demo: "Mininet N = 10, B4 vs. FG-GRU live" | §17 p.23, §18 item 15 | Production B4 and FG-GRU factories at N=10 | Missing | Factories; bank weights for N=10 are **not** defined by spec (bank has N=20) | J (ambiguity) |

## 2. Gateway architecture and protocol behaviour

| # | Requirement | § / page | Meaning | Existing | To implement | Dep. |
|---|---|---|---|---|---|---|
| 2.1 | Gateway: "per-client structures (Table 2); global in-flight expectation Φ; max-heap of candidate clients. Algorithm: Algorithm 1. Output: GRANT messages; the aggregate Ĝr." | §2.1 p.4 | One GatewayCore drives both simulator and daemon | `fedgrant/core/gateway.py` (`GatewayCore`) | Production provider assembly for the daemon | — |
| 2.2 | "The simulator instantiates the same GatewayCore used by the real daemon, and feeds it events with timestamps." | §6 p.13 | Daemon must use the same core **and** the same providers as the simulator | Simulator providers in `fedgrant/sim/run_net.py` (`GatewaySim`) | Reuse `GatewaySim` providers in the daemon | — |
| 2.3 | GatewayCore methods `on_round_start(t)`, `on_ready(c, energies, t)`, `on_data(c, grant_id, pos, block_idx, payload, t)`, `on_tick(t)`, `on_deadline(t)`; actions `SendGrant`, `SendRoundEnd`, `EmitAggregate` | §6.1 p.14 | Public API is fixed | Implemented | None | — |
| 2.4 | Policy interface (`PredInput`, `Predictor`, `Selector`, `CandidateView`, `PolicyConfig`) "identical in the simulator and the gateway daemon" | §6.1 p.13 | Same `PolicyConfig` object configures both | `fedgrant/policy/types.py` | Production factory accepts `PolicyConfig` per policy | P (owner) |
| 2.5 | Table 2: estimator state persistent across rounds; per-round state reset at ROUND_START | §2 p.5 | Gateway object must persist across rounds in the daemon | `ClientState`, `GatewayCore.on_round_start` | Factory returns one long-lived core | — |
| 2.6 | Timeout τg = min(D, tg + srtt + max(4 rttvar, 10 ms)); cap; 100 ms rate window | §3.7 pp.6–7 | Provider behaviour | `fedgrant/core/estimators.py` | Reused via providers | — |
| 2.7 | Pacing P_target = C(median srtt + 5 ms)/(8S); issue only if Φ + p̂ ≤ P_target | §3.8 p.7 | Provider behaviour | `policy/pacing.py` | Reused | P |
| 2.8 | Eligibility (i)–(v), θ = 0.05 | §3.9 p.7 | Provider behaviour | `policy/eligibility.py` | Reused | P |
| 2.9 | "Predictions are recomputed for all ready clients on a 5 ms tick, and for one client at each resolution event" | §4.6 p.11; Alg. 1 line 26 | Daemon ticks are not aligned to round start; the 5 ms refresh must still fire | **Defect found**: core refreshes only when `(t − start) % 5 ms == 0` | Fix to 5 ms bucket crossing | — |
| 2.10 | Fallback: predictor exception / non-finite / > 2 ms ⇒ P-EWMA for the rest of the round, restored next round | §9 p.15 | Production must wrap predictor | `FallbackPredictor` (auto-wrapped by GatewayCore) | Reused | P |
| 2.11 | "ϵ = 0 in every experiment of Section 11 and in production." | §9 p.15 | No exploration in Mininet | `PolicyConfig.explore_eps` default 0.0 | Factory must reject ϵ ≠ 0 | — |
| 2.12 | Value-integrity rule (≥3 energy violations) | §9 p.15 | Optional checker | `ValueIntegrityChecker` hook | Not required for this task | — |
| 2.13 | Wire format (common header `!HBBIHHQ`, six messages, ROUND_START `mode` 0 grant / 1 push / 2 plan / 3 tcp, `rho_us`) | §8 p.15 | Baselines signal mode in ROUND_START | `fedgrant/common/wire.py` | Baseline processes use the standard messages | — |
| 2.14 | "Messages with a wrong magic, version or round_id are dropped and counted." | §8 p.15 | Count rejections in every process | Grant/B3 count; B2 accumulator counts | Count in new control endpoints | — |

## 3. B1, B2 and B3 baselines

| # | Requirement | § / page | Meaning | Existing | To implement | Dep. |
|---|---|---|---|---|---|---|
| 3.1 | B1: "Mininet only. Each client opens a TCP connection (Linux default CUBIC) at readiness and streams all K blocks in energy order. At D the gateway closes all connections; complete blocks received by D are delivered." TCP stream "per block: block_idx u16, n_coords u16, payload f16×L" | §10 p.17; §8 p.15 | Real TCP; gateway closes at D | `fedgrant/baselines/b1_tcp.py` (`B1TcpGateway`, `B1TcpClient`), single-round | Multi-round Mininet process with ROUND_START (mode 3) / ROUND_END | — |
| 3.2 | B2 (Mode 1): "At readiness each client sends all K blocks in energy order, paced at b̄i, with no grants and no retransmission, until D." | §10 p.17 | Pacing at the configured mean access rate b̄i | `b2_udp.py` (`B2UdpClient`, `B2RoundAccumulator`, single-round `B2UdpGateway`) | Multi-round process on ports 9000/9001 | — |
| 3.3 | B3 (Mode 2): t_plan = 0.1D; M = ⌊C(D − t_plan − median srtt)/(8S)⌋; srtt sample = first DATA arrival − PLAN send; caps ci; greedy ki; PLAN(ki, b̂i); clients send first ki blocks paced at b̂i, no retransmission | §10 p.17 | Existing core implements this | `b3_plan.py` (`B3GatewayCore`, multi-round `B3UdpGateway`, `B3UdpClient`) | ρi in ROUND_START, READY after ρi, ROUND_END for error feedback | — |
| 3.4 | "Held constant across every baseline and FedGrant variant. Encoder and energy order; fp16 payload; error feedback …; zero-fill aggregation; FL hyperparameters; C, Q, δi, b̄i, loss scenario, D, ρi, Tc; seeds." | §10 p.16 | Baselines must use the same ρi draws, Tc and D as FedGrant; error feedback needs ROUND_END | ρi = 0 in existing Mininet grant path and in `B3UdpGateway` | Shared seeded ρi derivation for all policies | — |
| 3.5 | Client agent: "On ROUND_END, apply error feedback"; "If ROUND_END does not arrive within 200 ms after D, the client treats every block as undelivered." | §2.1 p.4; §9 p.15 | Applies to baselines too (error feedback held constant) | Grant `UdpClient` implements; B1/B2/B3 clients do not | ROUND_END handling and 200 ms fallback in baseline clients | — |

**Ambiguity A3-1.** §8 defines READY as the client's message after ρi, but §10 B1/B2 say clients start sending "at readiness" without mentioning READY. The specification does not say whether B1/B2 clients send READY. Decision recorded in the design register (D-07): B1/B2 do not send READY.

## 4. B4, B6 and FG-GRU policies

| # | Requirement | § / page | Meaning | Existing | To implement | Dep. |
|---|---|---|---|---|---|---|
| 4.1 | B4: "FedGrant with selector rr … and P-EWMA." B6: "selector value and P-EWMA." FG-GRU: "selector value and P-GRU." | §10 p.17 | `PolicyConfig("rr","ewma")`, `("value","ewma")`, `("value","gru")` | Same mapping in `scripts/run_grid.py` and `fedgrant/fl/run_fl.py` | Same mapping in the production factory | — |
| 4.2 | M1 policies: "B1, B2, B3, B4, B6, FG-GRU"; M2 policies: "B3, FG-GRU" | §11 p.17 | Mininet factories required for exactly these | Stub only | B4, B6, FG-GRU factories; B1/B2/B3 processes | — |
| 4.3 | P-GRU production checkpoint `models/gru_prod.npz` = seed with lowest validation loss; NumPy float32 inference, no PyTorch at the gateway; ring buffer of 16 observations and cached hᵢ | §4.4 p.9; §4.6 p.10 | Factory loads via existing predictor factory | `PGRUNumpy`, `models/gru_prod.npz` (tracked), `models/export_summary.json` (seed 3) | Load through `make_predictor("gru")`; fail if missing | P (confirm final checkpoint) |
| 4.4 | Production checkpoints "needed by the FG-GRU and FG-MLP grids and by the Mininet runs" | §16 p.22 | Hard dependency for FG-GRU runs | Present | Error clearly if absent | P |

## 5. Value-profile bank and client update generation

| # | Requirement | § / page | Meaning | Existing | To implement | Dep. |
|---|---|---|---|---|---|---|
| 5.1 | "The lossless reference run (FULL, seed 0 …) stores, for every round and client, the sorted energy list … and wi, in `data/bank/femnist_full_s0.npz`. Network-only runs … draw, for each (round, client), a round index uniformly from 1–300 and use that client's stored profile." | §5.2 p.12 | M1 (network-only) values must come from this bank; weights wi from the bank | Loader `fedgrant/sim/value_bank.py` (`load_value_bank`, `energies_for_run_round`, `client_weights`); **bank file absent** in this checkout | Client-side provider + gateway weights from the bank; no synthetic fallback | J (bank file and provenance) |
| 5.2 | M1 is "network-only" (§11 p.17) | §11 | Payload content is not FL; only its block energies matter | `profile_update.synthetic_update` builds a payload with prescribed block energies | Use bank profile → synthetic payload (energies match the real profile) | — |
| 5.3 | M2 is "end-to-end FL" | §11 | Real FEMNIST training deltas, not the bank | FL server/training exist for the simulator path only | Not implementable without FL-over-Mininet orchestration | J |

**Ambiguity A5-1.** The bank stores N = 20 clients. The specification does not define profiles or weights for Mininet runs with N ≠ 20 (the N = 3 smoke round, the N = 10 demo). Real M1 uses N = 20 and is unaffected.

**Ambiguity A5-2.** The specification does not define how a network-only Mininet client turns a stored energy list into a DATA payload. The repository's existing convention (`synthetic_update` with seed `seed + 1009·client_id + round_id`, block j receiving the j-th sorted energy) is reused and documented as an assumption.

## 6. Round timing, compute window, deadline and late drain

| # | Requirement | § / page | Meaning | Existing | To implement | Dep. |
|---|---|---|---|---|---|---|
| 6.1 | Tc = "Compute window before the timed upload window: 1.0 s. Wall-clock time per round is Tc + D." | §3.1 p.5 | Each round: Tc, then ROUND_START, then D | `gateway_proc.py` sleeps `--compute-window-s` (default 1.0) | Shared schedule object | — |
| 6.2 | "After Tc = 1.0 s from broadcast, the gateway sends ROUND_START with the absolute deadline. Clients finish local training within Tc; the run aborts if any client exceeds it." | §7 p.14 | Abort rule applies to FL (M2) | Not implemented (no FL in Mininet) | M2 only | J |
| 6.3 | D = βT_full rounded to whole ms (199/398/797 ms; 996 ms for N=50) | §3.1 p.5 | Use `compute_deadline_ns` | `fedgrant/sim/run_net.compute_deadline_ns` | Reuse | — |
| 6.4 | ρi "drawn i.i.d. U(0, 0.1D) per client per round" | §3.1 p.5 | Gateway sends `rho_us` per client in ROUND_START | `GatewayCore.on_round_start(rho_us=…)` supported; `gateway_proc` never passes it (ρ=0) | Seeded per-client ρ from the simulator's readiness stream | — |
| 6.5 | Late DATA: "(b) arriving after D" counted; OnData: "if t > D then discard; count late-after-deadline" | §12 p.18; Alg. 1 line 18 | Gateway must keep receiving after D | `gateway_proc` waits `--late-drain-s` (0.1 s) | Late-drain phase in shared loop | — |
| 6.6 | Client ROUND_END fallback 200 ms after D | §9 p.15 | Late drain should not delay ROUND_END beyond that | Grant ROUND_END sent at D | Baselines send ROUND_END at D | — |

**Ambiguity A6-1.** The late-drain duration is not specified; the existing 0.1 s default is retained.

## 7. UDP/TCP communication and process architecture

| # | Requirement | § / page | Meaning | Existing | To implement | Dep. |
|---|---|---|---|---|---|---|
| 7.1 | `net/udp_gateway.py` "asyncio, 1 ms tick" | §16 p.21 | Real asyncio daemon | `fedgrant/net/udp_gateway.py`, `LoggingUdpGateway` | Reused | — |
| 7.2 | Client: "On each GRANT(g, s), send DATA for send position s within 0.1 ms" | §2.1 p.4 | Grant client latency | `fedgrant/net/udp_client.py` | None | — |
| 7.3 | "The server broadcasts the model to the clients over TCP"; "Broadcasts x_{r+1} over TCP outside the timed window" | §7 p.14; §2.1 p.4 | M2 model broadcast | `fedgrant/net/fl_tcp.py` (transport only) | M2 only | J |
| 7.4 | Baselines module layout: `tcp_baseline.py` (B1), `push_client.py` (B2, B3 client side) | §15 p.20; §16 p.21 | Suggested layout; repository uses `fedgrant/baselines/` | Different but equivalent layout | Keep existing modules | — |

## 8. Logging, metrics, telemetry and metadata

| # | Requirement | § / page | Meaning | Existing | To implement | Dep. |
|---|---|---|---|---|---|---|
| 8.1 | "Per-grant, per-packet and per-round CSVs (schema in `fedgrant/common/logschema.py`); tc -s qdisc sampled every 10 ms on the bottleneck; psutil CPU and RSS of the gateway every 100 ms." | §7 p.14 | Baseline gateways write the same CSV files | Grant path: `LoggingUdpGateway`, `telemetry.py` | Baseline gateway writes `packets.csv`, `rtt_samples.csv`, `round_metrics.csv`, `rounds.csv` with existing columns | T (telemetry) |
| 8.2 | "`analysis/` reads only results/raw CSVs whose schema is defined in logschema.py." | §15 p.20 | No schema drift | `logschema.SCHEMAS` | Do not add columns | — |
| 8.3 | Metric definitions (η, delivered blocks, clients represented, deadline completion, goodput, utilization, queue drops, wasted grants, late packets, RTT, queueing delay, gateway CPU/RSS, scheduler latency, protocol overhead) | §12 pp.18–19 | Each Mininet metric must follow these | Grant path computes subset; utilization/queue from tc telemetry | Baselines: counts from real sockets; B1/B2 η needs client energies (gateway never sees them) | — |
| 8.4 | "Per-run scalars are averaged over rounds 21–last." | §12 p.18 | M1 (100 rounds) uses rounds 21–100 | `analysis/raw_metrics.py` | None | — |
| 8.5 | RTT: "Median and 99th percentile of grant RTT samples" | §12 | B3 PLAN→DATA samples are a separate type | `rtt_type` column distinguishes `grant_resolution` / `plan_data` | Baseline gateway writes `plan_data` rows | — |

**Ambiguity A8-1.** §12 does not say how η is measured for B1/B2 in Mininet, where the gateway never receives energies (no READY). Section 8.3 above records the gap; the design register (D-09) records the chosen post-run join.

## 9. M1 requirements and acceptance

| Requirement | § / page | Meaning |
|---|---|---|
| "network-only, 100 rounds, β = 0.10, {L2, L4}", policies B1, B2, B3, B4, B6, FG-GRU, seeds 0–2, 36 runs | §11 p.17 | 2 scenarios × 6 policies × 3 seeds |
| M-sim: simulator twins (B2, B3, B4, B6, FG-GRU; seeds 0–2; 30 runs) "used for calibration" | §11 p.18 | Twins must share seeds and therefore ρi draws and trace offsets (§14 "Pairing") |
| "For each M1 configuration and policy, the paired difference (Mininet minus simulator twin) of η, utilization and wasted grants" | §14 p.19 | η, utilization and wasted grants must be measured in Mininet |
| "one M1 configuration reproduced in Docker by a non-author" | §17 p.23 | Reproducibility acceptance |
| "netem realized loss within ±1 percentage point over 10⁵ packets" | §17 p.22 | Testbed acceptance (T) |

**Acceptance for M1 (derived):** 36 completed runs with production policies (no stub), bank-sourced values, spec ρi, logs in `logschema` format, and `results/summary` rows; paired Mininet − simulator analysis.

## 10. M2 requirements and acceptance

| Requirement | § / page | Meaning |
|---|---|---|
| "end-to-end FL, 300 rounds, β = 0.10, L2", policies B3 and FG-GRU, seed 0, 2 runs | §11 p.17 | Real FEMNIST training over Mininet |
| Server broadcast over TCP, Tc = 1.0 s, abort if a client exceeds Tc | §7 p.14 | FL orchestration on `srv` and clients |
| Evaluate test accuracy every 5 rounds; TTA in seconds = r(Tc + D) | §5.1 p.11; §12 p.18 | FL metrics |
| Prajin owns "run M2" | §16 p.21 | — |

**Acceptance for M2 (derived):** two 300-round runs whose updates are real FEMNIST deltas, delivered over the Mininet B3 and FG-GRU transports, with FL history and TTA. Depends on J (FEMNIST data, FL client/server orchestration boundary) — see the repository audit.

## 11. Reproducibility, seeds, configuration, provenance

| # | Requirement | § / page | Meaning | Dep. |
|---|---|---|---|---|
| 11.1 | Master seed s split by `SeedSequence(s).spawn(6)`: trace offsets, GE chains, readiness, exploration, FL, cross-traffic phase | §6 p.13 | Mininet ρi should come from the readiness stream so twins share readiness draws | — |
| 11.2 | "Pairing. All policies within a configuration share seeds, and therefore share trace offsets, readiness draws and FL data order." | §14 p.19 | ρi derivation identical across policies and between Mininet and simulator | — |
| 11.3 | Configuration files per policy and grid (`configs/policies/*.yaml`, `configs/grids/mn_m1.yaml`, `mn_m2.yaml`) | §15 p.20 | `configs/policies/` and `mn_m2.yaml` absent in repository | T/J |
| 11.4 | Bank provenance: FEMNIST LEAF commit pinned, preprocessing command fixed | §5.1 p.11 | Bank loader verifies provenance hash | J |
| 11.5 | Change control: parameter changes need a CHANGELOG amendment approved by all four | §19 p.24 | No spec parameter may be changed in this work | all |
