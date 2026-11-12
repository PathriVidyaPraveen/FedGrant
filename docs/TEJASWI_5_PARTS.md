# Tejaswi work split: five independently finishable parts

This split covers the testbed and trace work described in Tejaswi's implementation plan. Each part has its own deliverables and acceptance gate. Parts 1 and 2 are pure Python and need neither root nor Mininet. Parts 3 and 4 require a working local Mininet/tc environment. Part 5 can be built with synthetic logs; the final simulator comparison needs Jayadev's M-sim rows.

The shared interfaces to preserve are `fedgrant/analysis/runs_schema.py`, `docs/SUMMARY_CSV_SCHEMA.md`, `fedgrant/policy/constants.py`, and Jayadev's `fedgrant/sim/traces.py`. The implementation in this checkout adds `fedgrant/traces/` and `fedgrant/mininet/` against those contracts. The Fedora N=3/L2 acceptance test passed on 2026-11-02 after the receiver-window measurement correction.

## Part 1 — Trace acquisition, preparation, and deterministic replay inputs

**Goal:** produce trace files and a reusable deterministic client-rate API that a Mininet runner can consume without simulator-specific code.

**Deliverables**

- `scripts/get_traces.sh`: accept an explicit source directory and discover supported local sources; copy the five required `.up` traces and their license; record source, revision/version, date, and SHA-256 hashes. Fail with an actionable listing if any required trace is absent. Network download is optional, never required for the pure tests.
- `fedgrant/traces/binning.py` and `scripts/bin_traces.py`: validate monotone finite timestamps, count 1500-byte opportunities in complete 100 ms bins, convert each count to bit/s, split ID traces contiguously 70/15/15 using floor for train and validation, and keep OOD traces whole. Emit CSV headers and a summary JSON.
- `fedgrant/traces/client_trace.py`: choose the documented ID/OOD trace, deterministic client offset, wrap only within the selected segment, scale to the client target mean, and enforce the 100 kbit/s floor. Keep seed streams independent of client count.
- Unit tests and a checked-in golden vector covering binning, boundaries, deterministic offsets, scaling, floor, and malformed input.

**Acceptance gate**

- Synthetic `.up` fixtures produce exact expected complete-bin rates and split boundaries.
- The same `(client, split, seed)` always yields the same array; changing the seed changes the offset; asking for 20 versus 50 clients does not change a given client's offset.
- The scaled mean equals the target when flooring does not alter values; every output is at least 100 kbit/s; wrapping preserves the selected segment's values.
- The trace pipeline and tests run without root, Mininet, or downloaded data.

## Part 2 — Loss model, tc command builders, and scenario definitions

**Goal:** make network shaping parameters explicit, reviewable, and testable before executing any system command.

**Deliverables**

- `fedgrant/mininet/ge.py`: convert target average loss and mean burst to netem Gilbert–Elliott `(p, r)` parameters, compute observed loss/burst statistics, and report moving-block-bootstrap 95% confidence intervals that preserve temporal loss dependence.
- `fedgrant/mininet/tcsetup.py`: pure command builders for client uplink/downlink, bottleneck, gateway egress, and replay rate changes. Import frame size `S` from `fedgrant.policy.constants`.
- `configs/scenarios/`: machine-readable L1–L5 and cross-traffic settings, plus M1 grid descriptions that do not duplicate run-schema definitions.
- Unit tests with exact expected command strings, frame/queue byte sizes, GE parameter round trips, and malformed parameter rejection.

**Acceptance gate**

- L1, L2/L4/L5/X2, and L3 reproduce the specified `(p, r)` values and target loss within the documented tolerance.
- A 32-frame access queue is `32 * S` bytes and a 64-frame bottleneck queue is `64 * S` bytes.
- Builders validate interface names, finite nonnegative delays, valid probabilities, and positive integer bit rates; they execute no shell commands.
- Tests pass without root or Mininet.

## Part 3 — Mininet topology and measured link validation

**Goal:** create and verify the actual isolated network described by the plan.

**Deliverables**

- `fedgrant/mininet/topo.py`: explicit client, gateway, cross-traffic, and server interfaces/addresses; static ARP; access shaping and the shared 10 Mbit/s bottleneck; offloads disabled where supported.
- `verify_tc`: inspect applied qdiscs and fail before a run if delay, loss, rate, or queue limits differ from the requested setup.
- `fedgrant/mininet/validate_ge.py`: sequence-numbered UDP sender/receiver and analysis using the Part 2 statistics.
- A documented Fedora setup/check command list and small root-required integration tests for N=3 ping RTT, access rate cap, and bottleneck payload throughput.

**Acceptance gate**

- The topology starts and tears down repeatedly without stale namespaces or switches.
- `verify_tc` checks every expected interface and qdisc before data collection.
- Ping RTT is within 3 ms of the configured table, access shaping stays within 3%, and bottleneck UDP payload throughput is 9.6–9.8 Mbit/s for the specified 1432-byte payload test.
- GE validation uses at least 100,000 sequence-numbered packets per tested setting and records measured loss and burst statistics with uncertainty; discrepancies are reported, not silently accepted.

**Environment gate:** this part requires Mininet, Open vSwitch, `iproute-tc`, the netem/tbf kernel modules, and sudo. If those are unavailable, finish the code and pure tests, but label the measured acceptance gate pending rather than claiming it passed.

## Part 4 — Trace replay, traffic, telemetry, and a self-contained runner

**Goal:** run a complete round against the real Mininet topology using local stand-ins where production gateway wiring is not ready.

**Deliverables**

- `fedgrant/mininet/replay.py`: monotonic-clock, 100 ms client rate replay from Part 1 CSVs; validate each requested tbf change and stop cleanly.
- `fedgrant/mininet/crosstraffic.py`: start/stop iperf3 traffic and verify the offered/delivered rate.
- `fedgrant/mininet/telemetry.py`: sample queue backlog/drops and process CPU/RSS with monotonic timestamps.
- `fedgrant/common/logschema.py` and a logging tap: define raw CSV columns once and record round, packet, queue, process, and timing events with headers.
- `fedgrant/mininet/stub_policy.py`, synthetic profile/update input, gateway/client process entry points, idempotent teardown, and `run_mininet.py`. The runner must accept a gateway factory so later production wiring can plug in without changing the topology or run protocol.
- Integration test for one stub-policy round, including clean teardown after both success and injected failure.

**Acceptance gate**

- A local run produces trace replay, optional cross traffic, raw logs, and metadata in a new run directory; the run directory records configuration, seed, trace hashes, and CPU placement.
- The runner checks topology/qdiscs before rounds and terminates child processes and Mininet state on every exit path.
- The stub-policy round test runs without any teammate's policy/predictor wiring. Production M1 results are not claimed from a stub.

## Part 5 — Raw-log metrics, calibration, packaging, and report evidence

**Goal:** turn raw testbed artifacts into reproducible M1 evidence and a clean-machine reproduction path.

**Deliverables**

- `fedgrant/analysis/raw_metrics.py`: parse the Part 4 raw logs, compute documented per-run metrics, and build rows exclusively through `fedgrant.analysis.runs_schema.make_row()` / `write_runs_csv()`.
- `fedgrant/analysis/calibrate.py`: pair Mininet M1 and simulator M-sim rows by the shared run keys and report paired differences and confidence intervals; fail clearly on missing or duplicate pairs. Test using synthetic rows now; consume real simulator rows when available.
- `docker/Dockerfile`, pinned testbed requirements, and `Makefile` targets for setup, pure tests, root-required tests, trace preparation, and a smoke run. Keep host kernel/Mininet requirements explicit; a container cannot supply the host's kernel qdiscs.
- Testbed/traces report section outline and generated tables/figures sourced from the recorded run artifacts, including trace summaries, GE validation, and calibration results.

**Acceptance gate**

- Synthetic fixtures cover each metric and calibration edge case; emitted rows pass the existing summary-schema validator.
- Re-running analysis from the same raw logs produces byte-stable CSV/table outputs, aside from explicitly documented timestamps.
- A clean-machine checklist succeeds through pure tests and identifies host-only requirements before any privileged test.
- M1 and M-sim calibration is marked complete only when real paired rows exist; otherwise the script and synthetic tests are complete and the empirical comparison is pending.

## Work that is intentionally not part of these five independent packages

- The production policy/predictor provider wiring and B1/B2/B3 client implementations owned by Prajin. Part 4 uses the documented stub interface until then.
- Simulator M-sim executions and energy-profile bank format owned by Jayadev. Part 5 defines and tests the join/calibration path without inventing their outputs.
- Group-wide M1/M2 claims, report conclusions, or clean-machine success claims that require real runs. Each claim must be backed by the corresponding acceptance artifact above.

## Suggested order

Finish Parts 1 and 2 first because they are pure and unblock both the topology and reproducible simulator inputs. Build Part 3 next to retire the Fedora kernel risk early. Complete Part 4 with the stub before production wiring arrives. Build Part 5 alongside the runner using fixtures, then replace the fixtures with measured results when the independent run dependencies are available.

## Implementation state in this checkout (updated 2026-11-02)

- Part 1 has implementation and unit tests, including a checked-in golden vector. The five licensed upstream traces were acquired, their source commit and SHA-256 hashes recorded, and all 11 ID/OOD splits generated under `data/traces/`. Part 2 has implementation and unit tests, including a shared deterministic trace-offset path used by Jayadev's simulator.
- Part 3 has lazy-import Mininet topology/runtime code, GE measurement commands, cleanup, and root-required acceptance tests. The N=3/L2 acceptance test passed on 2026-11-02. The L1/L2/L3 100,000-packet trials passed the target-in-95%-interval check for both loss rate and mean burst; results are in `results/mininet/ge_L*.json` and the report table.
- Part 4 has a stub-policy UDP runner, replay, cross-traffic process, telemetry, logs, and cleanup. The runner fails closed if it produces no valid grant/DATA exchange; a previous smoke run accepted 14 DATA blocks across three clients with zero invalid events. The required injected-failure teardown test is still absent. The runner also does not yet accept the specification's `--config <yaml> --seed <s>` command or execute the M1 grid.
- Part 5 has raw-metric and paired calibration code with synthetic-fixture tests, scenario/grid files, a Dockerfile, Makefile, lock file, and report/setup drafts. Non-ML and ML tests passed in the current checkout during the previous verification pass; container clean-machine and Mininet reproduction still need a recorded run. Empirical M1/M-sim results remain pending and calibration requires production M1 rows plus matching Jayadev M-sim rows.
- PyTorch-only FL tests are kept behind `make test-ml`; the testbed suite does not require downloading Torch.

## Update 2026-11-12 (local, uncommitted)

- `run_mininet` accepts `--config <yaml> --seed <s>` (scenario or grid YAML; CLI overrides; grid values validated), plus `--experiment-id`, `--policy`, `--compute-window-s`. `meta.json` now records experiment id, policy, `n_rounds`, `config_id` (M1/M2), config path/hash, and the compute window actually passed to the gateway. The stub factory cannot be labelled M1/M2.
- Round state is initialised before the gateway starts (removes the start-up race). Failed runs write `run_failure.json` and no `meta.json`.
- `tests/unit/test_mininet_teardown.py` adds the injected-failure teardown tests (build failure, `verify_tc` failure, client crash with a hung gateway, spawn failure) and teardown idempotency/kill tests, all without root.
- `validate_ge measure` passes PYTHONPATH to its child processes.
- Added `configs/grids/mn_m2.yaml`, `make mininet-run`, and a proposed Fedora platform amendment in `CHANGELOG.md`.
- Still open: a production gateway factory and production client process for M1 (B1/B2/B3 adapters, B4/B6/FG-GRU providers), the M1 runs themselves, accepted M-sim rows, calibration, and the Docker clean-machine reproduction.
