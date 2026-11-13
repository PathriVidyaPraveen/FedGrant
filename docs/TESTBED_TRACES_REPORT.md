# Testbed and traces

## Testbed design

The Mininet network uses one OVS bridge. Each client has a shaped access link to the gateway, which connects to a shared 10 Mbit/s bottleneck; the cross-traffic source joins at that bottleneck. The gateway has a separate point-to-point link to the server. Interface names and IP addresses are assigned explicitly, and static ARP prevents address-resolution packets from advancing the configured loss process.

Client one-way access delay is `10 * (1 + client_id mod 4)` ms, access rate is selected cyclically from 2, 4, and 6 Mbit/s, and the measured baseline RTT is `2 * (access_delay + 5 ms)`. The uplink uses netem's two-state Gilbert–Elliott model followed by TBF; the reverse client path uses delay only. The shared bottleneck uses 5 ms netem delay followed by a 10 Mbit/s TBF. Queue limits are 32 and 64 Ethernet frames, respectively, using the project frame size `S = 1474` bytes.

The setup code reads back the qdisc state before a run begins and aborts when the expected delay, loss model, TBF rate, or queue limit cannot be verified. Hardware offloads are disabled on veth interfaces where the driver supports those controls. The UDP smoke runner uses the repository's UDP gateway/client cores and an explicitly labeled first-eligible stub policy; its results are protocol and topology checks, not production-policy M1 results.

The Fedora N=3/L2 root-required acceptance test passed on 2026-11-02 (1 passed, 8.56 s) after correcting the access-rate measurement to count receiver arrivals only during the offered interval. This closes the topology, RTT, access-rate, and shared-bottleneck acceptance gate. It does not constitute a production M1 policy result.

The stub smoke round subsequently completed with 14 accepted DATA blocks across all three clients, zero rejected DATA, and zero invalid events (`results/mininet/smoke-fixed/`). It is a protocol smoke result using synthetic profiles and the stub policy, not an M1 policy result.

## Gilbert–Elliott validation

Tables and figures in this section are regenerated from the raw trial files by `make testbed-report` (`scripts/testbed_report.py`), which writes `results/summary/testbed_*.csv`, `results/summary/testbed_tables.tex` and `results/figures/testbed_*.pdf`; reruns on the same inputs are byte-identical.


For stationary loss probability `π` and mean bad-state burst length `B`, the configured transition probabilities are `r = 1/B` and `p = πr/(1−π)`. With a 5% target, L1 uses `B = 1/0.95`, L2/L4/L5/X2 use `B = 8`, and L3 uses `B = 32`. The validator sends sequence-numbered UDP datagrams, counts missing sequence numbers, and reports empirical loss and burst statistics with dependence-aware 95% confidence intervals.

| Scenario | Target loss | Target mean burst | Packets | Measured loss (95% CI) | Measured mean burst (95% CI) | Target covered by both intervals |
|---|---:|---:|---:|---:|---:|---|
| L1 | 5% | 1.05 | 100,000 | 4.949% [4.824%, 5.079%] | 1.053 [1.047, 1.060] | yes |
| L2 | 5% | 8 | 100,000 | 5.342% [4.828%, 5.881%] | 8.057 [7.509, 8.599] | yes |
| L3 | 5% | 32 | 100,000 | 4.377% [3.501%, 5.227%] | 28.608 [24.431, 32.290] | yes |

These values are from 100,000-packet Fedora trials. Loss-rate intervals use circular moving-block bootstrap; burst-mean intervals resample complete uncensored bad runs. Each target lies within its corresponding 95% interval.

Figure: `results/figures/testbed_ge_validation.pdf` (measured loss rate and mean burst with 95% intervals against the targets).

## Trace preparation

The raw cellular `.up` files contain 1500-byte delivery opportunities with millisecond timestamps. The trace builder validates nondecreasing finite timestamps, counts opportunities in zero-anchored 100 ms bins, converts each count to bit/s, and drops the partial final bin. ID traces are split contiguously 70/15/15 into train, validation, and test; the two OOD traces are retained whole. Every output CSV has `t_ms,rate_bps` columns, and `traces_summary.json` records bin count, duration, mean rate, and 5th/95th percentiles.

The five licensed Mahimahi traces were acquired from upstream commit `f1346c38c415abfc98626dd82450f49a57033603`; `data/traces/SOURCE` records the revision, retrieval time, and per-file SHA-256 values, and `data/traces/COPYRIGHT` preserves the license. Processing produced 11 CSV splits plus `traces_summary.json`.

For each client, the selected segment is rotated by a deterministic offset from the trace child seed stream, scaled to the 2/4/6 Mbit/s target mean by client ID, and floored at 100 kbit/s. The same fixed 50-draw offset vector is used regardless of experiment client count. The checked-in golden vector fixes the offsets for clients 0–4 at seeds 0 and 1. Raw trace licensing and SHA-256 hashes are copied into the data directory by the acquisition script.

Figure: `results/figures/testbed_traces.pdf` shows the five segments that L4 (ID test splits) and L5 (OOD traces) replay, before per-client scaling; `results/summary/testbed_trace_summary.csv` lists every split's duration, mean, and 5th/95th percentile rate.

## Trace replay timing

Each client's TBF rate changes at every 100 ms bin boundary on the monotonic clock, restarting the segment at each round start, and the scheduled and actual time of every change is logged (`trace_replay_client_<i>.csv`). At N = 20 all clients change rate at the same instant, which made the implementation matter (`results/summary/testbed_replay_lag.csv`, `results/figures/testbed_replay_lag.pdf`; L4, 25 rounds, 2,000 changes per run):

| Implementation | Median lag | p99 | Max | Changes > 10 ms late |
|---|---:|---:|---:|---:|
| Threads in the runner, `tc` through the Mininet host shell | 9.0 ms | 15.4 ms | 19.9 ms | 38% |
| One agent process per client, spawning `tc` | 11.5 ms | 25.3 ms | 37.4 ms | 65% |
| One agent process per client, rtnetlink (final) | 3.5 ms | 12.3 ms | 17.0 ms | 2.0% |

Spawning `tc` costs about 2.3 ms even on an idle host, and 20 simultaneous spawns compete for the client CPUs; an rtnetlink change (pyroute2) takes 0.6–1.0 ms, sets identical TBF parameters, and returns only after the kernel acknowledged it, so the logged time is the applied time. The runner starts the agents inside the client namespaces and waits until all report ready before the gateway starts; without that wait, all 20 agents missed round 1. With 3 clients the original implementation lagged a median 4 ms.

## Experiment and analysis procedure

`make mininet-smoke` exercises a small three-client UDP round. `make test-mininet` runs the root-required RTT, access-rate, and bottleneck checks on a host with Mininet, Open vSwitch, iperf3, netem, and TBF available. The runner samples the shared bottleneck with `tc -s` every 10 ms and the gateway process every 100 ms. It records per-grant, per-packet, round, queue, process, RTT, scheduler, and trace-rate-replay events in headered CSV files with monotonic timestamps. Metadata records configured and read-back qdiscs, queue capacities, scenario loss targets, trace inputs/hashes, CPU affinity, and run times.

Round summaries report delivered energy, delivered blocks, represented clients, deadline completion measured before deadline cleanup, delivered DATA payload goodput (the full fixed-size float16 block payload, including zero padding in the final block), wasted grants, late packets, and the round's monotonic start/deadline. The analyzer derives utilization and bottleneck queue drops from cumulative TBF counters, and queueing delay from sampled backlog bytes divided by the 10 Mbit/s capacity. Per-run queue-delay mean and p99 are calculated over the pooled 10 ms samples in warm rounds. A shared TBF does not attribute drops by input flow. The cross-traffic path (xt → s1 → gw) has no loss point other than the bottleneck, and iperf3 numbers every datagram, so the runner captures cross datagrams at the gateway: each missing sequence number is one cross-traffic drop, placed in time by its send timestamp plus the calibrated pre-queue delay. FedGrant drops per round are the TBF drop total minus the cross drops in [start, D]; with no cross traffic the total is FedGrant's. Protocol overhead uses sender-side packet records and subtracts delivered payload bytes from DATA datagrams.

The raw-metrics reader applies `runs_schema.py`, excludes rounds 1–20 from per-round means, and calculates RTT, scheduler latency, and queue-delay percentiles from pooled sample-level records. The calibration tool compares the three primary metrics (eta, utilization, wasted grants) by configuration, policy, and seed and reports paired confidence intervals. Holm correction is used for the primary policy comparisons in the general statistical analysis, not for the Mininet-versus-simulator calibration intervals. Historical runs.csv schema-v1 rows remain readable, with newly added queue-delay columns left blank when the old files did not measure them.

| Item | Status |
|---|---|
| Paired M1/M-sim runs | pending: production Mininet gateway/baseline processes (Prajin) and accepted M-sim rows |
| Python 3.11 container build and test suite (`make docker-test`, podman or docker) | passed 2026-11-13 on podman 5.8: image builds from a clean context (no local data), 843 passed, 1 skipped (root-only Mininet test); `test_fl_smoke` runs separately with regenerated FEMNIST |
| Clean-machine Mininet reproduction | pending: needs a privileged container or a second Fedora host |

## Reproducibility notes and limitations

- **Platform.** Spec Section 7 names Ubuntu 22.04 with Mininet 2.3; this testbed runs on Fedora 44 with Fedora's `mininet` and `openvswitch` packages and `kernel-modules-extra` for `sch_netem`/`sch_tbf`. Topology, qdiscs and every Section 7 parameter are unchanged; the acceptance test and GE trials above were measured on this host. The change is recorded as an amendment in `CHANGELOG.md`.
- **Loss realisations.** The netem Gilbert–Elliott generator cannot be seeded (spec Section 7), so Mininet runs are not bit-identical; all other randomness (trace offsets, readiness, profiles, payloads) is seeded, and the run's configuration, trace hashes, value-bank hash and `tc` read-back are recorded in `meta.json`.
- **Cross-traffic drops.** A shared TBF cannot attribute drops by flow. Cross traffic reaches the bottleneck with no other loss point and iperf3 numbers every datagram, so a capture at the gateway identifies each cross-traffic drop and its time; FedGrant drops are the TBF total minus these (`fedgrant/mininet/cross_drops.py`). Drops before the first or after the last received cross datagram of a run cannot be detected. Validation (N = 20, 25 rounds, 12 Mbit/s cross traffic, i.e. deliberate overload above C): the capture identified 8,330 cross drops against 8,432 reported by iperf3's receiver (1.2% apart), the TBF counted 8,901 drops in total, and in no round did the attributed cross drops exceed the TBF total for that round (`results/mininet/validate-X2-cross12`). At the specified 3 Mbit/s with the stub gateway the queue never filled (peak 53 of 64 frames) and both counts were 0.
- **Value-profile bank.** Mininet clients use the frozen FULL seed-0 bank with `energies_for_run_round(seed, r)[i]`, exactly the profile the M-sim twin receives, so M1/M-sim pairs share update profiles (Section 14 pairing). Each run records the bank's SHA-256.
- **FEMNIST regeneration is machine-dependent.** Regenerating FEMNIST with the pinned script, LEAF commit and seeds on this host produced a byte-identical `provenance.json` but a different set of 200 writers (25 in common) than the bank's FULL run. LEAF's `data/utils/sample.py` reads its input files with `os.listdir()` (filesystem order) and samples them with one seeded RNG in that order, so the sample depends on the file system. The canonical FEMNIST shards are therefore the bank's, identified by the client manifest hash recorded in the bank, and are copied rather than regenerated.
- **B3 RTT sample.** B3 takes one RTT sample per client per round: first DATA arrival minus PLAN send time (spec Section 10). When send position 1 is lost the sample includes the client's pacing offset `(s-1)·8S/b̂`; Mininet and the simulator use the same `B3GatewayCore`, so this definition is identical on both platforms.

The recorded 100,000-packet GE validations and N=3 topology tests are Fedora measurements, the accepted host platform for this project. The current UDP smoke runner intentionally uses a test stub; production M1/M2 execution and paired calibration require Prajin's production gateway/baseline adapters. No M1 or production-policy claim is made from the stub output.
