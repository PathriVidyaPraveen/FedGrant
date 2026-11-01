# Testbed and traces

## Testbed design

The Mininet network uses one OVS bridge. Each client has a shaped access link to the gateway, which connects to a shared 10 Mbit/s bottleneck; the cross-traffic source joins at that bottleneck. The gateway has a separate point-to-point link to the server. Interface names and IP addresses are assigned explicitly, and static ARP prevents address-resolution packets from advancing the configured loss process.

Client one-way access delay is `10 * (1 + client_id mod 4)` ms, access rate is selected cyclically from 2, 4, and 6 Mbit/s, and the measured baseline RTT is `2 * (access_delay + 5 ms)`. The uplink uses netem's two-state Gilbert–Elliott model followed by TBF; the reverse client path uses delay only. The shared bottleneck uses 5 ms netem delay followed by a 10 Mbit/s TBF. Queue limits are 32 and 64 Ethernet frames, respectively, using the project frame size `S = 1474` bytes.

The setup code reads back the qdisc state before a run begins and aborts when the expected delay, loss model, TBF rate, or queue limit cannot be verified. Hardware offloads are disabled on veth interfaces where the driver supports those controls. The UDP smoke runner uses the repository's UDP gateway/client cores and an explicitly labeled first-eligible stub policy; its results are protocol and topology checks, not production-policy M1 results.

The Fedora host N=3/L2 topology acceptance test passed on 2026-11-01: configured RTT, client access throughput, and shared-bottleneck payload throughput met the test's acceptance bounds. This does not constitute GE statistical validation, a completed stub-runner experiment, or an M1 policy result.

The stub smoke round subsequently completed with 14 accepted DATA blocks across all three clients, zero rejected DATA, and zero invalid events (`results/mininet/smoke-fixed/`). It is a protocol smoke result using synthetic profiles and the stub policy, not an M1 policy result.

## Gilbert–Elliott validation

For stationary loss probability `π` and mean bad-state burst length `B`, the configured transition probabilities are `r = 1/B` and `p = πr/(1−π)`. With a 5% target, L1 uses `B = 1/0.95`, L2/L4/L5/X2 use `B = 8`, and L3 uses `B = 32`. The validator sends sequence-numbered UDP datagrams, counts missing sequence numbers, and reports empirical loss and burst statistics with dependence-aware 95% confidence intervals.

| Scenario | Target loss | Target mean burst | Packets | Measured loss (95% CI) | Measured mean burst (95% CI) | Target covered by both intervals |
|---|---:|---:|---:|---:|---:|---|
| L1 | 5% | 1.05 | 100,000 | 4.949% [4.824%, 5.079%] | 1.053 [1.047, 1.060] | yes |
| L2 | 5% | 8 | 100,000 | 5.342% [4.828%, 5.881%] | 8.057 [7.509, 8.599] | yes |
| L3 | 5% | 32 | 100,000 | 4.377% [3.501%, 5.227%] | 28.608 [24.431, 32.290] | yes |

These values are from 100,000-packet Fedora trials. Loss-rate intervals use circular moving-block bootstrap; burst-mean intervals resample complete uncensored bad runs. Each target lies within its corresponding 95% interval.

## Trace preparation

The raw cellular `.up` files contain 1500-byte delivery opportunities with millisecond timestamps. The trace builder validates nondecreasing finite timestamps, counts opportunities in zero-anchored 100 ms bins, converts each count to bit/s, and drops the partial final bin. ID traces are split contiguously 70/15/15 into train, validation, and test; the two OOD traces are retained whole. Every output CSV has `t_ms,rate_bps` columns, and `traces_summary.json` records bin count, duration, mean rate, and 5th/95th percentiles.

The five licensed Mahimahi traces were acquired from upstream commit `f1346c38c415abfc98626dd82450f49a57033603`; `data/traces/SOURCE` records the revision, retrieval time, and per-file SHA-256 values, and `data/traces/COPYRIGHT` preserves the license. Processing produced 11 CSV splits plus `traces_summary.json`.

For each client, the selected segment is rotated by a deterministic offset from the trace child seed stream, scaled to the 2/4/6 Mbit/s target mean by client ID, and floored at 100 kbit/s. The same fixed 50-draw offset vector is used regardless of experiment client count. The checked-in golden vector fixes the offsets for clients 0–4 at seeds 0 and 1. Raw trace licensing and SHA-256 hashes are copied into the data directory by the acquisition script.

## Experiment and analysis procedure

`make mininet-smoke` exercises a small three-client UDP round. `make test-mininet` runs the root-required RTT, access-rate, and bottleneck checks on a host with Mininet, Open vSwitch, iperf3, netem, and TBF available. Queue, gateway process, packet/action, and round-level measurements are written as headered CSV files with monotonic timestamps. Run metadata records configuration, seed, trace inputs, CPU affinity, and run times.

The raw-metrics reader applies the existing `runs_schema.py` contract and excludes rounds 1–20 from per-round means. RTT and scheduler percentiles use sample-level CSVs. The calibration tool pairs M1 and M-sim rows by configuration, policy, and seed, reports Mininet-minus-simulator differences with paired 95% t intervals, and applies Holm correction to the paired tests.

| Calibration output | Status |
|---|---|
| Paired M1/M-sim runs | pending real runs |
| Python 3.14 container build and unit suite | passed (Podman; 552 passed, 6 skipped) |
| Clean-machine Fedora Mininet reproduction | pending full runner and GE measurements |

No simulator-calibration or clean-machine result is claimed until the matching run artifacts are available and the relevant acceptance gates pass.
