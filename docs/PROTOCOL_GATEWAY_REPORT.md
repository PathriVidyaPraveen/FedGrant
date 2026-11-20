# Protocol and Gateway Report

Report-ready sections for the protocol (§1–§2) and the gateway (§3–§6), plus
gateway profiling (§7), requirement traceability (§8) and open issues (§9).
Every statement below cites code or tests in this checkout. The specification
(`docs/Networks Final Project Specification.pdf`) and `CHANGELOG.md` are the
source of requirements.

**Evidence levels used throughout.** *Implemented*: code exists. *Unit-tested*:
deterministic tests on the component. *Integration-tested*: real sockets on
localhost (`127.0.0.1`), no topology, shaping or loss. *Mininet-validated*:
run on the testbed. **No protocol or gateway behaviour in this report has been
validated on Mininet from this environment**, and no M1/M2 result is claimed.
*Testbed evidence* (§6.1) means CSV summaries produced by the testbed owner's
runs and committed under `results/summary/`; they are cited, not re-run here.
*Simulator evidence* (§5.1) means simulator runs committed with their scripts.

**Symbols.** N clients; K blocks per update (169); L FP16 coordinates per
block (700); d = 117 694 model coordinates; S = 1474-byte DATA frame on the
wire (Ethernet + IPv4 + UDP + 32-byte FedGrant DATA header + 2L payload);
C = 10 Mbit/s bottleneck; D round duration; Tc compute window; ρᵢ readiness
delay of client i; δᵢ one-way access delay; srttᵢ / rttvarᵢ RFC 6298
estimates; Φ in-flight probability mass; p̂ᵢ predicted delivery probability.

---

## 1. Wire protocol

### 1.1 Common header

Every message is one UDP payload: a 20-byte header followed by a body.
All integers are unsigned, **network byte order (big-endian)**; the header is
`struct` format `!HBBIHHQ` (`fedgrant/common/constants.py`,
`fedgrant/common/wire.py::COMMON_HEADER_FIELDS`).

| Offset | Field | Type | Meaning / constraint |
|---|---|---|---|
| 0 | `magic` | u16 | `0x4647` ("FG") |
| 2 | `version` | u8 | 1 |
| 3 | `type` | u8 | `MessageType`: 1 ROUND_START, 2 READY, 3 GRANT, 4 DATA, 5 ROUND_END, 6 PLAN |
| 4 | `round_id` | u32 | ≥ 1 |
| 8 | `client_id` | u16 | addressee (gateway → client) or sender (client → gateway) |
| 10 | `flags` | u16 | reserved, must be 0 |
| 12 | `t_send_ns` | u64 | sender `CLOCK_MONOTONIC` in ns |

### 1.2 The six messages

Body fields follow the header in the order listed (`MESSAGE_DEFINITIONS`).
Sizes are UDP payload bytes at K = 169, L = 700
(`test_wire.py::test_specified_udp_payload_lengths_at_default_block_count`).

| Type | Direction | Body fields (in order) | Payload bytes |
|---|---|---|---|
| ROUND_START (1) | gateway → client | `deadline_ns` u64 · `K` u16 (≥1) · `L` u16 (≥1) · `rho_us` u32 · `mode` u8 (`TransferMode`: 0 GRANT, 1 PUSH, 2 PLAN, 3 TCP) · `pad` 3×u8 = 0 | 40 |
| READY (2) | client → gateway | `K` u16 (≥1) · `pad` u16 = 0 · `energy` K×f32 | 24 + 4K = 700 |
| GRANT (3) | gateway → client | `grant_id` u32 (≥1) · `send_pos` u16 (≥1, 1-based) · `pad` u16 = 0 | 28 |
| DATA (4) | client → gateway | `grant_id` u32 (≥1; `0xFFFFFFFF` in PUSH/PLAN) · `send_pos` u16 (≥1) · `block_idx` u16 (0-based) · `n_coords` u16 (1..L) · `pad` u16 = 0 · `payload` L×f16 | 32 + 2L = 1432 |
| ROUND_END (5) | gateway → client | `K` u16 (≥1) · `n_delivered` u16 · `bitmap` ⌈K/8⌉ bytes | 24 + ⌈K/8⌉ = 46 |
| PLAN (6) | gateway → client (B3 only) | `k` u16 · `pad` u16 = 0 · `rate_bps` u32 | 28 |

Field semantics:

- **ROUND_START** opens round `round_id` for one client. `deadline_ns` is the
  *absolute* deadline in the gateway's monotonic clock; `rho_us` is that
  client's readiness delay ρᵢ in µs; `mode` selects the transfer mode. The
  client builds its DATA payloads with the `L` from this message.
- **READY** carries the client's K block energies in **send-position order**;
  they must be finite, ≥ 0 and non-increasing (the client sends blocks in
  decreasing-energy order).
- **GRANT** authorises exactly one DATA: the block at 1-based send position
  `send_pos`. `grant_id` is unique within the round and starts at 1.
- **DATA** carries one block: `block_idx` is the original block index,
  `send_pos` its position in the client's send order, `n_coords` the number of
  real coordinates (the last block has 94 at d = 117 694, L = 700). The payload
  is always L big-endian IEEE-754 binary16 values; coordinates beyond
  `n_coords` are zero padding.
- **ROUND_END** returns the delivered block indices as a bitmap
  (block index b ↔ bit `b mod 8` of byte `b // 8`, least-significant bit first);
  padding bits must be 0 and `n_delivered` must equal the population count.
- **PLAN** (B3) tells a client to send its first `k` blocks paced at `rate_bps`.

The protocol has no ACK message: delivery feedback is the GRANT stream during
the round and the ROUND_END bitmap after it.

### 1.3 Encoding, decoding and validation

`encode_message` and `decode_message` (`fedgrant/common/wire.py`) are the only
codec. Both validate the same body constraints (`_validate_body_values`).

- **Encoder** inserts magic/version, zeroes every reserved field and rejects
  missing or unexpected body keys, out-of-range integers, booleans used as
  integers, non-finite floats and values not representable as f16/f32. The
  encoded length is checked against the schema.
- **Decoder** rejects: short datagrams, bad magic, unsupported version or
  type, `round_id` 0, non-zero `flags` or padding, truncated fields, trailing
  bytes, and every body-constraint violation. It raises a single
  `WireProtocolError` (no `struct` exceptions leak). DATA is decoded with the
  negotiated `block_length` (L).
- **Round-trip guarantee**: for every message type, `decode(encode(m)) == m`,
  with repeated float fields returned as tuples and the body as a read-only
  mapping (`test_all_messages_round_trip`, `test_float_representations_and_zero_are_preserved`).

Wire validity is separate from application validity. A well-formed DATA can
still be rejected by `GatewayCore` (wrong round, unknown grant, wrong position,
after D, duplicate; see §4.3).

**Malformed-packet handling by layer.** `UdpGateway.datagram_received` drops
undecodable datagrams (`stats.malformed`), message types other than READY/DATA
(`unexpected_type`), unknown client ids (`unknown_client`) and a datagram whose
source endpoint differs from the one bound to that client (`endpoint_conflict`).
`UdpClient` likewise counts `malformed`, `wrong_endpoint`, `wrong_client`,
`unexpected_type` and `stale_or_invalid`. No invalid input raises out of a
datagram callback.

**B1 framing (not one of the six messages).** B1 runs over TCP
(`fedgrant/baselines/b1_tcp.py`): a 10-byte hello `!4sIH` ("FGB1", `round_id`,
`client_id`) and then fixed records `!HH` (`block_idx`, `n_coords`) + 2L
big-endian f16 = 1404 bytes per block, with the same zero-padding rule.

---

## 2. Message sequence of a normal round (GRANT mode)

The production Mininet path is `fedgrant.mininet.gateway_proc` (gateway) and
`fedgrant.mininet.client_proc` (clients).

```
gateway (GatewayCore + UdpGateway)                         client i (ClientCore + UdpClient)
  wait Tc = 1.0 s
  on_round_start(t0, D, rho_us)  ── ROUND_START(deadline, K, L, rho_i, GRANT) ──▶  receipt time r_i captured;
                                                                                  start_round(): encode, energies
                                 ◀───────────── READY(energies) ──────────────── sent at r_i + rho_i
  on_ready: validate, val_i = w_i^2 * energy,
  pump → eligibility → selector → pacing → GRANT
                                 ───────── GRANT(grant_id, send_pos) ─────────▶  data_for_grant(send_pos)
                                 ◀──── DATA(grant_id, send_pos, block_idx, …) ── sent immediately
  on_data: validate, deliver, resolve grant,
  pump → next GRANT(s)            …  (repeats; 1 ms ticks expire timeouts and pump) …
  at D: resolve outstanding as losses
                                 ──────── ROUND_END(K, n_delivered, bitmap) ────▶  finish_round(bitmap): error feedback
  drain until D + 0.1 s (late DATA counted, never delivered)                      (no ROUND_END by D + 200 ms →
  finish_round → metrics; idle until next Tc                                       all blocks treated as undelivered)
```

**B2 (PUSH)**: no READY or GRANT. After ρᵢ the client sends DATA in send order
at its configured rate with `grant_id = 0xFFFFFFFF`. **B3 (PLAN)**: the client
sends READY; at t_plan = 0.1·D the gateway sends one PLAN(kᵢ, b̂ᵢ) per
client, and the client sends its first kᵢ blocks paced at b̂ᵢ, with no
retransmission. Both end with ROUND_END.

---

## 3. Gateway architecture

```
               fedgrant.mininet.gateway_proc  (process; Tejaswi)
                 │ build_context → contract.build_gateway(--gateway-factory, ctx)
                 ▼
 fedgrant.mininet.production:create_gateway ──▶ GatewayCore  (fedgrant/core/gateway.py)
   LiveClockGatewaySim providers:                   │  on_round_start / on_ready(_message) /
   eligibility, selector, pacing, timeout,          │  on_data(_message) / on_tick / on_deadline
   estimator hooks, predictor, aggregate            │  → actions: SendRoundStart, SendGrant,
                                                    │            SendRoundEnd, EmitAggregate
                 round_loop.run_rounds              ▼
   + gateway_proc.GrantRoundProtocol ──▶ LoggingUdpGateway ─▶ UdpGateway (fedgrant/net/udp_gateway.py)
   (Tc, ROUND_START, D, drain,            (raw CSV logs)       asyncio UDP socket, endpoint table,
    round_state file, metrics)                                  1 ms tick loop, action encoding/sending
```

- **`GatewayCore`** is the socket-free state machine of Algorithm 1. It owns
  round state (`RoundContext`, `GatewayPhase` IDLE/ACTIVE/DEADLINE), per-client
  `ClientState` (status per send position, outstanding grants, srtt/rttvar,
  b̂, cap), grant ids, delivered blocks and payloads, and the counters in
  `GatewayStats`. Policy behaviour enters only through injected providers:
  `EligibilityProvider`, `Selector`, `PacingProvider`, `TimeoutProvider`,
  `EstimatorHooks`, a predictor or `PredictionProvider`, `AggregateProvider`,
  and an optional `ValueIntegrityChecker`.
- **Production providers** (`fedgrant/mininet/production.py`) reuse the
  simulator's `GatewaySim` provider assembly, so Mininet and simulator twins
  schedule identically; `LiveClockGatewaySim` only reads the round origin and
  D from the live core. Construction raises on a missing model or value bank;
  the runtime P-EWMA fallback is `FallbackPredictor`.
- **`UdpGateway`** decodes datagrams, enforces endpoint binding, passes the
  gateway receive time (`received_at_ns`) to the core, encodes and sends the
  returned actions, and runs the tick loop. `LoggingUdpGateway`
  (`fedgrant/mininet/logging_tap.py`) adds the raw packet, grant, RTT and
  scheduler CSV logs.
- **Selectors** (`fedgrant/policy/selectors.py`): `value` (argmax ṽᵢ, Eq. 13;
  B6, FG-GRU), `rr` (cyclic pointer, B4; the pointer advances only when a grant
  is issued), `srpt` (B5).
- **Aggregation**: at the deadline the core emits `EmitAggregate` with the
  provider's aggregate and per-client coordinate counts. `UdpGateway` hands it
  to an `aggregate_handler` or `aggregate_sender`. The FL transfer path is
  exercised over localhost by `test_fl_round_transport.py`. For network-only
  runs (M1, smoke) `gateway_proc.serve` installs `NetworkOnlyAggregateSink`,
  which writes one line per round to `aggregates.jsonl` (round id, dimension,
  clients, delivered coordinates) and does not apply the aggregate to a model
  (`test_run_m1.py::test_network_only_aggregate_sink_records_each_round`).
  M2 still needs a real sender to the FL server (§9).
- **Probe logging** (CHANGELOG 2026-11-14 eligibility amendment). The raw
  grant log gains a `probe` column, 1 on the *issued* row of a probe grant and
  0 otherwise (`logging_tap._probe_flag` → `GatewayCore.is_probe_grant`).
  `round_metrics.csv` gains `probe_grants`, the round's count
  (`GatewayCore.round_probe_grants`), on the Mininet path
  (`gateway_proc.finish_round`) and the simulator path
  (`scripts/run_grid.py`, which prefers a simulator `probe_grants` metric when
  one exists). Both columns are appended last in `common/logschema.py`, and both
  are blank for B1–B3 or for a core without probe support
  (`test_probe_logging.py`, `test_run_grid.py::test_*probe*`).

---

## 4. Round lifecycle

### 4.1 Phases

`run_rounds` (`fedgrant/mininet/round_loop.py`) drives each round and
publishes the round-state file read by trace replay and the samplers:

1. **Idle / Tc**: wait `compute_window_s` (M1: Tc = 1.0 s).
2. **Active [start, D]**: `on_round_start` resets per-round state
   (`ClientState.reset_round`, grant ids from 1, delivered sets, resolved and
   timed-out grant maps, predictor and estimator round hooks) and returns one
   ROUND_START per client; the state file is set active before they are sent.
3. **Deadline**: the first `on_tick` at t ≥ D (or `GrantRoundProtocol.at_deadline`,
   whichever runs first) calls `on_deadline`: every outstanding grant is
   resolved as a loss (estimator notified), starvation counters are updated,
   one ROUND_END bitmap per client and `EmitAggregate` are returned, and the
   phase becomes DEADLINE (a second call is a no-op).
4. **Drain (D, D + late_drain]** (M1: 0.1 s): sockets and logs stay open; the
   state file is in the *drain* phase (`drain_until_ns`). Same-round DATA is
   counted as `late_after_deadline`, never delivered.
5. **Finish**: `finish_round` writes the round metrics row; the state becomes
   idle; the next round starts after the next Tc.

`round_at(t)` returns a round id only on [start, D], so per-round metrics
cover [0, D]; `phase_at(t)` labels drain samples. B1/B2/B3 use the hook-based
`run_round_loop`, which publishes the same active → drain → idle phases and
also clears the active flag if a round hook fails or is cancelled.

### 4.2 READY registration and grant generation

`on_ready` rejects READY outside [start, D) or outside the ACTIVE phase, from unknown clients, with a
wrong K, non-finite, negative or increasing energies (`rejected_ready`); a
second READY in the round is counted as `duplicate_ready`. A valid READY sets
`valᵢ = wᵢ² · energy` (Eq. 2, `values_from_energies`) and calls `_pump`.

`_pump` (GRANT mode only, t < D) loops: `eligibility.candidates(...)` →
`selector.select` → p̂ᵢ → `pacing.allow`. A blocked pacing test ends the loop.
Otherwise the core takes the next send position, the timeout τ_g, allocates
`grant_id`, inserts the outstanding grant and its probability into Φ, and
emits `SendGrant`.

**Bounded probe (rule (v) amendment, CHANGELOG 2026-11-13; `POLICY_REVISION = "2026-11-17-probe-minrtt"`).**
A client that passes rules (i)–(iv) and fails only rule (v) (p̂ᵢ < θ) may get
at most one *probe* grant per round, so its predictor can recover from a
lockout. A probe still goes through the selector and pacing and is accounted
like any other grant. `is_eligible(..., probe_available=True)` returns
`Reason.PROBE` for such a client (`policy/eligibility.py`). A NaN p̂ᵢ is never
a probe; it fails rule (v). The simulator's eligibility provider
(`sim/run_net.py::_CoreEligibility`) passes
`probe_available=core.probe_available(i)` and sets the optional
`CandidateView.probe` mask from `Reason.PROBE`. `CandidateView` validates the
mask as a 1-D bool array aligned with the candidates. The core classifies a
grant as a probe only when the mask is set **and** p̂ᵢ < θ. The allowance is consumed at issue,
after the outstanding grant and its Φ entry exist, so a probe blocked by
pacing keeps its allowance. Offering a probe after the allowance is used
raises `GatewayError` without mutating state. The allowance resets only at a
valid ROUND_START and survives the deadline and drain. Ablation A4 (no
rule (v)) never probes. Decision records carry a `probe` field. Tests:
`test_gateway.py::test_t1…t10`, `test_probe_mask_must_align_with_candidates`,
`test_policy_amendments.py`, `test_candidate_view_rejects_malformed_probe_mask`.
A provider that supplies no mask gets only normal grants
(`test_t7_missing_probe_mask_means_every_grant_is_normal`). The production
gateway inherits the simulator's provider through `LiveClockGatewaySim`.
`run_mininet.probe_wired()` checks functionally that a client with p̂ = 0
receives exactly one probe; the result is recorded in each run's
`meta.json`, and `scripts/run_m1.py` refuses grant runs without it.

### 4.3 DATA validation and delivery

Order of checks for a wire DATA (`GatewayCore.on_data_message` → `on_data`):

1. decode with the round's L; DATA type; known client
   (`bad_client_id`);
2. **round identity**: no current round, or `round_id` ≠ current round →
   rejected and counted once in **`stale_round_data`** (also `rejected_data`,
   `invalid_events`). Nothing in the round, grants, deliveries or the aggregate
   is touched;
3. receive time > D → **`late_after_deadline`** (same round, drain), returned
   without delivery;
4. phase must be ACTIVE; time, grant id, `send_pos` ∈ [1, K],
   `block_idx` ∈ [0, K), payload of exactly L finite numbers,
   `n_coords` ∈ [1, L];
5. GRANT mode: the grant must be outstanding (or previously resolved) for this
   client with the same `send_pos`; PUSH/PLAN: `grant_id` must be the sentinel;
6. already-delivered position → `duplicate_data` (no second delivery);
7. otherwise the block is delivered once (`accepted_data`), its payload and
   `n_coords` stored, the grant resolved (estimator `resolve(…, delivered=True)`)
   and `_pump` called. DATA for a grant whose timeout had already passed is
   still delivered but counted as `late_after_timeout`, and the estimator is
   not given a success for it.

### 4.4 Timeouts, retransmission and failures

- On each tick, outstanding grants with `timeout_ns ≤ t` are resolved as
  losses (`mark_timeout`: the position becomes LOST). With
  `PolicyConfig.retransmit = True` (default; ablation A2 disables it) a LOST
  position is eligible again and gets a new grant id. A grant resolved before D
  is not resolved again at D
  (`test_grant_timed_out_before_deadline_is_not_resolved_again_at_deadline`).
- Client-side failures: a client that never receives ROUND_END finishes the
  round at D + 200 ms with every block treated as undelivered
  (`UdpClient._finish_round_timeout`); READY is skipped (counted) if its time
  falls at or after D.
- Gateway-side robustness: invalid events are counted and dropped; transport
  callbacks catch `GatewayError`/`ValueError`/`TypeError`; a failing tick is
  logged and counted (`tick_errors`) and the loop continues.

---

## 5. Timing, deadlines and scheduling

| Quantity | Definition (source) | M1 value |
|---|---|---|
| D | `round(β · T_full)`, T_full = 3.985 s · (N/20) · (K/169) (`sim/run_net.py::compute_deadline_ns`) | β = 0.10, N = 20: 398 ms |
| Tc | compute window before each ROUND_START (`RoundTiming.compute_window_s`) | 1.0 s |
| late drain | `RoundTiming.late_drain_s` | 0.1 s |
| ρᵢ | U(0, 0.1·D) ns, child 2 of `SeedSequence(network_round_seed(seed, r)).spawn(6)`, floored to µs (`round_loop.readiness_delays_us`) | seeded |
| READY time | client sends at (ROUND_START receipt) + ρᵢ; encoding time is not added (`UdpClient._handle_round_start`, `loop.call_at`) | — |
| Simulator READY arrival | (δᵢ + 5 ms) + ρᵢ + (δᵢ + 5 ms) (`run_net.py`, CHANGELOG 2026-11-13) | — |
| Tick | `UdpGateway` stable-cadence loop, 1 ms grid, never overlapping, overruns skip to the next grid point | 1 ms |
| Prediction refresh | all ready clients when a tick enters a new 5 ms bucket since round start, plus one client per resolution | 5 ms |
| Eligibility | rules (i)–(v): READY, pending position, outstandingᵢ < capᵢ, **t + srttᵢ + 2 ms ≤ D**, p̂ᵢ ≥ θ = 0.05 (`policy/eligibility.py`) | — |
| Timeout | τ_g = min(D, t_g + srttᵢ + max(4·rttvarᵢ, 10 ms)) (Eq. 10, `core/estimators.py::grant_timeout`) | — |
| Cap | capᵢ = max(4, ⌈1.25 · b̂ᵢ · srttᵢ / (8S)⌉ + 2) (Eq. 11) | — |
| Pacing | issue iff Φ + p̂ᵢ ≤ P_target, P_target = C · (RTT~ + 5 ms) / (8S). Under the min-RTT amendment (CHANGELOG 2026-11-17), RTT~ is the median, over ready clients with at least one sample, of each client's minimum RTT sample (`ClientState.rtt_min_ns`, set in `estimators.rtt_update`, kept across rounds); it is 100 ms if no ready client has a sample (Eq. 12, `policy/pacing.py::p_target_min_rtt`). B3 keeps median srtt. | ≈ 57–60 packets in late rounds (§5.1) |
| Client fallback | ROUND_END missing by D + 200 ms (`ROUND_END_GRACE_NS`) | 200 ms |

Timestamp origins: the header `t_send_ns` is the sender's clock and is used
only as a fallback; the gateway uses its own receive time for READY and DATA.
ROUND_START's `deadline_ns` is in the gateway's monotonic clock; clients
compare it with their own `time.monotonic_ns()`, which is the same clock only
when gateway and clients share a kernel (one Mininet host, or localhost).

**ρᵢ in the Mininet grant path (verified in code).** `GrantRoundProtocol.begin_round`
passes `readiness_delays_us(seed, r, N, D)` as `rho_us`, and `UdpClient`
waits ρᵢ from ROUND_START receipt. The B1/B2/B3 processes use the same draws.
Mininet READY packets bypass GE loss but keep the access delay
(`tcsetup.py`, `ready_lossless`). None of this has been run on the testbed
from this environment.

### 5.1 Rule (v) lockout, the probe and min-RTT pacing (simulator evidence)

Rule (v) is absorbing without a probe. A client whose p̂ᵢ falls below θ gets
no grants, so its predictor gets no new observations and p̂ᵢ cannot recover.
This was found in the profiling harness: at N = 80 the issued-grant count
collapses round by round (§7.1). The `current` column below shows the same
lockout at M1 settings.

`results/summary/eligibility_variants_sim/` compares seven variants, each
applied in memory to B4 (rr/EWMA) and B6 (value/EWMA) as M1 configures them.
The configuration is N = 20, β = 0.10, 100 rounds, seeds 0–4, and loss
scenarios L2 and L4. The scripts are `elig_variants.py` and
`elig_summary.py`; "late" means rounds 51–100. Every job runs in a fresh
process. A first run reused worker processes, so min-RTT patches leaked into
later jobs; it is kept under `superseded_worker_patch_leak/` and is not
cited. Means over seeds:

| Late rounds | current | minrtt | starve | minrtt+starve | probe | **minrtt+probe** | noRuleV (A4) |
|---|---|---|---|---|---|---|---|
| B4 L2: η | 0.203 | 0.192 | 0.449 | 0.429 | 0.448 | **0.554** | 0.442 |
| B4 L4: η | 0.008 | 0.007 | 0.388 | 0.419 | 0.458 | **0.503** | 0.458 |
| B6 L2: η | 0.187 | 0.403 | 0.439 | 0.561 | 0.454 | **0.593** | 0.454 |
| B6 L4: η | 0.003 | 0.004 | 0.382 | 0.372 | 0.451 | **0.525** | 0.459 |
| B4 L2: queue drops per round | 15.8 | 1.1 | 51.8 | 13.0 | 59.3 | **1.9** | 65.7 |
| B4 L4: queue drops per round | 0.0 | 0.0 | 19.1 | 0.5 | 34.8 | **3.7** | 36.2 |
| B6 L2: queue drops per round | 14.1 | 0.0 | 55.0 | 3.9 | 56.3 | **0.4** | 63.6 |
| B6 L4: queue drops per round | 0.0 | 0.0 | 18.7 | 0.7 | 36.1 | **1.5** | 37.2 |
| B6 L4: clients granted per round (of 20) | 0.1 | 0.1 | 15.4 | 13.5 | 20.0 | **20.0** | 20.0 |
| Rounds with no grant, L4 (of 100; B4 / B6) | 62.8 / 83.0 | 62.4 / 80.8 | 0 / 0 | 0 / 0 | 0 / 0 | **0 / 0** | 0 / 0 |

Reading:

- **The lockout is real.** Under the current rule (v), B4 and B6 grant almost
  no client in L4: η ≈ 0 in late rounds, and 63–83 of 100 rounds issue no
  grant at all.
- **The probe alone fixes the lockout but not the queue.** All 20 clients
  are granted again, and η rises to 0.45–0.46, the same as with rule (v)
  removed (A4). But it leaves 35–59 queue drops per round, close to A4's
  36–66. The cause is Eq. 12 with median srtt: srtt includes the queueing
  delay pacing is meant to bound, so P_target grows with the queue, to about
  114–116 packets.
- **Probe + min-RTT is the best variant in every cell.** η is 0.50–0.59,
  which is 0.77–0.91 of the per-round upper bound, against about 0.69–0.70
  for probe alone. Queue drops fall to 0.4–3.7 per round, and the queue's
  p99 delay falls from about 76 ms to 36–42 ms. P_target settles near 57–60
  packets. Probes are under 1 % of grants.
- **min-RTT alone does not fix the lockout** in L4, and the starvation
  override (`starve`) leaves 1–5 clients per round ungranted while adding
  queue drops.

**Adopted** (CHANGELOG 2026-11-13 and 2026-11-17;
`POLICY_REVISION = "2026-11-17-probe-minrtt"`): the bounded probe (§4.2) and
min-RTT pacing (§5, Pacing row). Both are wired into the simulator
(`sim/run_net.py`), and production inherits them through
`LiveClockGatewaySim`.

The integrated implementation reproduces the clean probe-only study:
`results/summary/probe_validation_sim/probe_summary.csv` gives late η of
0.451–0.457, 20/20 clients, at most one probe per client per round, and
`round_probe_grants` equal to the logged probes. Its `probe_validation.csv`
compares against the superseded study, which is how the leak was found.

ML data and the P-GRU/P-MLP models are regenerated and retrained under the
new revision. `ml/datagen.py` refuses to generate data unless the simulator
functionally issues probes and follows min-RTT.

---

## 6. Baselines and accounting

| | B1 | B2 | B3 |
|---|---|---|---|
| Transport | TCP, one connection per client per round (`b1_tcp.py`) | UDP DATA push, sentinel grant id (`b2_udp.py`) | UDP READY → PLAN → paced DATA (`b3_plan.py`, `b3_pacing.py`) |
| Gateway | `B1TcpGateway`, `B1RoundAccumulator` | `B2RoundAccumulator` | `B3GatewayCore`, `B3UdpGateway` |
| Retransmission | TCP's own | none | none |

- **B3 RTT** (Section 10): one sample per client per round = arrival time of
  the client's first *accepted* DATA − the PLAN send time. The send time is
  stamped immediately before `sendto` and recorded only if the send succeeds
  (`mark_plan_sent`). Nothing is subtracted for the pacing offset, so a sample
  produced by position 2 after position 1 was lost includes the offset.
  Duplicate, late, pre-PLAN DATA and clients without a successful PLAN produce
  no sample. Samples are written with `rtt_type = plan_data` and the producing
  `send_pos` (`rounds.csv`, FL history `plan_data_rtt_send_pos`).
- **Goodput** (Section 12): unique delivered blocks × 2L bytes × 8 / D, i.e.
  the full padded 1400-byte payload per block including the zero padding of
  the last block; duplicates and failed sends are not counted. The same rule is
  used by the grant path (`gateway_proc.finish_round`), the baseline processes
  and the simulators.
- **DATA classification** in the Mininet baseline gateway
  (`baseline_proc.BaselineGateway._classify`): accepted; same-round after D →
  `late_after_deadline`; anything else (duplicate, wrong round, invalid) →
  rejected. READY is logged but not counted in the DATA counters. The B2 core
  keeps `rejected_rounds` (stale) separate from `rejected_late_packets`.
- **Transfer records**: each B3 client round produces one immutable
  `B3ClientRoundRecord` with `transfer_status` ∈ {no_plan, completed,
  cancelled, failed}; partial transfers keep exact sent counts.

### 6.1 B3 READY cutoff margin (testbed evidence)

B3 currently plans at t_plan = 0.1·D (`policy/plan.py::plan_time_ns`).
Since ρᵢ ~ U(0, 0.1·D) and a READY reaches the gateway only after
ROUND_START delivery (δᵢ + 5 ms), ρᵢ and the READY's own path (δᵢ + 5 ms),
a client with a large ρᵢ or δᵢ misses the plan and gets k = 0. That is a
property of the cutoff, not of the client. The proposed cutoff covers the
whole READY arrival bound with δ_max = 40 ms:
t_plan = 0.1·D + 2(δ_max + 5 ms) + margin, which is 129.8 ms + margin at
β = 0.10, N = 20 (D = 398 ms).

The summary is `results/summary/testbed_b3_ready_cutoff.csv`, generated by
`scripts/testbed_report.py::b3_ready_cutoff` from the testbed's packet
logs. A round counts as affected if any READY misses the cutoff.

| READY path (runs) | margin 0 | 5 ms | 10 ms | **15 ms** | 20 ms | 30 ms |
|---|---|---|---|---|---|---|
| Quiet: no DATA ahead of READY, as in B3 before t_plan (500 READYs, 25 rounds) | 2.00 % / 28 % | 0 / 0 | 0 / 0 | **0 / 0** | 0 / 0 | 0 / 0 |
| Loaded, M1 B4/B6 L2: READY queues behind granted DATA | 7.25 % / 64.8 % | 4.95 / 44.5 | 3.57 / 33.3 | **2.42 / 24.8** | 1.63 / 17.8 | 0.78 / 9.5 |
| Loaded, M1 B4/B6 L4 | 3.85 % / 42.8 % | 1.79 / 14.5 | 1.33 / 10.8 | **0.97 / 8.0** | 0.74 / 6.5 | 0.41 / 4.2 |

Each cell is READYs missed / rounds affected.

On the quiet path, which is the one that applies to B3, READY lateness beyond
the bound is p50 2.97 ms, p99 7.30 ms and max 8.64 ms. A margin of 5 ms
already misses nothing; the proposed **15 ms** leaves more than 6 ms of
headroom over the observed maximum. The loaded rows are not B3 conditions:
there, READYs queue behind granted DATA. They show that no fixed margin
would cover a loaded READY path.

Consequences that are still open (§9): the B3 budget
M = ⌊C(D − t_plan − RTT~)/8S⌋ shrinks as t_plan moves later, and the
per-client caps use the nominal plan offset. At β = 0.05 the later cutoff
leaves no feasible budget under the current formulas: remaining time minus
the initial median RTT is −10.9 ms. The simulator diagnostic of the fixed
0.1·D + 90 ms cutoff (`docs/M1_MSIM_AUDIT_2026-11-13.md` §4.1) is the
evidence. Every READY registered. L4 at β = 0.10 delivered 86.0 % of planned
blocks, and 83 % of the shortfall was DATA arriving after D. L2 at β = 0.20
lost 25.7 % of planned blocks in the network, which is a separate problem. The
cutoff is
**not changed in code** yet. The agreed reporting conventions and the
remaining vote are in §9, item 9.

---

## 7. Gateway profiling

Two tools exist; they measure different boundaries.

### 7.1 Localhost profiling harness (Task C)

`fedgrant/profiling/gateway.py` profiles the **production grant gateway** over
**localhost UDP** for several client counts. It is not a Mininet measurement.

- *Gateway process*: `contract.build_gateway("fedgrant.mininet.production:create_gateway", ctx)`
  with the chosen policy and uniform weights, the real `UdpGateway` tick loop
  and the production round lifecycle (`run_rounds` +
  `GrantRoundProtocol`: Tc, seeded ρᵢ, D, drain).
- *Clients*: N `UdpClient`s with synthetic updates in **one separate process**
  (so client CPU never counts as gateway CPU). Each client delays its uplink
  datagrams by the spec access RTT 2(δᵢ + 5 ms) = 30/50/70/90 ms; there is no
  bandwidth limit, loss or queueing.
- *Excluded*: Mininet/tc/GE, `LoggingUdpGateway`'s per-packet CSV logging, and
  the FL aggregate transfer (the aggregate is computed at D and discarded).
- *Probes*: a `UdpGateway` subclass plus per-instance wrappers of
  `GatewayCore._pump`, `on_tick` and `on_data_message`. Samples are kept in
  memory and written after the run. The only change to production code is
  `UdpGateway.scheduled_tick_ns`, which records the grid time the loop was
  waiting for; nothing reads it back.

| Metric | Interval | Clock |
|---|---|---|
| Gateway CPU | 100 · Δ(user + system CPU time) / Δwall, per process, % of one logical CPU; over the whole measurement window and over active [start, D] intervals only | `psutil`, monotonic |
| Gateway RSS | resident set size (not VMS): baseline = median between socket bind and round 1; peak and median during measurement | `psutil` |
| Tick scheduling delay | actual tick start − scheduled grid time (signed) | monotonic |
| Tick execution | whole `_run_tick` (core `on_tick` + sending its actions); `tick_core` = `on_tick` alone. The tick that reaches D runs `on_deadline` (resolution, ROUND_END, production aggregate), so it is included and dominates the maximum | perf counter |
| Grant decision | duration of a `_pump` call that issued ≥ 1 grant (eligibility, selection, prediction, pacing, timeout); also per grant, and pumps that issued none | perf counter |
| Grant send | encode + `sendto` of one GRANT; decision end → GRANT handed to the socket | perf counter |
| DATA processing | datagram callback entry → `on_data_message` return (adapter decode + core); `data_core` = core only; `data_validation` = core − decision | perf counter |

Kernel receive queueing and network time are outside every interval.
Percentiles need ≥ 1 / 10 / 20 / 100 samples (p50 / p90 / p95 / p99); otherwise
they are `null`. Missing values stay empty. Warm-up rounds are excluded by
round id.

Command (defaults: N = 5 10 20 40 80, 12 rounds with 2 warm-up, policy B6,
seed 0, β = 0.10, Tc = 1.0 s, drain 0.1 s, 1 ms tick, 50 ms CPU/RSS sampling,
`spec_rtt` delay, L = 700):

```sh
.venv/bin/python -m fedgrant.profiling.gateway run
.venv/bin/python -m fedgrant.profiling.gateway run --clients 20 --rounds 30 --warmup-rounds 5 --policy FG-GRU
```

Other options: `--seed`, `--beta`, `--compute-window-s`, `--late-drain-s`,
`--tick-interval-ns`, `--sample-interval-s`, `--delay-model {spec_rtt,none}`,
`--block-length` (non-default L departs from the spec), `--models-dir`,
`--output-root`. Each run creates a new directory
`results/raw/gateway_profile/<UTC time>-<git rev>-<random>/` (never reused)
with `manifest.json` (config, git revision and dirty flag, environment,
boundaries), `summary.json`, `summary_resources.csv`, `summary_latency.csv`,
`summary.md`, and one `nNNN/` directory per N holding the raw samples
(`processes.csv`, `tick_samples.csv`, `pump_samples.csv`, `grant_samples.csv`,
`data_samples.csv`), `round_metrics.csv`, `gateway_summary.json`,
`clients.json` and process logs. `results/raw/` is git-ignored.

**Localhost measurement, run `20261113T132558Z-a4b5aa07-b65cf4`** (default
configuration, B6; WSL2 on a 16-logical-CPU laptop, Python 3.14; working tree
dirty at a4b5aa07). These are localhost numbers on a development machine, not
testbed or M2 results. Latencies are in µs, rounds 3–12.

| N | D (ms) | Gateway CPU, active / window mean (%) | RSS base → peak (MiB) | Tick delay p50 / p99 | Tick exec p50 / p99 | Grant decision p50 / p99 | DATA core p50 / p99 | DATA validation p50 |
|---|---|---|---|---|---|---|---|---|
| 5 | 100 | 8.0 / 4.6 | 46.3 → 50.2 | 630 / 3350 | 80 / 312 | — (no grants) | — | — |
| 10 | 199 | 59.3 / 12.8 | 46.4 → 51.5 | 809 / 2709 | 89 / 565 | 152 / 1017 | 458 / 930 | 329 |
| 20 | 398 | 71.3 / 24.3 | 46.4 → 53.7 | 825 / 2317 | 109 / 1037 | 173 / 785 | 482 / 938 | 326 |
| 40 | 797 | 83.2 / 41.9 | 46.5 → 58.6 | 927 / 2326 | 142 / 1993 | 212 / 1067 | 538 / 1011 | 335 |
| 80 | 1594 | 96.3 / 60.8 | 46.3 → 68.1 | 1094 / 2204 | 183 / 1523 | 258 / 631 | 591 / 997 | 336 |

What the run shows, without claiming more:

- At N = 5, D = 100 ms < srtt₀ + 2 ms = 102 ms, so eligibility rule (iv)
  admits no grant before the first RTT sample, and none can arrive without a
  grant: no grant or DATA samples. This follows from the spec parameters.
- Tick scheduling delay is ~0.6–1.1 ms at the median even when idle (N = 5).
  Python's `EpollSelector` rounds every timeout up to a whole millisecond, so
  the 1 ms tick loop waits up to one extra millisecond and then skips grid
  points. This applies to the production `UdpGateway` on Linux, not only to
  the harness.
- DATA validation inside `GatewayCore` (~0.33 ms median) does not grow with
  N. Timed in isolation on the same host, `GatewayCore`'s own decode of a DATA
  takes ~66 µs and `on_data`'s per-value payload type/finiteness checks
  ~190 µs, which together account for most of it. `UdpGateway` decodes the
  datagram once more before calling the core; `data_rx_to_core_done` −
  `data_core` ≈ 0.2 ms covers that decode and callback overhead.
- Active-phase gateway CPU rises with N and approaches one full core at
  N = 80 on this host. The client process stayed below the 90 % saturation
  flag (p95 80 % at N = 80).

### 7.2 Mininet profiling path

`run_mininet --profile-gateway` attaches `telemetry.ProcessSampler` to the
gateway process (`processes.csv`), records datagram/tick-to-GRANT-dispatch
latency in `scheduler_samples.csv` (`LoggingUdpGateway`), and writes
`gateway_profile.json` excluding the first 20 rounds. That path includes the
per-packet CSV logging. It requires the Mininet host and root; **it has not
been run from this environment**, so no testbed profile or M2 measurement
exists yet.

---

## 8. Requirement traceability

Status: **U** unit-tested, **I** integration-tested on localhost,
**M** Mininet-validated. No item is M.

| Requirement | Implementation | Test / evidence | Status |
|---|---|---|---|
| Six message definitions, header layout | `common/wire.py` `MESSAGE_DEFINITIONS`, `COMMON_HEADER_FIELDS` | `test_wire.py::test_all_six_message_body_definitions`, `test_common_header_layout`, `test_specified_udp_payload_lengths_at_default_block_count` | U |
| Network byte order, exact bytes | `encode_message` | `test_round_start_exact_network_order_bytes`, `test_ready_data_and_round_end_exact_variable_layouts` | U |
| Round trip | `encode_message` / `decode_message` | `test_all_messages_round_trip`, `test_float_representations_and_zero_are_preserved` | U |
| Malformed rejection | `decode_message`, `UdpGateway.datagram_received`, `UdpClient.datagram_received` | `test_decoder_rejects_*`, `test_round_end_bitmap_invariants_are_checked`, `test_udp_gateway.py::test_malformed_unexpected_and_conflicting_endpoint_packets_are_dropped`, `test_udp_client.py::test_malformed_wrong_client_and_unexpected_messages_are_ignored` | U |
| Round / client validation | `GatewayCore.on_*_message`, `_known_client` | `test_gateway.py::test_round_start_message_rejects_wrong_client_and_duplicate_or_stale_rounds`, `test_ready_rejects_unknown_or_invalid_events_without_mutating_state` | U |
| Stale-round DATA rejected and counted | `on_data_message`, `GatewayStats.stale_round_data` | `test_previous_round_data_is_stale_and_mutates_nothing`, `test_future_and_unrelated_round_data_count_once_per_packet`, `test_data_before_any_round_started_is_stale`, `test_stale_and_late_counters_are_cumulative_across_round_transitions` | U |
| Same-round late DATA | `on_data` (`late_after_deadline`) | `test_same_round_data_in_drain_is_late_not_stale`, `test_wire_data_uses_gateway_receive_timestamp_when_provided`; B2 process: `test_b2_gateway_counts_late_data_during_drain_without_delivering_it` | U, I |
| Grant issuance and validation | `GatewayCore._pump`, `on_data` | `test_ready_installs_weighted_values_and_issues_first_grant`, `test_data_at_grant_timeout_boundary_is_classified_by_arrival_time`; `test_udp_gateway.py::test_ready_dispatch_uses_gateway_receive_time_and_sends_grant` | U |
| Timeout and retransmission | `on_tick`, `_resolve_timeout`, `grant_timeout` | `test_timeout_marks_lost_and_retransmits_through_state_transition`, `test_grant_timeout_is_capped_at_round_deadline`, `test_grant_timed_out_before_deadline_is_not_resolved_again_at_deadline`, `test_estimators.py` | U |
| Deadline completion, ROUND_END, aggregate | `on_deadline`, `UdpGateway` | `test_early_deadline_callback_is_rejected_without_ending_round`, `test_udp_gateway.py::test_scheduler_deadline_and_close_cleanup`, `test_fl_round_transport.py::test_full_round_broadcast_udp_and_aggregate_transfer_loopback` | U, I |
| Round transitions | `on_round_start`, `run_rounds`, `run_round_loop` | `test_round_reset_clears_timed_out_grants_and_rejects_old_round_data`, `test_mininet_round_loop.py`, `test_sim_cross_round.py` | U, I |
| Full localhost round (grant mode) | `UdpGateway`, `UdpClient` | `test_udp_client_round.py::test_localhost_udp_round_from_round_start_through_error_feedback`; production factory through `gateway_proc.serve`: `test_mininet_gateway_integration.py` | I |
| READY timed from ROUND_START receipt | `UdpClient._handle_round_start` | `test_udp_client.py::test_ready_deadline_is_receipt_plus_rho_when_encoding_finishes_first`, `test_real_loop_ready_gap_excludes_encoding_time` | U |
| Simulator ↔ GatewayCore decision parity | `production.py`, `sim/run_net.py` | `test_sim_gateway_parity.py`, `test_mininet_production.py` | U, I |
| B3 RTT definition | `B3GatewayCore.accept_data`, `mark_plan_sent` | `test_b3_rtt_producer.py`, `test_b3_rtt_fl_path.py`, `test_b3_plan.py::test_first_data_position_after_one_accounts_for_pacing_offset` | U, I |
| B2/B3 goodput (padded 1400 B) | `baseline_proc.finalize_round`, `sim_adapter` | `test_b2_b3_accounting.py::test_goodput_boundary_matches_grant_simulator_for_padded_final_block`, `test_mininet_baseline_procs.py` (packets ↔ summary ↔ goodput) | U, I |
| B1/B2/B3 processes over real sockets | `baseline_proc.py` | `test_mininet_baseline_procs.py::test_baseline_processes_complete_rounds_over_real_sockets` | I |
| Gateway profiling harness | `profiling/gateway.py` | `test_gateway_profile_harness.py`; localhost run in §7.1 | U, I |
| Bounded probe: ≤ 1 per client per round, rules (i)–(iv) required, consumed at issue | `GatewayCore._pump`, `probe_available`, `is_probe_grant`; `eligibility.is_eligible(probe_available=…)` | `test_gateway.py::test_t1_one_probe_per_round_second_is_blocked` … `test_t10_cumulative_and_per_round_probe_counts`, `test_probe_mask_must_align_with_candidates`; `test_policy_amendments.py` | U, I (`test_probe_end_to_end.py`: M-sim and production issue one probe per round to a locked-out client, decision-for-decision equal; `test_sim_gateway_parity.py`; `probe_validation_sim/`) |
| Min-RTT pacing (Eq. 12 amendment) | `policy/pacing.py::p_target_min_rtt`, `ClientState.rtt_min_ns`, `estimators.rtt_update`, `GatewaySim.get_p_target` | `test_min_rtt_pacing.py`, `test_sim_gateway_parity.py` | U, I |
| Probe logging (`probe`, `probe_grants`) | `logging_tap._probe_flag`, `gateway_proc.finish_round`, `scripts/run_grid.py::_round_probe_grants`, `common/logschema.py` | `test_probe_logging.py`, `test_run_grid.py::test_grant_round_metrics_record_the_rounds_probe_grants` and three more; `test_mininet_gateway_integration.py` (grants.csv `probe` rows = round_metrics `probe_grants` over real sockets) | U, I |
| Network-only aggregate receiver | `gateway_proc.NetworkOnlyAggregateSink` | `test_run_m1.py::test_network_only_aggregate_sink_records_each_round` | U |
| Predictor feedback exactly once per grant | `GatewaySim.resolve` via `_CoreEstimator` | `test_predictor_feedback_exactly_once.py` (unit sequences + localhost UDP round) | U, I |

---

## 9. Discrepancies and open items

1. **1 ms tick vs. epoll resolution.** Spec §16 specifies "asyncio, 1 ms tick".
   On Linux the asyncio selector waits in whole milliseconds, so the measured
   tick start lags its grid by ~0.6–1.1 ms at the median and grid points are
   skipped (§7.1). Whether this matters for scheduling quality has not been
   measured; it needs a team decision (accept, or change the loop's timer).
2. **N = 5 at β = 0.10.** D = 100 ms is below srtt₀ + 2 ms, so no grant is ever
   eligible. This only affects runs outside the M1 configuration (N = 20).
3. **Aggregate receiver: resolved for M1, open for M2.** `serve` now
   installs `NetworkOnlyAggregateSink` (§3), so network-only runs no longer
   count `aggregate_transfer_errors`. M2 (FL over Mininet) still needs a real
   aggregate sender to the FL server.
4. **`stale_round_data` is not exported** to `gateway_summary.json` by
   `gateway_proc.finish_round` (Tejaswi; contract §2 key list). It is visible
   in `GatewayStats` and, on the baseline path, as `rejected` packets with
   their own round id.
5. **Value-integrity check not enabled.** Neither the production gateway nor
   the simulator configures a `ValueIntegrityChecker` (spec §9's 2 % check).
6. **Shared clock assumption.** ROUND_START carries an absolute gateway
   monotonic deadline, which is valid only when gateway and clients share a
   kernel clock (true for one Mininet host and for localhost).
7. **Not yet verified on Mininet:** every protocol/gateway behaviour in this
   report, the READY-loss filter and nonzero ρᵢ in the grant path, the
   Mininet profiling path, and M2 (FL over Mininet).
8. **Probe and min-RTT: resolved.** Both are wired into the simulator and
   production providers (§4.2, §5.1) under
   `POLICY_REVISION = "2026-11-17-probe-minrtt"`. Data stamped with the
   earlier `2026-11-13-probe` revision has no probes and is superseded.
   `ml/datagen.py` and `scripts/run_m1.py` now refuse to produce grant-policy
   data unless probes (and, for datagen, min-RTT) are functionally present.
   The FG-GRU numbers wait for the retrained models; the current
   `models/gru_prod.npz` and `models/mlp_prod.npz` are pre-amendment.
9. **B3 cutoff.** Agreed so far:
   - **Margin:** 15 ms, supported by §6.1.
   - **β = 0.05:** reported as *not applicable* for B3, with a footnote
     explaining that its budget is M = 0 by construction. It is not averaged
     in as η = 0.
   - **L4 acceptance:** no absolute delivery target, since that would mean
     tuning the baseline. Instead, READY registration of at least 99 %, plus
     parity with M-sim within the M1 fingerprint tolerances: η and delivered
     blocks within ±0.03, late-DATA fraction within ±10 % relative.

   Still open: early close at the last READY capped at 0.1·D + 105 ms
   (Tejaswi, Praveen and Prajin vote yes), or the fixed 0.1·D + 105 ms. This
   waits for Jayadev's vote. B3 M1 runs wait for the change.
10. **Load-sensitive tests (diagnosed, fixed in the tests).** Several
    integration tests failed only when the machine was busy. They were
    stress-tested by running many pytest processes in parallel on a 16-core
    host. The causes, all in the tests rather than the protocol:
    - **Predictor wall-clock limit in parity tests.** `FallbackPredictor`
      times each prediction against a 2 ms wall-clock limit. Under load a
      P-EWMA call can exceed it, which switches that side to the fallback, and
      two otherwise identical decision streams then diverge. The simulator ↔
      production parity tests (`test_mininet_production.py`,
      `test_sim_gateway_parity.py`) now freeze the clock with a module
      fixture, and assert zero fallbacks. At 1.5× CPU oversubscription the
      old versions failed in 77 of 96 runs; the new ones failed in 0 of 96.
    - **Wrong deadline assumption.** `test_predictor_feedback_exactly_once.py`
      assumed some grant is always outstanding at D. On localhost nothing may
      be, so the test now requires exactly the grants outstanding when the
      deadline fires to resolve at D.
    - **Port race and fixed sleep.** The `gateway_proc` serve test picked
      "free" ports and released them before binding, which failed with
      `EADDRINUSE`. It now uses kernel-assigned ports and registers client
      endpoints after binding. The baseline-process tests now start clients
      when the gateway signals its sockets are bound, instead of after a
      fixed 20 ms sleep.
    - **Round length.** These multi-round tests run gateway and clients in
      one event loop, so the margins were widened: D = 600 ms in the serve
      test, and D = 500 ms with Tc = 100 ms in the baseline tests. Under load,
      previous-round DATA can still arrive after the next ROUND_START. The
      gateway correctly rejects it as `stale_round_data` (Section 8), so the
      serve test now requires every invalid event to be stale-round DATA,
      rather than zero invalid events.

    After the fixes, all touched tests ran 96 times at 1.5× oversubscription
    with one failure. In that run, one client in the UDP exactly-once round
    missed ROUND_END within its 200 ms grace, which is a real-time protocol
    constant. Real-time tests still need a host that is not overloaded.

    Under load the P-GRU inference also sometimes exceeds its 2 ms budget
    (§9 fallback, `predictor fallback … reason=timeout`). That is correct
    behaviour, but **M1 grant rounds on a loaded host are flagged**
    (`predictor_fallbacks > 0`). The idle testbed benchmark has a median of
    0.85 ms and p99 of 1.17 ms for 20 clients
    (`results/summary/pred_latency_fedora_m1.csv`).
