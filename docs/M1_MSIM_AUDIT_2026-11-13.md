# M1 / M-sim correctness audit and decisions needed (draft, 2026-11-13)

Scope: the issues raised after the Fedora Mininet run and Tejaswi's 40-round
simulator diagnostic. Each item is marked **Verified** (code or a run in this
checkout), **Hypothesis** (consistent with evidence, not proven) or
**Recommendation**. Neither Tejaswi's 40-round diagnostic nor the Fedora B1/B3
run logs are in this checkout; where numbers are needed, a local reproduction is
used and labelled as such.

**Local reproduction used below.** `run_network_round` for 40 consecutive
rounds on one reused `GatewaySim`, N = 20, β = 0.10, L2, seed 0, uniform weights
and **synthetic** energies (no value bank here), B6 and B4. It is a reproduction
of the mechanism, not of Tejaswi's numbers (η is lower in absolute terms because
the synthetic energies are flat).

---

## 1. Fixed now (independent of any policy decision)

### 1.1 B2/B3 simulator-adapter timing — Verified, fixed

`fedgrant/baselines/sim_adapter.py::run_baseline_network_round` used the
pre-amendment timing: B3 READY at ρᵢ + δᵢ + 5 ms and B2's first DATA at ρᵢ,
both omitting the ROUND_START downlink. The authoritative definition
(`sim/run_net.py`, CHANGELOG 2026-11-13; `UdpClient` and `baseline_proc` start ρᵢ
at ROUND_START receipt) is:

- one control leg = δᵢ + 5 ms (`control_leg_ns`, access delay + bottleneck propagation);
- B3 READY arrival = leg + ρᵢ + leg (ρᵢ from the wire, whole µs, as the real client);
- B2 first DATA = leg + ρᵢ, then paced at b̄ᵢ as before.

PLAN downlink (one leg) and DATA uplink (through the access and bottleneck
links) were already modelled and are unchanged; no leg is counted twice.
Tests: `tests/unit/test_sim_adapter_timing.py` (B2 first send, B3 READY times,
cross-check against `run_net`'s own READY times for the same seed, PLAN/DATA
unchanged). Three of them fail on the old code.

**Same bug, not fixed here (owner Jayadev):** `fedgrant/sim/run_baselines.py`
(the FL-over-simulator B2/B3 path used by `fl/server.py`) still schedules B2's
first send at ρᵢ and B3 READY at ρᵢ + δᵢ + 5 ms. The fix is the same two
lines: add `access_links[cid].prop_delay_ns + BN_PROP_NS` once before ρᵢ.

### 1.2 B1 `block_energies` — Partially confirmed, fixed

B1 already logged `block_energies`, but only when its transfer task returned
before ROUND_END. `B1TcpClient.send_round` deliberately holds the TCP stream
until D; `PushOrTcpClient._finish` (ROUND_END) writes the round record and
calls `finish_round`, which drops the encoded update. When ROUND_END wins the
race (likely on the shaped Fedora links, rarely on localhost) the record is
written without energies and the energies can no longer be computed.

Fix (`fedgrant/mininet/baseline_proc.py::PushOrTcpClient._finish`): before
`finish_round`, record the energies of the round's own encoded update (the same
`_block_energies(core.round.encoded)` B2 uses). Nothing is written if no round
was prepared, so η for that round stays `None`. Tests:
`tests/unit/test_b1_block_energy_logging.py` (both orderings, multiple and empty
deliveries, η via `round_eta`, unprepared round → η `None`, B1/B2 schema) and a
schema check in `test_mininet_baseline_procs.py` for B1/B2/B3. The race-order
tests fail on the old code. The η formula is unchanged.

### 1.3 Expected test failure caused by 1.1

`tests/unit/test_b2_b3_accounting.py::test_real_b2_b3_rows_follow_summary_schema`
now fails: with the corrected READY timing, B3's only measured round (21) has
no READY before t_plan, so it plans 0 blocks and has no PLAN→DATA RTT sample
(`plan_data_rtt_p50_ms = ""`). That is the §4 problem, not a regression. The test
was left unchanged pending the §4 decision.

---

## 2. Decision: Eq. 12 pacing target

**Verified.** `fedgrant/policy/pacing.py::p_target`:
P_target = C · (RTT~ + τ_q) / (8S), RTT~ = median of srttᵢ over ready clients,
τ_q = 5 ms, C = 10 Mbit/s, S = 1474 B. Grants are issued while
Φ + p̂ ≤ P_target. srttᵢ is the RFC 6298 estimate (`core/estimators.py::rtt_update`,
srtt₀ = 100 ms) fed with samples t_DATA − t_issue, which include every queue on
the grant's path. The same P_target is used by B4, B5, B6, B7, FG-GRU and FG-MLP;
B3's budget uses median srtt separately (`policy/plan.py`).

**Mechanism (Verified in code, reproduced locally).** Queueing created by the
granted traffic raises the RTT samples → srtt → RTT~ → P_target, which admits
more in-flight grants. There is no minimum-RTT term, so the target tracks the
queue it creates. Growth stops when the 64-frame bottleneck queue (≈ 75 ms at
C; same size in the simulator and in Mininet `tcsetup.BOTTLENECK_QUEUE_BYTES`)
starts dropping: a standing-queue equilibrium, not unbounded divergence.

Local reproduction (B6): median srtt 87 → 123–133 ms, P_target 78 → 109–118,
wasted grants 38–61 % in rounds 2–7 then 9–24 %. B4 is similar. This matches
the shape Tejaswi reported (srtt ≈ 90 → 125–135 ms, P_target ≈ 87 → 105).

**Measurement note (Verified).** `wasted_grants_ratio` = timed-out / issued
(`run_net.py`) also counts grants whose DATA arrived after the timeout
(`late_after_timeout`). In the local B6 run, round 2: ~230 timed-out grants and
125 `late_after_timeout` DATA arrivals (an arrival count, which can include
duplicates). "Wasted" therefore overstates loss; a substantial part is
spurious timeouts caused by queueing jitter.

**Classification.** A genuine feedback property of Eq. 12 as specified
(Verified), amplified by srtt₀ = 100 ms above the base RTT. Not a configuration
or measurement error, although the wasted-grant metric mixes losses and
spurious timeouts. No Mininet grant-path run exists yet to compare
(Fedora runs so far are stub/READY-validation runs).

**Praveen's minimum-RTT proposal** is not in the repository. **Praveen: please
send the exact formula** (which minimum, over what window, per client or
global, initialisation).

Options:

| Option | Expected effect | Spec change | Reruns |
|---|---|---|---|
| A. Keep Eq. 12 | standing queue, ~15–20 % wasted grants, feeds §3 | no | none |
| B. Praveen's min-RTT form (formula pending) | — | yes (CHANGELOG) | all grant policies (M-sim, M1), ML data, GRU |
| C. *Illustrative, Prajin's, not Praveen's:* RTT~ = median over ready clients of each client's minimum observed RTT (data already kept in `ClientState.rtt_last8` / RTT samples), srtt₀ until a sample exists | P_target near the base BDP (~60–65 frames at a ~60–70 ms base RTT instead of ~110), queue near τ_q, fewer spurious timeouts | yes | same as B |

Effects on policies: all grant policies share P_target, so B4/B6/B7/FG-GRU all
change; fewer spurious timeouts mean fewer y = 0 observations for P-EWMA, P-HMM
and P-GRU (and fewer §3 lockouts). Features in x48 that depend on srtt and
timeouts shift, so ML data must be generated after the change.
**Recommendation:** decide B (once the formula is supplied) before any further
grant-policy runs; validate with the 40-round diagnostic.

---

## 3. Decision: absorbing eligibility rule (v)

**Verified in code.**
- `policy/eligibility.py::is_eligible` rule (v): p̂ᵢ ≥ θ = 0.05 (when
  `pred_in_eligibility`, the default for every Mininet/M-sim policy).
- P-EWMA (`predictors/ewma.py`, B4/B6 and the fallback): p ← 0.9 p + 0.1 y only
  in `observe`, initial 0.9, never reset between rounds (`reset_round` is a
  no-op). From 0.9, 28 consecutive y = 0 bring p̂ below θ; a client may hold up to
  cap (≥ 45 at start) outstanding grants, so one timeout burst suffices.
- Observations come only from grant resolutions
  (`GatewaySim.resolve` → `predictor.observe`), and an ineligible client
  receives no grant, so its p̂ cannot change: absorbing for P-EWMA.
- Starvation boost (Eq. 13, `selectors.value_key`) and ε-exploration
  (`run_net._CoreSelector`) choose only among `CandidateView` entries, which are
  built from eligible clients (`_CoreEligibility.candidates`). Neither can
  rescue a locked-out client.
- P-GRU: p̂ also depends on time-varying context features, so it is not
  provably absorbing; not measured here. P-HMM: not verified.

**Quantified (local reproduction).** Clients with p̂ < θ: B6 0 → 15 of 20 by
round 7 and constant through round 40 (their srtt/rttvar identical in rounds 8
and 40, i.e. no grant for 32 rounds; starve counters 33–38); clients
represented 20 → 5. B4: 13 locked out, 7 represented. Lockouts accumulate
during the high-timeout rounds 2–7 (§2), so §2 and §3 interact.

Alternatives (none implemented):

| | A. Periodic probe grant | B. Starvation overrides rule (v) | C. Change θ / recovery semantics |
|---|---|---|---|
| Definition | an ineligible-by-(v) client gets one grant every k rounds (or after s ms without a grant) | if starveᵢ ≥ 5 and mᵢ = 0, rule (v) is skipped for that client (rules (i)–(iv) and pacing still apply) | lower θ, or decay p̂ toward a prior when unobserved, or reset per round |
| Fairness | bounded extra service to excluded clients | ties recovery to the existing starvation signal | changes every client's admission, not only locked-out ones |
| Wasted grants | ≤ 1 probe per locked client per k rounds | ≤ 1 grant per starved client per round until it delivers | lower θ admits more low-p̂ grants in normal operation |
| Recovery | one success lifts P-EWMA from 0.01 to 0.109 > θ | same | depends on the decay/prior chosen |
| Spec change | yes | yes (Section 3.9 rule (v)) | yes |
| ML data | adds probe samples (label them) | adds samples from starved clients | shifts p̂ distribution for all |
| Tests / reruns | eligibility, selector, parity; all grant-policy runs and ML data | same | same |

**Recommendation (for the team to choose):** B, limited to one grant per round
per starved client, because it reuses an existing, already-specified signal,
needs no new timer and is bounded. Validate with the 40-round diagnostic:
clients below θ must return to eligibility and η must not collapse.

---

## 4. Decision: B3 plan time

**Verified.** `fedgrant/policy/plan.py::plan_time_ns` returns 0.1 D
(Section 10, "t_plan = 0.1 D", per `docs/MININET_SPECIFICATION_AUDIT.md` §3.3,
so it is a specification value, not a default); `B3GatewayCore` accepts READY
only up to `plan_ns` and plans at that time; all grids use N = 20, β = 0.10,
D = 398 ms, t_plan = 39.8 ms.

With the approved READY timing a B3 READY reaches the gateway at
2(δᵢ + 5 ms) + ρᵢ, δᵢ ∈ {10, 20, 30, 40} ms, ρᵢ ~ U(0, 0.1 D): 30–129.8 ms at
N = 20 (no queueing), matching the reported 30–130 ms. Only δᵢ = 10 ms clients
with ρᵢ ≤ 9.8 ms can beat 39.8 ms. Computed with the seeded readiness draws
(`readiness_delays_ns`, 5 seeds × 100 rounds) and `policy/plan.py`'s budget/caps
(srtt = 2(δᵢ + 5 ms), b̂ = trace mean rate — model estimates):

| t_plan | READYs registered (of 20) | rounds with none | budget M | planned blocks/round |
|---|---|---|---|---|
| 0.1 D = 39.8 ms | mean 1.23 (0–5) | 108 / 500 | 252 | mean 119 (p10 0) |
| 0.1 D + 2(δ_max + 5 ms) = 129.8 ms | 20 | 0 / 500 | 176 | 176 |

Planning later leaves D − t_plan = 268 ms; the budget and caps already subtract
the remaining time and srtt, so transfers still fit before D. For small D the
fixed bound fails (N = 5: 0.1 D + 90 ms = D), so the rule needs the guard below.
Fairness: today B3 effectively serves only near clients (δ = 10 ms) and none in
~22 % of rounds, which makes it an unrepresentative baseline; the change
restores participation for all clients.

**Proposed CHANGELOG row (not applied):**

| 2026-11-13 | B3 plan time t_plan (Section 10) | t_plan = 0.1 D | t_plan = min(arrival of the last of the N READYs, 0.1 D + 2(δ_max + 5 ms)) with δ_max = 40 ms (129.8 ms at N = 20, β = 0.10); valid only if t_plan + median srtt < D, otherwise the run is rejected | Under the approved READY timing (2026-11-13) READY arrives at 2(δᵢ + 5 ms) + ρᵢ ∈ [30, 129.8] ms at N = 20; at 0.1 D = 39.8 ms on average 1.2 of 20 clients are registered and 22 % of rounds plan nobody, so B3 is not a meaningful baseline. Rerun all B3 M-sim, M1 and FL runs | PROPOSED by Prajin; approvals required: Tejaswi, Jayadev, Praveen, Prajin |

Affected code: `policy/plan.py::plan_time_ns` (and its callers
`B3GatewayCore`, `B3UdpGateway`, `sim_adapter`, `sim/run_baselines.py`), plus
an "all READYs received" early-close in `B3GatewayCore`. Affected tests:
`test_b3_plan.py` (t_plan cut-off, no-ready, budget/caps), `test_b3_rtt_producer.py`,
`test_baseline_sim_adapter.py` (B3), `test_b2_b3_accounting.py`,
`test_b3_rtt_fl_path.py`, `test_b3_plan_loopback.py`,
`test_mininet_baseline_procs.py` (B3 assertions), `test_sim_adapter_timing.py`
(acceptance rule only). Results: every B3 row in M-sim, N-main and M1.

---

## 5. ML dataset

**Verified.** `fedgrant/ml/datagen.py` runs `run_network_round` (the affected
simulator gateway: Eq. 12, rule (v), the corrected READY timing from Jayadev's
merge) with `PolicyConfig(selector="value", predictor="ewma", explore_eps=0.1)`,
N = 20, 50 rounds per run, β ∈ {0.05, 0.10, 0.20}, the FULL value bank
(`data/bank/femnist_full_s0.npz`), one gateway reused across the 50 rounds.
ε-exploration picks among eligible candidates only. In the local B6
reproduction lockout is complete by round 7, so most of each 50-round run is in
the post-lockout regime: samples come mainly from the few still-eligible
clients.

No ML dataset is present in this checkout (`data/` absent); `results/summary/ml_eval_meta.json`
reports test splits from `fedgrant.ml.synthetic` (2 runs each), and
`models/train_summary.json` does not record which data trained
`models/gru_prod.npz`. **Recommendation:** treat all current ML data and the
current P-GRU as **provisional (pipeline validation only)**, not final.

Dependency sequence: (1) team approves Eq. 12 and rule (v) changes →
(2) implement → (3) 40-round diagnostic → (4) confirm η does not collapse and
locked clients recover → (5) Jayadev generates final ML data with the approved
timing and policy → (6) Praveen retrains the GRU once → (7) production gateway
and Mininet smoke tests → (8) FG-GRU M1.

---

## 6. Other findings for owners

- **Jayadev** — `sim/run_baselines.py` B2/B3 timing (§1.1).
- **Jayadev** — `sim/run_net.py::run_network_round` wraps the reused gateway's
  predictor in a *new* `FallbackPredictor` every round (line ~690), so the
  fallback P-EWMA shadow restarts at 0.9 each round in the simulator (and in
  datagen), contrary to its "stays warm" contract. Production (`create_gateway`)
  wraps once. Matters only when the primary trips the 2 ms fallback.

## 7. Experiment readiness

| Policy | M-sim | M1 (Mininet) | Blocked by |
|---|---|---|---|
| B1 | n/a | ready (B1 energies fixed) | — |
| B2 | ready after §1.1 (earlier M-sim B2 rows used old timing: rerun) | ready | `run_baselines.py` fix for FL runs |
| B3 | blocked | blocked | §4 decision |
| B4, B6 | runnable but affected | runnable but affected | §2, §3 decisions; rerun after |
| FG-GRU | blocked | blocked | §2, §3, final data, retrain |
| M2 | — | blocked | FL over Mininet not implemented |
