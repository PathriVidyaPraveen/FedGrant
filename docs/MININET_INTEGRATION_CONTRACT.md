# Mininet production integration contract (M1)

Status: **DRAFT for agreement** (Tejaswi → Prajin, Jayadev, Praveen), 2026-11-13.
Runner side implemented in `fedgrant/mininet/contract.py` (shared definitions), `gateway_proc.py`, `client_proc.py` and `run_mininet.py`; tests in `tests/unit/test_mininet_contract.py`.
Spec: Sections 2.1, 6.1, 7, 8, 10, 11 (M1/M-sim), 12.

M1 stays blocked until a real Mininet run of each policy produces valid results.
Order: one B6 round (N=3, then N=20) → FG-GRU → B4 → B2 → B3 → B1 → the M1 grid.

## 0. Ownership

| Piece | Owner | Lives in |
|---|---|---|
| Runner, topology, qdiscs, process lifecycle, round-state file, logging, telemetry, run directory, `meta.json` | Tejaswi | `fedgrant/mininet/run_mininet.py`, `gateway_proc.py`, `client_proc.py`, `topo.py`, `telemetry.py`, `logging_tap.py` |
| Production grant gateway factory (B4, B6, FG-GRU) | Prajin (+ Praveen for predictors) | new `fedgrant/mininet/production.py` (or `fedgrant/net/`) |
| B1/B2/B3 gateway and client processes (real sockets, no simulator adapters) | Prajin | new `fedgrant/mininet/baseline_proc.py` |
| Value-profile bank file and loader | Jayadev | `data/bank/femnist_full_s0.npz`, `fedgrant/sim/value_bank.py` |
| Bank → Mininet client path | Tejaswi (runner/client side) | `client_proc.py` |

## 1. Grant gateway factory (B4, B6, FG-GRU)

Passed as `--gateway-factory module:function`. Called **once** inside the gateway
process. The same object serves every round, so estimator and predictor state
persist across rounds (spec Table 2).

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class MininetGatewayContext:
    num_clients: int              # N (20 for M1; 3 for smoke)
    policy: str                   # "B4" | "B6" | "FG-GRU"
    seed: int                     # master seed s (0..2 for M1)
    policy_config: PolicyConfig | None  # contract.POLICY_CONFIGS[policy]; None only for the smoke stub
    client_weights: np.ndarray    # float32 [N], from the value bank, sums to 1
    deadline_ns: int              # D = compute_deadline_ns(beta, N)
    K: int                        # 169
    L: int                        # 700
    models_dir: str               # "models" (gru_prod.npz, norm_stats_gru.json)
    log_dir: str                  # run directory, for factory-side logs if any

def create_gateway(ctx: MininetGatewayContext) -> "fedgrant.core.gateway.GatewayCore": ...
```

- **Import paths:** `from fedgrant.core.gateway import GatewayCore` (the module is `core/gateway.py`, not the spec layout's `gateway_core.py`), and `from fedgrant.mininet.contract import MininetGatewayContext`.
- **`policy_config`:** taken from `contract.POLICY_CONFIGS` (B4 = rr/ewma, B6 = value/ewma, FG-GRU = value/gru). A unit test asserts these equal `scripts/run_grid.py::_SIMULATOR_POLICY_CONFIGS`, and the context rejects any other value. Use `ctx.policy_config` directly and do not rebuild it from `ctx.policy`.

Requirements on the returned `GatewayCore`:
- It runs in GRANT mode with the production selector, eligibility, pacing, timeout and estimator hooks for `ctx.policy`. That is the same `PolicyConfig` the simulator uses for that policy, so M1 and M-sim run identical policy code.
- It uses `ctx.client_weights` for `ClientState` weights. It must not assume `1/N`.
- It is deterministic given `ctx.seed`; exploration is ε = 0.
- It opens no sockets and starts no threads or event loops. `gateway_proc` drives it through `on_round_start / on_ready / on_data / on_tick (1 ms) / on_deadline`.
- It may set `decision_hook`; `gateway_proc` will chain its own logging after it.
- It raises a clear error if a required model file is missing. It does not fall back silently.

Runner side (Tejaswi, **done**): `gateway_proc.build_context()` builds the context
from its CLI (`--policy --seed --value-bank --models-dir`), and
`contract.build_gateway(spec, ctx)` calls the factory. It checks that the result is a
`GatewayCore` with clients `0..N-1`. Import `MininetGatewayContext` from
`fedgrant.mininet.contract`. The stub `fedgrant.mininet.stub_policy:create_gateway`
accepts the context (and its weights) and stays the default for smoke runs only.
The runner refuses to label stub output as M1/M2.

## 2. Baseline processes (B1 TCP, B2 push, B3 plan)

These are not `GatewayCore`, so they run their own process entry points. The
runner chooses the entry point from `--policy`:

| Policy | Gateway process | Client process | Ports |
|---|---|---|---|
| B4, B6, FG-GRU | `fedgrant.mininet.gateway_proc --gateway-factory ...` | `fedgrant.mininet.client_proc` (grant mode) | UDP gw 9000, clients 9001 |
| B2 | `fedgrant.mininet.baseline_proc gateway --mode push` | `fedgrant.mininet.baseline_proc client --mode push` | UDP 9000 / 9001 |
| B3 | `... gateway --mode plan` | `... client --mode plan` | UDP 9000 / 9001 |
| B1 | `... gateway --mode tcp` | `... client --mode tcp` | TCP gw 9100 |

The exact command lines are built by `run_mininet.process_commands()`, and the policy table
is in `contract.BASELINE_MODES` / `contract.GRANT_POLICIES`. The runner fails before building
the network if `fedgrant.mininet.baseline_proc` does not exist yet. Every entry point must
accept this common CLI (the runner passes all of it):

```
gateway: --clients N --rounds R --seed S --duration-ns D --compute-window-s 1.0
         --late-drain-s 0.1 --log-dir DIR --round-state DIR/.round_state.json
         --value-bank PATH [--profile-gateway]
client:  --client-id I --rounds R --seed S --round-timeout-s T --log-file DIR/client_I.json
         [--clients N --value-bank PATH]
```

Required behaviour of every gateway process:
- **Round loop:** use `fedgrant.mininet.round_loop.run_rounds(protocol, RoundTiming(...), round_state=..., log_dir=...)` and implement only the four protocol steps: `begin_round`, `send_round_start`, `at_deadline` and `finish_round`. See `gateway_proc.GrantRoundProtocol` for the grant version. The loop owns:
  - the Tc wait and the round-state file (active over [start, D], then a drain phase until D + `late_drain_s`, then idle);
  - sleeping until D and through the drain;
  - writing `round_metrics.csv` and `gateway_summary.json`.
- **Late drain:** sockets, packet/RTT logs and the samplers keep running through the drain, and arrivals after D are counted against that round in `finish_round`. Per-round metrics stay defined on [0, D] (spec Section 12), so `RoundStateFile.round_at` returns 0 after D; `phase_at(t)` labels drain samples as `(round_id, "drain")`.
- **Logs:** write these files in `--log-dir` using the column tuples in `fedgrant/common/logschema.py`:
  - `round_metrics.csv` (`ROUND_METRIC_COLUMNS`; `wasted_grants` blank for push/plan/tcp, per the summary schema);
  - `packets.csv` (`PACKET_COLUMNS`);
  - `rtt_samples.csv` (`RTT_COLUMNS`; B3 writes its one PLAN→first-DATA sample per client per round with `rtt_type=plan_first_data`);
  - `gateway_summary.json` (a list with one dict per round: `round_id`, `accepted_data`, `rejected_data`, `invalid_events`).

  Reusing `LoggingUdpGateway` or its writers is preferred.
- **Exit code:** 0 on success and non-zero on any protocol error, so the runner records the run as failed.

Every client process must:
- write `client_I_packets.csv` (`PACKET_COLUMNS`, `direction=sent`) next to `--log-file`, because the runner merges these into `packets.csv`;
- write `client_I.json` with `completed_rounds` on exit;
- use the bank profile of Section 3, and never `profile_update.profile_bank`.

## 3. Value-profile bank (spec Section 5.2)

- File: `data/bank/femnist_full_s0.npz`, loaded with `fedgrant.sim.value_bank.load_value_bank(path, N)`. It validates provenance and the seed-0 FULL run.
- **Profile of client i in round r:** `bank.energies_for_run_round(seed, r)[i]`. This is exactly what `scripts/run_grid.py` gives the M-sim twins, so each M1 run and its simulator twin see identical update profiles (required for the Section 14 paired calibration).
- **Client weights:** `bank.client_weights` (gateway factory and baseline gateways).
- **DATA payload:** a float32 update whose block energies equal the profile, built with `fedgrant.mininet.profile_update.synthetic_update(energies, seed=...)`. The seed is `int(np.random.SeedSequence([seed, r - 1, 2, i]).generate_state(1)[0])`. The payload energies then pass the gateway's 2% value-integrity check.
- For M1/M2 there is no fallback: if the bank is missing or fails validation, the run fails. Synthetic `profile_bank` stays only for the stub smoke run.
- Implemented: `contract.bank_update_provider(bank, seed, i)` (used by `client_proc --value-bank`) and `contract.payload_seed`. Baseline clients should call the same function.
- Needed from Jayadev: the bank file (or the command that regenerates it) and the FEMNIST provenance/manifest files that `load_value_bank` checks.

## 4. What the runner records (already implemented)

`meta.json`: `experiment_id`, `policy`, `seed`, `N`, `scenario`, `beta`, `cross_mbps`, `n_rounds`, `config_id`, config path and hash, `compute_window_s`, `tc` readback, trace hashes, CPU roles. Also recorded: `gateway_entry_point`, `client_entry_point`, `value_bank` (path and sha256), `profile_source`, and `model_sha256` for FG-GRU.
A failed run writes `run_failure.json` and no `meta.json`. For M1/M2 the runner turns on `--profile-gateway` and defaults the bank to `data/bank/femnist_full_s0.npz` automatically.

## 5. Acceptance per step

1. **B6, N=3, L2, 1 round:** the gateway issues grants, `accepted_data > 0`, `invalid_events == 0`, and READY energies match the bank profile.
2. **B6, N=20, L2, 25 rounds:** `make metrics` produces a valid `runs.csv` row (`eta`, `utilization`, `wasted_grants`, `gw_cpu_pct` filled).
3. Repeat step 2 for FG-GRU, then B4, B2, B3 and B1 (B1 leaves the grant-only columns blank).
4. Run the M1 grid: `make mininet-run CONFIG=configs/grids/mn_m1.yaml SEED=s ARGS="--scenario L2 --policy B6 --gateway-factory ... --profile-gateway"` for 36 runs, then `make metrics`, then `make calibrate` against the accepted M-sim rows.

## Open questions

1. Is the `MininetGatewayContext` field list enough for FG-GRU (Praveen)?
2. Will baseline processes reuse `gateway_proc`'s round loop, or a shared helper extracted from it? I can extract `run_rounds(core_like, args)` if that helps.
3. Does M2 (FL over Mininet) need the FL server on `srv`? That is outside this contract; it is Prajin's M2 task.
