# ML dataset schema (contract for `ml/datagen.py`)

Spec: Section 4.5 (dataset generation), 4.2 (observations), 4.3 (features), 9 (exploration).
Code: `fedgrant/ml/schema.py` is the executable form of this document. **Write every split
through `SplitRecorder`**; it builds the windows and validates the file, so the layout below
cannot drift. `fedgrant/ml/synthetic.py` writes the same schema with a simplified network.

Schema version: **1**.

## 1. Files

One split per file or directory, under `data/mldata/`:

| Split | Seeds (run seeds, disjoint) | Scenarios (Section 4.5) |
|---|---|---|
| `train` | 1000–1999 | train trace segments + constant capacity |
| `val` | 2000–2099 | validation segments + constant capacity |
| `test_id` | 3000–3099 | test segments + constant capacity |
| `test_ood` | 4000–4099 | OOD traces; π ∈ {0.05, 0.15}; burst ∈ {8, 32} |

- **Single file:** `data/mldata/<split>.npz` (written with `np.savez_compressed`, loadable with
  `allow_pickle=False`).
- **Shards:** a directory `data/mldata/<split>/` of `shard_000.npz`, `shard_001.npz`, …
  Each shard is self-contained: it holds whole runs, and its `sample_end` indexes its own
  `events`. `load_split()` concatenates shards in file-name order and offsets `sample_end`.

## 2. Arrays

`E` = number of events, `ns` = number of samples, `R` = number of runs.

### Events (one per resolved grant)

| Name | dtype | shape | Meaning |
|---|---|---|---|
| `events` | float32 | [E, 8] | raw observation `o` returned by `features.resolve_event` (o1..o8, Section 4.2) |
| `event_t_ns` | int64 | [E] | resolution time, run-absolute ns (see §3) |
| `event_run` | int32 | [E] | run id |
| `event_client` | int16 | [E] | client id |

Events are **grouped by (run, client), groups sorted by (run, client) ascending, and in
resolution order within a group** (`event_t_ns` non-decreasing).

> SPEC-NOTE: `event_t_ns`, `event_run`, `event_client` (and `sample_t_ns`, `sample_run`,
> `sample_client`) are added to Section 4.5's list so that `tests/unit/test_causality.py`
> can check causality from the file alone.

### Samples (one per issued grant)

| Name | dtype | shape | Meaning |
|---|---|---|---|
| `sample_end` | int64 | [ns] | exclusive end index of the window in `events` |
| `sample_len` | int8 | [ns] | window length n ∈ [0, 16]; window = `events[end - n : end]` |
| `sample_t_ns` | int64 | [ns] | grant issue time t_g, run-absolute ns |
| `sample_run` | int32 | [ns] | run id |
| `sample_client` | int16 | [ns] | client id |
| `ctx` | float32 | [ns, 16] | context c_i(t_g) = `x48[:, 32:48]` (features 33–48) |
| `x48` | float32 | [ns, 48] | `features.features48` at t_g (Section 4.3) |
| `y` | uint8 | [ns] | label, Eq. (8): 1 iff the DATA answering g reaches the gateway by min(D, τ_g) |
| `p_ewma` | float32 | [ns] | P-EWMA p̂ at t_g |
| `p_hmm` | float32 | [ns] | P-HMM p̂ at t_g (shadow mode) |
| `p_oracle` | float32 | [ns] | P-ORACLE p̂ at t_g (shadow mode) |

Samples are stored in issue order within each run. When `sample_len` = 0, `sample_end` is
still the position where the window would end (the start of that (run, client)'s events).

### Runs and metadata

| Name | dtype | shape | Meaning |
|---|---|---|---|
| `run_id` | int32 | [R] | run id (= the run's seed, so ids are unique across splits) |
| `run_seed` | int64 | [R] | master seed s of the run (Section 6, Seeds) |
| `run_burst` | int32 | [R] | mean burst length 1/r |
| `run_pi` | float64 | [R] | average loss π |
| `run_beta` | float64 | [R] | budget ratio β |
| `run_D_ns` | int64 | [R] | deadline D |
| `run_capacity` | `<U32` | [R] | `"constant"` or the trace name |
| `meta_json` | `<U…` | [] | JSON: `schema_version`, `split`, `seed_range` [lo, hi), `generator`, `n_runs`, `n_events`, `n_samples` |

## 3. Time

All `*_t_ns` are **run-absolute**: ns since the start of the run on one clock, non-decreasing
across rounds (round r starts at its absolute `round_start_ns`; a time t inside the round is
stored as `round_start_ns + t`). Inside the gateway logic times stay round-relative;
`ClientState.reset_round(K, round_start_ns=...)` keeps the persistent timestamps consistent.

## 4. Causality rules (Section 4.5) — enforced

1. **Window.** A sample's window holds only events of the **same (run, client)** with
   resolution time **strictly before** t_g: the last n = min(16, #such events), oldest to
   newest. Windows never cross runs.
2. **Outstanding grants are excluded.** A grant still outstanding at t_g has no event yet, so
   it cannot be in the window; its outcome is unknown at t_g.
3. **One event per grant, never edited.** Each grant resolves exactly once (DATA before its
   timeout → y = 1; timeout or deadline → y = 0). A late DATA after the timeout marks the block
   delivered but **never changes or adds an event**, and never changes the label.
4. **Features before the update.** `x48`, `ctx` and the three p̂ are computed at t_g **before**
   the grant changes any state (before `take_position`, `outstanding`, Φ). Each observation is
   computed by `resolve_event` from the state **before** that event updates it (o2 uses the old
   srtt; o5/o6 are taken after removing the resolved grant, as Section 4.2 defines).
5. **Split isolation.** Every run belongs to exactly one split; run seeds lie in that split's
   range; trace time segments are disjoint between splits (generator's responsibility);
   standardization statistics use the train split only.

> SPEC-NOTE: an event resolved at exactly t_g (same ns, processed before the grant) is already
> reflected in `x48`/`ctx` (the state was updated) but excluded from the window (rule 1 is
> strict). This matches the spec's wording; it is rare and harmless.

`tests/unit/test_causality.py` checks rules 1–5 directly from the arrays (set
`FEDGRANT_ML_SPLITS=path1,path2` to run it on real data).

## 5. Hook for the simulator (`ml/datagen.py`)

```python
from fedgrant.ml.schema import SplitRecorder
from fedgrant.policy.features import features48, resolve_event

rec = SplitRecorder(split, generator="fedgrant.ml.datagen")
for seed in seeds_of_split:                       # one run per master seed
    rec.begin_run(seed, seed, burst=..., pi=..., beta=..., D_ns=D, capacity=...)
    ...
    # Algorithm 1, Pump: client i selected, pacing test passed, grant about to be issued at t
    x48 = features48(state_i, view)                # BEFORE any state change
    h = rec.add_sample(seed, i, round_start_ns + t, x48,
                       p_ewma_i, p_hmm_i, p_oracle_i)      # shadow predictors, all at t
    # ... take_position, outstanding[g] = ..., Phi += p, send GRANT; remember h with grant g

    # Algorithm 1, Resolve: DATA of g before its timeout (y=1), or timeout / deadline (y=0)
    obs = resolve_event(state_i, view, g, y, t)    # the only way to resolve a grant
    rec.add_event(h, y, round_start_ns + t, obs)
    predictor.observe(i, y, t, obs); p_hmm.observe(i, y, t, obs)   # shadow predictors too

    # late DATA after the timeout: mark_delivered only; NO add_event
    rec.end_run(seed)                              # fails if any grant is unresolved
rec.save(f"data/mldata/{split}.npz")               # validates, then writes
```

Generating policy (Section 4.5): selector `value`, P-EWMA for eligibility and pacing,
exploration ε = 0.1 (Section 9), N = 20, 50 rounds per run, scenario drawn uniformly per run.
Record every DATA arriving at the gateway (FedGrant DATA only, not cross traffic) in
`view.arrivals` and call `view.arrivals.reset()` at ROUND_START.

## 6. Loading

```python
from fedgrant.ml.schema import load_split
d = load_split("data/mldata/train.npz")   # or a shard directory; validated on load
w = d["events"][d["sample_end"][k] - d["sample_len"][k] : d["sample_end"][k]]   # window of sample k
```
