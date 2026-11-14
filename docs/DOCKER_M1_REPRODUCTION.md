# Docker reproduction of M1 Mininet runs (non-author check)

Owner: Tejaswi. Reproducer: Praveen (or anyone who did not produce the M1 runs).
Spec Section 14 asks for a clean-machine reproduction; this is the recipe.

The reproducer builds the image from a clean clone, rebuilds the trace CSVs from the
upstream Mahimahi traces, reruns two committed M1 configurations inside a privileged
container, and checks them automatically against the author's runs.

| Step | Time |
|---|---|
| Build image (first time, downloads ~2 GB incl. CPU PyTorch) | 10–20 min |
| Prepare inputs | < 1 min |
| Two M1 runs (L2-B2-s0, L4-B2-s0, 100 rounds each) | ~6 min |

## What is checked

`scripts/m1_fingerprint.py check` compares the new run with the committed reference
`results/summary/m1_reference/<scenario>-<policy>-s<seed>.json`:

* **Identity, must match exactly:** experiment, scenario, policy, seed, N, β, rounds,
  deadline, READY-lossless flag, M1 grid YAML sha256, value-bank sha256, every trace CSV
  sha256, and the exact `tc` commands configured on every interface.
* **Network metrics, within tolerance** (rounds 21–100, computed as `runs.csv` rows):

  | Metric | Tolerance |
  |---|---|
  | η | ±0.03 absolute |
  | delivered blocks, goodput | ±5 % |
  | utilization | ±0.05 absolute |
  | clients represented | ±2 |
  | FedGrant queue drops | ±15 % |
  | late after deadline | ±10 % |
  | queueing delay p99 | ±10 % |

  Mininet runs are not bit-reproducible: netem draws GE loss from the kernel's PRNG
  (`tc` shows a fresh random `seed` per qdisc) and process timing differs. The tolerances
  cover the spread of the author's own B2 runs across seeds 0–2. Different seeds pass
  every metric check and fail only the identity check on `seed`.
* **Reported, not gated:** gateway CPU %, RSS and scheduler latency (machine-dependent).

B2 (UDP push) is the reproduction target because its M1 rows are final. B4/B6 are
waiting for the eligibility fix and FG-GRU for the retrained model; their references
will be added when those runs are final.

## Host requirements

* Linux x86-64 with ≥ 4 cores (12 used on the author's machine), ≥ 8 GB RAM, root (sudo).
* Docker, or podman (Fedora). With podman, run every step under `sudo`: root has its own
  image store, and the run needs a rootful privileged container.
* Kernel modules on the **host** (a container shares the host kernel):

  ```sh
  sudo modprobe -a openvswitch sch_netem sch_tbf sch_prio cls_u32
  ```

* No Open vSwitch or Mininet is needed on the host; both run inside the container.
* Nothing else heavy running during the runs (one Mininet run per machine, Section 7).

## Steps

```sh
# 0. Clean clone of the branch under test
git clone https://github.com/PathriVidyaPraveen/FedGrant.git && cd FedGrant
git checkout tejaswi            # or main once merged

# 1. Upstream Mahimahi traces at the commit recorded in the M1 metadata
git clone https://github.com/ravinet/mahimahi.git ~/mahimahi
git -C ~/mahimahi checkout f1346c38c415abfc98626dd82450f49a57033603

# 2. The value bank (from Jayadev; 3.3 MB). Put it in any directory, e.g. ~/fg-bank/
sha256sum ~/fg-bank/femnist_full_s0.npz
# expect 037a54a870b8a7bee6acb3240d07b02cc8aa883f15c9ac8707abadb464adde9d

# 3. Build the image, copy and bin the traces, install the bank, verify all hashes
sudo make docker-m1-prepare TRACES=$HOME/mahimahi/traces BANK=$HOME/fg-bank

# 4. Run and check L2-B2-s0 and L4-B2-s0 (about 6 minutes)
sudo make docker-m1
```

Step 3 must end with `inputs verified: value bank and 11 trace CSVs match the reference runs`.
Step 4 prints, per run, an `ok`/`FAIL` line for each identity and metric check and
`PASS` or `FAIL`; `make` exits non-zero if any run fails. To choose other runs:
`sudo make docker-m1 M1_RUNS="L2 B2 1 L4 B2 2"` (each needs a committed reference).

## What to send back

```sh
sudo chown -R "$USER" results/mininet/docker-M1
tar czf m1-repro-$(hostname).tgz results/mininet/docker-M1/*/reproduction.json \
    results/mininet/docker-M1/*/meta.json results/mininet/docker-M1/run_m1_log.jsonl
```

Also send the terminal output of step 4, `uname -r`, `docker --version` or
`podman --version`, and the CPU model (`lscpu | head -15`).

## Troubleshooting

| Message | Fix |
|---|---|
| `host kernel modules not loaded: …` | run the `modprobe` line above on the host |
| `Open vSwitch cannot create a kernel bridge` | the container is not privileged: use `sudo make docker-m1`, not a hand-written `docker run` |
| `value bank sha256 does not match` | wrong bank file; get `femnist_full_s0.npz` from Jayadev again |
| `trace CSVs missing or different` | wrong Mahimahi commit, or `data/traces` from an older build: delete `data/traces` and redo step 3 |
| `Could not locate the Mahimahi trace license` | `TRACES` must be the `traces/` folder inside the Mahimahi clone (its `COPYING` sits one level up) |
| `docker-m1 needs root` | prefix with `sudo` |
| interrupted with Ctrl+C | press it once and wait: the container passes it to `run_mininet`, which tears the network down; the half-finished run is kept as `<run>.failed-N` and redone next time |
| a metric `FAIL` with identity `ok` | send the output anyway; one rerun of that configuration is allowed (Section 14 "Outliers"), and both results are reported |

## Files

* `docker/Dockerfile`: image (Python 3.11, locked dependencies, Debian Mininet + Open vSwitch).
* `docker/m1_repro.sh`: in-container driver (`prepare`, `run`); preflight checks, OVS start,
  interrupt forwarding.
* `scripts/m1_fingerprint.py`: `make` writes a reference, `check` compares a run with one.
* `results/summary/m1_reference/*.json`: the author's reference fingerprints (B2, L2/L4, seeds 0–2).
* `tests/unit/test_m1_fingerprint.py`: tolerance and identity rules, reference completeness.
