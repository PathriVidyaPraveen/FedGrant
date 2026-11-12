# FedGrant

FedGrant studies deadline-bounded, receiver-driven allocation of federated-learning updates under bursty loss. The frozen execution contract is Networks Final Project Specification v3.1 (the PDF is maintained outside this repository). The code separates the Python 3.11 simulator, FEMNIST training, UDP gateway/client, Mininet testbed, trace pipeline, ML predictors, and result analysis.

## Local development

The project requires Python 3.11 or newer. The simulator and analysis use NumPy and SciPy; FEMNIST/ML additionally needs PyTorch; telemetry needs psutil. Install the locked dependencies and the editable package:

```sh
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

Run the suites separately:

```sh
make test
make test-ml                 # needs the PyTorch extra and prepared FEMNIST data
make test-mininet            # sudo, Open vSwitch, sch_netem and sch_tbf
```

`make test` excludes the two ML integration modules; `make test-ml` runs them. The third command checks the root-required N=3/L2 topology, RTT, client access shaping, and shared bottleneck. It is a topology acceptance test, not an M1 experiment.

## FEMNIST data

The project specification uses LEAF commit `09ec454a5675e32e1f0546b456b77857fdece018` and these preprocessing settings:

```bash
./preprocess.sh -s niid -sf 0.1 -k 100 -t sample --smplseed 1549786595 --spltseed 1549786796
```

From the repository root, prepare a reproducible copy with:

```bash
bash scripts/get_femnist.sh
```

The script writes to `data/femnist/reproduced/` and refuses to overwrite that path. Pass another output path as its first argument to prepare a separate copy. The output contains the LEAF train/test JSON files, memory-mappable per-client NumPy shards, a client manifest, and `provenance.json` with the exact commit and settings. It selects the first 200 sorted writer IDs and groups 10 consecutive writers into each of 20 clients. Images are 28×28 grayscale values in `[0,1]`, with no augmentation. The source `data/femnist/clients/` and any existing runs are left untouched.

To run a full-participation reference experiment on that generated dataset, pass its paths explicitly, for example:

```bash
.venv/bin/python -m fedgrant.fl.run_fl --seed 0 \
  --data-root data/femnist/reproduced/leaf/data \
  --shard-root data/femnist/reproduced/clients \
  --bank data/bank/femnist_full_s0.npz
```

The FL runner defaults to those same `reproduced/` paths and refuses datasets without embedded provenance unless `--allow-unverified-data` is explicitly passed. Every run record stores the preprocessing version, the provenance document and its SHA-256, the client-manifest hash, and the per-shard hashes. Value banks store the same dataset metadata. Existing runs and banks created from `data/femnist/clients/` remain unverified; their data source cannot be recovered from the artifacts alone.

## Simulator ML splits

Generate the four event/feature datasets from the discrete-event simulator:

```bash
.venv/bin/python -m fedgrant.ml.datagen
```

By default this uses every seed in the split ranges from `fedgrant.ml.schema.SPLITS`, runs 50 rounds per seed, and writes bounded shards under `data/mldata/{train,val,test_id,test_ood}/`. In-distribution splits use the required `*_train.csv`, `*_val.csv`, and `*_test_id.csv` cellular traces; the OOD split uses `*_test_ood.csv` OOD traces. Missing trace files are reported before generation rather than silently replaced by constant-rate links. Use `--runs-per-split 1` for a small smoke run, `--trace-dir` to locate traces, or `--out` to choose a new output directory. The generator will not overwrite existing shards.

The simulator and gateway feature code share `fedgrant.policy.features.features48`. The parity integration test checks that `GatewaySim` imports that exact implementation. This checkout does not contain a separate gateway-daemon implementation, so parity against a deployed daemon still requires running the same test vectors against that external implementation.

## Trace preparation

`make trace-fetch` is intentionally offline. It copies the five licensed raw `.up` files and license from `data/traces/raw/`, the installed Mahimahi trace directory, or an explicit source directory. It does not contact GitHub. The repository may contain local trace data; if not, install/source the licensed Mahimahi trace package and pass its directory with `scripts/get_traces.sh --from DIR`.

```sh
make data
```

This copies the raw files, writes source/hash metadata, bins timestamps to 100 ms, and writes the deterministic train/validation/test and OOD CSVs. Scenario L4 replays ID test traces; L5 replays OOD traces. The Mininet replay restarts from the same selected trace offset at each round boundary and writes actual qdisc change times to `trace_replay_client_*.csv`.

## Mininet host setup

Fedora 44 is the accepted host for this project. Mininet uses the host kernel; a container cannot supply missing `sch_netem`/`sch_tbf` support.

On Fedora, install the packages in [docs/FEDORA_TESTBED_SETUP.md](docs/FEDORA_TESTBED_SETUP.md). Then run:

```sh
sudo .venv/bin/python -m pytest -q tests/integration/test_mininet_round.py
sudo .venv/bin/python -m fedgrant.mininet.run_mininet \
  --clients 3 --rounds 1 --scenario L2 --output results/mininet/smoke
```

The specified form is `sudo .venv/bin/python -m fedgrant.mininet.run_mininet --config <yaml> --seed <s>` (or `make mininet-run CONFIG=<yaml> SEED=<s>`). The config may be a scenario file (`configs/scenarios/L3.yaml`) or a grid file (`configs/grids/mn_m1.yaml`); for a grid, pick one value of each listed factor with `--scenario`/`--policy`. Command-line values override the file, and grid values must be in the file's lists. The runner refuses to label stub output as M1/M2: those runs need `--gateway-factory module:function` for the production gateway. A failed run leaves `run_failure.json` instead of `meta.json`, so `make metrics` cannot mistake it for a complete run.

The default gateway is an explicitly labeled first-eligible stub policy with synthetic client updates. Its output validates transport/topology/logging only; it must not be labeled production M1 data. It records effective `tc` readbacks, 10 ms bottleneck queue samples, 100 ms process samples, per-grant and per-packet logs, round summaries, and trace-replay updates. For cross traffic, `--cross-mbps 3` sends UDP from `xt` to the gateway sink on port 9999. A single TBF qdisc does not reveal which flow caused a drop, so the analyzer reports the foreground/cross-traffic split as unmeasured when both share the queue.

## Experiment grid and analysis

Long-running simulator grids are explicit Make targets:

```sh
make grid-net
make grid-fl
```

`grid-fl` runs the three FULL reference seeds first because their final accuracy defines the FL target. B2 UDP push and B3 plan-then-send remain unavailable until Prajin's transport adapters are integrated; those rows are marked missing rather than silently run as ordinary grant-mode FedGrant. M1/M2 also require the production gateway/baseline extension and are not launched by the network simulator grid.

For completed raw Mininet run directories, use `make metrics RUN_DIRS="dir1 dir2"`. The resulting `runs.csv` is validated against [docs/SUMMARY_CSV_SCHEMA.md](docs/SUMMARY_CSV_SCHEMA.md). Per-round means use rounds 21 onward; RTT and scheduler percentiles use pooled sample rows. Queue utilization/drop/queueing-delay summaries use the 10 ms bottleneck counters and backlog samples. `make calibrate` compares M1/M-sim pairs per configuration and policy for eta, utilization, and wasted grants. The supplied Jayadev M-sim bundle has only round-level RTT quantiles, so its RTT percentiles cannot be corrected to pooled sample-level values without the raw RTT samples or rerunning those simulator seeds.

```sh
make plots
make docker-build
docker run --rm fedgrant-testbed
```

The Docker image pins the project runtime to Python 3.11 and runs the non-root unit suite. Mininet tests still require a compatible host kernel, Open vSwitch, and elevated network-namespace access. There is no final LaTeX report in this checkout yet; the Tejaswi testbed/traces report draft is [docs/TESTBED_TRACES_REPORT.md](docs/TESTBED_TRACES_REPORT.md).
