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

The runner currently uses an explicitly labeled first-eligible stub policy and synthetic client updates. Its output validates transport/topology/logging only; it must not be labeled production M1 data. It records effective `tc` readbacks, 10 ms bottleneck queue samples, 100 ms process samples, per-grant and per-packet logs, round summaries, and trace-replay updates. For cross traffic, `--cross-mbps 3` sends UDP from `xt` to the gateway sink on port 9999. A single TBF qdisc does not reveal which flow caused a drop, so the analyzer reports the foreground/cross-traffic split as unmeasured when both share the queue.

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
