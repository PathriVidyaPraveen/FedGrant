PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
RUNS ?= results/summary/runs.csv

.PHONY: help install install-ml test test-pure test-ml test-mininet trace-fetch trace-bin data mininet-smoke mininet mininet-run metrics calibrate plots grid-net grid-fl docker-build docker-test docker-m1-prepare docker-m1 testbed-report

help:
	@printf '%s\n' 'install       Install Python dependencies from docker/requirements.lock' \
	  'install-ml    Install the optional PyTorch ML dependencies' \
	  'test          Run non-ML tests (Mininet tests skip when host support is absent)' \
	  'test-pure     Run unit tests that do not require PyTorch' \
	  'test-ml       Run the PyTorch FEMNIST and simulator integration tests' \
	  'test-mininet  Run root-required Mininet integration tests' \
	  'trace-fetch   Copy licensed Mahimahi traces (offline; skips if present, FORCE=1 replaces)' \
	  'trace-bin     Bin raw .up traces into CSVs' \
	  'data          Prepare trace CSV splits from local raw traces' \
	  'mininet-smoke Run one three-client Mininet round (requires root)' \
	  'mininet       Run the root-required Mininet topology acceptance test' \
	  'mininet-run   One Mininet run: CONFIG=<yaml> SEED=<s> [ARGS="--scenario L2 --policy B6 ..."]' \
	  'grid-net      Execute the simulator network grid (long-running)' \
	  'grid-fl       Run FULL references, then FL simulator jobs (long-running)' \
	  'metrics       Build summary CSV rows from run directories (RUN_DIRS=...)' \
	  'calibrate     Compare paired M1 and M-sim rows' \
	  'plots         Generate analysis plots from RUNS' \
	  'testbed-report Regenerate testbed/trace tables and figures from raw artifacts' \
	  'docker-build  Build the reproducibility image (docker or podman)' \
	  'docker-test   Build the image and run the unit suite inside it' \
	  'docker-m1-prepare  TRACES=<mahimahi traces dir> BANK=<dir with femnist_full_s0.npz>: inputs -> data/' \
	  'docker-m1     As root: reproduce M1 runs in a privileged container [M1_RUNS="L2 B2 0 L4 B2 0"]'

install:
	$(PYTHON) -m pip install -r docker/requirements.lock
	$(PYTHON) -m pip install --no-deps -e .

install-ml:
	$(PYTHON) -m pip install -e '.[ml,test]'

test:
	$(PYTEST) -q --ignore=tests/integration/test_fl_smoke.py --ignore=tests/unit/test_simulator_fixes.py

test-pure:
	$(PYTEST) -q tests/unit --ignore=tests/unit/test_simulator_fixes.py

test-ml:
	$(PYTEST) -q tests/integration/test_fl_smoke.py tests/unit/test_simulator_fixes.py

test-mininet:
	sudo $(PYTEST) -q tests/integration/test_mininet_round.py

mininet-run:
	@test -n "$(CONFIG)" || (echo 'Set CONFIG=<yaml> (e.g. configs/grids/mn_m1.yaml) and SEED=<s>' >&2; exit 2)
	sudo $(PYTHON) -m fedgrant.mininet.run_mininet --config $(CONFIG) --seed $(or $(SEED),0) $(ARGS)

# get_traces.sh writes SOURCE last, so its presence means a completed copy.
# Re-running `make data` keeps those traces; FORCE=1 replaces them.
trace-fetch:
	@if [ -f data/traces/SOURCE ] && [ -z "$(FORCE)" ]; then \
	  echo 'data/traces/SOURCE exists; keeping traces (make trace-fetch FORCE=1 to replace)'; \
	else bash scripts/get_traces.sh $(if $(FORCE),--force); fi

trace-bin:
	$(PYTHON) scripts/bin_traces.py

data: trace-fetch trace-bin

mininet-smoke:
	sudo $(PYTHON) -m fedgrant.mininet.run_mininet --clients 3 --rounds 1 --scenario L2 --output results/mininet/smoke-$$(date +%Y%m%d-%H%M%S)

mininet:
	sudo $(PYTEST) -q tests/integration/test_mininet_round.py

grid-net:
	$(PYTHON) scripts/run_grid.py --exp N-main N-X1 N-X2 N-abl --run --output-csv $(RUNS)

grid-fl:
	$(PYTHON) scripts/run_grid.py --exp F-ref --run --output-csv $(RUNS)
	$(PYTHON) scripts/run_grid.py --exp F-main F-abl --run --output-csv $(RUNS)

metrics:
	@test -n "$(RUN_DIRS)" || (echo 'Set RUN_DIRS to one or more run directories' >&2; exit 2)
	$(PYTHON) -m fedgrant.analysis.raw_metrics $(RUN_DIRS) --output $(RUNS)

calibrate:
	$(PYTHON) -m fedgrant.analysis.calibrate --runs $(RUNS)

testbed-report:
	$(PYTHON) scripts/testbed_report.py

plots:
	$(PYTHON) -m fedgrant.analysis.plots --runs $(RUNS)

# docker or podman (Fedora ships podman); both read docker/Dockerfile and .dockerignore.
CONTAINER ?= $(shell command -v docker 2>/dev/null || command -v podman 2>/dev/null)
IMAGE ?= fedgrant-testbed

docker-build:
	@test -n "$(CONTAINER)" || (echo 'Install docker or podman' >&2; exit 2)
	$(CONTAINER) build -f docker/Dockerfile -t $(IMAGE) .

docker-test: docker-build
	$(CONTAINER) run --rm $(IMAGE)

# Non-author M1 reproduction (docs/DOCKER_M1_REPRODUCTION.md). Run both targets
# with sudo: docker-m1 needs a privileged container (network namespaces, tc, Open
# vSwitch), and with podman root has its own image store, built by docker-build
# under the same sudo. The container keeps its own PID namespace, so cleanup
# inside it can never signal host processes.
M1_RUNS ?= L2 B2 0 L4 B2 0
M1_MOUNTS = -v $(CURDIR)/data:/workspace/FedGrant/data -v $(CURDIR)/results/mininet:/workspace/FedGrant/results/mininet

docker-m1-prepare: docker-build
	@test -n "$(TRACES)" -a -n "$(BANK)" || (echo 'Set TRACES=<mahimahi traces dir> BANK=<dir with femnist_full_s0.npz>' >&2; exit 2)
	mkdir -p data results/mininet
	$(CONTAINER) run --rm --security-opt label=disable $(M1_MOUNTS) \
	  -v $(abspath $(TRACES)/..):/in/src:ro -v $(abspath $(BANK)):/in/bank:ro \
	  -e TRACES_SUBDIR=$(notdir $(abspath $(TRACES))) $(IMAGE) docker/m1_repro.sh prepare

docker-m1: docker-build
	@test "$$(id -u)" = 0 || (echo 'docker-m1 needs root (privileged container): sudo make docker-m1' >&2; exit 2)
	mkdir -p results/mininet
	$(CONTAINER) run --rm --privileged --init --stop-timeout 150 $(M1_MOUNTS) \
	  $(IMAGE) docker/m1_repro.sh run $(M1_RUNS)
