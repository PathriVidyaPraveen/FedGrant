PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
RUNS ?= results/summary/runs.csv

.PHONY: help install install-ml test test-pure test-ml test-mininet trace-fetch trace-bin data mininet-smoke mininet mininet-run metrics calibrate plots grid-net grid-fl docker-build

help:
	@printf '%s\n' 'install       Install Python dependencies from requirements.lock' \
	  'install-ml    Install the optional PyTorch ML dependencies' \
	  'test          Run non-ML tests (Mininet tests skip when host support is absent)' \
	  'test-pure     Run unit tests that do not require PyTorch' \
	  'test-ml       Run the PyTorch FEMNIST and simulator integration tests' \
	  'test-mininet  Run root-required Mininet integration tests' \
	  'trace-fetch   Copy existing licensed Mahimahi traces (offline)' \
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
	  'docker-build  Build the reproducibility image'

install:
	$(PYTHON) -m pip install -r requirements.lock
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

trace-fetch:
	bash scripts/get_traces.sh

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

plots:
	$(PYTHON) -m fedgrant.analysis.plots --runs $(RUNS)

docker-build:
	docker build -f docker/Dockerfile -t fedgrant-testbed .
