# Fedora testbed setup and validation

The Python trace, schema, statistics, and command-builder code runs without root. Mininet tests need a Linux host with network namespaces, Open vSwitch, `sch_netem`, `sch_tbf`, and sudo. A container shares the host kernel and cannot install missing kernel qdiscs by itself.

## Host packages

On Fedora 44, install the host tools and the extra modules package matching the running kernel:

```sh
sudo dnf install -y mininet openvswitch iperf3 iproute iproute-tc ethtool tcpdump \
  kernel-modules-extra python3 python3-pip python3-devel git make gcc tmux
sudo systemctl enable --now openvswitch
systemctl is-active openvswitch
rpm -q kernel-modules-extra
sudo modprobe sch_netem sch_tbf
lsmod | grep -E 'sch_netem|sch_tbf'
```

If the module package does not match `uname -r`, install the matching kernel modules and reboot before continuing. Do not run shaping checks against `lo`; use a throwaway veth pair if verifying raw `tc` syntax manually.

## Python environment

Use the system interpreter so it can import Fedora's Mininet package:

```sh
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest -q tests/unit
```

The unit suite does not need root. The PyTorch FL tests require the optional `ml` extra and are outside the testbed lock file.

## Mininet acceptance checks

Run tests from the repository root and pass the venv interpreter explicitly to sudo:

```sh
sudo PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/integration/test_mininet_round.py
sudo PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m fedgrant.mininet.run_mininet \
  --clients 3 --rounds 1 --scenario L2 --output results/mininet/smoke
```

The test checks the configured RTT, client access shaping, and shared bottleneck payload throughput. The smoke runner uses the local first-eligible stub policy; it verifies the topology and UDP round protocol and must not be reported as production-policy M1 data.

For a GE measurement, start `fedgrant.mininet.validate_ge receive` in the gateway namespace, then `send` from a client namespace through the configured L1/L2/L3 uplink. Run at least 100,000 packets per setting and retain each JSON result. Trace acquisition stores the raw files, source/revision, license, and SHA-256 hashes under `data/traces/`; `python scripts/bin_traces.py` writes the prepared splits there.

## Reproduction boundaries

- `make test-pure` runs code-only checks.
- `make test-mininet` and `make mininet-smoke` need sudo, Open vSwitch, and the host kernel modules.
- `make docker-build` creates a convenience image. Running its Mininet checks still needs host kernel qdiscs and privileged container settings.
- `make calibrate` requires paired real M1 and M-sim rows; synthetic fixtures validate the analysis code but do not substitute for those runs.
