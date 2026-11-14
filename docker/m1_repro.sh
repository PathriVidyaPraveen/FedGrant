#!/usr/bin/env bash
# Non-author reproduction of M1 Mininet runs inside the fedgrant-testbed image.
# Run through `make docker-m1-prepare` / `make docker-m1` (see docs/DOCKER_M1_REPRODUCTION.md);
# the container must be privileged and the host must have the kernel modules loaded.
#
#   docker/m1_repro.sh prepare                 traces from /in/src/$TRACES_SUBDIR, bank from /in/bank -> data/
#   docker/m1_repro.sh run L2 B2 0 [L4 B2 0 ...] each run -> results/mininet/docker-M1/<sc>-<pol>-s<seed>,
#                                               then checked against results/summary/m1_reference/
set -euo pipefail
cd /workspace/FedGrant

REF_DIR=results/summary/m1_reference
OUT_ROOT=results/mininet/docker-M1
ANY_REF="$REF_DIR/L2-B2-s0.json"

die() { echo "m1_repro: $*" >&2; exit 2; }

verify_inputs() {
  python - "$ANY_REF" <<'EOF'
import hashlib, json, sys
from pathlib import Path
identity = json.load(open(sys.argv[1]))["identity"]
bank = Path("data/bank/femnist_full_s0.npz")
if not bank.is_file():
    sys.exit("m1_repro: data/bank/femnist_full_s0.npz missing: run prepare first")
if hashlib.sha256(bank.read_bytes()).hexdigest() != identity["value_bank_sha256"]:
    sys.exit("m1_repro: value bank sha256 does not match the reference runs")
expected = identity["trace_sha256"]
bad = [name for name, sha in expected.items()
       if not (Path("data/traces") / name).is_file()
       or hashlib.sha256((Path("data/traces") / name).read_bytes()).hexdigest() != sha]
if bad:
    sys.exit(f"m1_repro: trace CSVs missing or different from the reference runs: {bad}")
print(f"inputs verified: value bank and {len(expected)} trace CSVs match the reference runs")
EOF
}

prepare() {
  # /in/src is the parent of the traces directory, so get_traces.sh also finds the
  # license that Mahimahi keeps beside traces/ (COPYING/COPYRIGHT/LICENSE).
  local traces="/in/src/${TRACES_SUBDIR:-traces}"
  [[ -d "$traces" ]] || die "traces directory $traces not mounted (make docker-m1-prepare TRACES=...)"
  [[ -f /in/bank/femnist_full_s0.npz ]] || die "mount the directory holding femnist_full_s0.npz at /in/bank"
  if [[ -f data/traces/SOURCE ]]; then
    echo "data/traces already prepared; keeping it (delete data/traces to redo)"
  else
    bash scripts/get_traces.sh --from "$traces"
    python scripts/bin_traces.py
  fi
  mkdir -p data/bank
  cp -n /in/bank/femnist_full_s0.npz data/bank/
  verify_inputs
}

start_network_stack() {
  [[ $(id -u) -eq 0 ]] || die "must run as root inside a --privileged container"
  python -c "import mininet.net" 2>/dev/null || die "mininet is not importable by $(command -v python)"
  local missing=()
  for mod in openvswitch sch_netem sch_tbf sch_prio cls_u32; do
    [[ -d /sys/module/$mod ]] || missing+=("$mod")
  done
  ((${#missing[@]} == 0)) || die "host kernel modules not loaded: ${missing[*]} (on the host: sudo modprobe -a ${missing[*]})"
  /usr/share/openvswitch/scripts/ovs-ctl start --system-id=random --no-mlockall >/dev/null \
    || die "could not start Open vSwitch inside the container"
  ovs-vsctl --timeout=10 show >/dev/null || die "Open vSwitch is not answering"
  # Without --privileged, OVS cannot create its kernel datapath and Mininet's
  # first ovs-vsctl blocks forever; probe it with a timeout instead.
  if ! ovs-vsctl --timeout=10 add-br fgprobe0 >/dev/null 2>&1 || ! ip link show fgprobe0 >/dev/null 2>&1; then
    die "Open vSwitch cannot create a kernel bridge: run the container with --privileged (sudo make docker-m1)"
  fi
  ovs-vsctl --timeout=10 del-br fgprobe0
  mn -c >/dev/null 2>&1 || true
}

run_forwarding_interrupt() {
  # Ctrl+C / docker stop reach this script (PID 1 under --init); pass them on as
  # SIGINT so run_m1.py and run_mininet tear the network down instead of dying.
  "$@" &
  local child=$! rc=0
  trap 'kill -INT "$child" 2>/dev/null || true' INT TERM
  wait "$child" || rc=$?
  while kill -0 "$child" 2>/dev/null; do wait "$child" || rc=$?; done
  trap - INT TERM
  return "$rc"
}

run() {
  (($# >= 3 && $# % 3 == 0)) || die "usage: run <scenario> <policy> <seed> [...]"
  verify_inputs
  start_network_stack
  local status=0
  while (($#)); do
    local sc=$1 pol=$2 seed=$3 name ref
    shift 3
    name="$sc-$pol-s$seed"
    ref="$REF_DIR/$name.json"
    [[ -f "$ref" ]] || die "no reference fingerprint $ref"
    run_forwarding_interrupt python scripts/run_m1.py --out-root "$OUT_ROOT" \
      --scenarios "$sc" --policies "$pol" --seeds "$seed" || status=1
    [[ -f "$OUT_ROOT/$name/meta.json" ]] || { echo "$name: run failed"; status=1; continue; }
    python scripts/m1_fingerprint.py check "$OUT_ROOT/$name" "$ref" --report "$OUT_ROOT/$name/reproduction.json" \
      || status=1
  done
  /usr/share/openvswitch/scripts/ovs-ctl stop >/dev/null 2>&1 || true
  return $status
}

case "${1:-}" in
  prepare) prepare ;;
  run) shift; run "$@" ;;
  *) die "usage: $0 prepare | run <scenario> <policy> <seed> [...]" ;;
esac
