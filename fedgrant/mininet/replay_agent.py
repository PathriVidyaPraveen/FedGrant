"""Per-client trace replay process, run inside the client's network namespace.

At N = 20 every 100 ms boundary fires 20 rate changes at once. Measured on the
Fedora testbed: replaying from runner threads through the Mininet host shell
landed changes a median 9 ms (max 20 ms) late; one agent per client spawning
``tc`` was worse (median 11.5 ms, max 37 ms), because 20 simultaneous
fork/execs of ``tc`` (2.3 ms each when idle) compete for the client CPUs. Each
agent therefore changes its TBF over rtnetlink (pyroute2, ~0.6 ms, no process
spawn). The call returns after the kernel acknowledged the change, so the
logged time is the applied time. ``tc`` is the fallback when pyroute2 is
unavailable; both set identical parameters (rate, burst, limit).

Log format (``trace_replay_client_<i>.csv``) is unchanged:
t_mono_ns, round_id, bin_index, rate_bps, scheduled_mono_ns, lag_ns, where
``t_mono_ns`` is taken after the change call returned, i.e. when the change is applied.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from fedgrant.mininet.replay import RateReplay
from fedgrant.mininet.tcsetup import ACCESS_QUEUE_BYTES, CLIENT_TBF_PARENT, tbf_change

TBF_HANDLE = 0x00100000   # 10:
TBF_BURST = 3000


def tc_handle(text: str) -> int:
    """'2:1' -> 0x00020001 (tc major:minor in hex, as tc parses it)."""
    major, minor = text.split(":")
    return (int(major, 16) << 16) | int(minor or "0", 16)

READY_MARKER = "replay agent ready:"
REPLAY_COLUMNS = ("t_mono_ns", "round_id", "bin_index", "rate_bps", "scheduled_mono_ns", "lag_ns")


def tc_applier(iface: str, run: Callable[..., subprocess.CompletedProcess] = subprocess.run, *,
               parent: str = CLIENT_TBF_PARENT) -> Callable[[int], None]:
    """Return ``rate -> None`` that runs the validated ``tc qdisc change`` and fails loudly."""
    def apply(rate_bps: int) -> None:
        command = shlex.split(tbf_change(iface, rate_bps, parent=parent))
        result = run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"{' '.join(command)} failed ({result.returncode}): {result.stderr.strip()}")
    return apply


def netlink_applier(iface: str, ipr=None, *, parent: str = CLIENT_TBF_PARENT) -> Callable[[int], None]:
    """Return ``rate -> None`` that changes the client TBF over rtnetlink in this namespace."""
    if ipr is None:
        from pyroute2 import IPRoute
        ipr = IPRoute()
    links = ipr.link_lookup(ifname=iface)
    if not links:
        raise RuntimeError(f"interface {iface} not found in this network namespace")
    index = links[0]
    parent_handle = tc_handle(parent)

    def apply(rate_bps: int) -> None:
        if isinstance(rate_bps, bool) or int(rate_bps) <= 0:
            raise ValueError("rate_bps must be a positive integer")
        try:
            ipr.tc("change", "tbf", index, handle=TBF_HANDLE, parent=parent_handle,
                   rate=f"{int(rate_bps)}bit", burst=TBF_BURST, limit=ACCESS_QUEUE_BYTES)
        except Exception as exc:
            raise RuntimeError(f"netlink tbf change on {iface} to {rate_bps} bit/s failed: {exc}") from exc
    return apply


def default_applier(iface: str, parent: str = CLIENT_TBF_PARENT) -> Callable[[int], None]:
    """Netlink when pyroute2 is importable (e.g. dnf install python3-pyroute2), else ``tc``.

    The choice is printed to the agent log so a slow fallback is visible in the run.
    """
    try:
        apply = netlink_applier(iface, parent=parent)
        print(f"replay agent {iface}: applier=netlink", flush=True)
    except ImportError:
        apply = tc_applier(iface, parent=parent)
        print(f"replay agent {iface}: applier=tc (pyroute2 not importable as this user; "
              "install python3-pyroute2 for lower lag)", flush=True)
    return apply


def run_agent(rates: list[float], iface: str, round_state: str, log_path: str | Path,
              stop: threading.Event, *, apply_rate: Callable[[int], None] | None = None,
              tbf_parent: str = CLIENT_TBF_PARENT) -> None:
    """Replay ``rates`` round by round until ``stop`` is set; raise if any change failed."""
    log_path = Path(log_path)
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(REPLAY_COLUMNS)
        stream.flush()

        def log_change(actual_ns: int, round_id: int, bin_index: int, rate_bps: int, scheduled_ns: int) -> None:
            writer.writerow((actual_ns, round_id, bin_index, rate_bps, scheduled_ns, actual_ns - scheduled_ns))
            stream.flush()

        replay = RateReplay(rates, apply_rate or default_applier(iface, tbf_parent), repeat=True,
                            round_state_path=round_state, on_change=log_change).start()
        # The runner starts the gateway only after every agent printed this line.
        print(f"{READY_MARKER} {iface}", flush=True)
        try:
            while not stop.wait(0.2):
                if replay.error is not None:
                    break
        finally:
            replay.stop()
        if replay.error is not None:
            raise RuntimeError("trace replay failed") from replay.error


def main(argv: list[str] | None = None) -> None:
    from fedgrant.traces.client_trace import client_trace

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--split", choices=("test", "ood"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--iface", required=True)
    parser.add_argument("--round-state", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--tbf-parent", default=CLIENT_TBF_PARENT, choices=("2:1", "1:1"),
                        help="client TBF parent: 2:1 READY-lossless layout (default), 1:1 original layout")
    args = parser.parse_args(argv)
    rates = [float(r) for r in client_trace(args.client_id, args.split, args.seed, data_dir=args.trace_dir)]
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    try:
        run_agent(rates, args.iface, args.round_state, args.log, stop, tbf_parent=args.tbf_parent)
    except Exception as exc:  # reported to the runner through the exit code and log
        print(f"replay agent client {args.client_id}: {exc!r}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
