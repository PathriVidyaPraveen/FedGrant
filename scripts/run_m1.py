#!/usr/bin/env python3
"""Run the M1 Mininet grid (spec Section 11) one run at a time, resumably.

Reads configs/grids/mn_m1.yaml (scenarios x policies x seeds, 100 rounds, N = 20)
and runs each configuration through ``fedgrant.mininet.run_mininet`` into
``results/mininet/M1/<scenario>-<policy>-s<seed>``.

* A run is complete when its directory has ``meta.json`` with status ok and, for
  grant policies (B4, B6, FG-GRU), was produced by the current policy revision
  (``meta.json`` provenance.policy_revision == POLICY_REVISION). A completed run
  from another policy revision is *superseded*: it is moved to
  ``<out-root>-superseded/`` (never deleted) and rerun.
  A grant run is also superseded unless provenance.probe_wired is true: the
  revision string can name the probe before the providers actually issue probes.
* Grant policies are refused while the code has no POLICY_REVISION or does not
  issue probe grants (``run_mininet.probe_wired``), unless --allow-unrevised is given.
* A failed run is moved aside to ``<dir>.failed-<n>`` and retried once with the
  same seed. A second failure is recorded in ``missing_runs.json``:
  ``missing`` if rounds had started (an outlier, spec Section 14) or
  ``infrastructure_failed`` if no round completed (testbed unavailable: root,
  Mininet/OVS, tc, kernel modules). The file accumulates across invocations;
  a later success removes the entry.
* Every run uses the value bank and gateway profiling. Nothing is deleted.

Run with root, one run per machine at a time (Section 7):

    sudo python3 scripts/run_m1.py --policies B2 B4 B6          # a subset
    sudo python3 scripts/run_m1.py --dry-run                     # print the plan only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GRANT_FACTORY = "fedgrant.mininet.production:create_gateway"
GRANT_POLICIES = {"B4", "B6", "FG-GRU"}
_UNRECORDED = "<not recorded>"


def plan(grid_path: Path, policies: list[str] | None, seeds: list[int] | None,
         scenarios: list[str] | None) -> tuple[dict, list[tuple[str, str, int]]]:
    grid = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
    if grid.get("experiment_id") != "M1" or grid.get("platform") != "mininet":
        raise SystemExit(f"{grid_path} is not the M1 Mininet grid")
    for name, chosen, allowed in (("policy", policies, grid["policies"]), ("seed", seeds, grid["seeds"]),
                                  ("scenario", scenarios, grid["scenario"])):
        unknown = sorted(set(chosen or ()) - set(allowed))
        if unknown:
            raise SystemExit(f"{name} {unknown} not in {grid_path} ({allowed})")
    jobs = [(sc, pol, seed) for sc in (scenarios or grid["scenario"]) for pol in (policies or grid["policies"])
            for seed in (seeds or grid["seeds"])]
    return grid, jobs


def command(grid_path: Path, scenario: str, policy: str, seed: int, out: Path) -> list[str]:
    cmd = [sys.executable, "-m", "fedgrant.mininet.run_mininet", "--config", str(grid_path),
           "--scenario", scenario, "--policy", policy, "--seed", str(seed), "--output", str(out)]
    if policy in GRANT_POLICIES:
        cmd += ["--gateway-factory", GRANT_FACTORY]
    return cmd


def run_state(out: Path, policy: str, revision: str | None) -> str:
    """'todo' (no completed run), 'complete', or 'superseded' (other policy revision)."""
    meta_path = out / "meta.json"
    if not meta_path.is_file():
        return "todo"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "todo"   # unreadable: treated as a failed attempt and moved aside
    if meta.get("status", "ok") != "ok":
        return "todo"
    if policy in GRANT_POLICIES:
        provenance = meta.get("provenance") or {}
        if provenance.get("policy_revision", _UNRECORDED) != revision:
            return "superseded"
        if provenance.get("probe_wired") is not True:   # stamped with the revision, run without probes
            return "superseded"
    return "complete"


def _aside(path: Path, parent: Path, suffix: str = "") -> Path:
    """Move path into parent (created) under a free name; never overwrite or delete."""
    parent.mkdir(parents=True, exist_ok=True)
    target, n = parent / f"{path.name}{suffix}", 1
    while target.exists():
        n += 1
        target = parent / f"{path.name}{suffix}.{n}"
    path.rename(target)
    return target


def completed_rounds(out: Path) -> int:
    try:
        with (out / "round_metrics.csv").open(newline="", encoding="utf-8") as stream:
            return sum(1 for _ in csv.DictReader(stream))
    except OSError:
        return 0


def load_missing(path: Path) -> dict[str, list[str]]:
    record = {"missing": [], "infrastructure_failed": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return record
    if isinstance(data, list):           # format before 2026-11-14: a plain list of missing runs
        record["missing"] = [str(x) for x in data]
    elif isinstance(data, dict):
        for key in record:
            record[key] = [str(x) for x in data.get(key, [])]
    return record


def save_missing(path: Path, record: dict[str, list[str]]) -> None:
    path.write_text(json.dumps({k: sorted(set(v)) for k, v in record.items()}, indent=2) + "\n",
                    encoding="utf-8")


def predictor_fallbacks(out: Path) -> int | None:
    """meta.json predictor_fallbacks_total (None for B1-B3 or when not recorded)."""
    try:
        value = json.loads((out / "meta.json").read_text(encoding="utf-8")).get("predictor_fallbacks_total")
    except (OSError, ValueError):
        return None
    return None if value is None else int(value)


def gateway_rcvbuf_drops(out: Path) -> int | None:
    """meta.json gateway_udp.RcvbufErrors: datagrams lost because the gateway read too slowly."""
    try:
        udp = json.loads((out / "meta.json").read_text(encoding="utf-8")).get("gateway_udp") or {}
    except (OSError, ValueError):
        return None
    value = udp.get("RcvbufErrors")
    return None if value is None else int(value)


def busy_host_warning(threshold_per_cpu: float = 0.25) -> str | None:
    """A warning when the 1-minute load average says the host is not idle (grant runs time the predictor)."""
    try:
        load = os.getloadavg()[0]
    except OSError:
        return None
    cpus = os.cpu_count() or 1
    if load > threshold_per_cpu * cpus:
        return (f"WARNING: host load {load:.1f} on {cpus} CPUs; grant policies time the predictor "
                "(2 ms budget), so stop other jobs before B4/B6/FG-GRU runs")
    return None


def run_one(cmd: list[str], out: Path, log) -> bool:
    started = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        _, stderr = proc.communicate()
    except KeyboardInterrupt:
        # Never SIGKILL run_mininet (subprocess.run would): it must tear down
        # the Mininet network, the replay agents and the gateway/client
        # processes. Forward the interrupt and wait for its cleanup.
        print("\ninterrupted: waiting for run_mininet to tear down the network ...", flush=True)
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        try:
            proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            print("run_mininet did not exit; run `sudo mn -c` and "
                  "`sudo pkill -f fedgrant.mininet` before the next run", flush=True)
        raise
    ok = proc.returncode == 0 and (out / "meta.json").is_file()
    fallbacks = predictor_fallbacks(out) if ok else None
    if fallbacks:
        print(f"  WARNING: {fallbacks} predictor fallbacks to P-EWMA (Section 9 wall-clock budget); "
              "the host was probably not idle. Keep it idle during grant runs.", flush=True)
    rcvbuf = gateway_rcvbuf_drops(out) if ok else None
    if rcvbuf:
        print(f"  WARNING: gateway socket dropped {rcvbuf} datagrams (receive buffer full): the gateway "
              "process could not keep up, so RTT/loss include a gateway-side queue the tc samples miss.",
              flush=True)
    log.write(json.dumps({"output": str(out), "returncode": proc.returncode, "ok": ok,
                          "seconds": round(time.monotonic() - started, 1),
                          "rounds_completed": completed_rounds(out),
                          "predictor_fallbacks_total": fallbacks,
                          "gateway_rcvbuf_drops": rcvbuf,
                          "stderr_tail": (stderr or "")[-2000:]}) + "\n")
    log.flush()
    return ok


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", default="configs/grids/mn_m1.yaml")
    parser.add_argument("--out-root", default="results/mininet/M1")
    parser.add_argument("--policies", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and commands without running")
    parser.add_argument("--allow-unrevised", action="store_true",
                        help="run grant policies although the code has no POLICY_REVISION (diagnostics only)")
    args = parser.parse_args(argv)
    from fedgrant.mininet.run_mininet import policy_revision, probe_wired

    revision = policy_revision()
    grid_path = (ROOT / args.grid).resolve()
    out_root = ROOT / args.out_root
    superseded_root = out_root.with_name(out_root.name + "-superseded")
    grid, jobs = plan(grid_path, args.policies, args.seeds, args.scenarios)
    if revision is None and not args.allow_unrevised:
        blocked = sorted({pol for _, pol, _ in jobs} & GRANT_POLICIES)
        if blocked:
            raise SystemExit(f"{blocked} need the eligibility amendment (no POLICY_REVISION in this code); "
                             "merge it first, or pass --allow-unrevised for a diagnostic run")
    if not args.allow_unrevised:
        blocked = sorted({pol for _, pol, _ in jobs} & GRANT_POLICIES)
        if blocked and not probe_wired():
            raise SystemExit(f"{blocked} need the bounded probe wired into the providers "
                             f"(POLICY_REVISION {revision!r}, but no probe grant is issued: "
                             "CandidateView.probe pass-through missing); "
                             "or pass --allow-unrevised for a diagnostic run")
    states = {job: run_state(out_root / f"{job[0]}-{job[1]}-s{job[2]}", job[1], revision) for job in jobs}
    todo = [job for job in jobs if states[job] != "complete"]
    superseded = [job for job in jobs if states[job] == "superseded"]
    print(f"M1: {len(jobs)} selected, {len(jobs) - len(todo)} already complete, {len(todo)} to run "
          f"({len(superseded)} superseded by policy revision {revision}; "
          f"{grid['rounds']} rounds each, about {len(todo) * 2.7:.0f} min)")
    if args.dry_run:
        for sc, pol, seed in todo:
            note = "  # superseded run moves aside" if (sc, pol, seed) in superseded else ""
            print("  " + " ".join(command(grid_path, sc, pol, seed, out_root / f"{sc}-{pol}-s{seed}")) + note)
        return
    out_root.mkdir(parents=True, exist_ok=True)
    for sc, pol, seed in superseded:
        moved = _aside(out_root / f"{sc}-{pol}-s{seed}", superseded_root)
        print(f"  superseded (other policy revision): moved to {moved}")
    missing_path = out_root / "missing_runs.json"
    record = load_missing(missing_path)
    with (out_root / "run_m1_log.jsonl").open("a", encoding="utf-8") as log:
        for index, (sc, pol, seed) in enumerate(todo, start=1):
            out = out_root / f"{sc}-{pol}-s{seed}"
            for attempt in (1, 2):
                if out.exists():  # leftover from a failed attempt: keep it aside, never delete
                    n = 1
                    while out.with_name(f"{out.name}.failed-{n}").exists():
                        n += 1
                    out.rename(out.with_name(f"{out.name}.failed-{n}"))
                print(f"[{index}/{len(todo)}] {out.name} (attempt {attempt}) ...", flush=True)
                if pol in GRANT_POLICIES and (warning := busy_host_warning()):
                    print("  " + warning, flush=True)
                if run_one(command(grid_path, sc, pol, seed, out), out, log):
                    for key in record:
                        record[key] = [name for name in record[key] if name != out.name]
                    break
            else:
                kind = "missing" if completed_rounds(out) > 0 else "infrastructure_failed"
                record[kind].append(out.name)
                print(f"  {out.name}: failed twice, recorded as {kind}", flush=True)
            save_missing(missing_path, record)
    save_missing(missing_path, record)
    print(f"done; missing: {sorted(set(record['missing'])) or 'none'}; "
          f"infrastructure_failed: {sorted(set(record['infrastructure_failed'])) or 'none'}")


if __name__ == "__main__":
    main()
