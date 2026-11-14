#!/usr/bin/env python3
"""Run the M1 Mininet grid (spec Section 11) one run at a time, resumably.

Reads configs/grids/mn_m1.yaml (scenarios x policies x seeds, 100 rounds, N = 20)
and runs each configuration through ``fedgrant.mininet.run_mininet`` into
``results/mininet/M1/<scenario>-<policy>-s<seed>``.

* A run whose directory has ``meta.json`` is complete and skipped (resume).
* A failed run is moved aside to ``<dir>.failed-<n>`` and retried once with the
  same seed; a second failure is recorded as missing (spec Section 14,
  "Outliers"). Nothing is deleted.
* Grant policies (B4, B6, FG-GRU) use the production gateway factory; B1-B3 use
  their own processes. Every run uses the value bank and gateway profiling.

Run with root, one run per machine at a time (Section 7):

    sudo python3 scripts/run_m1.py --policies B2 B4 B6          # a subset
    sudo python3 scripts/run_m1.py --dry-run                     # print the plan only
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GRANT_FACTORY = "fedgrant.mininet.production:create_gateway"
GRANT_POLICIES = {"B4", "B6", "FG-GRU"}


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
    log.write(json.dumps({"output": str(out), "returncode": proc.returncode, "ok": ok,
                          "seconds": round(time.monotonic() - started, 1),
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
    args = parser.parse_args(argv)
    grid_path = (ROOT / args.grid).resolve()
    out_root = ROOT / args.out_root
    grid, jobs = plan(grid_path, args.policies, args.seeds, args.scenarios)
    todo = [(sc, pol, seed) for sc, pol, seed in jobs
            if not (out_root / f"{sc}-{pol}-s{seed}" / "meta.json").is_file()]
    print(f"M1: {len(jobs)} selected, {len(jobs) - len(todo)} already complete, {len(todo)} to run "
          f"({grid['rounds']} rounds each, about {len(todo) * 2.5:.0f} min)")
    if args.dry_run:
        for sc, pol, seed in todo:
            print("  " + " ".join(command(grid_path, sc, pol, seed, out_root / f"{sc}-{pol}-s{seed}")))
        return
    out_root.mkdir(parents=True, exist_ok=True)
    missing = []
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
                if run_one(command(grid_path, sc, pol, seed, out), out, log):
                    break
            else:
                missing.append(out.name)
                print(f"  {out.name}: failed twice, recorded as missing", flush=True)
    print(f"done; missing: {missing or 'none'}")
    (out_root / "missing_runs.json").write_text(json.dumps(sorted(missing), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
