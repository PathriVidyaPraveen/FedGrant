#!/usr/bin/env python3
"""Fingerprint an M1 Mininet run and check a reproduction against it (spec Section 14).

A fingerprint holds the run's identity (config, value bank, traces, tc settings) and
its network metrics, computed exactly as runs.csv rows (rounds 21..100, via
fedgrant.analysis.raw_metrics). A reproduction passes when its identity matches
exactly and every gated metric is within tolerance.

Mininet runs are not bit-reproducible: netem draws GE loss from the kernel PRNG and
process timing varies, so the tolerances below are set from the spread of the
author's own runs (B2, L2/L4, seeds 0-2: eta 0.37-0.41, goodput within 3.3%).
Gateway CPU, memory and scheduler latency depend on the machine; they are reported
but never gated.

    python3 scripts/m1_fingerprint.py make  results/mininet/M1/L2-B2-s0 -o ref.json
    python3 scripts/m1_fingerprint.py check results/mininet/docker-L2-B2-s0 ref.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORMAT = "fedgrant-m1-fingerprint/1"
# Identity: must match exactly (same experiment, same inputs, same emulated network).
IDENTITY_KEYS = ("experiment_id", "platform", "scenario", "policy", "seed", "N", "beta", "cross_mbps",
                 "n_rounds", "ready_lossless", "compute_window_s", "duration_ns", "profile_source")
# metric -> ("abs" | "rel", tolerance)
TOLERANCES: dict[str, tuple[str, float]] = {
    "eta": ("abs", 0.03),
    "delivered_blocks": ("rel", 0.05),
    "clients_represented": ("abs", 2.0),
    "goodput_bps": ("rel", 0.05),
    "utilization": ("abs", 0.05),
    "queue_drops_fg": ("rel", 0.15),
    "late_after_deadline": ("rel", 0.10),
    "queueing_delay_p99_ms": ("rel", 0.10),
}
REPORTED_ONLY = ("gw_cpu_pct", "gw_rss_mb", "sched_lat_p50_us", "sched_lat_p99_us", "rtt_p50_ms", "rtt_p99_ms")


def _num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _effective_queue(meta: dict) -> dict:
    raw = meta.get("effective_queue_settings")
    if isinstance(raw, str):
        import ast
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {"raw": raw}
    if not isinstance(raw, dict):
        return {}
    # Keep the tc commands that were configured; drop the kernel readback text. It
    # carries netem's per-qdisc random PRNG seed and refcounts, and its format
    # depends on the iproute2 version, so it differs on every run and machine.
    settings = {key: value for key, value in raw.items() if key != "tc_readback"}
    settings["tc_configured"] = {entry.get("interface"): entry.get("configured_commands")
                                 for entry in raw.get("tc_readback") or ()}
    return settings


def _literal(raw):
    if isinstance(raw, str):
        import ast
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw
    return raw


def fingerprint(run_dir: str | Path) -> dict:
    from fedgrant.analysis.raw_metrics import run_row

    root = Path(run_dir)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    if meta.get("status", "ok") != "ok":
        raise SystemExit(f"{root}: run status is {meta.get('status')!r}, not ok")
    row = run_row(root)
    identity = {key: meta.get(key) for key in IDENTITY_KEYS}
    identity["config_sha256"] = (meta.get("config") or {}).get("config_sha256")
    identity["value_bank_sha256"] = (meta.get("value_bank") or {}).get("sha256")
    identity["trace_sha256"] = _literal(meta.get("trace_sha256"))
    identity["effective_queue_settings"] = _effective_queue(meta)
    metrics = {key: _num(row.get(key)) for key in (*TOLERANCES, *REPORTED_ONLY)}
    return {"format": FORMAT, "run": root.name, "identity": identity, "metrics": metrics,
            "environment": {"python": platform.python_version(), "kernel": platform.release(),
                            "machine": platform.machine(), "in_container": Path("/.dockerenv").exists()
                            or Path("/run/.containerenv").exists()}}


def compare(candidate: dict, reference: dict) -> tuple[bool, list[str]]:
    lines, ok = [], True
    for key, ref_value in reference["identity"].items():
        value = candidate["identity"].get(key)
        if value != ref_value:
            ok = False
            lines.append(f"FAIL identity {key}: {value!r} != reference {ref_value!r}")
    if ok:
        lines.append("ok   identity: config, value bank, traces, tc settings and run parameters match")
    for key, (kind, tol) in TOLERANCES.items():
        ref_value, value = reference["metrics"].get(key), candidate["metrics"].get(key)
        if ref_value is None:
            lines.append(f"skip {key}: not in reference")
            continue
        if value is None:
            ok = False
            lines.append(f"FAIL {key}: missing (reference {ref_value:.4g})")
            continue
        diff = abs(value - ref_value)
        limit = tol if kind == "abs" else tol * abs(ref_value)
        passed = diff <= limit
        ok &= passed
        shown = f"±{tol:g}" if kind == "abs" else f"±{100 * tol:g}%"
        lines.append(f"{'ok  ' if passed else 'FAIL'} {key}: {value:.4g} vs reference {ref_value:.4g} "
                     f"(|diff| {diff:.3g}, tolerance {shown})")
    for key in REPORTED_ONLY:
        ref_value, value = reference["metrics"].get(key), candidate["metrics"].get(key)
        if ref_value is not None or value is not None:
            fmt = lambda v: "n/a" if v is None else f"{v:.4g}"
            lines.append(f"info {key}: {fmt(value)} vs reference {fmt(ref_value)} (machine-dependent, not gated)")
    return ok, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    make = sub.add_parser("make", help="write a fingerprint of a run directory")
    make.add_argument("run_dir")
    make.add_argument("-o", "--output", required=True)
    check = sub.add_parser("check", help="compare a run directory against a reference fingerprint")
    check.add_argument("run_dir")
    check.add_argument("reference")
    check.add_argument("--report", help="also write the comparison as JSON here")
    args = parser.parse_args(argv)
    if args.cmd == "make":
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fingerprint(args.run_dir), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out}")
        return 0
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    if reference.get("format") != FORMAT:
        raise SystemExit(f"{args.reference}: not a {FORMAT} file")
    candidate = fingerprint(args.run_dir)
    ok, lines = compare(candidate, reference)
    print(f"reproduction of {reference['run']} by {Path(args.run_dir).name}:")
    print("\n".join("  " + line for line in lines))
    print("PASS" if ok else "FAIL")
    if args.report:
        Path(args.report).write_text(json.dumps({"pass": ok, "lines": lines, "candidate": candidate,
                                                 "reference": reference}, indent=2) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
