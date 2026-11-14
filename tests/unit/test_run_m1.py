"""scripts/run_m1.py: resume rules, policy-revision supersession, failure records."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("run_m1", ROOT / "scripts" / "run_m1.py")
m1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m1)


def _run(path: Path, *, revision="__absent__", status="ok", rounds=0):
    path.mkdir(parents=True)
    meta = {"status": status}
    if revision != "__absent__":
        meta["provenance"] = {"policy_revision": revision}
    (path / "meta.json").write_text(json.dumps(meta))
    if rounds:
        (path / "round_metrics.csv").write_text("round_id\n" + "".join(f"{r}\n" for r in range(1, rounds + 1)))
    return path


def test_run_state(tmp_path):
    assert m1.run_state(tmp_path / "none", "B6", "r1") == "todo"
    old = _run(tmp_path / "old")                          # pre-provenance grant run
    assert m1.run_state(old, "B6", "r1") == "superseded"
    assert m1.run_state(old, "B6", None) == "superseded"  # absent != None: old runs are never final
    assert m1.run_state(_run(tmp_path / "cur", revision="r1"), "B4", "r1") == "complete"
    assert m1.run_state(_run(tmp_path / "other", revision="r0"), "FG-GRU", "r1") == "superseded"
    assert m1.run_state(old, "B2", "r1") == "complete"    # baselines do not depend on the policy
    assert m1.run_state(_run(tmp_path / "bad", status="failed"), "B2", "r1") == "todo"


def test_missing_record_reads_old_list_and_accumulates(tmp_path):
    path = tmp_path / "missing_runs.json"
    path.write_text(json.dumps(["L2-B2-s1"]))
    record = m1.load_missing(path)
    assert record == {"missing": ["L2-B2-s1"], "infrastructure_failed": []}
    record["infrastructure_failed"].append("L4-B1-s0")
    m1.save_missing(path, record)
    assert m1.load_missing(path) == {"missing": ["L2-B2-s1"], "infrastructure_failed": ["L4-B1-s0"]}


def test_aside_never_overwrites(tmp_path):
    for _ in range(3):
        _run(tmp_path / "M1" / "L2-B6-s0")
        m1._aside(tmp_path / "M1" / "L2-B6-s0", tmp_path / "M1-superseded")
    names = sorted(p.name for p in (tmp_path / "M1-superseded").iterdir())
    assert names == ["L2-B6-s0", "L2-B6-s0.2", "L2-B6-s0.3"]


@pytest.fixture
def grid(tmp_path):
    path = tmp_path / "mn_m1.yaml"
    path.write_text("experiment_id: M1\nplatform: mininet\nscenario: [L2]\npolicies: [B2, B6]\n"
                    "seeds: [0]\nrounds: 100\n")
    return path


def _main(monkeypatch, tmp_path, grid, revision, outcomes, *extra):
    """Run main() with run_one replaced: outcomes[name] = list of (ok, rounds) per attempt."""
    import fedgrant.mininet.run_mininet as rm
    monkeypatch.setattr(rm, "policy_revision", lambda: revision)
    calls = []

    def fake_run_one(cmd, out, log):
        ok, rounds = outcomes[out.name].pop(0)
        calls.append(out.name)
        _run(out, revision=revision, status="ok" if ok else "failed", rounds=rounds)
        if not ok:
            (out / "meta.json").unlink()
        return ok

    monkeypatch.setattr(m1, "run_one", fake_run_one)
    out_root = tmp_path / "M1"
    m1.main(["--grid", str(grid), "--out-root", str(out_root), *extra])
    return calls, out_root


def test_grant_policies_refused_without_revision(monkeypatch, tmp_path, grid):
    with pytest.raises(SystemExit, match="eligibility amendment"):
        _main(monkeypatch, tmp_path, grid, None, {})


def test_superseded_runs_move_aside_and_failures_are_classified(monkeypatch, tmp_path, grid):
    _run(tmp_path / "M1" / "L2-B6-s0")                    # pre-probe B6 run
    _run(tmp_path / "M1" / "L2-B2-s0")                    # final B2 run
    calls, out_root = _main(monkeypatch, tmp_path, grid, "r1",
                            {"L2-B6-s0": [(False, 0), (False, 0)]})
    assert calls == ["L2-B6-s0", "L2-B6-s0"]              # B2 resumed, B6 rerun twice
    assert (tmp_path / "M1-superseded" / "L2-B6-s0" / "meta.json").is_file()
    assert json.loads((out_root / "missing_runs.json").read_text()) == {
        "missing": [], "infrastructure_failed": ["L2-B6-s0"]}

    # Next invocation: a failure after rounds started is an outlier; a success clears the record.
    calls, _ = _main(monkeypatch, tmp_path, grid, "r1", {"L2-B6-s0": [(False, 37), (True, 100)]})
    assert json.loads((out_root / "missing_runs.json").read_text()) == {
        "missing": [], "infrastructure_failed": []}
    assert (out_root / "L2-B6-s0.failed-1").is_dir()


def test_network_only_aggregate_sink_records_each_round(tmp_path):
    import asyncio

    import numpy as np

    from fedgrant.mininet.gateway_proc import NetworkOnlyAggregateSink

    sink = NetworkOnlyAggregateSink(tmp_path)
    asyncio.run(sink.send_aggregate(3, np.zeros(117_694, np.float32), {0: {1: 700, 2: 700}, 5: {1: 700}}))
    asyncio.run(sink.send_aggregate(4, np.zeros(117_694, np.float32), {}))
    rows = [json.loads(line) for line in (tmp_path / "aggregates.jsonl").read_text().splitlines()]
    assert rows == [{"round_id": 3, "dimension": 117_694, "clients": 2, "delivered_coords": 2100},
                    {"round_id": 4, "dimension": 117_694, "clients": 0, "delivered_coords": 0}]
