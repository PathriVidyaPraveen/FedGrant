"""Atomic shared run/round state for Mininet child processes and samplers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class RoundStateFile:
    """Publish absolute monotonic round windows to concurrent telemetry readers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def write(self, *, round_id: int, start_ns: int = 0, deadline_ns: int = 0, active: bool,
              drain_until_ns: int = 0) -> None:
        """Publish the round window; ``drain_until_ns`` marks the late-drain period after D."""
        if round_id < 0 or start_ns < 0 or deadline_ns < 0 or drain_until_ns < 0:
            raise ValueError("round state values must be non-negative")
        if active and (round_id < 1 or deadline_ns <= start_ns):
            raise ValueError("active round state needs a valid round id and time window")
        if drain_until_ns and (active or round_id < 1 or drain_until_ns < deadline_ns):
            raise ValueError("a drain window follows an ended round and cannot precede its deadline")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"round_id": int(round_id), "start_ns": int(start_ns),
                   "deadline_ns": int(deadline_ns), "active": bool(active),
                   "drain_until_ns": int(drain_until_ns)}
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def read(self) -> dict[str, Any]:
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"round_id": 0, "start_ns": 0, "deadline_ns": 0, "active": False}
        if not isinstance(state, dict):
            raise ValueError(f"invalid round state in {self.path}")
        return state

    def round_at(self, t_ns: int) -> int:
        """Round id for a sample inside [start, D]; 0 otherwise (metrics use [0, D])."""
        state = self.read()
        if state.get("active") and int(state["start_ns"]) <= t_ns <= int(state["deadline_ns"]):
            return int(state["round_id"])
        return 0

    def phase_at(self, t_ns: int) -> tuple[int, str]:
        """Return (round_id, "active" | "drain" | "idle") for a sample time."""
        state = self.read()
        round_id = int(state.get("round_id", 0))
        start, deadline = int(state.get("start_ns", 0)), int(state.get("deadline_ns", 0))
        if state.get("active") and start <= t_ns <= deadline:
            return round_id, "active"
        drain_until = int(state.get("drain_until_ns", 0))
        if drain_until and deadline < t_ns <= drain_until:
            return round_id, "drain"
        return 0, "idle"


__all__ = ["RoundStateFile"]
