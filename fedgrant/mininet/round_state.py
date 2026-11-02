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

    def write(self, *, round_id: int, start_ns: int = 0, deadline_ns: int = 0, active: bool) -> None:
        if round_id < 0 or start_ns < 0 or deadline_ns < 0:
            raise ValueError("round state values must be non-negative")
        if active and (round_id < 1 or deadline_ns <= start_ns):
            raise ValueError("active round state needs a valid round id and time window")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"round_id": int(round_id), "start_ns": int(start_ns),
                   "deadline_ns": int(deadline_ns), "active": bool(active)}
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
        state = self.read()
        if state.get("active") and int(state["start_ns"]) <= t_ns <= int(state["deadline_ns"]):
            return int(state["round_id"])
        return 0


__all__ = ["RoundStateFile"]
