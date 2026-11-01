"""Idempotent cleanup helpers for spawned processes and a Mininet network."""

from __future__ import annotations

import subprocess
from typing import Any, Iterable


def stop_processes(processes: Iterable[subprocess.Popen], *, timeout_s: float = 2.0) -> None:
    procs = tuple(processes)
    for proc in procs:
        if proc is not None and proc.poll() is None:
            proc.terminate()
    for proc in procs:
        if proc is None or proc.poll() is not None:
            continue
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout_s)


def teardown(net: Any | None = None, processes: Iterable[subprocess.Popen] = ()) -> None:
    """Stop children before removing the network; safe to call more than once."""
    stop_processes(processes)
    if net is not None:
        if getattr(net, "_fedgrant_stopped", False):
            return
        net.stop()
        try:
            net._fedgrant_stopped = True
        except (AttributeError, TypeError):
            pass


__all__ = ["stop_processes", "teardown"]
