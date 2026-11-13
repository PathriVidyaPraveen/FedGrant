"""Post-run delivered-energy ratio for Mininet baseline runs (design decision D-09).

B1/B2 clients send no READY, so the gateway never learns block energies. Eq. (7):

    eta_r = sum_i w_i^2 sum_{j in S_i} eps_ij / sum_i w_i^2 sum_j eps_ij

is therefore computed by joining the gateway's delivered sets
(``gateway_rounds.jsonl``) with each client's per-block energies
(``client_<i>_rounds.jsonl``) and weights (``client_<i>.json``). A round is
reported only when every client's energies are present; otherwise None.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def _jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def round_eta(
    delivered: Mapping[int, Mapping[int, list[int]]],
    block_energies: Mapping[int, Mapping[int, list[float]]],
    client_weights: Mapping[int, float],
) -> dict[int, float | None]:
    """eta per round from {round: {client: delivered idx}}, {round: {client: eps by block}}, w."""
    result: dict[int, float | None] = {}
    for round_id, by_client in sorted(delivered.items()):
        energies = block_energies.get(round_id, {})
        if set(energies) != set(client_weights):
            result[round_id] = None
            continue
        total = sum(client_weights[c] ** 2 * sum(energies[c]) for c in client_weights)
        got = sum(client_weights[c] ** 2 * sum(energies[c][j] for j in by_client.get(c, ())) for c in client_weights)
        result[round_id] = got / total if total > 0 else None
    return result


def eta_from_run_dir(run_dir: str | Path, num_clients: int) -> dict[int, float | None]:
    """Join the raw logs written by ``fedgrant.mininet.baseline_proc``."""
    root = Path(run_dir)
    delivered = {
        int(rec["round_id"]): {int(c): list(v) for c, v in rec["delivered"].items()}
        for rec in _jsonl(root / "gateway_rounds.jsonl")
    }
    energies: dict[int, dict[int, list[float]]] = {}
    weights: dict[int, float] = {}
    for cid in range(num_clients):
        summary = json.loads((root / f"client_{cid}.json").read_text(encoding="utf-8"))
        if summary.get("client_weight") is None:
            raise ValueError(f"client_{cid}.json has no client_weight; eta cannot be computed")
        weights[cid] = float(summary["client_weight"])
        for rec in _jsonl(root / f"client_{cid}_rounds.jsonl"):
            if "block_energies" in rec:
                energies.setdefault(int(rec["round_id"]), {})[cid] = list(rec["block_energies"])
    return round_eta(delivered, energies, weights)


__all__ = ["eta_from_run_dir", "round_eta"]
