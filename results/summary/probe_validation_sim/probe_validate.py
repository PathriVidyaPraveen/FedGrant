"""Validate the integrated bounded probe (main 4de21f7 + run_net CandidateView.probe pass-through)
against the in-memory "probe" variant of results/summary/eligibility_variants_sim/.

Nothing is patched here: this is the repository's own eligibility/gateway code. Same setup as
elig_variants.py (B4/B6 as M1 configures them, bank energies, beta 0.10, 20 clients, seeds 0-4).
Probe grants are counted from the core's own per-round counter (GatewayCore.round_probe_grants).

Usage: python3 probe_validate.py OUT.csv [rounds] [seeds...]
"""
from __future__ import annotations

import csv
import sys
from multiprocessing import Pool

import numpy as np

ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SEEDS = [int(s) for s in sys.argv[3:]] or [0, 1, 2, 3, 4]


def run_job(job):
    policy, scenario, seed = job
    import fedgrant.sim.run_net as rn
    from fedgrant.mininet.contract import POLICY_CONFIGS
    from fedgrant.policy.constants import POLICY_REVISION, C, S, THETA
    from fedgrant.sim.seeding import network_round_seed
    from fedgrant.sim.value_bank import load_value_bank

    cfg = POLICY_CONFIGS[policy]
    bank = load_value_bank("data/bank/femnist_full_s0.npz", 20)
    w2 = bank.client_weights.astype(np.float64) ** 2
    gw, abs_ns, rows = None, 0, []
    for rid in range(1, ROUNDS + 1):
        decisions: list[dict] = []
        energies = bank.energies_for_run_round(seed, rid)
        res = rn.run_network_round(round_id=rid, num_clients=20, client_weights=bank.client_weights,
                                   client_energies=energies, beta=0.10, scenario=scenario, policy_config=cfg,
                                   seed=network_round_seed(seed, rid), gateway=gw, round_start_abs_ns=abs_ns,
                                   decision_hook=decisions.append)
        gw, m = res["gateway"], res["metrics"]
        abs_ns += 1_000_000_000 + int(res["deadline_ms"] * 1e6)
        B = int(C * res["deadline_ms"] / 1000 / (8 * S))
        weighted = (w2[:, None] * np.asarray(energies, np.float64)).ravel()
        opt = float(np.sort(weighted)[::-1][:B].sum() / weighted.sum())
        issued = [d for d in decisions if d.get("outcome") == "grant_issued"]
        per_client: dict[int, int] = {}
        for d in issued:
            if d.get("probe"):
                per_client[d["client_id"]] = per_client.get(d["client_id"], 0) + 1
        p_hat = [gw.core._predict_probability(c, abs_ns) for c in range(20)]
        rows.append({"variant": "integrated-probe", "policy": policy, "scenario": scenario, "seed": seed,
                     "round": rid, "eta": float(res["eta"]), "eta_bound": opt,
                     "clients_granted": len({d["client_id"] for d in issued}),
                     "probe_grants": gw.core.round_probe_grants,
                     "probe_grants_logged": sum(per_client.values()),
                     "max_probes_one_client": max(per_client.values(), default=0),
                     "grants_issued": m["grants_issued"],
                     "wasted_pct": 100 * m["wasted_grants_ratio"],
                     "queue_drops": m["drops_queue_bn_client"], "ge_drops": m["drops_ge"],
                     "utilization": m["utilization"], "q_p99_ms": m["queueing_delay_p99_ms"],
                     "low_phat_clients": int(sum(p < THETA for p in p_hat)),
                     "p_target": gw.get_p_target(), "policy_revision": POLICY_REVISION})
    return rows


if __name__ == "__main__":
    jobs = [(p, sc, s) for p in ("B4", "B6") for sc in ("L2", "L4") for s in SEEDS]
    with Pool(10) as pool:
        out = [r for rows in pool.imap_unordered(run_job, jobs) for r in rows]
    out.sort(key=lambda r: (r["policy"], r["scenario"], r["seed"], r["round"]))
    with open(sys.argv[1], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("wrote", len(out), "rows")
