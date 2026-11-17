"""Eligibility/pacing variants in the simulator, patched in memory (no repo file changes).

Variants (applied to B4 = rr/ewma and B6 = value/ewma exactly as M1 configures them):
  current      Eq. 12 with median srtt, rule (v) p_hat >= 0.05            (spec today)
  minrtt       Eq. 12 with RTT_min = median over ready clients of each client's
               minimum RTT sample (fallback 100 ms before any sample)        (Praveen)
  starve       rule (v) skipped when starve_i >= 5 and m_i = 0, i.e. the Eq. 13
               starvation condition                                           (override only)
  minrtt+starve  both                                                         (Tejaswi's vote)
  probe        a client failing only rule (v) may get at most 1 grant per round,
               still through pacing, normal accounting                        (Prajin)
  minrtt+probe both
  noRuleV      rule (v) disabled (ablation A4), reference only

Usage: python3 elig_variants.py OUT.csv [rounds] [seeds...]
"""
from __future__ import annotations

import csv
import dataclasses
import sys
from multiprocessing import Pool

import numpy as np

ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SEEDS = [int(s) for s in sys.argv[3:]] or [0, 1, 2]
VARIANTS = ["current", "minrtt", "starve", "minrtt+starve", "probe", "minrtt+probe", "noRuleV"]
FALLBACK_RTT_NS = 100_000_000


def run_job(job):
    variant, policy, scenario, seed = job
    import fedgrant.policy.features as feat
    import fedgrant.sim.run_net as rn
    from fedgrant.mininet.contract import POLICY_CONFIGS
    from fedgrant.policy.constants import C, S, TAU_Q_NS, THETA
    from fedgrant.policy.eligibility import Reason, is_eligible as orig_elig
    from fedgrant.sim.seeding import network_round_seed
    from fedgrant.sim.value_bank import load_value_bank

    use_min = variant.startswith("minrtt")
    mode = "starve" if "starve" in variant else "probe" if "probe" in variant else None
    rtt_min: dict[int, int] = {}
    probed: set[int] = set()
    issued: list[tuple[int, bool]] = []  # (client, was_probe) this round

    orig_rtt = feat.rtt_update
    def rtt_update(state, sample_ns):
        orig_rtt(state, sample_ns)
        c = state.client_id
        rtt_min[c] = min(rtt_min.get(c, sample_ns), int(sample_ns))
    feat.rtt_update = rtt_update

    def is_eligible(state, t_ns, D_ns, p_hat, cfg, probe_available=False):
        # The repository's own probe (probe_available) is ignored: each variant here
        # implements rule (v) itself, starting from the unamended rule.
        ok, reason = orig_elig(state, t_ns, D_ns, p_hat, cfg)
        if ok or reason != Reason.LOW_PREDICTION:
            return ok, reason
        if mode == "starve" and state.starve >= 5 and state.m == 0:
            return True, Reason.ELIGIBLE
        if mode == "probe" and state.client_id not in probed:
            return True, Reason.ELIGIBLE
        return ok, reason
    rn.is_eligible = is_eligible

    orig_issue = rn._CorePacing.issue_probability
    def issue_probability(self, client_id, p_hat):
        probe = p_hat < THETA and self.sim.cfg.pred_in_eligibility
        if probe:
            probed.add(int(client_id))
        issued.append((int(client_id), probe))
        return orig_issue(self, client_id, p_hat)
    rn._CorePacing.issue_probability = issue_probability

    if use_min:
        def get_p_target(self):
            ready = [s.client_id for s in self.client_states if s.ready] or [s.client_id for s in self.client_states]
            samples = [rtt_min[c] for c in ready if c in rtt_min]
            rtt = float(np.median(samples)) if samples else FALLBACK_RTT_NS
            return C * (rtt + TAU_Q_NS) / (8 * S * 1e9)
        rn.GatewaySim.get_p_target = get_p_target

    cfg = POLICY_CONFIGS[policy]
    if variant == "noRuleV":
        cfg = dataclasses.replace(cfg, pred_in_eligibility=False)
    bank = load_value_bank("data/bank/femnist_full_s0.npz", 20)
    w2 = bank.client_weights.astype(np.float64) ** 2
    gw, abs_ns, rows = None, 0, []
    for rid in range(1, ROUNDS + 1):
        probed.clear(); issued.clear()
        energies = bank.energies_for_run_round(seed, rid)
        res = rn.run_network_round(round_id=rid, num_clients=20, client_weights=bank.client_weights,
                                   client_energies=energies, beta=0.10, scenario=scenario, policy_config=cfg,
                                   seed=network_round_seed(seed, rid), gateway=gw, round_start_abs_ns=abs_ns)
        gw, m = res["gateway"], res["metrics"]
        abs_ns += 1_000_000_000 + int(res["deadline_ms"] * 1e6)
        # loss-free capacity bound: the B best weighted blocks, B = C*D/(8S)
        B = int(C * res["deadline_ms"] / 1000 / (8 * S))
        weighted = (w2[:, None] * np.asarray(energies, np.float64)).ravel()
        opt = float(np.sort(weighted)[::-1][:B].sum() / weighted.sum())
        p_hat = [gw.core._predict_probability(c, abs_ns) for c in range(20)]
        rows.append({"variant": variant, "policy": policy, "scenario": scenario, "seed": seed, "round": rid,
                     "eta": float(res["eta"]), "eta_bound": opt,
                     "clients_granted": len({c for c, _ in issued}),
                     "probe_grants": sum(p for _, p in issued),
                     "grants_issued": m["grants_issued"],
                     "wasted_pct": 100 * m["wasted_grants_ratio"],
                     "queue_drops": m["drops_queue_bn_client"], "ge_drops": m["drops_ge"],
                     "utilization": m["utilization"], "q_p99_ms": m["queueing_delay_p99_ms"],
                     "low_phat_clients": int(sum(p < THETA for p in p_hat)),
                     "p_target": gw.get_p_target()})
    return rows


if __name__ == "__main__":
    jobs = [(v, p, sc, s) for v in VARIANTS for p in ("B4", "B6") for sc in ("L2", "L4") for s in SEEDS]
    # maxtasksperchild=1: every job monkeypatches module/class attributes, so a reused
    # worker would carry an earlier variant's patches (e.g. minrtt get_p_target) into later jobs.
    with Pool(10, maxtasksperchild=1) as pool:
        out = [r for rows in pool.imap_unordered(run_job, jobs) for r in rows]
    with open(sys.argv[1], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("wrote", len(out), "rows")
