"""Summarise elig_variants.csv: per (variant, policy, scenario), mean over seeds of per-run means."""
import csv
import sys
from collections import defaultdict

import numpy as np

rows = list(csv.DictReader(open(sys.argv[1])))
runs = defaultdict(list)
for r in rows:
    runs[(r["variant"], r["policy"], r["scenario"], r["seed"])].append(r)

def run_stats(rs):
    rs = sorted(rs, key=lambda r: int(r["round"]))
    late = rs[50:]
    g = lambda k, x=rs: np.mean([float(r[k]) for r in x])
    return {
        "eta": g("eta"), "eta_late": g("eta", late), "eta_ratio_late": g("eta", late) / g("eta_bound", late),
        "clients_late": g("clients_granted", late), "low_phat_end": float(rs[-1]["low_phat_clients"]),
        "zero_grant_rounds": sum(int(r["grants_issued"]) == 0 for r in rs),
        "wasted_late": g("wasted_pct", late), "qdrops_late": g("queue_drops", late),
        "util_late": g("utilization", late), "q_p99_late": g("q_p99_ms", late),
        "probe_share": sum(int(r["probe_grants"]) for r in rs) / max(1, sum(int(r["grants_issued"]) for r in rs)),
        "p_target_late": g("p_target", late),
    }

agg = defaultdict(list)
for (v, p, sc, s), rs in runs.items():
    agg[(v, p, sc)].append(run_stats(rs))

order = ["current", "minrtt", "starve", "minrtt+starve", "probe", "minrtt+probe", "noRuleV"]
cols = ["eta", "eta_late", "eta_ratio_late", "clients_late", "low_phat_end", "zero_grant_rounds",
        "wasted_late", "qdrops_late", "util_late", "q_p99_late", "probe_share", "p_target_late"]
out = []
for p in ("B4", "B6"):
    for sc in ("L2", "L4"):
        for v in order:
            st = agg.get((v, p, sc))
            if not st:
                continue
            row = {"policy": p, "scenario": sc, "variant": v, "seeds": len(st)}
            for c in cols:
                vals = np.array([x[c] for x in st])
                row[c] = round(float(vals.mean()), 3)
                if c in ("eta_late",):
                    row[c + "_sd"] = round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 3)
            out.append(row)
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
hdr = f"{'pol':3} {'sc':2} {'variant':14} {'eta':>5} {'eta51+':>6} {'±sd':>5} {'/bound':>6} {'clients':>7} {'lowp':>4} {'zero':>4} {'waste%':>6} {'qdrop':>5} {'util':>4} {'qp99':>5} {'probe%':>6} {'Ptgt':>5}"
print(hdr)
for r in out:
    print(f"{r['policy']:3} {r['scenario']:2} {r['variant']:14} {r['eta']:5.3f} {r['eta_late']:6.3f} {r['eta_late_sd']:5.3f} "
          f"{r['eta_ratio_late']:6.2f} {r['clients_late']:7.1f} {r['low_phat_end']:4.1f} {r['zero_grant_rounds']:4.1f} "
          f"{r['wasted_late']:6.1f} {r['qdrops_late']:5.1f} {r['util_late']:4.2f} {r['q_p99_late']:5.1f} "
          f"{100*r['probe_share']:6.2f} {r['p_target_late']:5.1f}")
