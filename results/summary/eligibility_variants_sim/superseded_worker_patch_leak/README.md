# Superseded: worker patch leak (2026-11-14)

These files came from the first run of `../elig_variants.py`, which used a reused
`multiprocessing.Pool`. Each job monkeypatches module/class attributes in its worker,
so a worker that had run a `minrtt*` job kept the patched `GatewaySim.get_p_target`
(P_target from min-RTT, 59.5 ms) for later jobs. Rows labelled `probe`, `starve`,
`current` and `noRuleV` were therefore partly min-RTT runs; e.g. "probe" B6/L2
eta51+ 0.592 here is really minRTT+probe (clean: probe 0.454, minRTT+probe 0.593).

The script now uses `Pool(10, maxtasksperchild=1)` (one fresh process per job); the
corrected results are `../elig_summary.csv` and `../elig_variants_per_round.csv.gz`.
Kept only as a record of what the team saw on 2026-11-13; do not cite.
