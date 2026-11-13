# Specification change log

Section 19 of the specification requires an amendment line here, approved by
all four members, for any change to a binding parameter.

| Date | Parameter | Old value | New value | Reason | Status |
|---|---|---|---|---|---|
| 2026-11-12 | Mininet platform (Section 7) | Ubuntu 22.04, Mininet 2.3 | Fedora 44 host with Fedora's `mininet` and `openvswitch` packages, `kernel-modules-extra` for `sch_netem`/`sch_tbf`; topology, qdiscs and all other Section 7 parameters unchanged | Testbed host is Fedora; acceptance test (N=3/L2 RTT, access rate, bottleneck) and 100k-packet GE validation (L1–L3) passed on it (docs/TESTBED_TRACES_REPORT.md) | PROPOSED by Tejaswi; approved: Praveen (2026-11-13); pending: Prajin, Jayadev |
| 2026-11-13 | READY delivery and timer origin (Sections 6–7) | READY subject to Mininet GE loss; simulator treated ROUND_START as instantaneous and scheduled READY arrival at ρi + δi + 5 ms | READY bypasses GE loss but keeps access delay; ρi starts when the client receives ROUND_START; simulator READY arrival is (δi + 5 ms) + ρi + (δi + 5 ms) | Match the simulator and Mininet control paths. At N = 20, 13–16% of READYs were previously lost; the old simulator also omitted the ROUND_START downlink delay. Re-run affected simulator and Mininet results produced under the old timing/loss behavior. | APPROVED by all four (Tejaswi, Praveen, Jayadev, Prajin; 2026-11-13). READY-loss filter implemented in `fedgrant/mininet/tcsetup.py`, verified before each run; simulator timing updated in `fedgrant/sim/run_net.py`. Client timer correction pending Prajin; Fedora timing remeasurement pending Tejaswi. |
