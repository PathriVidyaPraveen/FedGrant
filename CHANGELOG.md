# Specification change log

Section 19 of the specification requires an amendment line here, approved by
all four members, for any change to a binding parameter.

| Date | Parameter | Old value | New value | Reason | Status |
|---|---|---|---|---|---|
| 2026-11-12 | Mininet platform (Section 7) | Ubuntu 22.04, Mininet 2.3 | Fedora 44 host with Fedora's `mininet` and `openvswitch` packages, `kernel-modules-extra` for `sch_netem`/`sch_tbf`; topology, qdiscs and all other Section 7 parameters unchanged | Testbed host is Fedora; acceptance test (N=3/L2 RTT, access rate, bottleneck) and 100k-packet GE validation (L1–L3) passed on it (docs/TESTBED_TRACES_REPORT.md) | PROPOSED by Tejaswi; needs approval from Praveen, Prajin and Jayadev |
