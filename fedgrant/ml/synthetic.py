"""Synthetic ML dataset generator (stand-in for ml/datagen.py until the simulator exists).

Writes EXACTLY the schema of docs/ML_DATASET_SCHEMA.md through ml/schema.SplitRecorder.
The gateway side is the real code: ClientState, estimators, eligibility (rules
i-v with P-EWMA), pacing, the value selector with exploration eps = 0.1
(Section 9), and features.resolve_event / features48 for every observation and
feature. Only the network is simplified (see NetworkModel).

    python -m fedgrant.ml.synthetic --split train --runs 10 --seed 1000 --out data/mldata_synth/

Per run (Section 4.5): N = 20 clients, 50 rounds, scenario drawn uniformly from
burst {1, 8, 32}, pi {0.02, 0.05, 0.10}, beta {0.05, 0.10, 0.20}; capacity is
constant b-bar_i (no traces). test_ood uses pi {0.05, 0.15}, burst {8, 32}.
"""

from __future__ import annotations

import argparse
import heapq
import time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from math import floor
from pathlib import Path

import numpy as np

from fedgrant.core.estimators import RateWindow, grant_timeout
from fedgrant.core.state import ClientState, OutstandingGrant, values_from_energies
from fedgrant.ml.schema import SPLITS, SplitRecorder
from fedgrant.policy.constants import C, K, S
from fedgrant.policy.eligibility import is_eligible
from fedgrant.policy.features import GatewayView, features48, resolve_event
from fedgrant.policy.pacing import p_target, pacing_p
from fedgrant.policy.selectors import ValueSelector, value_key
from fedgrant.policy.types import CandidateView, PolicyConfig
from fedgrant.predictors.ewma import PEWMA
from fedgrant.predictors.hmm import PHMM

MS = 1_000_000
TC_NS = 1_000 * MS                      # compute window before each timed window (Section 3.1)
N_CLIENTS = 20
N_ROUNDS = 50
EXPLORE_EPS = 0.1                       # data generation only (Section 9)
TICK_NS = 1 * MS
RATE_TICK_NS = 100 * MS
CLIENT_PROC_NS = 100_000                # 0.1 ms grant -> DATA (Section 6)
BN_DELAY_NS = 5 * MS
ACCESS_FIFO = 32                        # frames
FRAME_BITS = 8 * S

SCENARIOS: dict[str, dict[str, tuple]] = {
    "train": dict(burst=(1, 8, 32), pi=(0.02, 0.05, 0.10), beta=(0.05, 0.10, 0.20)),
    "val": dict(burst=(1, 8, 32), pi=(0.02, 0.05, 0.10), beta=(0.05, 0.10, 0.20)),
    "test_id": dict(burst=(1, 8, 32), pi=(0.02, 0.05, 0.10), beta=(0.05, 0.10, 0.20)),
    "test_ood": dict(burst=(8, 32), pi=(0.05, 0.15), beta=(0.05, 0.10, 0.20)),
}
DEFAULT_RUNS: dict[str, int] = {"train": 10, "val": 2, "test_id": 2, "test_ood": 2}

# event kinds; at equal times arrivals are processed before timeouts, so a DATA
# arriving exactly at tau_g counts as delivered (Eq. 8: arrival <= min(D, tau_g))
_ARRIVAL, _TIMEOUT, _READY, _TICK = 0, 1, 2, 3


def deadline_ns(beta: float, n_clients: int = N_CLIENTS) -> int:
    """D = beta * T_full in whole ms, T_full = 8 N K S / C.

    SPEC-NOTE: computed as floor(beta * T_full) in exact arithmetic; this gives the
    spec's listed 199 / 398 / 797 ms (N = 20) and 996 ms (N = 50, beta = 0.10).
    Rounding to nearest would give 399 ms for beta = 0.10.
    """
    t_full_ms = Fraction(8 * n_clients * K * S * 1000, C)
    return floor(Fraction(str(beta)) * t_full_ms) * MS


def ge_params(burst: int, pi: float) -> tuple[float, float]:
    """(p, r): mean burst 1/r, average loss pi = p / (p + r) (Section 3.5)."""
    r = 1.0 / burst
    return pi * r / (1.0 - pi), r


@dataclass(slots=True)
class _Grant:
    client: int
    pos: int
    handle: int


class NetworkModel:
    """Per-client uplink: grant reaches the client after delta_i, DATA leaves 0.1 ms later;
    the Gilbert-Elliott chain advances once per DATA (drop if Bad, Section 3.5/6);
    after delta_i the frame enters a drop-tail FIFO of 32 frames served at b-bar_i;
    then 5 ms and S*8/C serialization at the bottleneck to the gateway.

    Simplification (synthetic only): no bottleneck queueing and no cross traffic;
    clients interact only through the gateway's pacing.
    """

    def __init__(self, n: int, burst: int, pi: float, ge_rngs: list[np.random.Generator]) -> None:
        self.p, self.r = ge_params(burst, pi)
        self.delta = [(10, 20, 30, 40)[i % 4] * MS for i in range(n)]
        self.bbar = [(2_000_000, 4_000_000, 6_000_000)[i % 3] for i in range(n)]
        self.ser = [-(-FRAME_BITS * 1_000_000_000 // b) for b in self.bbar]    # ns, rounded up
        self.ser_bn = -(-FRAME_BITS * 1_000_000_000 // C)
        self.rng = ge_rngs
        g0 = self.r / (self.p + self.r)
        self.state = [0 if rng.random() < g0 else 1 for rng in ge_rngs]       # 0 = Good, 1 = Bad
        self.fifo: list[deque[int]] = [deque() for _ in range(n)]

    def new_round(self) -> None:
        for q in self.fifo:                 # a 1 s compute window empties every queue
            q.clear()

    def prob_next_good(self, i: int) -> float:
        """Pr[next GE state = G | true current state]."""
        return 1.0 - self.p if self.state[i] == 0 else self.r

    def _timing(self, i: int, t_grant: int) -> tuple[int, int]:
        """(FIFO entry time, gateway arrival time) if the frame is not dropped."""
        t_fifo = t_grant + self.delta[i] + CLIENT_PROC_NS + self.delta[i]
        q = self.fifo[i]
        start = max(t_fifo, q[-1]) if q else t_fifo
        return t_fifo, start + self.ser[i] + BN_DELAY_NS + self.ser_bn

    def predict_arrival(self, i: int, t_grant: int) -> int:
        return self._timing(i, t_grant)[1]

    def send(self, i: int, t_grant: int) -> int | None:
        """Transmit one DATA for a grant issued at t_grant; return its gateway arrival time or None if lost."""
        rng = self.rng[i]
        s = self.state[i]
        s = (1 if rng.random() < self.p else 0) if s == 0 else (0 if rng.random() < self.r else 1)
        self.state[i] = s
        t_fifo, t_arr = self._timing(i, t_grant)
        q = self.fifo[i]
        while q and q[0] <= t_fifo:          # frames that left the FIFO before this one arrives
            q.popleft()
        if s == 1 or len(q) >= ACCESS_FIFO:
            return None
        q.append(t_arr - BN_DELAY_NS - self.ser_bn)      # this frame's departure from the FIFO
        return t_arr


def simulate_run(rec: SplitRecorder, seed: int, split: str, n_clients: int = N_CLIENTS,
                 n_rounds: int = N_ROUNDS) -> int:
    """Simulate one run and record it; returns the number of samples."""
    ss = np.random.SeedSequence(seed)
    s_scen, s_ge, s_ready, s_expl, s_values, _ = ss.spawn(6)
    rng_scen = np.random.default_rng(s_scen)
    sc = SCENARIOS[split]
    burst = int(rng_scen.choice(sc["burst"]))
    pi = float(rng_scen.choice(sc["pi"]))
    beta = float(rng_scen.choice(sc["beta"]))
    D = deadline_ns(beta, n_clients)
    rec.begin_run(seed, seed, burst=burst, pi=pi, beta=beta, D_ns=D, capacity="constant")

    net = NetworkModel(n_clients, burst, pi, [np.random.default_rng(s) for s in s_ge.spawn(n_clients)])
    rng_ready = np.random.default_rng(s_ready)
    rng_expl = np.random.default_rng(s_expl)
    rng_val = np.random.default_rng(s_values)
    cfg = PolicyConfig(selector="value", predictor="ewma", explore_eps=EXPLORE_EPS)
    selector = ValueSelector()
    w = 1.0 / n_clients
    states = [ClientState(i, w) for i in range(n_clients)]
    pewma, phmm = PEWMA(), PHMM()
    n_samples = 0

    for rnd in range(n_rounds):
        t0_abs = TC_NS + rnd * (TC_NS + D)
        net.new_round()
        view = GatewayView(t_ns=0, D_ns=D, P_target=1.0)
        rws = [RateWindow(0, 0) for _ in range(n_clients)]
        grants: dict[int, _Grant] = {}                       # every grant of this round
        next_gid = 1
        heap: list[tuple[int, int, int, int]] = []          # (t, kind, seq, arg)
        seq = 0

        def push(t: int, kind: int, arg: int) -> None:
            nonlocal seq
            heapq.heappush(heap, (t, kind, seq, arg))
            seq += 1

        for i, st in enumerate(states):
            st.reset_round(K, round_start_ns=t0_abs)
            rho = int(rng_ready.random() * 0.1 * D)        # readiness delay U(0, 0.1 D)
            push(rho + net.delta[i] + BN_DELAY_NS, _READY, i)
        for t in range(TICK_NS, D + 1, TICK_NS):
            push(t, _TICK, 0)

        def ready_ids() -> list[int]:
            return [i for i, st in enumerate(states) if st.ready]

        def refresh_view(t: int) -> None:
            view.t_ns = t
            ids = ready_ids()
            if ids:
                view.P_target = p_target([states[i].srtt_ns for i in ids])

        def resolve(gid: int, y: int, t: int) -> None:
            g = grants[gid]
            st = states[g.client]
            refresh_view(t)
            obs = resolve_event(st, view, gid, y, t)
            rec.add_event(g.handle, y, t0_abs + t, obs)
            pewma.observe(g.client, y, t, obs)
            phmm.observe(g.client, y, t, obs)
            rws[g.client].set_outstanding(t, len(st.outstanding))

        def issue(i: int, t: int, p_pace: float) -> None:
            nonlocal next_gid, n_samples
            st = states[i]
            x48 = features48(st, view)                       # before the grant changes any state
            tau = grant_timeout(t, st, D)
            t_hat = net.predict_arrival(i, t)
            p_or = net.prob_next_good(i) * (1.0 if t_hat <= min(D, tau) else 0.0)
            h = rec.add_sample(seed, i, t0_abs + t, x48, pewma.value(i), phmm.value(i), p_or)
            n_samples += 1
            pos = st.next_pos()
            st.take_position(pos)
            gid = next_gid
            next_gid += 1
            st.outstanding[gid] = OutstandingGrant(pos, t, p_pace, tau)
            view.inflight.add(gid, p_pace)
            rws[i].set_outstanding(t, len(st.outstanding))
            grants[gid] = _Grant(i, pos, h)
            t_arr = net.send(i, t)
            if t_arr is not None and t_arr <= D:
                push(t_arr, _ARRIVAL, gid)
            push(tau, _TIMEOUT, gid)

        def pump(t: int) -> None:
            ids = ready_ids()
            if not ids:
                return
            refresh_view(t)
            while True:
                elig = [i for i in ids if is_eligible(states[i], t, D, pewma.value(i), cfg)[0]]
                if not elig:
                    return
                if rng_expl.random() < cfg.explore_eps:
                    i = elig[int(rng_expl.integers(len(elig)))]
                else:
                    cv = CandidateView(
                        client_id=np.array(elig, np.int32),
                        next_value=np.array([value_key(states[j]) for j in elig], np.float64),
                        undelivered=np.array([K - states[j].m - len(states[j].outstanding) for j in elig], np.int32),
                        rr_pointer=0,
                    )
                    i = selector.select(cv)
                p = pacing_p(pewma.value(i), cfg)
                if not view.inflight.can_issue(p, view.P_target):
                    return
                issue(i, t, p)

        while heap and heap[0][0] <= D:
            t, kind, _, arg = heapq.heappop(heap)
            if kind == _TICK:
                if t % RATE_TICK_NS == 0:
                    for i, st in enumerate(states):
                        rws[i].advance(st, t)
            elif kind == _READY:
                st = states[arg]
                rws[arg].advance(st, t)
                eps = np.sort(rng_val.exponential(1.0, K).astype(np.float32))[::-1]
                st.set_ready(values_from_energies(eps, st.w))
            elif kind == _ARRIVAL:                           # Algorithm 1, OnData (t <= D)
                g = grants[arg]
                st = states[g.client]
                rws[g.client].advance(st, t)
                view.arrivals.record(t)
                rws[g.client].add_delivery(t)
                st.mark_delivered(g.pos)                     # block counts even if late
                if arg in st.outstanding:
                    resolve(arg, 1, t)
                # else: late DATA after its timeout; the recorded event is never edited
            else:  # _TIMEOUT                                # Algorithm 1, OnTick timeouts
                g = grants[arg]
                st = states[g.client]
                if arg in st.outstanding:
                    rws[g.client].advance(st, t)
                    resolve(arg, 0, t)
                    st.mark_timeout(g.pos)
            pump(t)

        # OnDeadline: resolve every outstanding grant with y = 0, in issue order
        for i, st in enumerate(states):
            rws[i].advance(st, D)
        for gid in sorted(grants):
            if gid in states[grants[gid].client].outstanding:
                resolve(gid, 0, D)
        for st in states:
            st.update_starve()

    rec.end_run(seed)
    return n_samples


def generate_split(split: str, runs: int, seed: int, out: str | Path,
                   n_clients: int = N_CLIENTS, n_rounds: int = N_ROUNDS, verbose: bool = True) -> Path:
    """Simulate `runs` runs with seeds seed .. seed + runs - 1 and write out/<split>.npz."""
    rng_split = SPLITS[split]
    if seed not in rng_split or seed + runs - 1 not in rng_split:
        raise ValueError(f"seeds {seed}..{seed + runs - 1} leave the {split} range "
                         f"[{rng_split.start}, {rng_split.stop})")
    rec = SplitRecorder(split, generator="fedgrant.ml.synthetic")
    t0 = time.time()
    for k in range(runs):
        n = simulate_run(rec, seed + k, split, n_clients, n_rounds)
        if verbose:
            print(f"[{split}] run {k + 1}/{runs} seed={seed + k}: {n} samples ({time.time() - t0:.0f}s)")
    path = rec.save(Path(out) / f"{split}.npz")
    if verbose:
        print(f"[{split}] wrote {path}")
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--runs", type=int, default=None, help="default: train 10, val/test_id/test_ood 2")
    ap.add_argument("--seed", type=int, default=None, help="first seed (default: start of the split's range)")
    ap.add_argument("--out", default="data/mldata_synth/")
    ap.add_argument("--rounds", type=int, default=N_ROUNDS)
    ap.add_argument("--clients", type=int, default=N_CLIENTS)
    a = ap.parse_args(argv)
    runs = DEFAULT_RUNS[a.split] if a.runs is None else a.runs
    seed = SPLITS[a.split].start if a.seed is None else a.seed
    if runs < 1 or a.rounds < 1 or a.clients < 1:
        ap.error("--runs, --rounds and --clients must be >= 1")
    generate_split(a.split, runs, seed, a.out, a.clients, a.rounds)


if __name__ == "__main__":
    main()