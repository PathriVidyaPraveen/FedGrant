"""Unit tests for OraclePredictor (Section 4.6)."""

import numpy as np
import pytest

from fedgrant.core.state import ClientState
from fedgrant.predictors.base import PredInput, RoundCtx
from fedgrant.predictors.oracle import OraclePredictor
from fedgrant.sim.engine import SimulationEngine
from fedgrant.sim.ge import GilbertElliottModel
from fedgrant.sim.links import AccessLink, BottleneckLink


def test_oracle_predictor() -> None:
    engine = SimulationEngine()
    bn = BottleneckLink(engine=engine, capacity_bps=10_000_000, prop_delay_ns=5_000_000)
    ge = GilbertElliottModel(p=0.01, r=0.2, seed=42)
    link = AccessLink(
        client_id=0,
        engine=engine,
        prop_delay_ns=10_000_000,
        ge_model=ge,
        initial_rate_bps=2_000_000,
    )

    # OraclePredictor.predict() requires client_states to be attached so that
    # it can read the per-client RTO state (srtt_ns, rttvar_ns). Supply one
    # ClientState with default initial values (SRTT0=100 ms, RTTVAR0=50 ms).
    client_state = ClientState(client_id=0, w=1.0)
    oracle = OraclePredictor(access_links=[link], bottleneck=bn, client_states=[client_state])
    ctx = RoundCtx(round_id=1, D_ns=398_000_000, N=1, K=169)
    oracle.reset_round(ctx)

    # In Good state initially
    inp = PredInput.from_x48(np.array([0], dtype=np.int32), np.zeros((1, 48), dtype=np.float32), t_ns=0)
    p = oracle.predict(inp)
    assert len(p) == 1
    # State is G, transition to G is 1 - p = 0.99
    assert pytest.approx(p[0], 0.01) == 0.99

    # Far beyond deadline t_ns = 400 ms > D=398 ms -> p_hat = 0
    inp_late = PredInput.from_x48(np.array([0], dtype=np.int32), np.zeros((1, 48), dtype=np.float32), t_ns=395_000_000)
    p_late = oracle.predict(inp_late)
    # At 395ms, arrival time ~ 395ms + 10ms (prop) + 5.8ms (tx) + 10ms (prop) + 5ms (bn prop) + 1.1ms (bn tx) = ~426ms > D=398ms
    assert p_late[0] == 0.0

