"""Build the predictor named by PolicyConfig.predictor (Sections 4.6, 6.1).

    from fedgrant.predictors.factory import make_predictor
    pred = make_predictor(cfg.predictor, "models/")

"ewma" -> PEWMA, "hmm" -> PHMM, "mlp" -> PMLPNumpy(models/mlp_prod.npz),
"gru" -> PGRUNumpy(models/gru_prod.npz). "oracle" is simulator-only: it needs the
true simulator state, so the simulator constructs it itself and passes it in.

GatewayCore wraps any predictor in FallbackPredictor automatically; pass
fallback=True to get the wrapper here (e.g. for the simulator).
"""

from __future__ import annotations

from pathlib import Path

from fedgrant.predictors.base import Predictor
from fedgrant.predictors.ewma import PEWMA
from fedgrant.predictors.fallback import FallbackPredictor
from fedgrant.predictors.gru_numpy import PGRUNumpy
from fedgrant.predictors.hmm import PHMM
from fedgrant.predictors.mlp_numpy import PMLPNumpy

PROD_FILES: dict[str, str] = {"gru": "gru_prod.npz", "mlp": "mlp_prod.npz"}


def make_predictor(name: str, models_dir: str | Path = "models", *, fallback: bool = False) -> Predictor:
    """New predictor instance for one run (state is per instance)."""
    if name == "ewma":
        p: Predictor = PEWMA()
    elif name == "hmm":
        p = PHMM()
    elif name in PROD_FILES:
        f = Path(models_dir) / PROD_FILES[name]
        if not f.is_file():
            raise FileNotFoundError(f"{f} not found; run python -m fedgrant.ml.export --models {models_dir}")
        p = PGRUNumpy(f) if name == "gru" else PMLPNumpy(f)
    elif name == "oracle":
        raise ValueError("P-ORACLE is simulator-only; the simulator builds it from true state")
    else:
        raise ValueError(f"unknown predictor {name!r}")
    return FallbackPredictor(p) if fallback else p