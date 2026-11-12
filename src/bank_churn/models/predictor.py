"""Churn model loading and scoring.

The model is loaded once and cached, so an API request pays no file-parsing cost. Scoring
uses the booster's `inplace_predict`, which skips the DMatrix construction that
`predict_proba` performs on every call, and runs the booster with one thread, because a
thread pool costs more than it saves on a handful of rows.

Contract
--------
`predict_proba(features)` takes a DataFrame with the `FEATURE_ORDER` columns (or a numeric
array of that width) and returns one probability per row, in input order.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ..config.settings import (
    DECISION_THRESHOLD,
    FEATURE_ORDER,
    MODEL_METADATA_PATH,
    MODEL_PATH,
)

MODEL_VERSION = "unfitted"

_MODEL: XGBClassifier | None = None
_METADATA: dict | None = None


class ModelNotReadyError(RuntimeError):
    """Raised when the churn model is requested before it has been trained."""


def is_model_available() -> bool:
    """True when a fitted model file exists on disk."""
    return Path(MODEL_PATH).exists()


def load_metadata() -> dict:
    """Metadata written by `trainer.save_model` (metrics, cut-off, version)."""
    global _METADATA, MODEL_VERSION
    if _METADATA is None:
        if not Path(MODEL_METADATA_PATH).exists():
            raise ModelNotReadyError(f"model metadata not found: {MODEL_METADATA_PATH}")
        _METADATA = json.loads(Path(MODEL_METADATA_PATH).read_text(encoding="utf-8"))
        MODEL_VERSION = str(_METADATA.get("model_version", "unknown"))
    return _METADATA


def load_model() -> XGBClassifier:
    """Load the fitted model once and keep it in memory."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not Path(MODEL_PATH).exists():
        raise ModelNotReadyError(
            f"no fitted model at {MODEL_PATH}; run "
            "'python -m bank_churn.models.trainer' first"
        )

    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    model.set_params(n_jobs=1)  # single-threaded inference: lower latency on small batches
    _MODEL = model
    try:
        load_metadata()
    except ModelNotReadyError:
        MODEL_VERSION  # keep the default when metadata is absent
    return model


def get_threshold() -> float:
    """Operating cut-off from the metadata, else the project default."""
    try:
        return float(load_metadata().get("threshold", DECISION_THRESHOLD))
    except ModelNotReadyError:
        return DECISION_THRESHOLD


def _as_matrix(features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Validate the input and return a contiguous float32 matrix."""
    if isinstance(features, pd.DataFrame):
        missing = [col for col in FEATURE_ORDER if col not in features.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing}")
        matrix = features[FEATURE_ORDER].to_numpy()
    else:
        matrix = np.asarray(features)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] != len(FEATURE_ORDER):
            raise ValueError(
                f"expected {len(FEATURE_ORDER)} feature columns, got {matrix.shape[1]}"
            )
    return np.ascontiguousarray(matrix, dtype=np.float32)


def predict_proba(features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Churn probability for every row, in input order."""
    model = load_model()
    matrix = _as_matrix(features)
    proba = model.get_booster().inplace_predict(matrix, validate_features=False)
    return np.asarray(proba, dtype=float).ravel()


def predict_churn(features: pd.DataFrame | np.ndarray, threshold: float | None = None) -> np.ndarray:
    """0/1 churn flag for every row, using the stored operating cut-off."""
    cut = get_threshold() if threshold is None else threshold
    return (predict_proba(features) >= cut).astype(int)


def score(features: pd.DataFrame | np.ndarray, threshold: float | None = None) -> pd.DataFrame:
    """Frame with `churn_probability` and `churn_flag` per row."""
    cut = get_threshold() if threshold is None else threshold
    proba = predict_proba(features)
    return pd.DataFrame(
        {"churn_probability": proba, "churn_flag": (proba >= cut).astype(int)},
        index=getattr(features, "index", None),
    )


def warm_up() -> bool:
    """Preload the model, so the first request is as fast as the rest. True when loaded."""
    try:
        load_model()
        return True
    except ModelNotReadyError:
        return False


__all__ = [
    "FEATURE_ORDER", "MODEL_METADATA_PATH", "MODEL_PATH", "MODEL_VERSION",
    "ModelNotReadyError", "get_threshold", "is_model_available", "load_metadata",
    "load_model", "predict_churn", "predict_proba", "score", "warm_up",
]
