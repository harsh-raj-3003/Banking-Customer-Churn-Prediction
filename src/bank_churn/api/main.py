"""FastAPI service for churn scoring.

Run it with:
    .venv/Scripts/python.exe -m uvicorn bank_churn.api.main:app --reload --port 8000

Endpoints
---------
GET  /health    service state and whether a fitted model is available
GET  /features  the input contract (column order, allowed categories, cut-off)
POST /predict   score one customer; returns 503 until study 3 supplies the model
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, status

from ..config.settings import (
    BINARY_COLUMNS,
    DECISION_THRESHOLD,
    FEATURE_ORDER,
    GENDERS,
    GEOGRAPHIES,
    NUMERIC_COLUMNS,
)
from ..features.engineering import build_model_frame, to_feature_matrix
from ..models import predictor
from .schemas import (
    CustomerFeatures,
    FeatureSpecResponse,
    HealthResponse,
    PredictionResponse,
)

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model once at startup, so the first request pays no file cost."""
    predictor.warm_up()
    yield


app = FastAPI(
    title="Bank Churn API",
    version="0.1.0",
    description="Score a bank customer's churn risk.",
    lifespan=lifespan,
)


def _customer_matrix(customer: CustomerFeatures) -> pd.DataFrame:
    """One API payload -> the 14-column training matrix."""
    raw = pd.DataFrame([customer.to_record()])
    return to_feature_matrix(build_model_frame(raw))


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness plus model availability."""
    return HealthResponse(
        status="ok",
        model_ready=predictor.is_model_available(),
        model_version=predictor.MODEL_VERSION,
        threshold=DECISION_THRESHOLD,
    )


@app.get("/features", response_model=FeatureSpecResponse, tags=["meta"])
def features() -> FeatureSpecResponse:
    """Input contract for the UI: column order and allowed values."""
    return FeatureSpecResponse(
        features=FEATURE_ORDER,
        numeric=NUMERIC_COLUMNS,
        binary=BINARY_COLUMNS,
        categorical={"Geography": GEOGRAPHIES, "Gender": GENDERS},
        threshold=DECISION_THRESHOLD,
        model_ready=predictor.is_model_available(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["scoring"],
    responses={503: {"description": "Model not trained yet (study 3 pending)"}},
)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Churn probability and flag for one customer."""
    if not predictor.is_model_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Churn model is not trained yet (study 3 pending).",
        )

    matrix = _customer_matrix(customer)
    try:
        probability = float(predictor.predict_proba(matrix)[0])
    except predictor.ModelNotReadyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return PredictionResponse(
        churn_probability=probability,
        churn_flag=probability >= DECISION_THRESHOLD,
        threshold=DECISION_THRESHOLD,
        model_version=predictor.MODEL_VERSION,
    )
