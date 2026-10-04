from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.ml.runtime.artifacts import ModelArtifactPaths
from app.ml.runtime.model_registry import ModelRegistry
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
)
from app.services.prediction_service import (
    PredictionService,
    ReferenceSpeedProvider,
)


router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"],
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]

ARTIFACT_PATHS = ModelArtifactPaths.from_project_root(
    PROJECT_ROOT,
)

MODEL_REGISTRY = ModelRegistry(
    ARTIFACT_PATHS,
)

MODEL_REGISTRY.load()

REFERENCE_SPEED_PROVIDER = ReferenceSpeedProvider(
    PROJECT_ROOT
    / "results"
    / "classification"
    / "sensor_reference_speeds.csv",
)

PREDICTION_SERVICE = PredictionService(
    registry=MODEL_REGISTRY,
    reference_speed_provider=REFERENCE_SPEED_PROVIDER,
)


def get_prediction_service() -> PredictionService:
    """Provide the application prediction service."""

    return PREDICTION_SERVICE


@router.post(
    "",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
    service: PredictionService = Depends(
        get_prediction_service
    ),
) -> PredictionResponse:
    """Predict future traffic congestion."""

    try:
        return service.predict(request)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc