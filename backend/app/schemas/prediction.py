from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    sequence: list[list[float]] = Field(
        ...,
        description="Historical speed sequence: 12 timesteps × 207 sensors.",
        min_length=12,
        max_length=12,
    )
    timestamp: datetime


class SHAPFeatureContribution(BaseModel):
    feature: str
    contribution: float


class SHAPExplanationResponse(BaseModel):
    predicted_class: str
    predicted_class_id: int
    top_features: list[SHAPFeatureContribution]


class SensorPrediction(BaseModel):
    sensor_index: int
    predicted_speed: float
    congestion_index: float
    congestion_class: str
    probabilities: dict[str, float]
    explanation: SHAPExplanationResponse | None = None


class PredictionResponse(BaseModel):
    timestamp: datetime
    forecast_horizon: str
    sensors: list[SensorPrediction]