from typing import Literal
from pydantic import BaseModel

HealthStatus = Literal["healthy", "unhealthy"]


class ComponentHealth(BaseModel):
    """Health information for an individual infrastructure component."""

    status: HealthStatus


class HealthResponse(BaseModel):
    """Response returned by the health endpoints."""

    status: HealthStatus
    service: str
    version: str


class ReadinessResponse(BaseModel):
    """Response returned by the readiness endpoint."""

    status: HealthStatus
    service: str
    version: str
    postgres: ComponentHealth
    redis: ComponentHealth
