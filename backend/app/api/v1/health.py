from fastapi import APIRouter, status, Response

from app.core.config import get_settings
from app.db.connection import create_database_engine, check_database_connection
from app.db.redis import check_redis_connection
from app.schemas.health import (
    ComponentHealth,
    HealthResponse,
    ReadinessResponse,
)

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)

settings = get_settings()


@router.get(
    "/live",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
)
async def liveness_check() -> HealthResponse:
    """Check whether the API process is running.

    Returns:
        Basic service health information.
    """
    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
)
async def readiness_check(response: Response) -> ReadinessResponse:
    """Check whether required infrastructure is available.

    Returns:
        Service and infrastructure readiness information.
    """
    postgres_healthy = await check_database_connection(
        engine=create_database_engine(),
    )

    redis_healthy = check_redis_connection()

    overall_healthy = postgres_healthy and redis_healthy

    if not overall_healthy:
        response.status_code = 503

    return ReadinessResponse(
        status="healthy" if overall_healthy else "unhealthy",
        service=settings.app_name,
        version=settings.app_version,
        postgres=ComponentHealth(
            status="healthy" if postgres_healthy else "unhealthy",
        ),
        redis=ComponentHealth(
            status="healthy" if redis_healthy else "unhealthy",
        ),
    )
