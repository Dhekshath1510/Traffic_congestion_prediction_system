from fastapi import APIRouter
from app.api.v1.health import router as health_router
from app.api.v1.predictions import router as predictions_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health_router)
api_router.include_router(predictions_router)
