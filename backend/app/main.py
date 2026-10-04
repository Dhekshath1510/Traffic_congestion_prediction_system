from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from loguru import logger
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exception_handler import (
    application_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.core.exceptions import ApplicationError
from app.core.logging import configure_logging


def create_application() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    settings = get_settings()

    configure_logging(settings.log_level)

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-based traffic congestion prediction system.",
        debug=settings.debug,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(api_router)

    application.add_exception_handler(
        ApplicationError,
        application_exception_handler,
    )

    application.add_exception_handler(
        RequestValidationError,
        validation_exception_handler,
    )

    application.add_exception_handler(
        Exception,
        unhandled_exception_handler,
    )

    logger.info(
        "Application initialized: {} v{}",
        settings.app_name,
        settings.app_version,
    )

    return application


app = create_application()
