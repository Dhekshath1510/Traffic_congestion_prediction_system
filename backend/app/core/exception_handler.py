from typing import Any

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.exceptions import ApplicationError


async def application_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle expected application errors.

    Args:
        request: Incoming HTTP request.
        exc: Raised exception.

    Returns:
        Standardized JSON error response.
    """
    if not isinstance(exc, ApplicationError):
        logger.error(
            "Invalid exception type received by application handler: {}",
            type(exc).__name__,
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred.",
                "details": None,
            },
        )

    logger.warning(
        "Application error on {} {}: {}",
        request.method,
        request.url.path,
        exc.message,
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "error_code": exc.error_code,
            "message": exc.message,
            "details": None,
        },
    )


async def validation_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle FastAPI request validation errors.

    Args:
        request: Incoming HTTP request.
        exc: Raised exception.

    Returns:
        Standardized validation error response.
    """
    if not isinstance(exc, RequestValidationError):
        logger.error(
            "Invalid exception type received by validation handler: {}",
            type(exc).__name__,
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred.",
                "details": None,
            },
        )

    logger.warning(
        "Validation error on {} {}",
        request.method,
        request.url.path,
    )

    errors: list[dict[str, Any]] = []

    for error in exc.errors():
        errors.append(
            {
                "location": error.get("loc", []),
                "message": error.get("msg", ""),
                "type": error.get("type", ""),
            }
        )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "message": "Request validation failed.",
            "details": {
                "errors": errors,
            },
        },
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected application exceptions.

    Args:
        request: Incoming HTTP request.
        exc: Unexpected exception.

    Returns:
        Standardized internal-server-error response.
    """
    logger.exception(
        "Unhandled exception on {} {}",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred.",
            "details": None,
        },
    )
