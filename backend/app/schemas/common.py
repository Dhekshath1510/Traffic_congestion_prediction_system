from typing import Any
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard API error response."""

    success: bool = Field(
        default=False,
        description="Indicates whether the request succeeded.",
    )

    error_code: str = Field(
        description="Machine-readable error identifier.",
    )

    message: str = Field(
        description="Human-readable error message.",
    )

    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional additional error details.",
    )
