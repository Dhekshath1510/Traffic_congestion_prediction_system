class ApplicationError(Exception):
    """Base exception for expected application errors."""

    def __init__(
        self,
        message: str,
        error_code: str,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.error_code = error_code


class ResourceNotFoundError(ApplicationError):
    """Raised when a requested resource does not exist."""

    def __init__(
        self,
        message: str,
        error_code: str = "RESOURCE_NOT_FOUND",
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
        )
