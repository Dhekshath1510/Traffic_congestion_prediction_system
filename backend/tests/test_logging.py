from loguru import logger

from app.core.logging import configure_logging


def test_configure_logging() -> None:
    """Verify that application logging can be configured."""
    configure_logging("INFO")

    assert logger is not None
