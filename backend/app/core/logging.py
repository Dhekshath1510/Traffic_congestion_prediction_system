import sys

from loguru import logger


def configure_logging(log_level: str = "INFO") -> None:
    """Configure application-wide logging.

    Args:
        log_level: Minimum log level to emit.
    """
    logger.remove()

    logger.add(
        sys.stdout,
        level=log_level,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:"
            "<cyan>{function}</cyan>:"
            "<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )
