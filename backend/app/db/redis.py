from typing import Any, Callable, cast
from redis import Redis
from app.core.config import get_settings


def create_redis_client() -> Redis:
    """Create a Redis client from application settings.

    Returns:
        Configured Redis client.
    """
    settings = get_settings()

    from_url = cast(
        Callable[..., Redis],
        getattr(Redis, "from_url"),
    )

    return from_url(
        settings.redis_url,
        decode_responses=True,
    )


def check_redis_connection() -> bool:
    """Check whether Redis is reachable.

    Returns:
        True when Redis responds successfully; otherwise False.
    """
    client = create_redis_client()

    try:
        ping = cast(
            Callable[[], Any],
            getattr(client, "ping"),
        )

        return bool(ping())
    except Exception:
        return False
    finally:
        client.close()
