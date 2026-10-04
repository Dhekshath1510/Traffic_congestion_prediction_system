from collections.abc import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings


def create_database_engine() -> AsyncEngine:
    """Create the asynchronous PostgreSQL engine.

    Returns:
        Configured SQLAlchemy asynchronous engine.
    """
    settings = get_settings()

    return create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
    )


def create_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Create an asynchronous SQLAlchemy session factory.

    Args:
        engine: SQLAlchemy asynchronous engine.

    Returns:
        Configured asynchronous session factory.
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def check_database_connection(
    engine: AsyncEngine,
) -> bool:
    """Check whether PostgreSQL is reachable.

    Args:
        engine: SQLAlchemy asynchronous engine.

    Returns:
        True when PostgreSQL responds successfully.
    """
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

        return True

    except Exception:
        return False


async def get_database_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Provide an asynchronous database session.

    Args:
        session_factory: SQLAlchemy session factory.

    Yields:
        Active database session.
    """
    async with session_factory() as session:
        yield session
