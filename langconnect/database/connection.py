import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Optional, Union

import asyncpg
import sqlalchemy
from langchain_core.embeddings import Embeddings
from langchain_postgres.vectorstores import PGVector
from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import AsyncEngine

from langconnect import config

logger = logging.getLogger(__name__)


_pool: asyncpg.Pool | None = None
_engine: Engine | None = None


async def get_db_pool() -> asyncpg.Pool:
    """Get the pg connection pool."""
    global _pool
    if _pool is None:
        # Use parsed components for asyncpg connection
        _pool = await asyncpg.create_pool(
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            database=config.POSTGRES_DB,
        )
        logger.info("Database connection pool created using parsed URL components.")
    return _pool


async def close_db_pool():
    """Close the pg connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def get_db_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """Acquire a connection from the pool and release it when done."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        yield conn


def get_vectorstore_engine(
    host: str = config.POSTGRES_HOST,
    port: str = config.POSTGRES_PORT,
    user: str = config.POSTGRES_USER,
    password: str = config.POSTGRES_PASSWORD,
    dbname: str = config.POSTGRES_DB,
) -> Engine:
    """Creates and returns a cached sync SQLAlchemy engine for PostgreSQL."""
    global _engine
    if _engine is None:
        connection_string = (
            f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"
        )
        _engine = create_engine(connection_string, pool_size=5, max_overflow=10)
        logger.info("SQLAlchemy engine created (singleton).")
    return _engine


def close_engine():
    """Dispose the cached SQLAlchemy engine."""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None
        logger.info("SQLAlchemy engine disposed.")


DBConnection = Union[sqlalchemy.engine.Engine, str]


def get_vectorstore(
    collection_name: str = config.DEFAULT_COLLECTION_NAME,
    embeddings: Optional[Embeddings] = None,
    engine: Optional[Union[DBConnection, Engine, AsyncEngine]] = None,
    collection_metadata: Optional[dict[str, Any]] = None,
) -> PGVector:
    """Initializes and returns a PGVector store for a specific collection,
    using an existing engine or creating one from connection parameters.
    """
    if embeddings is None:
        embeddings = config.get_embeddings()
    if engine is None:
        engine = get_vectorstore_engine()

    store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=engine,
        use_jsonb=True,
        collection_metadata=collection_metadata,
    )
    return store
