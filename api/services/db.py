"""Async database utilities for the clip pipeline."""

from __future__ import annotations

import asyncio
from typing import Optional

import asyncpg

_lock = asyncio.Lock()


class Database:
    """Manage a shared asyncpg connection pool."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None

    async def get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            async with _lock:
                if self._pool is None:
                    self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


_database_instance: Optional[Database] = None


def get_database(dsn: str) -> Database:
    global _database_instance
    if _database_instance is None or _database_instance._dsn != dsn:
        _database_instance = Database(dsn)
    return _database_instance
