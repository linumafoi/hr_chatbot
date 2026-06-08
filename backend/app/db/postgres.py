"""PostgreSQL access for the 'chatbot' semantic store.

Holds the RAG corpus and SQL-agent knowledge:
  - hr_documents   (policy / handbook chunks + embeddings)
  - hr_faq         (question/answer pairs + embeddings)
  - schema_tables  (NL descriptions of hrms tables + embeddings)
  - schema_domains (business domain descriptions + embeddings)
  - sql_examples   (NL->SQL few-shot pairs; currently empty)

Uses a vector column when pgvector is available, otherwise falls back to
in-Python cosine similarity over a JSON/array embedding column.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..config import settings

logger = logging.getLogger("hr.pg")

try:
    import asyncpg  # type: ignore
except Exception:  # pragma: no cover
    asyncpg = None


class PostgresClient:
    def __init__(self) -> None:
        self._pool: Optional["asyncpg.Pool"] = None
        self.available = False

    async def connect(self) -> None:
        if asyncpg is None:
            logger.warning("asyncpg not installed - Postgres features disabled")
            return
        try:
            self._pool = await asyncpg.create_pool(
                dsn=settings.pg_dsn, min_size=1, max_size=8, command_timeout=30
            )
            self.available = True
            logger.info("Connected to PostgreSQL '%s'", settings.pg_db)
        except Exception as exc:
            self.available = False
            logger.warning("PostgreSQL unavailable (%s) - running without RAG store", exc)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def fetch(self, sql: str, *args) -> List[Dict[str, Any]]:
        if not self._pool:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
            return [dict(r) for r in rows]

    # ---------- Vector search ----------
    async def vector_search(
        self,
        table: str,
        query_embedding: List[float],
        top_k: int,
        select_cols: str = "*",
        embedding_col: str = "embedding",
        where: str = "",
    ) -> List[Dict[str, Any]]:
        """Return top_k rows ordered by cosine distance to the query embedding.

        Tries pgvector's `<=>` operator first; on failure falls back to a plain
        fetch (caller can rerank). Safe table/column names are expected from
        config, not user input.
        """
        if not self._pool:
            return []
        vec_literal = "[" + ",".join(f"{x:.6f}" for x in query_embedding) + "]"
        where_clause = f"WHERE {where}" if where else ""
        # pgvector path
        sql = (
            f"SELECT {select_cols}, ({embedding_col} <=> $1::vector) AS distance "
            f"FROM {table} {where_clause} "
            f"ORDER BY {embedding_col} <=> $1::vector ASC LIMIT {int(top_k)}"
        )
        try:
            return await self.fetch(sql, vec_literal)
        except Exception as exc:
            logger.debug("pgvector search failed on %s (%s); falling back", table, exc)
            try:
                fallback = f"SELECT {select_cols} FROM {table} {where_clause} LIMIT {int(top_k)}"
                return await self.fetch(fallback)
            except Exception as exc2:
                logger.warning("Fallback fetch failed on %s: %s", table, exc2)
                return []


pg = PostgresClient()
