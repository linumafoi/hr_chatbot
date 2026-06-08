"""BGE-M3 embeddings via an OpenAI-compatible /embeddings endpoint.

Used by both the RAG agent (step 8.1) and the SQL agent (step 9.1) to embed
the (rewritten) query before vector retrieval from PostgreSQL.
"""
from __future__ import annotations

import logging
from typing import List

import httpx

from ..config import settings

logger = logging.getLogger("hr.embed")


class EmbeddingClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.embed_base_url,
            timeout=settings.request_timeout_seconds,
            headers={"Authorization": f"Bearer {settings.embed_api_key}"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def embed(self, text: str) -> List[float]:
        out = await self.embed_batch([text])
        return out[0] if out else []

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            resp = await self._client.post(
                "/embeddings", json={"model": settings.embed_model, "input": texts}
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as exc:
            if settings.graceful_fallback:
                logger.warning("Embedding endpoint unavailable (%s)", exc)
                return []
            raise


embeddings = EmbeddingClient()
