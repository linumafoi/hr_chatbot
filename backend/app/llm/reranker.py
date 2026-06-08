"""qwen3-Reranker-8B cross-encoder reranking (steps 8.4 and 9.3).

Calls a /rerank endpoint (vLLM score API / TEI / custom) that scores each
(query, document) pair. Falls back to identity ordering (keeps retrieval order)
when the reranker is unavailable, so the pipeline still returns results.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from ..config import settings

logger = logging.getLogger("hr.rerank")


class RerankClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.rerank_base_url,
            timeout=settings.request_timeout_seconds,
            headers={"Authorization": f"Bearer {settings.rerank_api_key}"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def rerank(
        self, query: str, documents: List[Dict[str, Any]], text_key: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """Return the top_k documents reordered by relevance to the query.

        Each returned doc gets a 'rerank_score' field.
        """
        if not documents:
            return []
        texts = [str(d.get(text_key, "")) for d in documents]
        try:
            resp = await self._client.post(
                "/rerank",
                json={"model": settings.rerank_model, "query": query, "documents": texts},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            # results: [{index, relevance_score}, ...]
            ordered = sorted(results, key=lambda r: r["relevance_score"], reverse=True)
            out = []
            for r in ordered[:top_k]:
                doc = dict(documents[r["index"]])
                doc["rerank_score"] = float(r["relevance_score"])
                out.append(doc)
            return out
        except Exception as exc:
            if settings.graceful_fallback:
                logger.warning("Reranker unavailable (%s) - keeping retrieval order", exc)
                out = []
                for i, d in enumerate(documents[:top_k]):
                    doc = dict(d)
                    doc.setdefault("rerank_score", 1.0 - i * 0.01)
                    out.append(doc)
                return out
            raise


reranker = RerankClient()
