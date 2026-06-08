"""Reranking (steps 8.4 and 9.3).

Two modes (config RERANK_MODE):
  * "http" - call a dedicated cross-encoder rerank server (vLLM score API / TEI)
             that returns per-document relevance scores. Best quality.
  * "llm"  - a single *listwise* pass through an instruct model (default for
             Ollama, which has no native rerank API): the model is asked to
             return the most relevant document indices in order. Approximate but
             works with a plain Ollama install.

Both modes fall back to identity ordering (keep retrieval order) when the
endpoint is unavailable, so the pipeline always returns results.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

import httpx

from ..config import settings

logger = logging.getLogger("hr.rerank")

_LLM_SYSTEM = (
    "You are a search reranker. Given a query and a numbered list of documents, "
    "return the indices of the documents most relevant to the query, ordered "
    "from most to least relevant. Respond ONLY with a JSON array of integers, "
    'e.g. [3, 1, 5]. Do not include any other text.'
)


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
        """Return the top_k documents reordered by relevance (adds 'rerank_score')."""
        if not documents:
            return []
        try:
            if settings.rerank_mode.lower() == "http":
                return await self._rerank_http(query, documents, text_key, top_k)
            return await self._rerank_llm(query, documents, text_key, top_k)
        except Exception as exc:
            if settings.graceful_fallback:
                logger.warning("Reranker unavailable (%s) - keeping retrieval order", exc)
                return self._identity(documents, top_k)
            raise

    # ---------- HTTP cross-encoder ----------
    async def _rerank_http(self, query, documents, text_key, top_k):
        texts = [str(d.get(text_key, "")) for d in documents]
        resp = await self._client.post(
            "/rerank",
            json={"model": settings.rerank_model, "query": query, "documents": texts},
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        ordered = sorted(results, key=lambda r: r["relevance_score"], reverse=True)
        out = []
        for r in ordered[:top_k]:
            doc = dict(documents[r["index"]])
            doc["rerank_score"] = float(r["relevance_score"])
            out.append(doc)
        return out

    # ---------- LLM listwise (Ollama) ----------
    async def _rerank_llm(self, query, documents, text_key, top_k):
        listing = "\n".join(
            f"[{i}] {str(d.get(text_key, ''))[:500]}" for i, d in enumerate(documents)
        )
        user = f"Query: {query}\n\nDocuments:\n{listing}\n\nMost relevant indices (JSON array):"
        resp = await self._client.post(
            "/chat/completions",
            json={
                "model": settings.rerank_model,
                "messages": [
                    {"role": "system", "content": _LLM_SYSTEM},
                    {"role": "user", "content": user + " /no_think"},
                ],
                "temperature": 0.0,
                "max_tokens": 128,
                "stream": False,
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        order = self._parse_indices(content, len(documents))
        if not order:
            return self._identity(documents, top_k)
        out = []
        for rank, idx in enumerate(order[:top_k]):
            doc = dict(documents[idx])
            doc["rerank_score"] = round(1.0 - rank / max(len(order), 1), 4)
            out.append(doc)
        return out

    @staticmethod
    def _parse_indices(text: str, n: int) -> List[int]:
        if "</think>" in text:
            text = text.split("</think>")[-1]
        m = re.search(r"\[[\d,\s]*\]", text)
        if not m:
            return []
        try:
            idxs = json.loads(m.group(0))
        except Exception:
            return []
        seen, out = set(), []
        for i in idxs:
            if isinstance(i, int) and 0 <= i < n and i not in seen:
                seen.add(i)
                out.append(i)
        return out

    @staticmethod
    def _identity(documents, top_k):
        out = []
        for i, d in enumerate(documents[:top_k]):
            doc = dict(d)
            doc.setdefault("rerank_score", 1.0 - i * 0.01)
            out.append(doc)
        return out


reranker = RerankClient()
