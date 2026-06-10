"""Step 8 - RAG Agent (FAQ + Policy search).

8.1  Embed the rewritten query with BGE-M3.
8.2  Retrieve from PostgreSQL chatbot DB: hr_documents + hr_faq.
8.3  Keep the top 20 candidates.
8.4  Rerank with qwen3-Reranker-8B and keep the best few.
Then generate a grounded answer with qwen3-14B, citing sources.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from ...config import settings
from ...db.postgres import pg
from ...llm.client import llm
from ...llm.embeddings import embeddings
from ...llm.reranker import reranker
from ..observability import Trace

logger = logging.getLogger("hr.rag")

ANSWER_SYSTEM = (
    "You are a helpful HR assistant. Using ONLY the provided context passages, "
    "write a clear, well-structured answer to the employee's question in plain, "
    "easy-to-understand language.\n"
    "Guidelines:\n"
    "- Directly answer the question first, then add relevant details.\n"
    "- Use short paragraphs or bullet points for eligibility, limits, steps, etc.\n"
    "- Summarise and paraphrase the policy; do not paste raw fragments.\n"
    "- Preserve specific figures, levels and limits exactly as written.\n"
    "- If the answer is not in the context, say you don't have that information "
    "and suggest contacting HR. Do not invent policy numbers, dates or amounts."
)


async def _retrieve(query_vec: List[float], query_text: str, trace: Trace) -> List[Dict[str, Any]]:
    """8.2 + 8.3 - retrieve from hr_documents and hr_faq, merge top 20."""
    candidates: List[Dict[str, Any]] = []

    # hr_faq: question + answer
    faq = await pg.vector_search(
        table="hr_faq",
        query_embedding=query_vec,
        top_k=settings.rag_top_k,
        select_cols="id, question, answer",
    )
    for r in faq:
        candidates.append({
            "source": "hr_faq",
            "ref": f"FAQ#{r.get('id')}",
            "text": f"Q: {r.get('question','')}\nA: {r.get('answer','')}",
            "distance": r.get("distance", 1.0),
        })

    # hr_documents: policy/handbook chunks
    docs = await pg.vector_search(
        table="hr_documents",
        query_embedding=query_vec,
        top_k=settings.rag_top_k,
        select_cols="id, source_file, content",
    )
    for r in docs:
        candidates.append({
            "source": "hr_documents",
            "ref": r.get("source_file") or f"DOC#{r.get('id')}",
            "text": str(r.get("content", "")),
            "distance": r.get("distance", 1.0),
        })

    candidates.sort(key=lambda c: c.get("distance", 1.0))
    top = candidates[: settings.rag_top_k]
    trace.event("rag.retrieved", faq=len(faq), docs=len(docs), kept=len(top))
    return top


async def run_rag(rewritten_query: str, trace: Trace) -> Dict[str, Any]:
    result: Dict[str, Any] = {"agent": "rag", "sources": [], "confidence": 0.0}

    # 8.1 embed
    with trace.span("rag.embed"):
        query_vec = await embeddings.embed(rewritten_query)
    if not query_vec:
        result["answer"] = ""
        result["error"] = "embeddings_unavailable"
        return result

    # 8.2 / 8.3 retrieve + top20
    with trace.span("rag.retrieve"):
        candidates = await _retrieve(query_vec, rewritten_query, trace)
    if not candidates:
        result["answer"] = ""
        result["error"] = "no_documents"
        return result

    # 8.4 rerank
    with trace.span("rag.rerank"):
        reranked = await reranker.rerank(
            rewritten_query, candidates, text_key="text", top_k=settings.rag_rerank_k
        )
    trace.event("rag.reranked", kept=len(reranked))

    context_block = "\n\n---\n\n".join(
        f"[{d['ref']}]\n{d['text']}" for d in reranked
    )
    sources = list(dict.fromkeys(d["ref"] for d in reranked))  # unique, ordered

    # answer generation (qwen3-14B)
    with trace.span("rag.generate"):
        answer = await llm.chat_safe(
            settings.answer_model,
            [{"role": "system", "content": ANSWER_SYSTEM},
             {"role": "user",
              "content": f"Context:\n{context_block}\n\nQuestion: {rewritten_query}"}],
            temperature=0.2,
            max_tokens=600,
            fallback="",
        )

    if not answer:
        # Extractive fallback (answer model unavailable): surface the most
        # relevant passages, trimmed to whole sentences so it reads cleanly.
        answer = _extractive_answer(reranked)
        result["confidence"] = 0.4
    else:
        top_score = reranked[0].get("rerank_score", 0.6)
        result["confidence"] = round(min(0.95, 0.5 + top_score / 2), 2)

    result["answer"] = answer
    result["sources"] = sources
    return result


def _extractive_answer(reranked: List[Dict[str, Any]]) -> str:
    """Readable fallback when the LLM can't generate: top passages, trimmed."""
    parts: List[str] = []
    for d in reranked[:2]:
        text = " ".join(str(d.get("text", "")).split())  # collapse whitespace
        snippet = text[:500]
        # trim to the last sentence boundary so we don't cut mid-word
        cut = max(snippet.rfind(". "), snippet.rfind("? "), snippet.rfind("! "))
        if cut > 120:
            snippet = snippet[: cut + 1]
        parts.append(snippet.strip())
    body = "\n\n".join(parts)
    return (
        "Here's the most relevant information I found in our HR documents:\n\n"
        f"{body}\n\n(For full details please refer to the source documents above "
        "or contact HR.)"
    )
