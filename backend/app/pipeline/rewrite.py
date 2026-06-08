"""Step 5 - Query rewrite using qwen3-8B.

Rewrites the user query for clarity and better retrieval: resolves pronouns and
follow-ups using recent conversation history, expands implicit references, and
produces a standalone, search-optimized query. Falls back to the cleaned query
when the model is unavailable.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..config import settings
from ..llm.client import llm

SYSTEM = (
    "You rewrite an HR user's message into a single, standalone search query.\n"
    "Rules:\n"
    "- Resolve follow-ups/pronouns using the conversation history.\n"
    "- Keep it concise and keyword-rich for retrieval.\n"
    "- Do NOT answer the question. Output ONLY the rewritten query, nothing else."
)


def _format_history(history: List[Dict[str, Any]]) -> str:
    lines = []
    for turn in history[-6:]:
        role = turn.get("role", "user")
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)


async def rewrite_query(message: str, history: List[Dict[str, Any]]) -> str:
    hist = _format_history(history)
    user_block = (
        f"Conversation so far:\n{hist}\n\nLatest message: {message}\n\nRewritten query:"
        if hist else f"Message: {message}\n\nRewritten query:"
    )
    out = await llm.chat_safe(
        settings.rewrite_model,
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": user_block}],
        temperature=0.1,
        max_tokens=80,
        fallback=message,
    )
    out = (out or message).strip().strip('"').strip()
    # guard against the model echoing instructions or being empty
    if not out or len(out) > 400:
        return message
    return out
