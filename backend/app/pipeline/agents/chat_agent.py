"""Step 10 - General chat agent.

Handles greetings, smalltalk and capability questions using the main LLM
(qwen3-14B). Keeps responses short, friendly, and on-brand for an HR assistant.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ...config import settings
from ...llm.client import llm
from ..observability import Trace

SYSTEM = (
    "You are a friendly HR virtual assistant for employees. Keep replies short "
    "and warm. You can help with HR policies, FAQs, leave balances, payroll and "
    "employment details. If asked what you can do, briefly list those. Do not "
    "make up company-specific facts."
)


async def run_chat(message: str, history: List[Dict[str, Any]], trace: Trace) -> Dict[str, Any]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for turn in history[-4:]:
        role = "assistant" if turn.get("role") == "assistant" else "user"
        msgs.append({"role": role, "content": turn.get("content", "")})
    msgs.append({"role": "user", "content": message})

    with trace.span("chat.generate"):
        answer = await llm.chat_safe(
            settings.chat_model, msgs, temperature=0.6, max_tokens=300,
            fallback="Hello! I'm your HR assistant. I can help with policies, "
                     "FAQs, your leave balance and employment details. "
                     "What would you like to know?",
        )
    return {"agent": "chat", "answer": answer, "confidence": 0.8, "sources": []}
