"""Step 4 - Intent classification using qwen3-8B.

Classes: faq, policy_search, database_query, general_chat, workflow.

Uses the LLM for accuracy with a fast deterministic keyword pre-classifier that
short-circuits obvious cases (keeps latency down) and acts as the fallback when
the model server is unavailable.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Tuple

from ..config import settings
from ..llm.client import llm

logger = logging.getLogger("hr.intent")

INTENTS = ["faq", "policy_search", "database_query", "general_chat", "workflow"]

_DB_HINTS = re.compile(
    r"\b(my|show|list|how many|count|balance|salary|leave balance|attendance|"
    r"payslip|details|record|status|remaining|taken|joined|department|designation|"
    r"manager|employment)\b", re.IGNORECASE)
_POLICY_HINTS = re.compile(
    r"\b(policy|policies|rule|guideline|eligib|entitle|procedure|notice period|"
    r"probation|handbook)\b", re.IGNORECASE)
_FAQ_HINTS = re.compile(
    r"\b(what is|what are|how do i|how to|where|when|who|can i|working hours|"
    r"holiday list|contact)\b", re.IGNORECASE)
_WORKFLOW_HINTS = re.compile(
    r"\b(apply|raise|request|submit|initiate|approve|reset|update my|change my|"
    r"book|register a complaint)\b", re.IGNORECASE)
_CHAT_HINTS = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|good (morning|evening|afternoon)|bye|"
    r"how are you|who are you|what can you do)\b", re.IGNORECASE)

SYSTEM = (
    "You are an intent classifier for an HR assistant. "
    "Classify the user message into exactly one of: "
    "faq, policy_search, database_query, general_chat, workflow.\n"
    "- database_query: needs employee-specific data (leave balance, salary, "
    "employment details, attendance, personal records).\n"
    "- policy_search: questions about company policies/rules/procedures.\n"
    "- faq: general company info / how-tos answerable from a knowledge base.\n"
    "- workflow: the user wants to perform an action (apply leave, raise request).\n"
    "- general_chat: greetings, smalltalk, capability questions.\n"
    'Respond ONLY as JSON: {"intent": "<label>", "confidence": <0..1>}'
)


def _keyword_intent(text: str) -> Tuple[str, float]:
    if _CHAT_HINTS.search(text):
        return "general_chat", 0.7
    if _WORKFLOW_HINTS.search(text):
        return "workflow", 0.6
    if _DB_HINTS.search(text):
        return "database_query", 0.6
    if _POLICY_HINTS.search(text):
        return "policy_search", 0.6
    if _FAQ_HINTS.search(text):
        return "faq", 0.55
    return "faq", 0.4


async def classify_intent(message: str) -> Tuple[str, float]:
    kw_intent, kw_conf = _keyword_intent(message)
    # very confident greetings short-circuit (no LLM round-trip)
    if kw_intent == "general_chat" and kw_conf >= 0.7:
        return kw_intent, kw_conf

    raw = await llm.chat_safe(
        settings.intent_model,
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": message}],
        temperature=0.0,
        max_tokens=64,
        fallback="",
    )
    if not raw:
        return kw_intent, kw_conf
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0) if match else raw)
        intent = str(data.get("intent", "")).strip().lower()
        conf = float(data.get("confidence", 0.6))
        if intent in INTENTS:
            return intent, max(0.0, min(conf, 1.0))
    except Exception as exc:
        logger.debug("intent parse failed (%s); using keyword fallback", exc)
    return kw_intent, kw_conf
