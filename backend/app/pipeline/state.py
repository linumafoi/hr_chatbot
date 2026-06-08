"""Shared state object that flows through the LangGraph pipeline."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from ..schemas import UserContext


class PipelineState(TypedDict, total=False):
    # input
    raw_message: str
    session_id: str
    user: UserContext
    history: List[Dict[str, Any]]

    # step 3 - preprocessing
    clean_message: str

    # step 4 - intent
    intent: str
    intent_confidence: float

    # step 5 - rewrite
    rewritten_query: str

    # step 6 - context (kept inside `user` + this dict for routing)
    context: Dict[str, Any]

    # agent outputs
    rag_result: Optional[Dict[str, Any]]
    sql_result: Optional[Dict[str, Any]]
    chat_result: Optional[Dict[str, Any]]

    # step 11 - aggregated final
    answer: str
    agent: str
    confidence: float
    sources: List[str]
    table: Optional[List[Dict[str, Any]]]
    chart: Optional[Dict[str, Any]]
