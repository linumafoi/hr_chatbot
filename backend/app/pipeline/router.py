"""Step 7 - LangGraph router / orchestrator.

Builds a small state graph:

    preprocess -> intent -> rewrite -> context -> ROUTE -> {rag|sql|chat} -> aggregate

The conditional edge after `context` chooses the agent based on the classified
intent. The graph is compiled once at import time and reused per request.

If LangGraph isn't installed the same flow runs via a plain async fallback so
the system still works.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from ..schemas import UserContext
from . import aggregator, context as context_step, intent as intent_step
from . import preprocessor, rewrite as rewrite_step
from .agents import chat_agent, rag_agent, sql_agent
from .observability import Trace
from .state import PipelineState

logger = logging.getLogger("hr.router")

try:
    from langgraph.graph import END, START, StateGraph  # type: ignore
    _HAS_LANGGRAPH = True
except Exception:  # pragma: no cover
    _HAS_LANGGRAPH = False


# ---------------- Node implementations ----------------
async def _node_preprocess(state: PipelineState) -> PipelineState:
    trace: Trace = state["context"]["_trace"]
    with trace.span("preprocess"):
        state["clean_message"] = preprocessor.preprocess(state["raw_message"])
    trace.event("preprocess", clean=state["clean_message"])
    return state


async def _node_intent(state: PipelineState) -> PipelineState:
    trace: Trace = state["context"]["_trace"]
    with trace.span("intent"):
        intent, conf = await intent_step.classify_intent(state["clean_message"])
    state["intent"] = intent
    state["intent_confidence"] = conf
    trace.event("intent", intent=intent, confidence=conf)
    return state


async def _node_rewrite(state: PipelineState) -> PipelineState:
    trace: Trace = state["context"]["_trace"]
    # Skip the rewrite LLM call when there's nothing to resolve: greetings, or a
    # first-turn question with no prior history (rewrite mainly resolves
    # pronouns/follow-ups). Saves a full LLM round-trip on modest hardware.
    if state["intent"] == "general_chat" or not state.get("history"):
        state["rewritten_query"] = state["clean_message"]
        return state
    with trace.span("rewrite"):
        state["rewritten_query"] = await rewrite_step.rewrite_query(
            state["clean_message"], state.get("history", [])
        )
    trace.event("rewrite", rewritten=state["rewritten_query"])
    return state


async def _node_context(state: PipelineState) -> PipelineState:
    trace: Trace = state["context"]["_trace"]
    user: UserContext = state["user"]
    with trace.span("context"):
        ctx = await context_step.extract_context(
            user, state["session_id"], state.get("history", []),
            state["context"].get("session_ctx", {}),
        )
    state["context"].update(ctx)
    return state


async def _node_rag(state: PipelineState) -> PipelineState:
    trace: Trace = state["context"]["_trace"]
    state["rag_result"] = await rag_agent.run_rag(state["rewritten_query"], trace)
    return state


async def _node_sql(state: PipelineState) -> PipelineState:
    trace: Trace = state["context"]["_trace"]
    state["sql_result"] = await sql_agent.run_sql(
        state["rewritten_query"], state["user"], trace
    )
    return state


async def _node_chat(state: PipelineState) -> PipelineState:
    trace: Trace = state["context"]["_trace"]
    state["chat_result"] = await chat_agent.run_chat(
        state["clean_message"], state.get("history", []), trace
    )
    return state


async def _node_aggregate(state: PipelineState) -> PipelineState:
    trace: Trace = state["context"]["_trace"]
    merged = aggregator.aggregate(state)
    state.update(merged)
    trace.event("aggregate", agent=merged["agent"], confidence=merged["confidence"])
    return state


def _route(state: PipelineState) -> str:
    """Conditional edge: map intent -> agent node name."""
    intent = state.get("intent", "faq")
    if intent in ("faq", "policy_search"):
        return "rag"
    if intent == "database_query":
        return "sql"
    if intent == "workflow":
        # No write actions are permitted; explain via chat + RAG guidance.
        return "rag"
    return "chat"


# ---------------- Graph compilation ----------------
def _build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("preprocess", _node_preprocess)
    graph.add_node("classify_intent", _node_intent)
    graph.add_node("rewrite", _node_rewrite)
    graph.add_node("build_context", _node_context)
    graph.add_node("rag", _node_rag)
    graph.add_node("sql", _node_sql)
    graph.add_node("chat", _node_chat)
    graph.add_node("aggregate", _node_aggregate)

    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "classify_intent")
    graph.add_edge("classify_intent", "rewrite")
    graph.add_edge("rewrite", "build_context")
    graph.add_conditional_edges("build_context", _route,
                                {"rag": "rag", "sql": "sql", "chat": "chat"})
    graph.add_edge("rag", "aggregate")
    graph.add_edge("sql", "aggregate")
    graph.add_edge("chat", "aggregate")
    graph.add_edge("aggregate", END)
    return graph.compile()


_COMPILED = _build_graph() if _HAS_LANGGRAPH else None


async def _run_fallback(state: PipelineState) -> PipelineState:
    """Plain sequential execution when LangGraph isn't installed."""
    state = await _node_preprocess(state)
    state = await _node_intent(state)
    state = await _node_rewrite(state)
    state = await _node_context(state)
    target = _route(state)
    if target == "rag":
        state = await _node_rag(state)
    elif target == "sql":
        state = await _node_sql(state)
    else:
        state = await _node_chat(state)
    state = await _node_aggregate(state)
    return state


async def run_pipeline(state: PipelineState) -> PipelineState:
    if _COMPILED is not None:
        return await _COMPILED.ainvoke(state)
    return await _run_fallback(state)
