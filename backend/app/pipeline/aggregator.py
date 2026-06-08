"""Step 11 - Response aggregator.

Merges agent outputs (when more than one agent ran), attaches sources and a
confidence score, and shapes the final payload (text / table / chart) for the
UI (step 15).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _maybe_chart(table: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Suggest a simple chart spec when the table looks like label/number pairs."""
    if not table or len(table) < 2:
        return None
    cols = list(table[0].keys())
    if len(cols) != 2:
        return None
    label_col, value_col = cols
    numeric = []
    for row in table:
        try:
            numeric.append(float(row[value_col]))
        except (TypeError, ValueError):
            return None
    return {
        "type": "bar",
        "labels": [str(r[label_col]) for r in table],
        "values": numeric,
        "x": label_col,
        "y": value_col,
    }


def aggregate(state: Dict[str, Any]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = [
        r for r in (state.get("rag_result"), state.get("sql_result"), state.get("chat_result"))
        if r and r.get("answer")
    ]

    if not results:
        return {
            "answer": "I'm sorry, I couldn't find an answer to that. "
                      "Please try rephrasing or contact HR.",
            "agent": "none",
            "confidence": 0.0,
            "sources": [],
            "table": None,
            "chart": None,
        }

    if len(results) == 1:
        primary = results[0]
        table = primary.get("table")
        return {
            "answer": primary["answer"],
            "agent": primary.get("agent", "unknown"),
            "confidence": round(primary.get("confidence", 0.0), 2),
            "sources": primary.get("sources", []),
            "table": table,
            "chart": _maybe_chart(table),
        }

    # Multi-agent merge: order by confidence, concatenate distinct answers
    results.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)
    merged_answer = "\n\n".join(r["answer"] for r in results)
    sources: List[str] = []
    for r in results:
        for s in r.get("sources", []):
            if s not in sources:
                sources.append(s)
    table = next((r.get("table") for r in results if r.get("table")), None)
    confidence = round(sum(r.get("confidence", 0.0) for r in results) / len(results), 2)

    return {
        "answer": merged_answer,
        "agent": "+".join(r.get("agent", "?") for r in results),
        "confidence": confidence,
        "sources": sources,
        "table": table,
        "chart": _maybe_chart(table),
    }
