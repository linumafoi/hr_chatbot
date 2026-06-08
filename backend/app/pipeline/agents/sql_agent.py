"""Step 9 - SQL Agent (Natural language -> SQL).

9.1  Embed the query with BGE-M3.
9.2  Retrieve relevant tables from PostgreSQL chatbot DB:
     schema_tables + schema_domains.
9.3  Rerank candidate tables with qwen3-Reranker-8B.
9.4  Retrieve few-shot SQL examples from chatbot.sql_examples
     (table is currently empty -> handled gracefully).
9.5  Generate SQL with arctic-text2sql (or deepseek-R1-distill-32b).
9.6  Validate SQL with qwen3-14B + static guards (read-only, allow-list).
9.7  Execute on SQL Server hrms over the allow-listed tables only,
     with row-level security injected for employees.
9.8  Return only the rows the user is permitted to see (masking applied later).
9.9  Generate a natural-language answer with qwen3-14B.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from ...config import settings
from ...db.mssql import mssql
from ...db.postgres import pg
from ...llm.client import llm
from ...llm.embeddings import embeddings
from ...llm.reranker import reranker
from ...schemas import UserContext
from .. import security
from ..observability import Trace

logger = logging.getLogger("hr.sql")

GEN_SYSTEM = (
    "You are a precise Text-to-SQL generator for Microsoft SQL Server (T-SQL).\n"
    "Generate a single read-only SELECT statement that answers the question.\n"
    "Rules:\n"
    "- Use ONLY the tables and columns described in the provided schema.\n"
    "- Never write INSERT/UPDATE/DELETE/DROP or any DDL/DML.\n"
    "- Use TOP (N) instead of LIMIT.\n"
    "- Return ONLY the SQL, no explanation, no markdown fences."
)

VALIDATE_SYSTEM = (
    "You validate T-SQL for an HR system. Given a schema and a SQL query, reply "
    "with JSON: {\"valid\": true|false, \"reason\": \"...\", \"fixed_sql\": \"...\"}. "
    "Mark invalid if it is not a read-only SELECT, references unknown tables/columns, "
    "or could mutate data. Provide fixed_sql only if a safe correction is obvious."
)

ANSWER_SYSTEM = (
    "You are an HR assistant. Given the user's question and SQL result rows, "
    "write a clear, concise natural-language answer. If rows are empty, say no "
    "matching records were found. Never expose more than the rows provided."
)


async def _retrieve_schema(query_vec: List[float], query_text: str, trace: Trace) -> List[Dict[str, Any]]:
    """9.2 + 9.3 - retrieve and rerank relevant table/domain descriptions."""
    cands: List[Dict[str, Any]] = []

    tables = await pg.vector_search(
        table="schema_tables", query_embedding=query_vec, top_k=settings.rag_top_k,
        select_cols="id, table_name, description, columns",
    )
    for r in tables:
        cands.append({
            "kind": "table",
            "name": r.get("table_name"),
            "text": f"TABLE {r.get('table_name')}: {r.get('description','')}\n"
                    f"COLUMNS: {r.get('columns','')}",
            "distance": r.get("distance", 1.0),
        })

    domains = await pg.vector_search(
        table="schema_domains", query_embedding=query_vec, top_k=settings.rag_top_k,
        select_cols="id, domain_name, description",
    )
    for r in domains:
        cands.append({
            "kind": "domain",
            "name": r.get("domain_name"),
            "text": f"DOMAIN {r.get('domain_name')}: {r.get('description','')}",
            "distance": r.get("distance", 1.0),
        })

    cands.sort(key=lambda c: c.get("distance", 1.0))
    cands = cands[: settings.rag_top_k]
    reranked = await reranker.rerank(query_text, cands, text_key="text", top_k=8)
    trace.event("sql.schema_retrieved", candidates=len(cands), kept=len(reranked))
    return reranked


async def _retrieve_examples(query_vec: List[float], trace: Trace) -> List[Dict[str, Any]]:
    """9.4 - few-shot SQL examples. Table is currently empty; returns []."""
    try:
        rows = await pg.vector_search(
            table="sql_examples", query_embedding=query_vec, top_k=3,
            select_cols="id, question, sql",
        )
    except Exception:
        rows = []
    trace.event("sql.examples", count=len(rows))
    return rows


def _clean_sql(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"```$", "", sql).strip()
    # take first statement only
    if ";" in sql:
        sql = sql.split(";")[0]
    return sql.strip()


def _build_schema_block(schema_docs: List[Dict[str, Any]]) -> str:
    # Constrain to allow-listed tables for safety/clarity
    allowed = {t.lower() for t in settings.allowed_tables}
    lines = []
    for d in schema_docs:
        if d["kind"] == "table" and d.get("name", "").lower() not in allowed:
            continue
        lines.append(d["text"])
    if not lines:
        # minimal default schema hint so generation still works
        lines = [f"TABLE {t}: (HR table)" for t in settings.allowed_tables]
    return "\n\n".join(lines)


async def run_sql(rewritten_query: str, ctx: UserContext, trace: Trace) -> Dict[str, Any]:
    result: Dict[str, Any] = {"agent": "sql", "sources": ["hrms"], "confidence": 0.0}

    # RBAC: must be allowed to read data
    if not (security.check_permission(ctx, "self.data.read")
            or security.check_permission(ctx, "all.data.read")):
        result["answer"] = "You don't have permission to access employee data."
        result["error"] = "forbidden"
        return result

    # 9.1 embed
    with trace.span("sql.embed"):
        qvec = await embeddings.embed(rewritten_query)

    # 9.2 / 9.3 schema retrieval + rerank
    schema_docs: List[Dict[str, Any]] = []
    if qvec:
        with trace.span("sql.schema"):
            schema_docs = await _retrieve_schema(qvec, rewritten_query, trace)
    schema_block = _build_schema_block(schema_docs)

    # 9.4 few-shot examples (empty table handled)
    examples = await _retrieve_examples(qvec, trace) if qvec else []
    example_block = ""
    if examples:
        example_block = "\n\nExamples:\n" + "\n".join(
            f"-- {e.get('question','')}\n{e.get('sql','')}" for e in examples
        )

    rls_note = (
        "All results MUST be limited to the current employee only "
        f"(employee_id = {ctx.employee_id})." if not ctx.is_admin
        else "The user is an administrator and may query all employees."
    )

    # 9.5 generate SQL
    with trace.span("sql.generate"):
        gen_user = (
            f"Schema (only these tables may be used):\n{schema_block}\n"
            f"{example_block}\n\n"
            f"Access scope: {rls_note}\n"
            f"Allowed tables: {', '.join(settings.allowed_tables)}\n\n"
            f"Question: {rewritten_query}\n\nSQL:"
        )
        raw_sql = await llm.chat_safe(
            settings.sql_gen_model,
            [{"role": "system", "content": GEN_SYSTEM},
             {"role": "user", "content": gen_user}],
            temperature=0.0, max_tokens=400, fallback="",
        )
    if not raw_sql:
        result["answer"] = ("I couldn't generate a database query right now "
                            "(the SQL model is unavailable).")
        result["error"] = "sql_gen_unavailable"
        return result
    sql = _clean_sql(raw_sql)
    trace.event("sql.generated", sql=sql)

    # 9.6 validation (LLM + static guards)
    if not security.is_read_only_sql(sql):
        result["answer"] = "Only read-only lookups are permitted."
        result["error"] = "not_read_only"
        trace.error("sql.validate", Exception("non read-only SQL blocked"))
        return result
    if not security.uses_only_allowed_tables(sql):
        result["answer"] = "That query referenced tables I'm not allowed to access."
        result["error"] = "table_not_allowed"
        return result

    with trace.span("sql.validate"):
        verdict = await llm.chat_safe(
            settings.sql_validation_model,
            [{"role": "system", "content": VALIDATE_SYSTEM},
             {"role": "user", "content": f"Schema:\n{schema_block}\n\nSQL:\n{sql}"}],
            temperature=0.0, max_tokens=300, fallback="",
        )
    sql = _maybe_apply_fix(sql, verdict, trace)

    # 9.7 enforce RLS + execute on hrms (allow-listed tables only)
    safe_sql = security.enforce_rls_in_sql(sql, ctx)
    if not mssql.available:
        result["answer"] = ("I generated the query but the HRMS database is "
                            "currently unreachable.")
        result["error"] = "hrms_unavailable"
        result["debug_sql"] = safe_sql
        return result

    try:
        with trace.span("sql.execute"):
            rows = await mssql.run_select(safe_sql, settings.sql_max_rows)
    except Exception as exc:
        trace.error("sql.execute", exc)
        result["answer"] = "I ran into an error executing the database query."
        result["error"] = f"execution_error: {exc}"
        result["debug_sql"] = safe_sql
        return result

    # 9.8 mask + limit to permitted columns
    rows = security.mask_rows(rows, ctx)
    trace.event("sql.executed", rows=len(rows))

    # 9.9 answer generation
    preview = rows[:20]
    with trace.span("sql.answer"):
        answer = await llm.chat_safe(
            settings.answer_model,
            [{"role": "system", "content": ANSWER_SYSTEM},
             {"role": "user",
              "content": f"Question: {rewritten_query}\n\nResult rows (JSON): {preview}"}],
            temperature=0.2, max_tokens=400,
            fallback=(f"Found {len(rows)} record(s)." if rows
                      else "No matching records were found."),
        )

    result["answer"] = answer
    result["table"] = rows if rows else None
    result["confidence"] = 0.85 if rows else 0.5
    result["debug_sql"] = safe_sql
    return result


def _maybe_apply_fix(sql: str, verdict: str, trace: Trace) -> str:
    if not verdict:
        return sql
    import json
    try:
        m = re.search(r"\{.*\}", verdict, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        if data.get("valid") is False and data.get("fixed_sql"):
            fixed = _clean_sql(data["fixed_sql"])
            if security.is_read_only_sql(fixed):
                trace.event("sql.validation_fixed", reason=data.get("reason", ""))
                return fixed
    except Exception:
        pass
    return sql
