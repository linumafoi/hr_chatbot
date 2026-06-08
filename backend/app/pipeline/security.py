"""Step 12 - Security & access control.

Implements:
  * RBAC          - permission checks per intent/agent
  * Row-level     - rewrites each allow-listed table reference into an inline
    security (RLS)  view filtered by the employee's identity, so employees only
                    ever read their own rows from hrms (admins are unrestricted)
  * Data masking  - masks PII columns for non-privileged users
  * SQL guarding  - read-only + allow-listed tables only

RLS identity columns differ per table:
  - employees           -> Id
  - EmployeeEmployment  -> EmployeeId
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from ..config import settings
from ..schemas import UserContext

logger = logging.getLogger("hr.security")

# Per-table column that holds the employee's identity (lower-cased table name).
RLS_COLUMNS = {
    "employees": "Id",
    "employeeemployment": "EmployeeId",
}

# Words that can follow a table name but are NOT an alias.
_ALIAS_STOPWORDS = (
    "on", "where", "inner", "left", "right", "full", "cross", "join",
    "group", "order", "having", "union", "outer", "as", "and", "or",
)

# Columns considered sensitive PII; masked for users without 'pii.read'.
# Matched case-insensitively against the real hrms column names.
PII_COLUMNS = {
    "aadhaar", "pan", "uan", "esic", "pfnumber", "pon",
    "fathername", "dateofbirth", "marriagedate", "bloodgroup",
    "religion", "nationality", "disabilitypercentage",
    "monthlybillingamount", "photostorage",
    # generic fallbacks
    "ssn", "national_id", "bank_account", "salary", "ctc", "email", "phone",
}

# Forbidden SQL keywords (read-only enforcement)
FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|merge|grant|revoke|exec|execute)\b",
    re.IGNORECASE,
)


def check_permission(ctx: UserContext, permission: str) -> bool:
    return permission in ctx.permissions


def is_read_only_sql(sql: str) -> bool:
    if FORBIDDEN_SQL.search(sql):
        return False
    stripped = sql.strip().lower()
    return stripped.startswith("select") or stripped.startswith("with")


def uses_only_allowed_tables(sql: str) -> bool:
    """Best-effort check that only allow-listed tables are referenced."""
    referenced = set(re.findall(r"(?:from|join)\s+([\w\.\[\]]+)", sql, re.IGNORECASE))
    allowed = {t.lower() for t in settings.allowed_tables}
    for ref in referenced:
        name = ref.split(".")[-1].strip("[]").lower()
        if name not in allowed:
            logger.warning("SQL references non-allowed table: %s", ref)
            return False
    return True


def enforce_rls_in_sql(sql: str, ctx: UserContext) -> Optional[str]:
    """Rewrite allow-listed table references into employee-scoped inline views.

    Returns the secured SQL, or None to fail closed (no identity).

    Example (employee 1024):
      FROM EmployeeEmployment ee
        -> FROM (SELECT * FROM EmployeeEmployment WHERE EmployeeId = 1024) AS ee

    Because each base table is replaced by a pre-filtered derived table, the
    employee can never see another employee's rows regardless of the projection,
    joins, or WHERE clause the generator produced. The derived table keeps the
    original alias (or the table name itself) so column references still resolve.
    """
    if ctx.is_admin:
        return sql
    if not ctx.employee_id:
        return None  # fail closed
    emp = int(ctx.employee_id)

    tables = "|".join(re.escape(t) for t in settings.allowed_tables)
    stop = "|".join(_ALIAS_STOPWORDS)
    # (FROM|JOIN) <table> [optional alias that is not a stopword]
    pattern = re.compile(
        rf"\b(from|join)\s+\[?({tables})\]?\b"
        rf"(?:\s+(?:as\s+)?(?!(?:{stop})\b)([A-Za-z_]\w*))?",
        re.IGNORECASE,
    )

    def _repl(m: re.Match) -> str:
        kw, table, alias = m.group(1), m.group(2), m.group(3)
        col = RLS_COLUMNS.get(table.lower())
        if not col:
            return m.group(0)  # unknown table: leave for allow-list check to reject
        alias = alias or table
        return f"{kw} (SELECT * FROM {table} WHERE {col} = {emp}) AS {alias}"

    secured = pattern.sub(_repl, sql)
    return secured


def mask_rows(rows: List[Dict[str, Any]], ctx: UserContext) -> List[Dict[str, Any]]:
    """Mask PII columns for users lacking 'pii.read'."""
    if check_permission(ctx, "pii.read"):
        return rows
    masked = []
    for row in rows:
        new_row = {}
        for k, v in row.items():
            if k.lower() in PII_COLUMNS and v not in (None, ""):
                new_row[k] = _mask_value(str(v))
            else:
                new_row[k] = v
        masked.append(new_row)
    return masked


def _mask_value(v: str) -> str:
    if "@" in v:  # email
        name, _, domain = v.partition("@")
        return (name[:2] + "***@" + domain) if len(name) > 2 else "***@" + domain
    if len(v) <= 4:
        return "****"
    return v[:2] + "*" * (len(v) - 4) + v[-2:]
