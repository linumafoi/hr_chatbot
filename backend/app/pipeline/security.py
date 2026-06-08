"""Step 12 - Security & access control.

Implements:
  * RBAC          - permission checks per intent/agent
  * Row-level     - injects an employee_id predicate so employees only ever
    security (RLS)  read their own rows from hrms
  * Data masking  - masks PII columns for non-privileged users
  * SQL guarding  - read-only + allow-listed tables only

The SQL agent and RAG agent call into these helpers; the gateway also performs
a coarse RBAC check before routing.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from ..config import settings
from ..schemas import UserContext

logger = logging.getLogger("hr.security")

# Columns considered sensitive PII; masked for users without 'pii.read'
PII_COLUMNS = {
    "ssn", "national_id", "aadhaar", "pan", "bank_account", "account_no",
    "ifsc", "salary", "ctc", "gross_salary", "net_salary", "dob",
    "date_of_birth", "phone", "mobile", "email", "address",
}

# Forbidden SQL keywords (read-only enforcement)
FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|merge|grant|revoke|exec|execute)\b",
    re.IGNORECASE,
)


def check_permission(ctx: UserContext, permission: str) -> bool:
    return permission in ctx.permissions


def rls_predicate(ctx: UserContext) -> str:
    """Return a SQL WHERE fragment enforcing row-level security.

    Employees are scoped to their own employee_id; admins are unrestricted.
    """
    if ctx.is_admin:
        return ""
    if not ctx.employee_id:
        # Fail closed: no identity -> no rows
        return "1 = 0"
    # employee_id is validated numeric at login, safe to inline
    return f"employee_id = {int(ctx.employee_id)}"


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


def enforce_rls_in_sql(sql: str, ctx: UserContext) -> str:
    """Inject the RLS predicate into a SELECT if not already scoped.

    This is a safety net in addition to prompting the SQL generator with the
    scope. For employees we wrap the query so only their rows survive.
    """
    pred = rls_predicate(ctx)
    if not pred:  # admin
        return sql
    # Wrap as subquery to guarantee filtering regardless of generated SQL shape.
    sql_no_semi = sql.rstrip().rstrip(";")
    return f"SELECT * FROM ({sql_no_semi}) AS rls_scoped WHERE {pred}"


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
