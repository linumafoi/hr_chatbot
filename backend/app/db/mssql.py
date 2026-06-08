"""SQL Server access for the 'hrms' execution store (read-only).

Only the allow-listed tables may be queried (config.MSSQL_ALLOWED_TABLES).
Queries run through pyodbc in a thread (pyodbc is blocking) and are forced to
read-only by validation upstream (SQL agent / security layer).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..config import settings

logger = logging.getLogger("hr.mssql")

try:
    import pyodbc  # type: ignore
except Exception:  # pragma: no cover
    pyodbc = None


class MSSQLClient:
    def __init__(self) -> None:
        self.available = pyodbc is not None
        self._checked = False

    def _connect(self):
        if pyodbc is None:
            raise RuntimeError("pyodbc not installed")
        return pyodbc.connect(settings.mssql_conn_str, timeout=10)

    async def probe(self) -> None:
        """Best-effort connectivity check at startup."""
        if pyodbc is None:
            logger.warning("pyodbc not installed - SQL agent execution disabled")
            self.available = False
            return
        try:
            await asyncio.to_thread(self._probe_sync)
            self.available = True
            logger.info("Connected to SQL Server '%s'", settings.mssql_database)
        except Exception as exc:
            self.available = False
            logger.warning("SQL Server unavailable (%s) - SQL agent will degrade", exc)

    def _probe_sync(self) -> None:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        cur.close()
        conn.close()

    async def run_select(self, sql: str, max_rows: int) -> List[Dict[str, Any]]:
        if pyodbc is None:
            return []
        return await asyncio.to_thread(self._run_select_sync, sql, max_rows)

    def _run_select_sync(self, sql: str, max_rows: int) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(sql)
            cols = [c[0] for c in cur.description] if cur.description else []
            rows = cur.fetchmany(max_rows)
            return [dict(zip(cols, _coerce_row(r))) for r in rows]
        finally:
            conn.close()


def _coerce_row(row) -> list:
    """Make values JSON-serialisable (dates, decimals, bytes)."""
    out = []
    for v in row:
        if v is None or isinstance(v, (str, int, float, bool)):
            out.append(v)
        else:
            out.append(str(v))
    return out


mssql = MSSQLClient()
