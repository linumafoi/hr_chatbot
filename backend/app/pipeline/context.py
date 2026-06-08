"""Step 6 - Context extraction.

Assembles the working context for routing and the agents: user id, role,
department, permissions and session context (recent history + any sticky
context such as the last entity discussed). Department is looked up lazily and
cached in session context.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..schemas import UserContext


async def extract_context(
    user: UserContext,
    session_id: str,
    history: List[Dict[str, Any]],
    session_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "user_id": user.employee_id,
        "role": user.role,
        "username": user.username,
        "department": user.department or session_ctx.get("department"),
        "permissions": user.permissions,
        "session_id": session_id,
        "is_admin": user.is_admin,
        "history_len": len(history),
        "last_intent": session_ctx.get("last_intent"),
        "last_entity": session_ctx.get("last_entity"),
    }
