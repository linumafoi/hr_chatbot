"""Step 13 - Observability & logging.

Lightweight, structured logging for: queries, retrieved data, generated SQL +
execution time, token usage, errors, and an append-only audit trail. Writes
human-readable logs to the standard logger and JSON audit records to
logs/audit.log so they can be shipped to any sink later.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger("hr.observability")

LOG_DIR = os.environ.get("HR_LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
AUDIT_PATH = os.path.join(LOG_DIR, "audit.log")


class Trace:
    """Collects timing + events for a single request, emitted as one audit row."""

    def __init__(self, session_id: str, user_id: Optional[str], role: str):
        self.session_id = session_id
        self.user_id = user_id
        self.role = role
        self.t0 = time.perf_counter()
        self.events: list[Dict[str, Any]] = []
        self.timings: Dict[str, float] = {}
        self.tokens: Dict[str, int] = {}
        self.errors: list[str] = []

    def event(self, name: str, **data: Any) -> None:
        self.events.append({"event": name, **data})
        logger.info("[%s] %s %s", self.session_id, name,
                    {k: v for k, v in data.items() if k != "embedding"})

    def error(self, where: str, exc: Exception) -> None:
        msg = f"{where}: {exc}"
        self.errors.append(msg)
        logger.warning("[%s] ERROR %s", self.session_id, msg)

    def add_tokens(self, model: str, n: int) -> None:
        self.tokens[model] = self.tokens.get(model, 0) + n

    @contextmanager
    def span(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round((time.perf_counter() - start) * 1000, 1)

    @property
    def latency_ms(self) -> int:
        return int((time.perf_counter() - self.t0) * 1000)

    def finalize(self, question: str, intent: str, agent: str,
                 answer_preview: str, confidence: float) -> None:
        record = {
            "ts": time.time(),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "role": self.role,
            "question": question,
            "intent": intent,
            "agent": agent,
            "confidence": confidence,
            "latency_ms": self.latency_ms,
            "timings_ms": self.timings,
            "tokens": self.tokens,
            "errors": self.errors,
            "answer_preview": answer_preview[:200],
            "events": self.events,
        }
        try:
            with open(AUDIT_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to write audit log: %s", exc)
        logger.info("[%s] DONE intent=%s agent=%s %dms",
                    self.session_id, intent, agent, self.latency_ms)
