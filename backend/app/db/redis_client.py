"""Redis-backed conversation memory (step 14).

Stores per-session message history and lightweight session context for
follow-up handling. Falls back to an in-process dict if Redis is unreachable
so the chat still works during local development.
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List

from ..config import settings

logger = logging.getLogger("hr.redis")

try:
    import redis.asyncio as aioredis  # type: ignore
except Exception:  # pragma: no cover
    aioredis = None

MAX_TURNS = 20


class ConversationMemory:
    def __init__(self) -> None:
        self._redis = None
        self.available = False
        self._fallback: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_TURNS))
        self._fallback_ctx: Dict[str, Dict[str, Any]] = defaultdict(dict)

    async def connect(self) -> None:
        if aioredis is None:
            logger.warning("redis not installed - using in-memory conversation store")
            return
        try:
            self._redis = aioredis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password or None,
                decode_responses=True,
                socket_connect_timeout=3,
            )
            await self._redis.ping()
            self.available = True
            logger.info("Connected to Redis %s:%s", settings.redis_host, settings.redis_port)
        except Exception as exc:
            self.available = False
            self._redis = None
            logger.warning("Redis unavailable (%s) - using in-memory store", exc)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()

    def _key(self, session_id: str) -> str:
        return f"hr:conv:{session_id}"

    async def add_turn(self, session_id: str, role: str, content: str) -> None:
        item = json.dumps({"role": role, "content": content, "ts": time.time()})
        if self._redis:
            try:
                key = self._key(session_id)
                await self._redis.rpush(key, item)
                await self._redis.ltrim(key, -MAX_TURNS, -1)
                await self._redis.expire(key, settings.conversation_ttl_seconds)
                return
            except Exception as exc:
                logger.debug("Redis add_turn failed: %s", exc)
        self._fallback[session_id].append(json.loads(item))

    async def history(self, session_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        if self._redis:
            try:
                raw = await self._redis.lrange(self._key(session_id), -limit, -1)
                return [json.loads(r) for r in raw]
            except Exception as exc:
                logger.debug("Redis history failed: %s", exc)
        return list(self._fallback[session_id])[-limit:]

    async def set_context(self, session_id: str, ctx: Dict[str, Any]) -> None:
        if self._redis:
            try:
                await self._redis.set(
                    f"hr:ctx:{session_id}",
                    json.dumps(ctx),
                    ex=settings.conversation_ttl_seconds,
                )
                return
            except Exception:
                pass
        self._fallback_ctx[session_id] = ctx

    async def get_context(self, session_id: str) -> Dict[str, Any]:
        if self._redis:
            try:
                raw = await self._redis.get(f"hr:ctx:{session_id}")
                return json.loads(raw) if raw else {}
            except Exception:
                pass
        return self._fallback_ctx.get(session_id, {})


memory = ConversationMemory()
