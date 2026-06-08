"""OpenAI-compatible chat client for the qwen3 / text2sql family.

Works with any server exposing the OpenAI /chat/completions API
(vLLM, Ollama via /v1, LM Studio, TGI, etc.). One client instance is reused
for all models; the model name is passed per call.

For qwen3 "thinking" models we disable the reasoning trace (`/no_think` and
`enable_thinking=false`) to keep latency low for this interactive UI.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import httpx

from ..config import settings

logger = logging.getLogger("hr.llm")


class LLMClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.llm_base_url,
            timeout=settings.request_timeout_seconds,
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        no_think: bool = True,
    ) -> str:
        """Return assistant text. Raises on transport error (caller handles fallback)."""
        # qwen3-specific: append /no_think to the last user turn so the model
        # skips its long reasoning trace -> much lower latency. Only applied to
        # qwen models (other models would treat it as literal text).
        if no_think and "qwen" in model.lower() and messages and messages[-1]["role"] == "user":
            messages = messages[:-1] + [
                {"role": "user", "content": messages[-1]["content"] + " /no_think"}
            ]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        t0 = time.perf_counter()
        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        # strip any leaked <think> blocks
        if "<think>" in text:
            text = text.split("</think>")[-1].strip()
        logger.debug("LLM %s -> %d chars in %dms", model, len(text),
                     int((time.perf_counter() - t0) * 1000))
        return text

    async def chat_safe(self, model: str, messages: List[Dict[str, str]],
                        fallback: str = "", **kw) -> str:
        """chat() with graceful fallback when the model server is down."""
        try:
            return await self.chat(model, messages, **kw)
        except Exception as exc:
            if settings.graceful_fallback:
                logger.warning("LLM '%s' unavailable (%s) - using fallback", model, exc)
                return fallback
            raise


llm = LLMClient()
