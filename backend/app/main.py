"""Step 2 - FastAPI Gateway.

Connects the UI to the pipeline:
  POST /api/auth/login   -> issue JWT (admin common login / employee by ID)
  POST /api/chat         -> run the 15-step LangGraph pipeline
  GET  /api/health       -> service + dependency status
  GET  /                 -> serves the light-theme UI

It also owns step 14 (conversation memory read/write) and step 15 (final
response shaping), and emits the step 13 audit record per request.
"""
from __future__ import annotations

import logging
import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .auth import authenticate, get_current_user, issue_token
from .config import settings
from .db.mssql import mssql
from .db.postgres import pg
from .db.redis_client import memory
from .llm.client import llm
from .llm.embeddings import embeddings
from .llm.reranker import reranker
from .pipeline.observability import Trace
from .pipeline.router import run_pipeline
from .schemas import (ChatRequest, ChatResponse, LoginRequest, LoginResponse,
                      UserContext, UserInfo)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("hr.gateway")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
FRONTEND_DIR = os.path.abspath(FRONTEND_DIR)

app = FastAPI(title="HR AI Chatbot Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Lifecycle ----------------
@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Starting HR AI Chatbot gateway...")
    await pg.connect()
    await memory.connect()
    await mssql.probe()
    logger.info(
        "Dependencies | postgres=%s mssql=%s redis=%s",
        pg.available, mssql.available, memory.available,
    )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await pg.close()
    await memory.close()
    await llm.close()
    await embeddings.close()
    await reranker.close()


# ---------------- Auth ----------------
@app.post("/api/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest) -> LoginResponse:
    user = authenticate(req)
    token = issue_token(user)
    return LoginResponse(access_token=token, user=user)


# ---------------- Chat ----------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user: UserInfo = Depends(get_current_user)) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(400, "Empty message")

    trace = Trace(session_id=req.session_id, user_id=user.employee_id, role=user.role)

    # Step 14 - load conversation memory + session context
    history = await memory.history(req.session_id, limit=8)
    session_ctx = await memory.get_context(req.session_id)

    ctx = UserContext(
        role=user.role,
        employee_id=user.employee_id,
        username=user.username,
        department=user.department,
        permissions=user.permissions,
        session_id=req.session_id,
    )

    state = {
        "raw_message": req.message,
        "session_id": req.session_id,
        "user": ctx,
        "history": history,
        "context": {"_trace": trace, "session_ctx": session_ctx},
    }

    try:
        result = await run_pipeline(state)
    except Exception as exc:
        trace.error("pipeline", exc)
        logger.exception("Pipeline failed")
        if not settings.graceful_fallback:
            raise HTTPException(500, "Pipeline error")
        result = {
            "answer": "Sorry, something went wrong while processing your request.",
            "agent": "error", "confidence": 0.0, "sources": [],
            "table": None, "chart": None, "intent": "unknown",
        }

    # Step 14 - persist this turn + sticky context
    await memory.add_turn(req.session_id, "user", req.message)
    await memory.add_turn(req.session_id, "assistant", result.get("answer", ""))
    await memory.set_context(req.session_id, {
        "last_intent": result.get("intent", "unknown"),
        "department": ctx.department,
    })

    # Step 13 - audit record
    trace.finalize(
        question=req.message,
        intent=result.get("intent", "unknown"),
        agent=result.get("agent", "none"),
        answer_preview=result.get("answer", ""),
        confidence=result.get("confidence", 0.0),
    )

    # Step 15 - final shaped response
    return ChatResponse(
        answer=result.get("answer", ""),
        intent=result.get("intent", "unknown"),
        agent=result.get("agent", "none"),
        confidence=result.get("confidence", 0.0),
        sources=result.get("sources", []),
        table=result.get("table"),
        chart=result.get("chart"),
        latency_ms=trace.latency_ms,
        session_id=req.session_id,
    )


# ---------------- Health ----------------
@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "dependencies": {
            "postgres": pg.available,
            "mssql": mssql.available,
            "redis": memory.available,
        },
        "graceful_fallback": settings.graceful_fallback,
    }


# ---------------- Static frontend ----------------
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/chat.html")
async def chat_page() -> FileResponse:
    return FileResponse(os.path.join(FRONTEND_DIR, "chat.html"))


# serve css/js and any other static assets at root
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")
