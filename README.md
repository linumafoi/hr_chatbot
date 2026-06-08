# HR AI Chatbot

A fast, low-latency HR assistant with a light-theme web UI, employee/admin
login with **row-level security (RLS)**, a **FastAPI gateway**, and a
**LangGraph orchestrated** multi-agent pipeline (RAG + NL→SQL + general chat)
backed by qwen3 models, BGE-M3 embeddings, PostgreSQL and SQL Server.

> **Note on this build:** it was generated in a sandbox that cannot reach your
> local PostgreSQL, SQL Server, or model servers. Every dependency is configured
> via `.env` and the app is designed to run **on your machine**, where those
> services are reachable. It also degrades gracefully (`GRACEFUL_FALLBACK=true`)
> so the UI and gateway run even before all models/DBs are up.

---

## Architecture — how the 15 steps map to the code

| Step | What | Where |
|------|------|-------|
| 1  | Light-theme UI, employee/admin login, employee ID for RLS | `frontend/` |
| 2  | FastAPI gateway connecting UI ↔ pipeline | `backend/app/main.py` |
| 3  | Query preprocessor (clean, normalize, expand abbreviations, spellfix) | `pipeline/preprocessor.py` |
| 4  | Intent classification (qwen3-8B) | `pipeline/intent.py` |
| 5  | Query rewrite (qwen3-8B) | `pipeline/rewrite.py` |
| 6  | Context extraction (user id, role, dept, permissions, session) | `pipeline/context.py` |
| 7  | LangGraph router / orchestrator | `pipeline/router.py` |
| 8  | RAG agent (BGE-M3 → hr_documents/hr_faq → top20 → qwen3-reranker) | `pipeline/agents/rag_agent.py` |
| 9  | SQL agent (NL→SQL, schema retrieval, validate, execute, answer) | `pipeline/agents/sql_agent.py` |
| 10 | General chat agent (qwen3-14B) | `pipeline/agents/chat_agent.py` |
| 11 | Response aggregator (merge, sources, confidence, chart) | `pipeline/aggregator.py` |
| 12 | Security: RBAC, RLS, data masking, permission checks | `pipeline/security.py`, `auth.py` |
| 13 | Observability & logging (queries, SQL, timings, tokens, audit) | `pipeline/observability.py` |
| 14 | Conversation memory (Redis, in-memory fallback) | `db/redis_client.py` |
| 15 | Final response shaping (text/table/chart) | `main.py` + `frontend/app.js` |

```
UI ──► /api/chat ──► [preprocess → intent → rewrite → context]
                                          │
                              ┌───────────┼─────────────┐
                            RAG          SQL           Chat
                              └───────────┼─────────────┘
                                     aggregate ──► RLS/mask ──► response
```

---

## Prerequisites (on your machine)

1. **PostgreSQL** (database `chatbot`) with the [pgvector](https://github.com/pgvector/pgvector) extension.
2. **SQL Server** (database `hrms`) with the `employees` and `EmployeeEmployment` tables, and the **ODBC Driver 18 for SQL Server** installed.
3. **Redis** (optional — falls back to in-memory).
4. **[Ollama](https://ollama.com/)** running locally (`ollama serve`, default port 11434), with these models pulled:
   ```bash
   ollama pull qwen3:8b        # intent + query rewrite
   ollama pull qwen3:14b       # chat, SQL validation, answer generation
   ollama pull deepseek-r1:32b # NL->SQL generation (or an arctic-text2sql GGUF)
   ollama pull bge-m3          # embeddings
   ```
   Reranking uses a single listwise LLM pass (`RERANK_MODE=llm`, default model `qwen3:8b`) because Ollama has no native rerank API. To use a true cross-encoder (e.g. `qwen3-reranker-8b` on vLLM/TEI), set `RERANK_MODE=http` and point `RERANK_BASE_URL` at it.
5. **Python 3.11+**.

> Adjust the model tags in `.env` to whatever you have pulled (`ollama list`).

---

## Setup

```bash
cd hr_chatbot
cp .env.example .env          # then edit credentials/endpoints

# 1) Create the PostgreSQL semantic store (tables + pgvector + seed schema rows)
psql "postgresql://postgres:Mafoi%40123@localhost:5433/chatbot" -f sql/postgres_schema.sql

# 2) Embed the seeded schema rows (and any docs/FAQ you add) with BGE-M3 via Ollama
PYTHONPATH=backend python -m scripts.embed_backfill

# 3) Install deps + run (creates a venv automatically)
bash run.sh
```

Then open **http://localhost:8000**.

Whenever you insert new rows into `hr_documents` / `hr_faq` / `sql_examples`,
re-run the backfill to embed them:

```bash
PYTHONPATH=backend python -m scripts.embed_backfill
```

---

## Logins

- **Admin** (common login): username `admin`, password from `ADMIN_PASSWORD` (default `admin@123`). Sees all data, unmasked PII.
- **Employee**: Employee ID (numeric) + password from `EMPLOYEE_PASSWORD` (default `employee@123`). The Employee ID becomes the RLS scope — they only ever see their own rows, and PII is masked.

> Tip: replace the shared employee password with a real identity provider for production. The architecture already carries `employee_id` + `permissions` in the JWT.

---

## Configuration

All endpoints, models, credentials and tuning live in `.env` (see `.env.example`).
Key knobs:

- `LLM_BASE_URL`, `EMBED_BASE_URL`, `RERANK_BASE_URL` — point these at your inference server(s).
- `*_MODEL` — match the model names your server exposes.
- `MSSQL_ALLOWED_TABLES` — the SQL agent allow-list (defaults to `employees,EmployeeEmployment`).
- `RAG_TOP_K` / `RAG_RERANK_K` / `SQL_MAX_ROWS` — retrieval + result limits.
- `GRACEFUL_FALLBACK` — keep `true` during setup so missing services don't break the app.

---

## HRMS schema & row-level security

The SQL agent is locked to two SQL Server tables (`MSSQL_ALLOWED_TABLES`):

| Table | Primary key | Employee identity column (RLS) |
|-------|-------------|-------------------------------|
| `employees` | `Id` | `Id` |
| `EmployeeEmployment` | `Id` | `EmployeeId` (FK → `employees.Id`) |

They join on `employees.Id = EmployeeEmployment.EmployeeId`. The logged-in
**Employee ID is `employees.Id`**.

**RLS enforcement** (`pipeline/security.py`): before execution, every reference
to an allow-listed table is rewritten into an employee-scoped inline view, e.g.

```sql
FROM EmployeeEmployment ee
-- becomes -->
FROM (SELECT * FROM EmployeeEmployment WHERE EmployeeId = 1024) AS ee
```

This guarantees an employee only ever sees their own rows regardless of the
generated projection/joins/WHERE. Admins are unrestricted; a missing identity
fails closed. Sensitive columns (PAN, UAN, ESIC, Aadhaar, FatherName,
DateOfBirth, etc.) are masked for users without the `pii.read` permission.

---

## Low latency notes

- Greetings short-circuit the LLM (keyword fast-path in `intent.py`).
- qwen3 "thinking" is disabled (`/no_think`) for interactive turns.
- Only one agent runs per query (intent-routed); RAG uses ANN (pgvector ivfflat).
- Reranking is a single listwise pass (1 LLM call) and capped to the top-K candidates.
- Keep the smaller models (`qwen3:8b`) for intent/rewrite/rerank; reserve the 14B/32B for SQL + answers. On CPU-only Ollama, consider smaller tags (e.g. `qwen3:4b`) to cut latency.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/login` | `{role, password, employee_id?, username?}` → `{access_token, user}` |
| POST | `/api/chat` | `{message, session_id}` (Bearer token) → `{answer, intent, agent, confidence, sources, table, chart, latency_ms}` |
| GET  | `/api/health` | dependency status |

Audit logs are written to `logs/audit.log` (one JSON record per request).
