"""Typed application configuration loaded from environment / .env file.

Every external dependency (databases, LLM endpoints, models) is configured here
so the whole system can be pointed at your local machine without touching code.
"""
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------- App ----------------
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    # ---------------- Auth ----------------
    jwt_secret: str = "change-me-super-secret-key"
    jwt_alg: str = "HS256"
    jwt_expire_minutes: int = 720
    admin_username: str = "admin"
    admin_password: str = "admin@123"
    employee_password: str = "employee@123"

    # ---------------- PostgreSQL (chatbot) ----------------
    pg_db: str = "chatbot"
    pg_user: str = "postgres"
    pg_password: str = "Mafoi@123"
    pg_host: str = "localhost"
    pg_port: int = 5433

    # ---------------- SQL Server (hrms) ----------------
    mssql_server: str = "localhost"
    mssql_port: int = 1433
    mssql_database: str = "hrms"
    mssql_username: str = "sa"
    mssql_password: str = ""
    mssql_driver: str = "ODBC Driver 18 for SQL Server"
    mssql_allowed_tables: str = "employees,EmployeeEmployment"

    # ---------------- Redis ----------------
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    conversation_ttl_seconds: int = 86400

    # ---------------- LLM endpoints (Ollama, OpenAI-compatible) ----------------
    # Ollama exposes the OpenAI API at http://localhost:11434/v1
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    # Ollama model tags (pull these with `ollama pull <name>`).
    # Sized for 8GB GPU / 16GB RAM: one shared qwen3:8b for all qwen roles
    # (stays resident -> low latency) + a small deepseek distill for NL->SQL.
    intent_model: str = "qwen3:8b"
    rewrite_model: str = "qwen3:8b"
    sql_validation_model: str = "qwen3:8b"
    sql_gen_model: str = "deepseek-r1:7b"    # 32b/14b won't fit 8GB VRAM
    chat_model: str = "qwen3:8b"
    answer_model: str = "qwen3:8b"

    embed_base_url: str = "http://localhost:11434/v1"
    embed_api_key: str = "ollama"
    embed_model: str = "bge-m3"

    # Reranking. Ollama has no native rerank API, so the default mode is "llm"
    # (a single listwise pass with rerank_model). Set rerank_mode="http" to use
    # a dedicated cross-encoder rerank server (vLLM/TEI) at rerank_base_url.
    rerank_mode: str = "llm"
    rerank_base_url: str = "http://localhost:11434/v1"
    rerank_api_key: str = "ollama"
    rerank_model: str = "qwen3:8b"

    # ---------------- Pipeline tuning ----------------
    rag_top_k: int = 20
    rag_rerank_k: int = 5
    sql_max_rows: int = 100
    request_timeout_seconds: int = 60
    graceful_fallback: bool = True

    # ---------------- Derived helpers ----------------
    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_tables(self) -> List[str]:
        return [t.strip() for t in self.mssql_allowed_tables.split(",") if t.strip()]

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )

    @property
    def mssql_conn_str(self) -> str:
        return (
            f"DRIVER={{{self.mssql_driver}}};"
            f"SERVER={self.mssql_server},{self.mssql_port};"
            f"DATABASE={self.mssql_database};"
            f"UID={self.mssql_username};PWD={self.mssql_password};"
            f"TrustServerCertificate=yes;Encrypt=yes;"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
