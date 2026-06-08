"""Backfill BGE-M3 embeddings for rows in the chatbot store.

Run after inserting/seeding rows (or after running sql/postgres_schema.sql) to
populate the `embedding` column for any rows that are missing it.

Usage:
    python -m scripts.embed_backfill         # from the backend/ directory
"""
import asyncio
import sys
from pathlib import Path

# allow running from repo root or backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.db.postgres import pg          # noqa: E402
from app.llm.embeddings import embeddings  # noqa: E402

# table -> text columns used to build the embedding input
TARGETS = {
    "hr_documents": ["title", "content"],
    "hr_faq": ["question", "answer"],
    "schema_tables": ["table_name", "description", "columns"],
    "schema_domains": ["domain_name", "description"],
    "sql_examples": ["question", "sql"],
}


async def backfill_table(table: str, cols: list[str]) -> int:
    select_cols = ", ".join(["id"] + cols)
    rows = await pg.fetch(
        f"SELECT {select_cols} FROM {table} WHERE embedding IS NULL"
    )
    if not rows:
        return 0
    updated = 0
    for row in rows:
        text = " ".join(str(row.get(c, "") or "") for c in cols).strip()
        if not text:
            continue
        vec = await embeddings.embed(text)
        if not vec:
            print(f"  ! embedding endpoint returned nothing; aborting {table}")
            break
        vec_literal = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
        await pg.fetch(
            f"UPDATE {table} SET embedding = $1::vector WHERE id = $2",
            vec_literal, row["id"],
        )
        updated += 1
    return updated


async def main() -> None:
    await pg.connect()
    if not pg.available:
        print("PostgreSQL is not reachable. Check PG_* settings in .env")
        return
    for table, cols in TARGETS.items():
        try:
            n = await backfill_table(table, cols)
            print(f"{table}: embedded {n} row(s)")
        except Exception as exc:
            print(f"{table}: skipped ({exc})")
    await pg.close()


if __name__ == "__main__":
    asyncio.run(main())
