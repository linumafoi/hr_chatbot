-- ============================================================
--  PostgreSQL "chatbot" database schema (semantic / RAG store)
--  Run against the database referenced by PG_DB (default: chatbot).
--  Requires the pgvector extension for fast similarity search.
--  BGE-M3 produces 1024-dim embeddings.
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------- RAG corpus ----------
CREATE TABLE IF NOT EXISTS hr_documents (
    id          SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    category    TEXT,
    embedding   vector(1024),
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS hr_faq (
    id          SERIAL PRIMARY KEY,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    category    TEXT,
    embedding   vector(1024),
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- ---------- SQL-agent knowledge ----------
-- Natural-language descriptions of the hrms tables the SQL agent may query.
CREATE TABLE IF NOT EXISTS schema_tables (
    id          SERIAL PRIMARY KEY,
    table_name  TEXT NOT NULL,
    description TEXT NOT NULL,
    columns     TEXT,                 -- e.g. "employee_id INT, first_name VARCHAR, ..."
    embedding   vector(1024)
);

CREATE TABLE IF NOT EXISTS schema_domains (
    id          SERIAL PRIMARY KEY,
    domain_name TEXT NOT NULL,
    description TEXT NOT NULL,
    embedding   vector(1024)
);

-- Few-shot NL->SQL pairs (currently empty; populate to improve generation).
CREATE TABLE IF NOT EXISTS sql_examples (
    id          SERIAL PRIMARY KEY,
    question    TEXT NOT NULL,
    sql         TEXT NOT NULL,
    embedding   vector(1024)
);

-- ---------- ANN indexes (cosine) ----------
CREATE INDEX IF NOT EXISTS hr_documents_emb_idx
    ON hr_documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS hr_faq_emb_idx
    ON hr_faq USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS schema_tables_emb_idx
    ON schema_tables USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX IF NOT EXISTS schema_domains_emb_idx
    ON schema_domains USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX IF NOT EXISTS sql_examples_emb_idx
    ON sql_examples USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

-- ---------- Seed: describe the allow-listed hrms tables ----------
INSERT INTO schema_tables (table_name, description, columns) VALUES
('employees',
 'Master record of each employee: identity and personal details.',
 'employee_id INT (PK), first_name VARCHAR, last_name VARCHAR, email VARCHAR, department VARCHAR, designation VARCHAR, date_of_birth DATE, phone VARCHAR'),
('EmployeeEmployment',
 'Employment record: joining info, employment status, manager and pay band.',
 'employee_id INT (FK), date_of_joining DATE, employment_type VARCHAR, status VARCHAR, manager_id INT, grade VARCHAR, location VARCHAR')
ON CONFLICT DO NOTHING;

INSERT INTO schema_domains (domain_name, description) VALUES
('Employee Directory', 'Who works here, their roles, departments and contact info.'),
('Employment & Onboarding', 'Joining dates, employment type/status, reporting manager and work location.')
ON CONFLICT DO NOTHING;
