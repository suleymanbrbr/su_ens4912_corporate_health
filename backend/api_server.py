# api_server.py
# Description: Modern FastAPI Server for SUT Assistant. PostgreSQL Edition.
#
# This module is intentionally thin. All endpoints live in ``backend/routers/``.
# What stays here:
#   - FastAPI app + lifespan
#   - CORS middleware
#   - Database schema bootstrap (init_system_tables)
#   - include_router() wiring for each split-out router
#   - Re-exports for test patch points (SUT_RAG_Engine, engine, db_session, …)

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from sut_rag_core import SUT_RAG_Engine

# Shared dependencies + DB helpers live in deps.py. We re-import them at
# module scope so anything that still references ``api_server.db_session``
# etc. keeps working (this preserves the test patch surface as well).
from deps import (  # noqa: F401  (re-exported for tests and legacy callers)
    HISTORY_PAGE_LIMIT,
    MAX_PDF_UPLOAD_BYTES,
    MIN_PASSWORD_LENGTH,
    VALID_PROVIDERS as _VALID_PROVIDERS,
    VALID_USER_ROLES,
    db_execute,
    db_session,
    get_current_admin,
    get_current_user,
    get_db_conn,
    log_audit,
)
from routers import admin as _admin_router
from routers import auth as _auth_router
from routers import chat as _chat_router
from routers import health as _health_router
from routers import kg as _kg_router
from routers import policy as _policy_router
from routers import user_keys as _user_keys_router

# Re-export Pydantic models from the routers so existing imports
# (api_server.UserResponse, api_server.Token, …) keep working.
from routers.auth import PasswordChange, Token, UserRegister, UserResponse  # noqa: F401
from routers.admin import AnnouncementCreate, RoleUpdate  # noqa: F401


load_dotenv()

# Structured logging — operators can grep for [api_server] in HF Space logs.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api_server")

# --- Startup validation: required env vars must be present ---
# auth_utils raises on import if JWT_SECRET_KEY is missing; we also need
# API_KEY_ENCRYPTION_KEY for per-user key storage to work at all.
if not os.getenv("API_KEY_ENCRYPTION_KEY"):
    raise RuntimeError(
        "API_KEY_ENCRYPTION_KEY env var is required. "
        "Generate one with `python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"`"
    )

# --- Global Engine Instance ---
# Set by the lifespan handler below; exposed at module scope so tests can
# patch ``api_server.engine`` directly (see tests/conftest.py).
engine = None


# --- DB Init Helper ---
def init_system_tables():
    """Create / migrate all application tables and indexes.

    Idempotent — safe to call on every cold start (Hugging Face Spaces
    restarts the container often, so this absorbs ``IF NOT EXISTS`` cost).
    """
    conn = get_db_conn()
    cur = conn.cursor()
    # Ensure pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # Base tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            is_approved INTEGER DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            text_content TEXT NOT NULL,
            metadata_json JSONB,
            header_text TEXT,
            embedding vector(384)
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS chunks_fts_idx ON chunks USING GIN "
        "(to_tsvector('turkish', COALESCE(header_text, '') || ' ' || text_content));"
    )
    cur.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            conversation_id TEXT,
            query TEXT,
            response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_query_history_user_conv ON query_history(user_id, conversation_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_query_history_created ON query_history(created_at DESC)"
    )
    cur.execute("""
        CREATE TABLE IF NOT EXISTS saved_responses (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            query TEXT,
            response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS announcements (
            id TEXT PRIMARY KEY,
            message TEXT NOT NULL,
            created_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            active SMALLINT DEFAULT 1,
            FOREIGN KEY(created_by) REFERENCES users(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            feedback_id TEXT PRIMARY KEY,
            message_id TEXT,
            rating INTEGER,
            feedback_text TEXT,
            is_accurate INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(message_id) REFERENCES query_history(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS agent_runs (
            run_id TEXT PRIMARY KEY,
            trigger_message_id TEXT,
            agent_name TEXT,
            input_data TEXT,
            output_data TEXT,
            status TEXT,
            duration_ms INTEGER,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            FOREIGN KEY(trigger_message_id) REFERENCES query_history(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            log_id TEXT PRIMARY KEY,
            user_id TEXT,
            action_type TEXT NOT NULL,
            entity_type TEXT,
            entity_id TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    # --- Conversation Summary (Phase 1.1) ---
    cur.execute("ALTER TABLE query_history ADD COLUMN IF NOT EXISTS summary TEXT")
    # --- File Metadata (Phase 1.2) ---
    cur.execute("ALTER TABLE query_history ADD COLUMN IF NOT EXISTS file_metadata JSONB")

    # --- Knowledge Graph Tables (Phase 2) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_documents (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            conversation_id TEXT,
            filename TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_docs_user_conv ON user_documents(user_id, conversation_id)"
    )

    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT,
            favorited INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC)"
    )

    cur.execute("""
        CREATE TABLE IF NOT EXISTS kg_nodes (
            node_id      TEXT PRIMARY KEY,
            label        TEXT NOT NULL,
            type         TEXT NOT NULL,
            text_content TEXT DEFAULT '',
            atc_code     TEXT DEFAULT '',
            icd_code     TEXT DEFAULT '',
            embedding    vector(384),
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS kg_nodes_type_idx ON kg_nodes(type)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kg_edges (
            edge_id     TEXT PRIMARY KEY,
            source_id   TEXT NOT NULL,
            target_id   TEXT NOT NULL,
            relation    TEXT NOT NULL,
            confidence  REAL DEFAULT 1.0,
            source_rule TEXT DEFAULT '',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS kg_edges_source_idx ON kg_edges(source_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS kg_edges_target_idx ON kg_edges(target_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS kg_edges_relation_idx ON kg_edges(relation)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kg_build_log (
            log_id           TEXT PRIMARY KEY,
            started_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            finished_at      TIMESTAMP,
            status           TEXT DEFAULT 'running',
            nodes_created    INTEGER DEFAULT 0,
            edges_created    INTEGER DEFAULT 0,
            chunks_processed INTEGER DEFAULT 0,
            error_message    TEXT
        )
    """)

    # --- Per-user API keys (Multi-tenant SaaS) ---
    # NOTE: users.id is TEXT (UUID) in this schema, so user_id is TEXT here
    # (the cross-team contract used INT but our existing users table is TEXT).
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_api_keys (
            id              SERIAL PRIMARY KEY,
            user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            provider        VARCHAR(50) NOT NULL CHECK (provider IN ('gemini','openrouter','local')),
            encrypted_key   BYTEA NOT NULL,
            base_url        VARCHAR(500),
            key_hint        VARCHAR(8) NOT NULL,
            is_active       BOOLEAN DEFAULT TRUE,
            last_used_at    TIMESTAMP,
            created_at      TIMESTAMP DEFAULT NOW(),
            updated_at      TIMESTAMP DEFAULT NOW(),
            UNIQUE(user_id, provider)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_user_api_keys_user_id ON user_api_keys(user_id)")

    conn.commit()
    cur.close()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Boot the database schema and the RAG engine once per process."""
    init_system_tables()
    global engine
    engine = SUT_RAG_Engine()
    if not engine.load_database():
        logger.warning("SUT Database not loaded. Please populate it via Admin panel.")
    yield
    # Best-effort teardown — the engine owns a psycopg2 connection.
    try:
        if engine and getattr(engine, "conn", None):
            engine.conn.close()
    except Exception:  # noqa: BLE001
        pass


app = FastAPI(title="SUT Corporate Health API", lifespan=lifespan)

# --- CORS (env-driven, no wildcard in production) ---
_origins = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
# Order doesn't matter for routing (FastAPI matches on path), but we keep
# them grouped roughly the way the frontend touches them.
app.include_router(_health_router.router)
app.include_router(_auth_router.router)
app.include_router(_user_keys_router.router)
app.include_router(_chat_router.router)
app.include_router(_policy_router.router)
app.include_router(_kg_router.router)
app.include_router(_admin_router.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
