# api_server.py
# Description: Modern FastAPI Server for SUT Assistant. PostgreSQL Edition.

import logging
import os
import uuid
import json
import re
from typing import List, Optional
from contextlib import asynccontextmanager
from collections import Counter

import psycopg2
from psycopg2.extras import RealDictCursor

from fastapi import FastAPI, Depends, HTTPException, status, Body, BackgroundTasks, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

from sut_rag_core import SUT_RAG_Engine
from rag_storage import SUT_Storage_Manager
from secrets_utils import decrypt_api_key

# Shared dependencies + DB helpers live in deps.py since multiple routers
# need them. We re-import them at module scope so anything that still
# references ``api_server.db_session`` etc. keeps working.
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
from routers import auth as _auth_router
from routers import user_keys as _user_keys_router

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

# Constants (VALID_USER_ROLES, MIN_PASSWORD_LENGTH, MAX_PDF_UPLOAD_BYTES,
# HISTORY_PAGE_LIMIT) and helpers (get_db_conn, db_session, db_execute) are
# imported from ``deps`` above; we keep the names available at api_server
# module scope as re-exports for any legacy callers.

# --- DB Init Helper ---
def init_system_tables():
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
    cur.execute("CREATE INDEX IF NOT EXISTS chunks_fts_idx ON chunks USING GIN (to_tsvector('turkish', COALESCE(header_text, '') || ' ' || text_content));")
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
    cur.execute("CREATE INDEX IF NOT EXISTS idx_query_history_user_conv ON query_history(user_id, conversation_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_query_history_created ON query_history(created_at DESC)")
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
    cur.execute("CREATE INDEX IF NOT EXISTS idx_user_docs_user_conv ON user_documents(user_id, conversation_id)")

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
    cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC)")

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

# ``log_audit`` is imported from ``deps`` (see top of file).


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_system_tables()
    global engine
    engine = SUT_RAG_Engine()
    if not engine.load_database():
        logger.warning("SUT Database not loaded. Please populate it via Admin panel.")
    yield
    if engine and engine.conn:
        engine.conn.close()

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

# --- Routers wired in below ---
# Auth + user-key endpoints live in ``routers/auth.py`` and ``routers/user_keys.py``.
# Their Pydantic models (UserRegister, Token, UserResponse, PasswordChange,
# ApiKeyCreate, ApiKeyTest) are defined alongside the routers. We re-export
# the response models that other endpoints in this file still annotate with.
from routers.auth import Token, UserRegister, UserResponse  # noqa: F401,E402

app.include_router(_auth_router.router)
app.include_router(_user_keys_router.router)


@app.get("/api/admin/users", response_model=List[UserResponse])
async def list_users(admin: dict = Depends(get_current_admin)):
    with db_session() as conn:
        cur = db_execute(conn, "SELECT id, username, email, role, is_approved FROM users")
        users = cur.fetchall()
        cur.close()
    return [dict(u) for u in users]

# ─── Conversation Summarizer ──────────────────────────────────────────────────
def _summarize_history(history: list) -> str:
    """
    Summarize older conversation turns using Gemini.
    Called when chat history grows long (> 8 messages).
    Returns a compact Turkish summary string.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
        summarizer = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0
        )
        turns = ""
        for msg in history:
            role = "Kullanıcı" if msg["role"] == "user" else "Asistan"
            turns += f"{role}: {msg['content'][:400]}\n"

        prompt = (
            "Aşağıdaki SUT asistanı konuşmasını 2-3 cümleyle özetle. "
            "Konuşulan ana konular, sorulan sorular ve verilen önemli bilgileri kısaca belirt. "
            "Türkçe yaz.\n\n"
            f"KONUŞMA:\n{turns}"
        )
        resp = summarizer.invoke([HumanMessage(content=prompt)])
        return resp.content.strip()
    except Exception as e:
        logger.warning(f"History summarisation failed, falling back to truncation: {e}")
        # Fallback: naive truncation summary
        lines = [f"{m['role']}: {m['content'][:200]}" for m in history[-3:]]
        return " | ".join(lines)


def get_chat_history(user_id: str, conversation_id: str):
    """Retrieve chat history for a session."""
    with db_session() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT query, response, file_metadata FROM query_history
            WHERE user_id = %s AND conversation_id = %s
            ORDER BY created_at ASC
        """, (user_id, conversation_id))
        rows = cur.fetchall()
        cur.close()

    history = []
    for row in rows:
        history.append({"role": "user", "content": row["query"], "file_metadata": row["file_metadata"]})
        if row["response"]:
            history.append({"role": "assistant", "content": row["response"]})
    return history

def save_query_history(user_id: str, conversation_id: str, query: str, response: str, file_metadata: dict = None, query_id: str = None):
    """Save a new query and response pair to the history."""
    qid = query_id or str(uuid.uuid4())
    with db_session() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO query_history (id, user_id, conversation_id, query, response, file_metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (qid, user_id, conversation_id, query, response, json.dumps(file_metadata) if file_metadata else None))
        conn.commit()
        cur.close()


def ensure_conversation_row(user_id: str, conversation_id: str, title_hint: str):
    """Ensure conversations row exists; set title from first message if new."""
    if not conversation_id:
        return
    title = (title_hint or "Sohbet")[:80]
    with db_session() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT conversation_id FROM conversations WHERE conversation_id = %s AND user_id = %s",
            (conversation_id, user_id),
        )
        if cur.fetchone():
            cur.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE conversation_id = %s AND user_id = %s",
                (conversation_id, user_id),
            )
        else:
            cur.execute(
                """
                INSERT INTO conversations (conversation_id, user_id, title, favorited)
                VALUES (%s, %s, %s, 0)
                """,
                (conversation_id, user_id, title),
            )
        conn.commit()
        cur.close()


# --- Pydantic Models ---
class ChatQuery(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    role: Optional[str] = "PATIENT"  # Default to PATIENT if not provided
    k: Optional[int] = None  # RAG top-k; defaults handled in engine if None
    provider: Optional[str] = None  # 'gemini' | 'openrouter' | 'local' — picks which stored key to use

# ``PasswordChange`` lives in routers.auth.
class SaveResponse(BaseModel):
    query: str
    response: str

class RoleUpdate(BaseModel):
    role: str

class AnnouncementCreate(BaseModel):
    message: str
class FeedbackCreate(BaseModel):
    message_id: str
    rating: int  # 1-5 or -1, 1
    feedback_text: str = ""
    is_accurate: bool = True


class FeedbackReportCreate(BaseModel):
    message_id: str
    category: str = "other"  # wrong_info | missing_source | other
    feedback_text: str = ""


class ConversationPatch(BaseModel):
    title: str


class FavoriteBody(BaseModel):
    favorited: bool = True


# ``ApiKeyCreate`` / ``ApiKeyTest`` live in routers.user_keys.
# ``_VALID_PROVIDERS`` is re-exported from deps at module top.

def _fetch_user_api_key(user_id: str, requested_provider: Optional[str]):
    """Return (provider, plaintext_key, base_url, key_row_id) for the user.

    Selection rule:
    - If `requested_provider` is given, look up that exact row.
    - Otherwise, fall back to the most-recently-used active key
      (most recently created if none has been used yet).
    Returns (None, None, None, None) if no active key is found.
    """
    with db_session() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        if requested_provider:
            cur.execute(
                """
                SELECT id, provider, encrypted_key, base_url
                FROM user_api_keys
                WHERE user_id = %s AND provider = %s AND is_active = TRUE
                LIMIT 1
                """,
                (user_id, requested_provider),
            )
        else:
            cur.execute(
                """
                SELECT id, provider, encrypted_key, base_url
                FROM user_api_keys
                WHERE user_id = %s AND is_active = TRUE
                ORDER BY (last_used_at IS NULL), last_used_at DESC, created_at DESC
                LIMIT 1
                """,
                (user_id,),
            )
        row = cur.fetchone()
        cur.close()
        if not row:
            return None, None, None, None
        try:
            plaintext = decrypt_api_key(row["encrypted_key"])
        except Exception as e:  # InvalidToken / corrupted ciphertext
            logger.error(f"Failed to decrypt API key id={row['id']} for user_id={user_id}: {e}")
            # Map to 401 rather than 500 — the row is unusable for this user.
            raise HTTPException(
                status_code=401,
                detail="Kayıtlı API anahtarı çözülemedi. Lütfen anahtarı yeniden ekleyin.",
            )
        return row["provider"], plaintext, row["base_url"], row["id"]


def _mark_api_key_used(key_id: int):
    """Best-effort update of last_used_at. Errors are swallowed."""
    try:
        with db_session() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE user_api_keys SET last_used_at = NOW() WHERE id = %s",
                (key_id,),
            )
            conn.commit()
            cur.close()
    except Exception as e:
        logger.warning(f"Failed to update last_used_at for api_key {key_id}: {e}")


# --- Chat ---
@app.post("/api/chat/query")
async def chat_query(q: ChatQuery, current_user: dict = Depends(get_current_user)):
    """Main Agentic RAG chat endpoint (streaming)."""
    effective_conv_id = q.conversation_id or str(uuid.uuid4())
    outbound_query_id = str(uuid.uuid4())
    rag_k = q.k if q.k is not None else 5

    # 0. Resolve user's LLM API key (multi-tenant: each user brings their own).
    provider, user_api_key, user_base_url, key_row_id = _fetch_user_api_key(
        current_user["id"], q.provider
    )
    if not user_api_key:
        raise HTTPException(
            status_code=400,
            detail="Please configure your LLM API key in Settings → API Keys before chatting.",
        )

    # 1. Fetch user documents for this conversation to provide context
    user_docs_context = ""
    active_file_metadata = None
    try:
        with db_session() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT filename, content FROM user_documents WHERE user_id = %s AND conversation_id = %s",
                (current_user["id"], effective_conv_id),
            )
            rows = cur.fetchall()
            if rows:
                user_docs_context = "\n\n--- KULLANICI DÖKÜMANLARI ---\n" + "\n".join([r["content"] for r in rows])
                active_file_metadata = {"filename": rows[0]["filename"], "type": "pdf"}
            cur.close()
    except Exception as e:
        logger.warning(f"Error fetching user docs: {e}")

    full_query = q.query
    if user_docs_context:
        full_query = f"{user_docs_context}\n\nKullanıcı Sorusu: {q.query}"

    # 2. Get history
    history = get_chat_history(current_user["id"], effective_conv_id)
    if len(history) > 10:
        summary_text = _summarize_history(history)
        # Engine expects List[{"role","content"}], not a bare summary string.
        history = [
            {
                "role": "user",
                "content": f"Önceki konuşmanın özeti (bağlam):\n{summary_text}",
            }
        ]

    async def generate():
        meta = {
            "meta": {
                "conversation_id": effective_conv_id,
                "query_id": outbound_query_id,
            }
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
        try:
            for chunk in engine.query_agentic_rag_stream(
                full_query,
                chat_history=history,
                role=q.role or "PATIENT",
                k=rag_k,
                user_api_key=user_api_key,
                user_provider=provider,
                user_base_url=user_base_url,
            ):
                if "final_answer" in chunk:
                    ensure_conversation_row(current_user["id"], effective_conv_id, q.query)
                    save_query_history(
                        current_user["id"],
                        effective_conv_id,
                        q.query,
                        chunk["final_answer"],
                        file_metadata=active_file_metadata,
                        query_id=outbound_query_id,
                    )
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        finally:
            # Best-effort: record that this key was just used.
            if key_row_id is not None:
                _mark_api_key_used(key_row_id)

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/api/chat/upload")
async def upload_document(
    conversation_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload a medical report PDF and extract its text.

    Hardening:
    - filename extension check (PDF only)
    - server-side size cap (MAX_PDF_UPLOAD_BYTES)
    - generic error to the client; full traceback only in server logs
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Sadece PDF dosyaları yüklenebilir.")

    try:
        import io
        from pypdf import PdfReader

        # Enforce a server-side size limit before loading into memory.
        # FastAPI `UploadFile.read()` has no built-in cap; an unbounded read
        # on a 500 MB PDF would happily OOM the Space.
        content = await file.read(MAX_PDF_UPLOAD_BYTES + 1)
        if len(content) > MAX_PDF_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Dosya boyutu çok büyük (maks {MAX_PDF_UPLOAD_BYTES // (1024*1024)} MB).",
            )

        try:
            pdf = PdfReader(io.BytesIO(content))
            text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        except Exception as e:
            logger.warning(f"PDF parse failed for user={current_user.get('id')}: {e}")
            raise HTTPException(status_code=400, detail="PDF dosyası okunamadı veya bozuk.")

        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF'den metin çıkarılamadı.")

        doc_id = str(uuid.uuid4())
        with db_session() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO user_documents (id, user_id, conversation_id, filename, content)
                VALUES (%s, %s, %s, %s, %s)
            """, (doc_id, current_user["id"], conversation_id, file.filename, text))
            conn.commit()
            cur.close()

        return {
            "message": "Belge başarıyla yüklendi ve işlendi.",
            "doc_id": doc_id,
            "filename": file.filename,
            "char_count": len(text),
        }
    except HTTPException:
        raise
    except Exception as e:
        # Never leak raw exception text — that has leaked stack frames before.
        logger.exception(f"Upload failed for user={current_user.get('id')}: {e}")
        raise HTTPException(status_code=500, detail="Dosya yüklenirken bir hata oluştu.")

@app.post("/api/feedback")
async def submit_feedback(data: FeedbackCreate, current_user: dict = Depends(get_current_user)):
    with db_session() as conn:
        db_execute(conn,
            "INSERT INTO user_feedback (feedback_id, message_id, rating, feedback_text, is_accurate) VALUES (%s, %s, %s, %s, %s)",
            (str(uuid.uuid4()), data.message_id, data.rating, data.feedback_text, 1 if data.is_accurate else 0)
        )
        conn.commit()
    return {"message": "Geri bildiriminiz kaydedildi. Teşekkürler!"}

@app.get("/api/history")
async def get_history(current_user: dict = Depends(get_current_user)):
    with db_session() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT id, conversation_id, query, response, file_metadata, created_at FROM query_history WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
            (current_user["id"], HISTORY_PAGE_LIMIT),
        )
        history = cur.fetchall()
        cur.close()

        cur2 = conn.cursor(cursor_factory=RealDictCursor)
        cur2.execute(
            "SELECT query, response, created_at FROM saved_responses WHERE user_id = %s ORDER BY created_at DESC",
            (current_user["id"],)
        )
        saved = cur2.fetchall()
        cur2.close()
    return {"history": [dict(h) for h in history], "saved": [dict(s) for s in saved]}

@app.post("/api/history/save")
async def save_response(data: SaveResponse, current_user: dict = Depends(get_current_user)):
    with db_session() as conn:
        db_execute(conn,
            "INSERT INTO saved_responses (id, user_id, query, response) VALUES (%s, %s, %s, %s)",
            (str(uuid.uuid4()), current_user["id"], data.query, data.response)
        )
        conn.commit()
    return {"message": "Yanıt kaydedildi."}


@app.get("/api/config")
async def get_app_config(current_user: dict = Depends(get_current_user)):
    """Lightweight client config (model label for topbar)."""
    display = (
        os.getenv("SUT_MODEL_DISPLAY")
        or os.getenv("GEMINI_MODEL_NAME")
        or "Gemini 2.0 Flash"
    )
    return {
        "model_display_name": display,
        "provider": os.getenv("LLM_PROVIDER", "google"),
    }


@app.get("/api/conversations")
async def list_conversations(limit: int = 10, current_user: dict = Depends(get_current_user)):
    safe_limit = max(1, min(int(limit or 10), 50))
    with db_session() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT conversation_id, title, favorited, updated_at, created_at
            FROM conversations
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (current_user["id"], safe_limit),
        )
        rows = cur.fetchall()
        cur.close()
    return {"conversations": [dict(r) for r in rows]}


@app.get("/api/conversations/search")
async def search_conversations(q: str = "", limit: int = 20, current_user: dict = Depends(get_current_user)):
    if not q.strip():
        return {"conversations": []}
    safe_limit = max(1, min(int(limit or 20), 50))
    with db_session() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT DISTINCT c.conversation_id, c.title, c.favorited, c.updated_at
            FROM conversations c
            LEFT JOIN query_history qh ON qh.conversation_id = c.conversation_id AND qh.user_id = c.user_id
            WHERE c.user_id = %s AND (
                c.title ILIKE %s OR qh.query ILIKE %s OR qh.response ILIKE %s
            )
            ORDER BY c.updated_at DESC
            LIMIT %s
            """,
            (current_user["id"], f"%{q}%", f"%{q}%", f"%{q}%", safe_limit),
        )
        rows = cur.fetchall()
        cur.close()
    return {"conversations": [dict(r) for r in rows]}


@app.patch("/api/conversations/{conversation_id}")
async def patch_conversation(conversation_id: str, data: ConversationPatch, current_user: dict = Depends(get_current_user)):
    with db_session() as conn:
        cur = db_execute(
            conn,
            "UPDATE conversations SET title = %s, updated_at = CURRENT_TIMESTAMP WHERE conversation_id = %s AND user_id = %s",
            (data.title[:200], conversation_id, current_user["id"]),
        )
        if cur.rowcount == 0:
            cur.close()
            raise HTTPException(status_code=404, detail="Konuşma bulunamadı.")
        conn.commit()
        cur.close()
    return {"message": "Başlık güncellendi.", "conversation_id": conversation_id}


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, current_user: dict = Depends(get_current_user)):
    with db_session() as conn:
        db_execute(
            conn,
            """DELETE FROM agent_runs WHERE trigger_message_id IN
               (SELECT id FROM query_history WHERE conversation_id = %s AND user_id = %s)""",
            (conversation_id, current_user["id"]),
        )
        db_execute(
            conn,
            """DELETE FROM user_feedback WHERE message_id IN
               (SELECT id FROM query_history WHERE conversation_id = %s AND user_id = %s)""",
            (conversation_id, current_user["id"]),
        )
        db_execute(conn, "DELETE FROM query_history WHERE conversation_id = %s AND user_id = %s", (conversation_id, current_user["id"]))
        db_execute(conn, "DELETE FROM user_documents WHERE conversation_id = %s AND user_id = %s", (conversation_id, current_user["id"]))
        db_execute(conn, "DELETE FROM conversations WHERE conversation_id = %s AND user_id = %s", (conversation_id, current_user["id"]))
        conn.commit()
    return {"message": "Konuşma silindi."}


@app.put("/api/conversations/{conversation_id}/favorite")
async def favorite_conversation(conversation_id: str, body: FavoriteBody = Body(...), current_user: dict = Depends(get_current_user)):
    with db_session() as conn:
        cur = db_execute(
            conn,
            "UPDATE conversations SET favorited = %s, updated_at = CURRENT_TIMESTAMP WHERE conversation_id = %s AND user_id = %s",
            (1 if body.favorited else 0, conversation_id, current_user["id"]),
        )
        if cur.rowcount == 0:
            cur.close()
            raise HTTPException(status_code=404, detail="Konuşma bulunamadı.")
        conn.commit()
        cur.close()
    return {"message": "Güncellendi.", "favorited": body.favorited}


@app.post("/api/feedback/report")
async def submit_feedback_report(data: FeedbackReportCreate, current_user: dict = Depends(get_current_user)):
    """Structured issue report for admin evaluation pipeline."""
    prefix = f"[{data.category}] "
    text = prefix + (data.feedback_text or "").strip()
    with db_session() as conn:
        db_execute(
            conn,
            "INSERT INTO user_feedback (feedback_id, message_id, rating, feedback_text, is_accurate) VALUES (%s, %s, %s, %s, %s)",
            (str(uuid.uuid4()), data.message_id, -1, text, 0),
        )
        conn.commit()
    return {"message": "Hata bildirimi kaydedildi. Teşekkürler!"}

# Per-user API key endpoints now live in routers/user_keys.py.


# --- Health Check (no auth) ---

@app.get("/health")
async def health_check():
    """Liveness + DB readiness probe. Always returns 200 with JSON body so
    load balancers can inspect db status without false-positive failover."""
    db_status = "down"
    try:
        with db_session() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
            cur.close()
        db_status = "ok"
    except Exception:
        db_status = "down"
    return {"status": "ok", "db": db_status, "version": "1.0.0"}


# --- Admin Endpoints ---

@app.get("/api/admin/system")
async def get_system_metrics(admin: dict = Depends(get_current_admin)):
    with db_session() as conn:
        cur = db_execute(conn, "SELECT COUNT(*) FROM users")
        users_count = cur.fetchone()[0]
        cur.close()
        cur = db_execute(conn, "SELECT COUNT(*) FROM query_history")
        queries_count = cur.fetchone()[0]
        cur.close()
        chunks_count = 0
        try:
            cur = db_execute(conn, "SELECT COUNT(*) FROM chunks")
            chunks_count = cur.fetchone()[0]
            cur.close()
        except Exception as e:
            logger.debug(f"chunks count skipped: {e}")
            conn.rollback()
        cur = db_execute(conn, "SELECT message FROM announcements WHERE active = 1 ORDER BY created_at DESC LIMIT 1")
        active_announcement = cur.fetchone()
        cur.close()
        cur = db_execute(conn, "SELECT COUNT(*) FROM users WHERE is_approved = 0")
        pending_count = cur.fetchone()[0]
        cur.close()

    return {
        "users_count": users_count,
        "queries_count": queries_count,
        "chunks_count": chunks_count,
        "pending_count": pending_count,
        "active_announcement": dict(active_announcement) if active_announcement else None
    }

@app.get("/api/admin/activity")
async def get_admin_activity(admin: dict = Depends(get_current_admin)):
    with db_session() as conn:
        cur = db_execute(conn, """
            SELECT qh.query, qh.created_at, u.username, u.role
            FROM query_history qh
            JOIN users u ON qh.user_id = u.id
            ORDER BY qh.created_at DESC
            LIMIT 20
        """)
        rows = cur.fetchall()
        cur.close()
    return [dict(r) for r in rows]

_ANALYTICS_STOPWORDS = {
    "ve", "bir", "ile", "bu", "için", "da", "de", "mi", "ne", "ben", "sen",
    "biz", "siz", "o", "şu", "ki", "gibi", "ama", "veya", "ya", "daha",
    "olan", "nasıl", "nedir", "hakkında", "bilgi", "ver", "söyle",
    "the", "is", "a", "of", "in", "to", "what", "how", "about",
}


@app.get("/api/admin/analytics")
async def get_admin_analytics(admin: dict = Depends(get_current_admin)):
    with db_session() as conn:
        cur = db_execute(conn, "SELECT query FROM query_history")
        all_queries = cur.fetchall()
        cur.close()

        word_counter = Counter()
        for row in all_queries:
            words = re.findall(r'\b[a-zA-ZğüşıöçĞÜŞİÖÇ]{4,}\b', row["query"].lower())
            for w in words:
                if w not in _ANALYTICS_STOPWORDS:
                    word_counter[w] += 1
        top_keywords = [{"keyword": k, "count": v} for k, v in word_counter.most_common(10)]

        # Daily volume — last 7 days (PostgreSQL syntax)
        cur = db_execute(conn, """
            SELECT DATE(created_at) as day, COUNT(*) as count
            FROM query_history
            WHERE created_at >= NOW() - INTERVAL '7 days'
            GROUP BY DATE(created_at)
            ORDER BY day ASC
        """)
        daily_rows = cur.fetchall()
        cur.close()
        daily_volume = [dict(r) for r in daily_rows]

        cur = db_execute(conn, "SELECT COUNT(*) FROM users")
        total_users = cur.fetchone()[0]
        cur.close()

        cur = db_execute(conn, "SELECT COUNT(DISTINCT user_id) FROM query_history WHERE created_at >= NOW() - INTERVAL '1 day'")
        daily_active = cur.fetchone()[0]
        cur.close()
        daily_engagement_rate = round((daily_active / total_users * 100) if total_users > 0 else 0, 1)

        cur = db_execute(conn, "SELECT COUNT(DISTINCT user_id) FROM query_history WHERE created_at >= NOW() - INTERVAL '30 days'")
        monthly_active = cur.fetchone()[0]
        cur.close()
        monthly_engagement_rate = round((monthly_active / total_users * 100) if total_users > 0 else 0, 1)

    return {
        "top_keywords": top_keywords,
        "daily_volume": daily_volume,
        "daily_engagement_rate": daily_engagement_rate,
        "monthly_engagement_rate": monthly_engagement_rate,
        "daily_active_users": daily_active,
        "monthly_active_users": monthly_active,
        "total_users": total_users
    }

def _chunk_row_to_result(c) -> dict:
    meta = c["metadata_json"] if isinstance(c["metadata_json"], dict) else (json.loads(c["metadata_json"]) if c["metadata_json"] else {})
    title = " > ".join([v for k, v in meta.items() if k.startswith("Header")])
    return {
        "id": c["chunk_id"],
        "title": title or "Başlıksız Bölüm",
        "excerpt": c["text_content"][:300],
        "full_text": c["text_content"],
        "metadata": meta,
    }


def _policy_query_token(raw: str) -> str:
    """Keep letters, digits, Turkish chars; min length 2 after strip."""
    t = re.sub(r"[^\wğüşıöçĞÜŞİÖÇ0-9]+", " ", raw, flags=re.I).strip()
    return t[:120] if len(t) >= 2 else ""


def _search_policy_chunks(conn, q: str, q_mode: str, limit: int, offset: int):
    """
    Full-text + ILIKE fallback.
    q_mode: phrase | and | or
    If ``q`` contains a comma: OR across comma-separated groups; within each group,
    tokens (whitespace) are combined with AND (stronger matches).
    """
    fts_col = "to_tsvector('turkish', COALESCE(header_text,'') || ' ' || text_content)"
    mode = (q_mode or "phrase").lower()
    if mode not in ("phrase", "and", "or"):
        mode = "phrase"

    def run_sql(sql: str, params: tuple):
        cur = db_execute(conn, sql, params)
        rows = cur.fetchall()
        cur.close()
        return rows

    if "," in q:
        groups = [g.strip() for g in q.split(",") if g.strip()]
        or_ts, or_like, params_ts, params_like = [], [], [], []
        for g in groups:
            tokens = [_policy_query_token(w) for w in g.split()]
            tokens = [t for t in tokens if t]
            if not tokens:
                continue
            and_ts = " AND ".join([f"{fts_col} @@ plainto_tsquery('turkish', %s)" for _ in tokens])
            or_ts.append(f"({and_ts})")
            params_ts.extend(tokens)
            and_li = " AND ".join(["text_content ILIKE %s" for _ in tokens])
            or_like.append(f"({and_li})")
            params_like.extend([f"%{t}%" for t in tokens])
        if not or_ts:
            return []
        sql_ts = f"""
            SELECT chunk_id, text_content, metadata_json FROM chunks
            WHERE ({' OR '.join(or_ts)})
            LIMIT %s OFFSET %s
        """
        rows = run_sql(sql_ts, tuple(params_ts + [limit, offset]))
        if rows:
            return rows
        sql_li = f"""
            SELECT chunk_id, text_content, metadata_json FROM chunks
            WHERE ({' OR '.join(or_like)})
            LIMIT %s OFFSET %s
        """
        return run_sql(sql_li, tuple(params_like + [limit, offset]))

    tokens = [_policy_query_token(w) for w in q.split()]
    tokens = [t for t in tokens if t]

    if mode == "phrase" or len(tokens) <= 1:
        phrase = q.strip()
        rows = run_sql(
            f"""
            SELECT chunk_id, text_content, metadata_json FROM chunks
            WHERE {fts_col} @@ plainto_tsquery('turkish', %s)
            LIMIT %s OFFSET %s
            """,
            (phrase, limit, offset),
        )
        if rows:
            return rows
        return run_sql(
            """
            SELECT chunk_id, text_content, metadata_json FROM chunks
            WHERE text_content ILIKE %s LIMIT %s OFFSET %s
            """,
            (f"%{phrase}%", limit, offset),
        )

    joiner = " AND " if mode == "and" else " OR "
    fts_clause = joiner.join([f"{fts_col} @@ plainto_tsquery('turkish', %s)" for _ in tokens])
    rows = run_sql(
        f"""
        SELECT chunk_id, text_content, metadata_json FROM chunks
        WHERE ({fts_clause})
        LIMIT %s OFFSET %s
        """,
        tuple(tokens + [limit, offset]),
    )
    if rows:
        return rows
    like_clause = joiner.join(["text_content ILIKE %s" for _ in tokens])
    like_params = [f"%{t}%" for t in tokens]
    return run_sql(
        f"""
        SELECT chunk_id, text_content, metadata_json FROM chunks
        WHERE ({like_clause})
        LIMIT %s OFFSET %s
        """,
        tuple(like_params + [limit, offset]),
    )


@app.get("/api/policies")
async def search_policies(
    q: str = "",
    section: str = "",
    chunk_id: str = "",
    limit: int = 20,
    offset: int = 0,
    date_from: str = "",
    date_to: str = "",
    status_filter: str = "",
    q_mode: str = "phrase",
    current_user: dict = Depends(get_current_user),
):
    # Clamp pagination — limit cap (200) prevents accidental megabyte responses.
    try:
        safe_limit = max(1, min(int(limit), 200))
    except (TypeError, ValueError):
        safe_limit = 20
    try:
        safe_offset = max(0, int(offset))
    except (TypeError, ValueError):
        safe_offset = 0

    try:
        with db_session() as conn:
            chunks = []
            if chunk_id:
                cur = db_execute(
                    conn,
                    "SELECT chunk_id, text_content, metadata_json FROM chunks WHERE chunk_id = %s",
                    (chunk_id,),
                )
                row = cur.fetchone()
                cur.close()
                chunks = [row] if row else []
            elif q:
                chunks = _search_policy_chunks(conn, q, q_mode, safe_limit, safe_offset)
            else:
                cur = db_execute(conn,
                    "SELECT chunk_id, text_content, metadata_json FROM chunks LIMIT %s OFFSET %s",
                    (safe_limit, safe_offset)
                )
                chunks = cur.fetchall()
                cur.close()

            results = []
            for c in chunks:
                item = _chunk_row_to_result(c)
                meta_raw = json.dumps(item["metadata"], ensure_ascii=False) if item["metadata"] else ""
                text_blob = item["full_text"] + meta_raw
                if section and section.upper() not in item["title"].upper():
                    continue
                if date_from and date_from not in text_blob:
                    continue
                if date_to and date_to not in text_blob:
                    continue
                if status_filter:
                    st = (item["metadata"].get("status") or item["metadata"].get("durum") or "").lower()
                    if status_filter.lower() not in st and status_filter.lower() not in text_blob.lower():
                        continue
                results.append(item)

            cur = db_execute(conn, "SELECT COUNT(*) FROM chunks")
            total = cur.fetchone()[0]
            cur.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"policy search failed: {e}")
        raise HTTPException(status_code=500, detail="Arama hatası.")
    return {"results": results, "total": total, "offset": safe_offset, "limit": safe_limit}

@app.put("/api/admin/users/{user_id}/role")
async def update_user_role(user_id: str, data: RoleUpdate, admin: dict = Depends(get_current_admin)):
    if data.role not in VALID_USER_ROLES:
        raise HTTPException(status_code=400, detail="Geçersiz rol.")
    with db_session() as conn:
        db_execute(conn, "UPDATE users SET role = %s WHERE id = %s", (data.role, user_id))
        log_audit(conn, "role_updated", user_id=admin["id"], entity_type="user", entity_id=user_id, details={"new_role": data.role})
        conn.commit()
    return {"message": "Rol güncellendi."}

@app.delete("/api/admin/users/{user_id}")
async def delete_user(user_id: str, admin: dict = Depends(get_current_admin)):
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="Kendi hesabınızı silemezsiniz.")
    with db_session() as conn:
        # 1. Delete feedback linked to user's queries
        db_execute(conn, """
            DELETE FROM user_feedback WHERE message_id IN (
                SELECT id FROM query_history WHERE user_id = %s
            )
        """, (user_id,))
        # 2. Delete agent runs linked to user's queries
        db_execute(conn, """
            DELETE FROM agent_runs WHERE trigger_message_id IN (
                SELECT id FROM query_history WHERE user_id = %s
            )
        """, (user_id,))
        # 3. Delete user's documents and history
        db_execute(conn, "DELETE FROM user_documents WHERE user_id = %s", (user_id,))
        db_execute(conn, "DELETE FROM query_history WHERE user_id = %s", (user_id,))
        db_execute(conn, "DELETE FROM saved_responses WHERE user_id = %s", (user_id,))
        # 4. Delete user's own audit logs (or nullify, but delete is safer for clean tests)
        db_execute(conn, "DELETE FROM audit_logs WHERE user_id = %s", (user_id,))
        # 5. Finally delete the user
        db_execute(conn, "DELETE FROM users WHERE id = %s", (user_id,))
        log_audit(conn, "user_deleted", user_id=admin["id"], entity_type="user", entity_id=user_id)
        conn.commit()
    return {"message": "Kullanıcı silindi."}

@app.put("/api/admin/users/{user_id}/approve")
async def approve_user(user_id: str, admin: dict = Depends(get_current_admin)):
    with db_session() as conn:
        db_execute(conn, "UPDATE users SET is_approved = 1 WHERE id = %s", (user_id,))
        log_audit(conn, "user_approved", user_id=admin["id"], entity_type="user", entity_id=user_id)
        conn.commit()
    return {"message": "Kullanıcı onaylandı."}

@app.post("/api/admin/rebuild-index")
async def rebuild_index(background_tasks: BackgroundTasks, admin: dict = Depends(get_current_admin)):
    global engine

    def run_indexing():
        global engine
        try:
            logger.info("[BACKGROUND] Starting indexing task...")
            storage = SUT_Storage_Manager(engine.embeddings_model)
            success = storage.populate_database()
            if success:
                # Reload global engine connection
                new_engine = SUT_RAG_Engine()
                if new_engine.load_database():
                    engine = new_engine
                    logger.info("[BACKGROUND] Indexing and engine reload complete.")
                else:
                    logger.warning("[BACKGROUND] Indexing complete but engine reload failed.")
            else:
                logger.warning("[BACKGROUND] Indexing failed.")
        except Exception as e:
            logger.exception(f"[BACKGROUND] Indexing error: {e}")

    background_tasks.add_task(run_indexing)

    with db_session() as dbconn:
        log_audit(dbconn, "index_rebuild_started", user_id=admin["id"])
        dbconn.commit()

    return {"message": "İndeksleme işlemi arka planda başlatıldı. İlerlemeyi sistem günlüklerinden takip edebilirsiniz."}

# --- Announcements ---

@app.post("/api/admin/announcements")
async def create_announcement(data: AnnouncementCreate, admin: dict = Depends(get_current_admin)):
    with db_session() as conn:
        db_execute(conn, "UPDATE announcements SET active = 0")
        ann_id = str(uuid.uuid4())
        db_execute(conn,
            "INSERT INTO announcements (id, message, created_by, active) VALUES (%s, %s, %s, 1)",
            (ann_id, data.message, admin["id"])
        )
        log_audit(conn, "announcement_created", user_id=admin["id"], entity_type="announcement", entity_id=ann_id)
        conn.commit()
    return {"message": "Duyuru yayınlandı."}

@app.get("/api/announcements")
async def get_active_announcement(current_user: dict = Depends(get_current_user)):
    with db_session() as conn:
        cur = db_execute(conn,
            "SELECT id, message, created_at FROM announcements WHERE active = 1 ORDER BY created_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        cur.close()
    return dict(row) if row else {}

@app.delete("/api/admin/announcements/{ann_id}")
async def deactivate_announcement(ann_id: str, admin: dict = Depends(get_current_admin)):
    with db_session() as conn:
        db_execute(conn, "UPDATE announcements SET active = 0 WHERE id = %s", (ann_id,))
        log_audit(conn, "announcement_deactivated", user_id=admin["id"], entity_type="announcement", entity_id=ann_id)
        conn.commit()
    return {"message": "Duyuru kaldırıldı."}

# --- Knowledge Graph API (Postgres-backed) ---

from kg_storage import KG_Storage_Manager
_kg = None

def _get_kg() -> KG_Storage_Manager:
    """Lazy-initialize KG_Storage_Manager so it connects AFTER the DB is ready."""
    global _kg
    if _kg is None:
        _kg = KG_Storage_Manager()
    return _kg

@app.get("/api/kg/stats")
async def get_kg_stats(current_user: dict = Depends(get_current_user)):
    """Return node/edge counts by type/relation."""
    try:
        return _get_kg().get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/kg/nodes")
async def search_kg_nodes(
    q: str = "",
    type: str = "",
    limit: int = 20,
    current_user: dict = Depends(get_current_user)
):
    """Search KG nodes by label (string + semantic)."""
    try:
        safe_limit = max(1, min(int(limit), 500))
    except (TypeError, ValueError):
        safe_limit = 20
    try:
        type_filter = type.upper() if type else None
        if q:
            exact = _get_kg().find_nodes_by_label(q, k=safe_limit, type_filter=type_filter)
            semantic = _get_kg().find_nodes_semantic(q, k=safe_limit, type_filter=type_filter)
            seen = {r["node_id"] for r in exact}
            merged = exact + [r for r in semantic if r["node_id"] not in seen]
            return {"nodes": merged[:safe_limit]}
        # Parameterized to prevent SQL injection — never interpolate user input.
        with db_session() as conn:
            base_sql = "SELECT node_id, label, type, text_content, atc_code, icd_code FROM kg_nodes"
            if type_filter:
                cur = db_execute(
                    conn,
                    base_sql + " WHERE type = %s LIMIT %s",
                    (type_filter, safe_limit),
                )
            else:
                cur = db_execute(conn, base_sql + " LIMIT %s", (safe_limit,))
            rows = [dict(r) for r in cur.fetchall()]
            cur.close()
        return {"nodes": rows}
    except Exception as e:
        logger.exception(f"kg/nodes failed: {e}")
        raise HTTPException(status_code=500, detail="Bilgi grafiği araması başarısız.")

@app.get("/api/kg/node/{node_id}")
async def get_kg_node(node_id: str, current_user: dict = Depends(get_current_user)):
    """Get a single node with all its neighbors."""
    node = _get_kg().get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    neighbors = _get_kg().get_neighbors(node_id, limit=20)
    return {"node": node, "neighbors": neighbors}

@app.get("/api/kg/subgraph/{rule_id}")
async def get_kg_subgraph(rule_id: str, current_user: dict = Depends(get_current_user)):
    """Get the subgraph for a RULE node (all related nodes & edges)."""
    return _get_kg().get_rule_subgraph(rule_id)

@app.get("/api/kg/path")
async def find_kg_path(
    from_id: str,
    to_id: str,
    max_hops: int = 3,
    current_user: dict = Depends(get_current_user)
):
    """Find shortest path between two nodes."""
    # Cap max_hops — BFS over the full KG can explode at higher hop counts.
    try:
        safe_hops = max(1, min(int(max_hops), 5))
    except (TypeError, ValueError):
        safe_hops = 3
    path = _get_kg().find_path(from_id, to_id, max_hops=safe_hops)
    return {"path": path, "found": len(path) > 0}

@app.post("/api/admin/kg/rebuild")
async def rebuild_kg(background_tasks: BackgroundTasks, admin: dict = Depends(get_current_admin)):
    """Trigger a full KG rebuild in the background."""
    def run_kg_build():
        try:
            from kg_builder import KG_Builder, KG_Enricher
            builder = KG_Builder()
            builder.build(clear_existing=True)
            enricher = KG_Enricher()
            enricher.enrich()
            logger.info("[KG_REBUILD] Complete.")
        except Exception as e:
            logger.exception(f"[KG_REBUILD] Error: {e}")

    background_tasks.add_task(run_kg_build)
    with db_session() as dbconn:
        log_audit(dbconn, "kg_rebuild_started", user_id=admin["id"])
        dbconn.commit()
    return {"message": "KG yeniden oluşturma işlemi arka planda başlatıldı."}

@app.get("/api/admin/kg/stats")
async def get_admin_kg_stats(admin: dict = Depends(get_current_admin)):
    try:
        return _get_kg().get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/eval/results")
async def get_eval_results(admin: dict = Depends(get_current_admin)):
    """Return the persisted retrieval evaluation results JSON."""
    eval_path = os.path.join(os.path.dirname(__file__), "eval_results", "retrieval_results.json")
    if not os.path.exists(eval_path):
        raise HTTPException(status_code=404, detail="Eval results not found")
    with open(eval_path) as f:
        return json.load(f)


@app.post("/api/admin/kg/benchmark")
async def run_kg_benchmark(admin: dict = Depends(get_current_admin)):
    """Run a mini KG benchmark: 10 multi-hop questions against KG tools."""
    QUESTIONS = [
        {"q": "Pnömokok aşısı hangi yaş grubuna ödenir?", "expected": ["AGE_LIMIT", "DRUG"]},
        {"q": "Kanser tedavisinde hangi uzman raporu gereklidir?", "expected": ["SPECIALIST", "DOCUMENT"]},
        {"q": "Diyabet ilacı için endikasyon şartı var mı?", "expected": ["DRUG", "CONDITION"]},
        {"q": "Fizik tedavi seansları için seans limiti nedir?", "expected": ["RULE", "DOSAGE"]},
        {"q": "Ortopedik protez temin şartları nelerdir?", "expected": ["RULE", "DEVICE"]},
        {"q": "Kronik böbrek hastalarına hangi ilaçlar ödenir?", "expected": ["DIAGNOSIS", "DRUG"]},
        {"q": "MS hastalığında biyolojik ilaç kullanma koşulları?", "expected": ["DIAGNOSIS", "CONDITION"]},
        {"q": "İşitme cihazı için hangi uzman raporu gerekir?", "expected": ["SPECIALIST", "DEVICE"]},
        {"q": "Çocuklarda büyüme hormonu tedavisi şartları?", "expected": ["AGE_LIMIT", "CONDITION"]},
        {"q": "Psikolojik tedavi seans ücreti nasıl ödenir?", "expected": ["RULE", "SPECIALIST"]},
    ]
    results = []
    hits = 0
    for item in QUESTIONS:
        try:
            nodes = _get_kg().find_nodes_by_label(item["q"][:40], k=5)
            if not nodes:
                nodes = _get_kg().find_nodes_semantic(item["q"], k=5)
            found_types = {n["type"] for n in nodes}
            hit = bool(found_types & set(item["expected"]))
            if hit:
                hits += 1
            results.append({
                "question": item["q"],
                "expected_types": item["expected"],
                "found_types": list(found_types),
                "found_nodes": [n["label"] for n in nodes[:3]],
                "hit": hit,
            })
        except Exception as e:
            results.append({"question": item["q"], "error": str(e), "hit": False})
    return {
        "total": len(QUESTIONS),
        "hits": hits,
        "hit_rate": round(hits / len(QUESTIONS), 3),
        "results": results,
    }


@app.get("/api/admin/audit-logs")
async def get_audit_logs(admin: dict = Depends(get_current_admin), limit: int = 50, offset: int = 0):
    try:
        safe_limit = max(1, min(int(limit), 500))
    except (TypeError, ValueError):
        safe_limit = 50
    try:
        safe_offset = max(0, int(offset))
    except (TypeError, ValueError):
        safe_offset = 0

    with db_session() as conn:
        cur = db_execute(conn, """
            SELECT a.log_id, a.action_type, a.entity_type, a.entity_id, a.details, a.created_at, u.username as user_name
            FROM audit_logs a
            LEFT JOIN users u ON a.user_id = u.id
            ORDER BY a.created_at DESC
            LIMIT %s OFFSET %s
        """, (safe_limit, safe_offset))
        logs = cur.fetchall()
        cur.close()

        cur2 = db_execute(conn, "SELECT COUNT(*) FROM audit_logs")
        total = cur2.fetchone()[0]
        cur2.close()

    parsed_logs = []
    for log in logs:
        d = dict(log)
        if d["details"]:
            try:
                d["details"] = json.loads(d["details"])
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        parsed_logs.append(d)

    return {"logs": parsed_logs, "total": total}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
