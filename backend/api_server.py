# api_server.py
# Description: Modern FastAPI Server for SUT Assistant. PostgreSQL Edition.

import os
import uuid
import json
import asyncio
import re
from typing import List, Optional
from contextlib import asynccontextmanager
from collections import Counter

import psycopg2
from psycopg2.extras import RealDictCursor

from fastapi import FastAPI, Depends, HTTPException, status, Body, BackgroundTasks, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

from auth_utils import get_password_hash, verify_password, create_access_token, decode_access_token
from sut_rag_core import SUT_RAG_Engine
from rag_storage import SUT_Storage_Manager

load_dotenv()

# --- Global Engine Instance ---
engine = None

# --- DB Helper ---
def get_db_conn():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"), cursor_factory=psycopg2.extras.DictCursor)
    return conn

def db_execute(conn, query, params=None):
    """Helper: run a query and return cursor."""
    cur = conn.cursor()
    cur.execute(query, params)
    return cur

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
    conn.commit()
    cur.close()
    conn.close()

def log_audit(conn, action_type, user_id=None, entity_type=None, entity_id=None, details=None):
    try:
        details_json = json.dumps(details, ensure_ascii=False) if details else None
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO audit_logs (log_id, user_id, action_type, entity_type, entity_id, details) VALUES (%s, %s, %s, %s, %s, %s)",
            (str(uuid.uuid4()), user_id, action_type, entity_type, entity_id, details_json)
        )
        cur.close()
    except Exception as e:
        print(f"[WARN] Failed to log audit: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_system_tables()
    global engine
    engine = SUT_RAG_Engine()
    if not engine.load_database():
        print("[WARN] SUT Database not loaded. Please populate it via Admin panel.")
    yield
    if engine and engine.conn:
        engine.conn.close()

app = FastAPI(title="SUT Corporate Health API", lifespan=lifespan)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Auth Models ---
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: Optional[str] = "user"

class Token(BaseModel):
    access_token: str
    token_type: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str
    is_approved: int

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    username: str = payload.get("sub")
    conn = get_db_conn()
    cur = db_execute(conn, "SELECT * FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return dict(user)

async def get_current_admin(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return current_user

# --- Endpoints ---

@app.post("/api/auth/register", response_model=UserResponse)
async def register(user: UserRegister):
    conn = get_db_conn()
    try:
        cur = db_execute(conn, "SELECT id FROM users WHERE username = %s OR email = %s", (user.username, user.email))
        existing = cur.fetchone()
        cur.close()
        if existing:
            raise HTTPException(status_code=400, detail="User already exists")
        
        user_id = str(uuid.uuid4())
        hashed_pwd = get_password_hash(user.password)
        role = user.role if user.role in ["user", "admin"] else "user"
        
        cur = db_execute(conn, "SELECT COUNT(*) FROM users")
        user_count = cur.fetchone()[0]
        cur.close()
        is_approved = 1 if user_count == 0 else 0
        
        db_execute(conn,
            "INSERT INTO users (id, username, email, hashed_password, role, is_approved) VALUES (%s, %s, %s, %s, %s, %s)",
            (user_id, user.username, user.email, hashed_pwd, role, is_approved)
        )
        log_audit(conn, "register", user_id=user_id, entity_type="user", entity_id=user_id, details={"username": user.username, "roles": role})
        conn.commit()
        return {"id": user_id, "username": user.username, "email": user.email, "role": role, "is_approved": is_approved}
    finally:
        conn.close()

@app.post("/api/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = get_db_conn()
    cur = db_execute(conn, "SELECT * FROM users WHERE username = %s OR email = %s", (form_data.username, form_data.username))
    user = cur.fetchone()
    cur.close()
    
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        conn.close()
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    
    if user["is_approved"] == 0:
        conn.close()
        raise HTTPException(status_code=403, detail="Hesabınız henüz onaylanmamıştır.")
    
    log_audit(conn, "login", user_id=user["id"])
    conn.commit()
    conn.close()
    
    access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/auth/me", response_model=UserResponse)
async def me(current_user: dict = Depends(get_current_user)):
    return current_user

@app.get("/api/admin/users", response_model=List[UserResponse])
async def list_users(admin: dict = Depends(get_current_admin)):
    conn = get_db_conn()
    cur = db_execute(conn, "SELECT id, username, email, role, is_approved FROM users")
    users = cur.fetchall()
    cur.close()
    conn.close()
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
            model="gemini-2.0-flash",
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
        # Fallback: naive truncation summary
        lines = [f"{m['role']}: {m['content'][:200]}" for m in history[-3:]]
        return " | ".join(lines)


def get_chat_history(user_id: str, conversation_id: str):
    """Retrieve chat history for a session."""
    conn = get_db_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT query, response, file_metadata FROM query_history 
        WHERE user_id = %s AND conversation_id = %s 
        ORDER BY created_at ASC
    """, (user_id, conversation_id))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    history = []
    for row in rows:
        history.append({"role": "user", "content": row["query"], "file_metadata": row["file_metadata"]})
        if row["response"]:
            history.append({"role": "assistant", "content": row["response"]})
    return history

def save_query_history(user_id: str, conversation_id: str, query: str, response: str, file_metadata: dict = None):
    """Save a new query and response pair to the history."""
    query_id = str(uuid.uuid4())
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO query_history (id, user_id, conversation_id, query, response, file_metadata)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (query_id, user_id, conversation_id, query, response, json.dumps(file_metadata) if file_metadata else None))
    conn.commit()
    cur.close()
    conn.close()


# --- Pydantic Models ---
class ChatQuery(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    role: Optional[str] = "PATIENT"  # Default to PATIENT if not provided

class PasswordChange(BaseModel):
    old_password: str
    new_password: str

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

# --- Chat ---
@app.post("/api/chat/query")
async def chat_query(q: ChatQuery, current_user: dict = Depends(get_current_user)):
    """Main Agentic RAG chat endpoint (streaming)."""
    
    # 1. Fetch user documents for this conversation to provide context
    user_docs_context = ""
    active_file_metadata = None
    try:
        conn = get_db_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT filename, content FROM user_documents WHERE user_id = %s AND conversation_id = %s", (current_user["id"], q.conversation_id))
        rows = cur.fetchall()
        if rows:
            user_docs_context = "\n\n--- KULLANICI DÖKÜMANLARI ---\n" + "\n".join([r["content"] for r in rows])
            active_file_metadata = {"filename": rows[0]["filename"], "type": "pdf"}
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error fetching user docs: {e}")

    full_query = q.query
    if user_docs_context:
        full_query = f"{user_docs_context}\n\nKullanıcı Sorusu: {q.query}"

    # 2. Get history
    history = []
    if q.conversation_id:
        history = get_chat_history(current_user["id"], q.conversation_id)
        # Summarize if too long
        if len(history) > 10:
            history = await _summarize_history(history)

    async def generate():
        # Pass the selected role to the engine for personalization
        for chunk in engine.query_agentic_rag_stream(full_query, chat_history=history, role=q.role):
            if "final_answer" in chunk:
                # Save to DB when finished
                save_query_history(current_user["id"], q.conversation_id, q.query, chunk["final_answer"], file_metadata=active_file_metadata)
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/api/chat/upload")
async def upload_document(
    conversation_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload a medical report PDF and extract its text."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Sadece PDF dosyaları yüklenebilir.")
    
    try:
        import io
        from pypdf import PdfReader
        
        content = await file.read()
        pdf = PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF'den metin çıkarılamadı.")
        
        doc_id = str(uuid.uuid4())
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_documents (id, user_id, conversation_id, filename, content)
            VALUES (%s, %s, %s, %s, %s)
        """, (doc_id, current_user["id"], conversation_id, file.filename, text))
        conn.commit()
        cur.close()
        conn.close()
        
        return {
            "message": "Belge başarıyla yüklendi ve işlendi.",
            "doc_id": doc_id,
            "filename": file.filename,
            "char_count": len(text)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {str(e)}")

@app.post("/api/feedback")
async def submit_feedback(data: FeedbackCreate, current_user: dict = Depends(get_current_user)):
    conn = get_db_conn()
    db_execute(conn,
        "INSERT INTO user_feedback (feedback_id, message_id, rating, feedback_text, is_accurate) VALUES (%s, %s, %s, %s, %s)",
        (str(uuid.uuid4()), data.message_id, data.rating, data.feedback_text, 1 if data.is_accurate else 0)
    )
    conn.commit()
    conn.close()
    return {"message": "Geri bildiriminiz kaydedildi. Teşekkürler!"}

@app.put("/api/auth/password")
async def change_password(data: PasswordChange, current_user: dict = Depends(get_current_user)):
    conn = get_db_conn()
    cur = db_execute(conn, "SELECT hashed_password FROM users WHERE id = %s", (current_user["id"],))
    user = cur.fetchone()
    cur.close()
    if not user or not verify_password(data.old_password, user["hashed_password"]):
        conn.close()
        raise HTTPException(status_code=400, detail="Mevcut şifre yanlış")
    
    new_hashed = get_password_hash(data.new_password)
    db_execute(conn, "UPDATE users SET hashed_password = %s WHERE id = %s", (new_hashed, current_user["id"]))
    log_audit(conn, "password_change", user_id=current_user["id"])
    conn.commit()
    conn.close()
    return {"message": "Şifre başarıyla güncellendi."}

@app.get("/api/history")
async def get_history(current_user: dict = Depends(get_current_user)):
    conn = get_db_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        "SELECT id, conversation_id, query, response, file_metadata, created_at FROM query_history WHERE user_id = %s ORDER BY created_at DESC LIMIT 200",
        (current_user["id"],)
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
    conn.close()
    return {"history": [dict(h) for h in history], "saved": [dict(s) for s in saved]}

@app.post("/api/history/save")
async def save_response(data: SaveResponse, current_user: dict = Depends(get_current_user)):
    conn = get_db_conn()
    db_execute(conn,
        "INSERT INTO saved_responses (id, user_id, query, response) VALUES (%s, %s, %s, %s)",
        (str(uuid.uuid4()), current_user["id"], data.query, data.response)
    )
    conn.commit()
    conn.close()
    return {"message": "Yanıt kaydedildi."}

# --- Admin Endpoints ---

@app.get("/api/admin/system")
async def get_system_metrics(admin: dict = Depends(get_current_admin)):
    conn = get_db_conn()
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
    except:
        pass
    cur = db_execute(conn, "SELECT message FROM announcements WHERE active = 1 ORDER BY created_at DESC LIMIT 1")
    active_announcement = cur.fetchone()
    cur.close()
    cur = db_execute(conn, "SELECT COUNT(*) FROM users WHERE is_approved = 0")
    pending_count = cur.fetchone()[0]
    cur.close()
    conn.close()

    return {
        "users_count": users_count,
        "queries_count": queries_count,
        "chunks_count": chunks_count,
        "pending_count": pending_count,
        "active_announcement": dict(active_announcement) if active_announcement else None
    }

@app.get("/api/admin/activity")
async def get_admin_activity(admin: dict = Depends(get_current_admin)):
    conn = get_db_conn()
    cur = db_execute(conn, """
        SELECT qh.query, qh.created_at, u.username, u.role
        FROM query_history qh
        JOIN users u ON qh.user_id = u.id
        ORDER BY qh.created_at DESC
        LIMIT 20
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/admin/analytics")
async def get_admin_analytics(admin: dict = Depends(get_current_admin)):
    conn = get_db_conn()

    cur = db_execute(conn, "SELECT query FROM query_history")
    all_queries = cur.fetchall()
    cur.close()
    stopwords = {
        "ve", "bir", "ile", "bu", "için", "da", "de", "mi", "ne", "ben", "sen",
        "biz", "siz", "o", "bu", "şu", "ki", "gibi", "ama", "veya", "ya", "daha",
        "olan", "nasıl", "nedir", "hakkında", "bilgi", "ver", "söyle",
        "the", "is", "a", "of", "in", "to", "what", "how", "about"
    }
    word_counter = Counter()
    for row in all_queries:
        words = re.findall(r'\b[a-zA-ZğüşıöçĞÜŞİÖÇ]{4,}\b', row["query"].lower())
        for w in words:
            if w not in stopwords:
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

    conn.close()
    return {
        "top_keywords": top_keywords,
        "daily_volume": daily_volume,
        "daily_engagement_rate": daily_engagement_rate,
        "monthly_engagement_rate": monthly_engagement_rate,
        "daily_active_users": daily_active,
        "monthly_active_users": monthly_active,
        "total_users": total_users
    }

@app.get("/api/policies")
async def search_policies(q: str = "", section: str = "", limit: int = 20, offset: int = 0, admin: dict = Depends(get_current_admin)):
    conn = get_db_conn()
    try:
        if q:
            # PostgreSQL Full-Text Search with tsquery
            cur = db_execute(conn, """
                SELECT chunk_id, text_content, metadata_json
                FROM chunks
                WHERE to_tsvector('turkish', COALESCE(header_text,'') || ' ' || text_content) @@ plainto_tsquery('turkish', %s)
                LIMIT %s OFFSET %s
            """, (q, limit, offset))
            chunks = cur.fetchall()
            cur.close()
            if not chunks:
                # Fallback: ILIKE search
                cur = db_execute(conn,
                    "SELECT chunk_id, text_content, metadata_json FROM chunks WHERE text_content ILIKE %s LIMIT %s OFFSET %s",
                    (f"%{q}%", limit, offset)
                )
                chunks = cur.fetchall()
                cur.close()
        else:
            cur = db_execute(conn,
                "SELECT chunk_id, text_content, metadata_json FROM chunks LIMIT %s OFFSET %s",
                (limit, offset)
            )
            chunks = cur.fetchall()
            cur.close()

        results = []
        for c in chunks:
            meta = c["metadata_json"] if isinstance(c["metadata_json"], dict) else (json.loads(c["metadata_json"]) if c["metadata_json"] else {})
            title = " > ".join([v for k, v in meta.items() if k.startswith("Header")])
            if section and section.upper() not in title.upper():
                continue
            results.append({
                "id": c["chunk_id"],
                "title": title or "Başlıksız Bölüm",
                "excerpt": c["text_content"][:300],
                "full_text": c["text_content"],
                "metadata": meta
            })
        cur = db_execute(conn, "SELECT COUNT(*) FROM chunks")
        total = cur.fetchone()[0]
        cur.close()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Arama hatası: {str(e)}")
    conn.close()
    return {"results": results, "total": total, "offset": offset, "limit": limit}

@app.put("/api/admin/users/{user_id}/role")
async def update_user_role(user_id: str, data: RoleUpdate, admin: dict = Depends(get_current_admin)):
    if data.role not in ["user", "admin"]:
        raise HTTPException(status_code=400, detail="Geçersiz rol.")
    conn = get_db_conn()
    db_execute(conn, "UPDATE users SET role = %s WHERE id = %s", (data.role, user_id))
    log_audit(conn, "role_updated", user_id=admin["id"], entity_type="user", entity_id=user_id, details={"new_role": data.role})
    conn.commit()
    conn.close()
    return {"message": "Rol güncellendi."}

@app.delete("/api/admin/users/{user_id}")
async def delete_user(user_id: str, admin: dict = Depends(get_current_admin)):
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="Kendi hesabınızı silemezsiniz.")
    conn = get_db_conn()
    db_execute(conn, "DELETE FROM query_history WHERE user_id = %s", (user_id,))
    db_execute(conn, "DELETE FROM saved_responses WHERE user_id = %s", (user_id,))
    db_execute(conn, "DELETE FROM users WHERE id = %s", (user_id,))
    log_audit(conn, "user_deleted", user_id=admin["id"], entity_type="user", entity_id=user_id)
    conn.commit()
    conn.close()
    return {"message": "Kullanıcı silindi."}

@app.put("/api/admin/users/{user_id}/approve")
async def approve_user(user_id: str, admin: dict = Depends(get_current_admin)):
    conn = get_db_conn()
    db_execute(conn, "UPDATE users SET is_approved = 1 WHERE id = %s", (user_id,))
    log_audit(conn, "user_approved", user_id=admin["id"], entity_type="user", entity_id=user_id)
    conn.commit()
    conn.close()
    return {"message": "Kullanıcı onaylandı."}

@app.post("/api/admin/rebuild-index")
async def rebuild_index(background_tasks: BackgroundTasks, admin: dict = Depends(get_current_admin)):
    global engine
    
    def run_indexing():
        global engine
        try:
            print("[BACKGROUND] Starting indexing task...")
            storage = SUT_Storage_Manager(engine.embeddings_model)
            success = storage.populate_database()
            if success:
                # Reload global engine connection
                new_engine = SUT_RAG_Engine()
                if new_engine.load_database():
                    engine = new_engine
                    print("[BACKGROUND] Indexing and engine reload complete.")
                else:
                    print("[BACKGROUND] Indexing complete but engine reload failed.")
            else:
                print("[BACKGROUND] Indexing failed.")
        except Exception as e:
            print(f"[BACKGROUND] Indexing error: {str(e)}")

    background_tasks.add_task(run_indexing)
    
    dbconn = get_db_conn()
    log_audit(dbconn, "index_rebuild_started", user_id=admin["id"])
    dbconn.commit()
    dbconn.close()
    
    return {"message": "İndeksleme işlemi arka planda başlatıldı. İlerlemeyi sistem günlüklerinden takip edebilirsiniz."}

# --- Announcements ---

@app.post("/api/admin/announcements")
async def create_announcement(data: AnnouncementCreate, admin: dict = Depends(get_current_admin)):
    conn = get_db_conn()
    db_execute(conn, "UPDATE announcements SET active = 0")
    ann_id = str(uuid.uuid4())
    db_execute(conn,
        "INSERT INTO announcements (id, message, created_by, active) VALUES (%s, %s, %s, 1)",
        (ann_id, data.message, admin["id"])
    )
    log_audit(conn, "announcement_created", user_id=admin["id"], entity_type="announcement", entity_id=ann_id)
    conn.commit()
    conn.close()
    return {"message": "Duyuru yayınlandı."}

@app.get("/api/announcements")
async def get_active_announcement(current_user: dict = Depends(get_current_user)):
    conn = get_db_conn()
    cur = db_execute(conn,
        "SELECT id, message, created_at FROM announcements WHERE active = 1 ORDER BY created_at DESC LIMIT 1"
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return dict(row) if row else {}

@app.delete("/api/admin/announcements/{ann_id}")
async def deactivate_announcement(ann_id: str, admin: dict = Depends(get_current_admin)):
    conn = get_db_conn()
    db_execute(conn, "UPDATE announcements SET active = 0 WHERE id = %s", (ann_id,))
    log_audit(conn, "announcement_deactivated", user_id=admin["id"], entity_type="announcement", entity_id=ann_id)
    conn.commit()
    conn.close()
    return {"message": "Duyuru kaldırıldı."}

# --- Knowledge Graph API (Postgres-backed) ---

from kg_storage import KG_Storage_Manager
_kg = KG_Storage_Manager()

@app.get("/api/kg/stats")
async def get_kg_stats(current_user: dict = Depends(get_current_user)):
    """Return node/edge counts by type/relation."""
    try:
        return _kg.get_stats()
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
        type_filter = type.upper() if type else None
        if q:
            exact = _kg.find_nodes_by_label(q, k=limit, type_filter=type_filter)
            semantic = _kg.find_nodes_semantic(q, k=limit, type_filter=type_filter)
            seen = {r["node_id"] for r in exact}
            merged = exact + [r for r in semantic if r["node_id"] not in seen]
            return {"nodes": merged[:limit]}
        else:
            conn = get_db_conn()
            q_str = f"SELECT node_id, label, type, text_content, atc_code, icd_code FROM kg_nodes"
            if type_filter:
                q_str += f" WHERE type = '{type_filter}'"
            q_str += f" LIMIT {limit}"
            cur = db_execute(conn, q_str)
            rows = [dict(r) for r in cur.fetchall()]
            cur.close()
            conn.close()
            return {"nodes": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/kg/node/{node_id}")
async def get_kg_node(node_id: str, current_user: dict = Depends(get_current_user)):
    """Get a single node with all its neighbors."""
    node = _kg.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    neighbors = _kg.get_neighbors(node_id, limit=20)
    return {"node": node, "neighbors": neighbors}

@app.get("/api/kg/subgraph/{rule_id}")
async def get_kg_subgraph(rule_id: str, current_user: dict = Depends(get_current_user)):
    """Get the subgraph for a RULE node (all related nodes & edges)."""
    return _kg.get_rule_subgraph(rule_id)

@app.get("/api/kg/path")
async def find_kg_path(
    from_id: str,
    to_id: str,
    max_hops: int = 3,
    current_user: dict = Depends(get_current_user)
):
    """Find shortest path between two nodes."""
    path = _kg.find_path(from_id, to_id, max_hops=max_hops)
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
            print("[KG_REBUILD] Complete.")
        except Exception as e:
            print(f"[KG_REBUILD] Error: {e}")

    background_tasks.add_task(run_kg_build)
    dbconn = get_db_conn()
    log_audit(dbconn, "kg_rebuild_started", user_id=admin["id"])
    dbconn.commit()
    dbconn.close()
    return {"message": "KG yeniden oluşturma işlemi arka planda başlatıldı."}

@app.get("/api/admin/kg/stats")
async def get_admin_kg_stats(admin: dict = Depends(get_current_admin)):
    try:
        return _kg.get_stats()
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
            nodes = _kg.find_nodes_by_label(item["q"][:40], k=5)
            if not nodes:
                nodes = _kg.find_nodes_semantic(item["q"], k=5)
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
    conn = get_db_conn()
    cur = db_execute(conn, """
        SELECT a.log_id, a.action_type, a.entity_type, a.entity_id, a.details, a.created_at, u.username as user_name
        FROM audit_logs a
        LEFT JOIN users u ON a.user_id = u.id
        ORDER BY a.created_at DESC
        LIMIT %s OFFSET %s
    """, (limit, offset))
    logs = cur.fetchall()
    cur.close()
    
    cur2 = db_execute(conn, "SELECT COUNT(*) FROM audit_logs")
    total = cur2.fetchone()[0]
    cur2.close()
    conn.close()
    
    parsed_logs = []
    for log in logs:
        d = dict(log)
        if d["details"]:
            try:
                d["details"] = json.loads(d["details"])
            except:
                pass
        parsed_logs.append(d)
        
    return {"logs": parsed_logs, "total": total}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
