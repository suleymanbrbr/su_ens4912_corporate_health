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
import psycopg2.extras

from fastapi import FastAPI, Depends, HTTPException, status, Body
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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            hashed_password TEXT,
            role TEXT DEFAULT 'user',
            is_approved SMALLINT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
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
    cur = db_execute(conn, "SELECT * FROM users WHERE username = %s", (form_data.username,))
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

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str
    k: int = 5
    conversation_id: Optional[str] = None

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

class HistoryResponseUpdate(BaseModel):
    response: str

class FeedbackCreate(BaseModel):
    message_id: str
    rating: int  # 1-5 or -1, 1
    feedback_text: str = ""
    is_accurate: bool = True

# --- Chat ---
@app.post("/api/chat")
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    if not engine or not engine.conn:
        raise HTTPException(status_code=500, detail="Database not loaded. Please run Re-index from Admin Panel.")

    query_id = str(uuid.uuid4())
    conversation_id = request.conversation_id or str(uuid.uuid4())
    conn = get_db_conn()
    
    # Fetch chat history
    cur = db_execute(conn,
        "SELECT query, response FROM query_history WHERE conversation_id = %s ORDER BY created_at ASC",
        (conversation_id,)
    )
    history_rows = cur.fetchall()
    cur.close()
    
    chat_history = []
    for row in history_rows:
        chat_history.append({"role": "user", "content": row["query"]})
        if row["response"]:
            chat_history.append({"role": "assistant", "content": row["response"]})

    try:
        db_execute(conn,
            "INSERT INTO query_history (id, user_id, conversation_id, query) VALUES (%s, %s, %s, %s)",
            (query_id, current_user["id"], conversation_id, request.message)
        )
        log_audit(conn, "query_executed", user_id=current_user["id"], entity_type="query", entity_id=query_id)
        conn.commit()
    except Exception as e:
        print(f"[WARN] Failed to log query history: {e}")
    finally:
        conn.close()

    async def event_generator():
        start_time = asyncio.get_event_loop().time()
        yield f"data: {json.dumps({'query_id': query_id, 'conversation_id': conversation_id}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.01)
        
        full_response_text = ""
        for step in engine.query_agentic_rag_stream(request.message, chat_history=chat_history, k=request.k):
            if "final_answer" in step:
                full_response_text = step["final_answer"]
            yield f"data: {json.dumps(step, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)
            
        duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        save_conn = get_db_conn()
        try:
            if full_response_text:
                db_execute(save_conn,
                    "UPDATE query_history SET response = %s WHERE id = %s", (full_response_text, query_id)
                )
            db_execute(save_conn,
                "INSERT INTO agent_runs (run_id, trigger_message_id, agent_name, input_data, output_data, status, duration_ms, ended_at) VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)",
                (str(uuid.uuid4()), query_id, "SUT_RAG_Engine", request.message, full_response_text, "success" if full_response_text else "failure", duration_ms)
            )
            save_conn.commit()
        except Exception as e:
            print(f"[WARN] Failed to update query response or telemetry: {e}")
        finally:
            save_conn.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream")

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
    cur = db_execute(conn,
        "SELECT id, conversation_id, query, response, created_at FROM query_history WHERE user_id = %s ORDER BY created_at DESC LIMIT 200",
        (current_user["id"],)
    )
    history = cur.fetchall()
    cur.close()
    cur2 = db_execute(conn,
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
async def rebuild_index(admin: dict = Depends(get_current_admin)):
    global engine
    try:
        storage = SUT_Storage_Manager(engine.embeddings_model)
        success = storage.populate_database()
        if not success:
            raise HTTPException(status_code=500, detail="İndeksleme işlemi sırasında hata oluştu.")
        
        # Reload global engine connection
        new_engine = SUT_RAG_Engine()
        if new_engine.load_database():
            engine = new_engine
            dbconn = get_db_conn()
            log_audit(dbconn, "index_rebuilt", user_id=admin["id"])
            dbconn.commit()
            dbconn.close()
            return {"message": "Sistem başarıyla yeniden indekslendi ve yüklendi."}
        else:
            return {"message": "İndeksleme tamamlandı ancak motor yüklenemedi.", "status": "partial"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sistem hatası: {str(e)}")

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

@app.get("/api/kg")
async def get_knowledge_graph(current_user: dict = Depends(get_current_user)):
    kg_path = "sut_knowledge_graph.json"
    if not os.path.exists(kg_path):
        raise HTTPException(status_code=404, detail="Knowledge Graph verisi bulunamadı.")
    with open(kg_path, "r", encoding="utf-8") as f:
        kg_data = json.load(f)
    return kg_data

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
