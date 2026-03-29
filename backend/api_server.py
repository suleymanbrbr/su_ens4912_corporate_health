# api_server.py
# Description: Modern FastAPI Server for SUT Assistant.

import os
import uuid
import json
import sqlite3
import asyncio
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

from auth_utils import get_password_hash, verify_password, create_access_token, decode_access_token
from sut_rag_core import SUT_RAG_Engine
from rag_storage import DB_PATH

load_dotenv()

# --- Global Engine Instance ---
engine = None

# --- DB Init Helper ---
def init_system_tables():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            hashed_password TEXT,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            query TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS saved_responses (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            query TEXT,
            response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure system tables exist
    init_system_tables()
    # Startup: Load RAG Engine
    global engine
    engine = SUT_RAG_Engine()
    if not engine.load_database():
        print("[WARN] SUT Database not found. Please populate it first.")
    yield
    # Shutdown: Clean up
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

# --- DB Helper ---
def get_db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

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

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    username: str = payload.get("sub")
    conn = get_db_conn()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
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
        existing = conn.execute("SELECT id FROM users WHERE username = ? OR email = ?", (user.username, user.email)).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="User already exists")
        
        user_id = str(uuid.uuid4())
        hashed_pwd = get_password_hash(user.password)
        role = user.role if user.role in ["user", "admin"] else "user"
        conn.execute(
            "INSERT INTO users (id, username, email, hashed_password, role) VALUES (?, ?, ?, ?, ?)",
            (user_id, user.username, user.email, hashed_pwd, role)
        )
        conn.commit()
        return {"id": user_id, "username": user.username, "email": user.email, "role": role}
    finally:
        conn.close()

@app.post("/api/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = get_db_conn()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (form_data.username,)).fetchone()
    conn.close()
    
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    
    access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/auth/me", response_model=UserResponse)
async def me(current_user: dict = Depends(get_current_user)):
    return current_user

@app.get("/api/admin/users", response_model=List[UserResponse])
async def list_users(admin: dict = Depends(get_current_admin)):
    conn = get_db_conn()
    users = conn.execute("SELECT id, username, email, role FROM users").fetchall()
    conn.close()
    return [dict(u) for u in users]

class ChatRequest(BaseModel):
    message: str
    k: int = 5

class PasswordChange(BaseModel):
    old_password: str
    new_password: str

class SaveResponse(BaseModel):
    query: str
    response: str

class RoleUpdate(BaseModel):
    role: str

@app.post("/api/chat")
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    if not engine.faiss_index:
        raise HTTPException(status_code=500, detail="Database not loaded")

    conn = get_db_conn()
    try:
        conn.execute("INSERT INTO query_history (id, user_id, query) VALUES (?, ?, ?)", 
                     (str(uuid.uuid4()), current_user["id"], request.message))
        conn.commit()
    except Exception as e:
        print(f"[WARN] Failed to log query history: {e}")
    finally:
        conn.close()

    async def event_generator():
        for step in engine.query_agentic_rag_stream(request.message, k=request.k):
            yield f"data: {json.dumps(step, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
@app.put("/api/auth/password")
async def change_password(data: PasswordChange, current_user: dict = Depends(get_current_user)):
    conn = get_db_conn()
    user = conn.execute("SELECT hashed_password FROM users WHERE id = ?", (current_user["id"],)).fetchone()
    if not user or not verify_password(data.old_password, user["hashed_password"]):
        conn.close()
        raise HTTPException(status_code=400, detail="Mevcut şifre yanlış")
    
    new_hashed = get_password_hash(data.new_password)
    conn.execute("UPDATE users SET hashed_password = ? WHERE id = ?", (new_hashed, current_user["id"]))
    conn.commit()
    conn.close()
    return {"message": "Şifre başarıyla güncellendi."}

@app.get("/api/history")
async def get_history(current_user: dict = Depends(get_current_user)):
    conn = get_db_conn()
    history = conn.execute("SELECT query, created_at FROM query_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 50", (current_user["id"],)).fetchall()
    saved = conn.execute("SELECT query, response, created_at FROM saved_responses WHERE user_id = ? ORDER BY created_at DESC", (current_user["id"],)).fetchall()
    conn.close()
    return {"history": [dict(h) for h in history], "saved": [dict(s) for s in saved]}

@app.post("/api/history/save")
async def save_response(data: SaveResponse, current_user: dict = Depends(get_current_user)):
    conn = get_db_conn()
    conn.execute("INSERT INTO saved_responses (id, user_id, query, response) VALUES (?, ?, ?, ?)", 
                 (str(uuid.uuid4()), current_user["id"], data.query, data.response))
    conn.commit()
    conn.close()
    return {"message": "Yanıt kaydedildi."}

@app.get("/api/admin/system")
async def get_system_metrics(admin: dict = Depends(get_current_admin)):
    conn = get_db_conn()
    users_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    queries_count = conn.execute("SELECT COUNT(*) FROM query_history").fetchone()[0]
    chunks_count = 0
    try:
        chunks_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    except:
        pass
    conn.close()
    return {
        "users_count": users_count,
        "queries_count": queries_count,
        "chunks_count": chunks_count
    }

@app.put("/api/admin/users/{user_id}/role")
async def update_user_role(user_id: str, data: RoleUpdate, admin: dict = Depends(get_current_admin)):
    if data.role not in ["user", "admin"]:
        raise HTTPException(status_code=400, detail="Geçersiz rol.")
    conn = get_db_conn()
    conn.execute("UPDATE users SET role = ? WHERE id = ?", (data.role, user_id))
    conn.commit()
    conn.close()
    return {"message": "Rol güncellendi."}

@app.delete("/api/admin/users/{user_id}")
async def delete_user(user_id: str, admin: dict = Depends(get_current_admin)):
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="Kendi hesabınızı silemezsiniz.")
    conn = get_db_conn()
    conn.execute("DELETE FROM query_history WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM saved_responses WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return {"message": "Kullanıcı silindi."}


@app.get("/api/kg")
async def get_knowledge_graph(current_user: dict = Depends(get_current_user)):
    kg_path = "sut_knowledge_graph.json"
    if not os.path.exists(kg_path):
        raise HTTPException(status_code=404, detail="Knowledge Graph verisi bulunamadı.")
    with open(kg_path, "r", encoding="utf-8") as f:
        kg_data = json.load(f)
    return kg_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
