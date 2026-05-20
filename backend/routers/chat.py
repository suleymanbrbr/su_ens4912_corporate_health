# routers/chat.py — chat, upload, history, feedback, conversation endpoints.
#
# Routes:
#   POST   /api/chat/query                 (SSE)
#   POST   /api/chat/upload
#   POST   /api/feedback
#   POST   /api/feedback/report
#   GET    /api/history
#   POST   /api/history/save
#   GET    /api/conversations
#   GET    /api/conversations/search
#   PATCH  /api/conversations/{conversation_id}
#   DELETE /api/conversations/{conversation_id}
#   PUT    /api/conversations/{conversation_id}/favorite
#   GET    /api/config
#   GET    /api/announcements
#
# The chat endpoint pulls the live RAG engine via ``import api_server``
# (lazy, inside the request) so tests can patch ``api_server.engine``.
#
# TODO: a dedicated /api/chat/regenerate endpoint is intentionally NOT
# included. The current /api/chat/query streams + saves state during the
# response, so a clean regenerate would require either substantial code
# duplication or refactoring chat_query into a reusable inner helper. For
# now the frontend regenerates by replaying the previous user message
# through /api/chat/query.

import json
import logging
import os
import uuid
from typing import Optional

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel

from deps import (
    MAX_PDF_UPLOAD_BYTES,
    HISTORY_PAGE_LIMIT,
    db_execute,
    db_session,
    get_current_user,
)
from secrets_utils import decrypt_api_key


logger = logging.getLogger("api_server")

router = APIRouter(tags=["chat"])


# --- Pydantic models ---
class ChatQuery(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    role: Optional[str] = "PATIENT"  # Default to PATIENT if not provided
    k: Optional[int] = None          # RAG top-k; defaults handled in engine if None
    provider: Optional[str] = None   # 'gemini' | 'openrouter' | 'local' — picks which stored key to use


class SaveResponse(BaseModel):
    query: str
    response: str


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


# --- Internal helpers ---
def _summarize_history(history: list) -> str:
    """Summarize older conversation turns using Gemini.

    Called when chat history grows long (> 8 messages).
    Returns a compact Turkish summary string.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
        summarizer = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
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
    except Exception as e:  # noqa: BLE001
        logger.warning(f"History summarisation failed, falling back to truncation: {e}")
        # Fallback: naive truncation summary
        lines = [f"{m['role']}: {m['content'][:200]}" for m in history[-3:]]
        return " | ".join(lines)


def _get_chat_history(user_id: str, conversation_id: str):
    """Retrieve chat history for a session."""
    with db_session() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT query, response, file_metadata FROM query_history
            WHERE user_id = %s AND conversation_id = %s
            ORDER BY created_at ASC
            """,
            (user_id, conversation_id),
        )
        rows = cur.fetchall()
        cur.close()

    history = []
    for row in rows:
        history.append({"role": "user", "content": row["query"], "file_metadata": row["file_metadata"]})
        if row["response"]:
            history.append({"role": "assistant", "content": row["response"]})
    return history


def _save_query_history(
    user_id: str,
    conversation_id: str,
    query: str,
    response: str,
    file_metadata: dict = None,
    query_id: str = None,
):
    """Save a new query and response pair to the history."""
    qid = query_id or str(uuid.uuid4())
    with db_session() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO query_history (id, user_id, conversation_id, query, response, file_metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (qid, user_id, conversation_id, query, response, json.dumps(file_metadata) if file_metadata else None),
        )
        conn.commit()
        cur.close()


def _ensure_conversation_row(user_id: str, conversation_id: str, title_hint: str):
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
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP "
                "WHERE conversation_id = %s AND user_id = %s",
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


def _fetch_user_api_key(user_id: str, requested_provider: Optional[str]):
    """Return (provider, plaintext_key, base_url, key_row_id) for the user.

    Selection rule:
    - If ``requested_provider`` is given, look up that exact row.
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
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to update last_used_at for api_key {key_id}: {e}")


# --- Chat endpoints ---
@router.post("/api/chat/query")
async def chat_query(q: ChatQuery, current_user: dict = Depends(get_current_user)):
    """Main Agentic RAG chat endpoint (streaming)."""
    # Lazy import: tests patch ``api_server.engine`` after the module loads,
    # so we must dereference it at request-time (not import-time).
    import api_server

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
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Error fetching user docs: {e}")

    full_query = q.query
    if user_docs_context:
        full_query = f"{user_docs_context}\n\nKullanıcı Sorusu: {q.query}"

    # 2. Get history
    history = _get_chat_history(current_user["id"], effective_conv_id)
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
            for chunk in api_server.engine.query_agentic_rag_stream(
                full_query,
                chat_history=history,
                role=q.role or "PATIENT",
                k=rag_k,
                user_api_key=user_api_key,
                user_provider=provider,
                user_base_url=user_base_url,
            ):
                if "final_answer" in chunk:
                    _ensure_conversation_row(current_user["id"], effective_conv_id, q.query)
                    _save_query_history(
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


@router.post("/api/chat/upload")
async def upload_document(
    conversation_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
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
        # FastAPI ``UploadFile.read()`` has no built-in cap; an unbounded read
        # on a 500 MB PDF would happily OOM the Space.
        content = await file.read(MAX_PDF_UPLOAD_BYTES + 1)
        if len(content) > MAX_PDF_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Dosya boyutu çok büyük (maks {MAX_PDF_UPLOAD_BYTES // (1024 * 1024)} MB).",
            )

        try:
            pdf = PdfReader(io.BytesIO(content))
            text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"PDF parse failed for user={current_user.get('id')}: {e}")
            raise HTTPException(status_code=400, detail="PDF dosyası okunamadı veya bozuk.")

        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF'den metin çıkarılamadı.")

        doc_id = str(uuid.uuid4())
        with db_session() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO user_documents (id, user_id, conversation_id, filename, content)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (doc_id, current_user["id"], conversation_id, file.filename, text),
            )
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


@router.post("/api/feedback")
async def submit_feedback(data: FeedbackCreate, current_user: dict = Depends(get_current_user)):
    with db_session() as conn:
        db_execute(
            conn,
            "INSERT INTO user_feedback (feedback_id, message_id, rating, feedback_text, is_accurate) "
            "VALUES (%s, %s, %s, %s, %s)",
            (str(uuid.uuid4()), data.message_id, data.rating, data.feedback_text, 1 if data.is_accurate else 0),
        )
        conn.commit()
    return {"message": "Geri bildiriminiz kaydedildi. Teşekkürler!"}


@router.post("/api/feedback/report")
async def submit_feedback_report(data: FeedbackReportCreate, current_user: dict = Depends(get_current_user)):
    """Structured issue report for admin evaluation pipeline."""
    prefix = f"[{data.category}] "
    text = prefix + (data.feedback_text or "").strip()
    with db_session() as conn:
        db_execute(
            conn,
            "INSERT INTO user_feedback (feedback_id, message_id, rating, feedback_text, is_accurate) "
            "VALUES (%s, %s, %s, %s, %s)",
            (str(uuid.uuid4()), data.message_id, -1, text, 0),
        )
        conn.commit()
    return {"message": "Hata bildirimi kaydedildi. Teşekkürler!"}


@router.get("/api/history")
async def get_history(current_user: dict = Depends(get_current_user)):
    with db_session() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT id, conversation_id, query, response, file_metadata, created_at FROM query_history "
            "WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
            (current_user["id"], HISTORY_PAGE_LIMIT),
        )
        history = cur.fetchall()
        cur.close()

        cur2 = conn.cursor(cursor_factory=RealDictCursor)
        cur2.execute(
            "SELECT query, response, created_at FROM saved_responses WHERE user_id = %s ORDER BY created_at DESC",
            (current_user["id"],),
        )
        saved = cur2.fetchall()
        cur2.close()
    return {"history": [dict(h) for h in history], "saved": [dict(s) for s in saved]}


@router.post("/api/history/save")
async def save_response(data: SaveResponse, current_user: dict = Depends(get_current_user)):
    with db_session() as conn:
        db_execute(
            conn,
            "INSERT INTO saved_responses (id, user_id, query, response) VALUES (%s, %s, %s, %s)",
            (str(uuid.uuid4()), current_user["id"], data.query, data.response),
        )
        conn.commit()
    return {"message": "Yanıt kaydedildi."}


@router.get("/api/config")
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


# --- Conversations ---
@router.get("/api/conversations")
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


@router.get("/api/conversations/search")
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


@router.patch("/api/conversations/{conversation_id}")
async def patch_conversation(
    conversation_id: str,
    data: ConversationPatch,
    current_user: dict = Depends(get_current_user),
):
    with db_session() as conn:
        cur = db_execute(
            conn,
            "UPDATE conversations SET title = %s, updated_at = CURRENT_TIMESTAMP "
            "WHERE conversation_id = %s AND user_id = %s",
            (data.title[:200], conversation_id, current_user["id"]),
        )
        if cur.rowcount == 0:
            cur.close()
            raise HTTPException(status_code=404, detail="Konuşma bulunamadı.")
        conn.commit()
        cur.close()
    return {"message": "Başlık güncellendi.", "conversation_id": conversation_id}


@router.delete("/api/conversations/{conversation_id}")
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
        db_execute(
            conn,
            "DELETE FROM query_history WHERE conversation_id = %s AND user_id = %s",
            (conversation_id, current_user["id"]),
        )
        db_execute(
            conn,
            "DELETE FROM user_documents WHERE conversation_id = %s AND user_id = %s",
            (conversation_id, current_user["id"]),
        )
        db_execute(
            conn,
            "DELETE FROM conversations WHERE conversation_id = %s AND user_id = %s",
            (conversation_id, current_user["id"]),
        )
        conn.commit()
    return {"message": "Konuşma silindi."}


@router.put("/api/conversations/{conversation_id}/favorite")
async def favorite_conversation(
    conversation_id: str,
    body: FavoriteBody = Body(...),
    current_user: dict = Depends(get_current_user),
):
    with db_session() as conn:
        cur = db_execute(
            conn,
            "UPDATE conversations SET favorited = %s, updated_at = CURRENT_TIMESTAMP "
            "WHERE conversation_id = %s AND user_id = %s",
            (1 if body.favorited else 0, conversation_id, current_user["id"]),
        )
        if cur.rowcount == 0:
            cur.close()
            raise HTTPException(status_code=404, detail="Konuşma bulunamadı.")
        conn.commit()
        cur.close()
    return {"message": "Güncellendi.", "favorited": body.favorited}


@router.get("/api/announcements")
async def get_active_announcement(current_user: dict = Depends(get_current_user)):
    with db_session() as conn:
        cur = db_execute(
            conn,
            "SELECT id, message, created_at FROM announcements WHERE active = 1 "
            "ORDER BY created_at DESC LIMIT 1",
        )
        row = cur.fetchone()
        cur.close()
    return dict(row) if row else {}
