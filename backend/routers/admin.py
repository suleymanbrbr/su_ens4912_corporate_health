# routers/admin.py — administrator endpoints.
#
# Routes:
#   GET    /api/admin/users
#   PUT    /api/admin/users/{user_id}/role
#   DELETE /api/admin/users/{user_id}
#   PUT    /api/admin/users/{user_id}/approve
#   GET    /api/admin/system
#   GET    /api/admin/activity
#   GET    /api/admin/analytics
#   GET    /api/admin/audit-logs
#   POST   /api/admin/rebuild-index
#   POST   /api/admin/announcements
#   DELETE /api/admin/announcements/{ann_id}
#   GET    /api/admin/eval/results

import json
import logging
import os
import re
import uuid
from collections import Counter
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from deps import (
    VALID_USER_ROLES,
    db_execute,
    db_session,
    get_current_admin,
    log_audit,
)
from routers.auth import UserResponse


logger = logging.getLogger("api_server")

router = APIRouter(prefix="/api/admin", tags=["admin"])


# --- Pydantic models ---
class RoleUpdate(BaseModel):
    role: str


class AnnouncementCreate(BaseModel):
    message: str


# --- Stopwords used by /analytics for keyword extraction ---
_ANALYTICS_STOPWORDS = {
    "ve", "bir", "ile", "bu", "için", "da", "de", "mi", "ne", "ben", "sen",
    "biz", "siz", "o", "şu", "ki", "gibi", "ama", "veya", "ya", "daha",
    "olan", "nasıl", "nedir", "hakkında", "bilgi", "ver", "söyle",
    "the", "is", "a", "of", "in", "to", "what", "how", "about",
}


# --- User management ---
@router.get("/users", response_model=List[UserResponse])
async def list_users(admin: dict = Depends(get_current_admin)):
    with db_session() as conn:
        cur = db_execute(conn, "SELECT id, username, email, role, is_approved FROM users")
        users = cur.fetchall()
        cur.close()
    return [dict(u) for u in users]


@router.put("/users/{user_id}/role")
async def update_user_role(user_id: str, data: RoleUpdate, admin: dict = Depends(get_current_admin)):
    if data.role not in VALID_USER_ROLES:
        raise HTTPException(status_code=400, detail="Geçersiz rol.")
    with db_session() as conn:
        db_execute(conn, "UPDATE users SET role = %s WHERE id = %s", (data.role, user_id))
        log_audit(
            conn,
            "role_updated",
            user_id=admin["id"],
            entity_type="user",
            entity_id=user_id,
            details={"new_role": data.role},
        )
        conn.commit()
    return {"message": "Rol güncellendi."}


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, admin: dict = Depends(get_current_admin)):
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="Kendi hesabınızı silemezsiniz.")
    with db_session() as conn:
        # 1. Delete feedback linked to user's queries
        db_execute(
            conn,
            """
            DELETE FROM user_feedback WHERE message_id IN (
                SELECT id FROM query_history WHERE user_id = %s
            )
            """,
            (user_id,),
        )
        # 2. Delete agent runs linked to user's queries
        db_execute(
            conn,
            """
            DELETE FROM agent_runs WHERE trigger_message_id IN (
                SELECT id FROM query_history WHERE user_id = %s
            )
            """,
            (user_id,),
        )
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


@router.put("/users/{user_id}/approve")
async def approve_user(user_id: str, admin: dict = Depends(get_current_admin)):
    with db_session() as conn:
        db_execute(conn, "UPDATE users SET is_approved = 1 WHERE id = %s", (user_id,))
        log_audit(conn, "user_approved", user_id=admin["id"], entity_type="user", entity_id=user_id)
        conn.commit()
    return {"message": "Kullanıcı onaylandı."}


# --- System / activity / analytics ---
@router.get("/system")
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
        except Exception as e:  # noqa: BLE001
            logger.debug(f"chunks count skipped: {e}")
            conn.rollback()
        cur = db_execute(
            conn,
            "SELECT message FROM announcements WHERE active = 1 ORDER BY created_at DESC LIMIT 1",
        )
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
        "active_announcement": dict(active_announcement) if active_announcement else None,
    }


@router.get("/activity")
async def get_admin_activity(admin: dict = Depends(get_current_admin)):
    with db_session() as conn:
        cur = db_execute(
            conn,
            """
            SELECT qh.query, qh.created_at, u.username, u.role
            FROM query_history qh
            JOIN users u ON qh.user_id = u.id
            ORDER BY qh.created_at DESC
            LIMIT 20
            """,
        )
        rows = cur.fetchall()
        cur.close()
    return [dict(r) for r in rows]


@router.get("/analytics")
async def get_admin_analytics(admin: dict = Depends(get_current_admin)):
    with db_session() as conn:
        cur = db_execute(conn, "SELECT query FROM query_history")
        all_queries = cur.fetchall()
        cur.close()

        word_counter: Counter = Counter()
        for row in all_queries:
            words = re.findall(r"\b[a-zA-ZğüşıöçĞÜŞİÖÇ]{4,}\b", row["query"].lower())
            for w in words:
                if w not in _ANALYTICS_STOPWORDS:
                    word_counter[w] += 1
        top_keywords = [{"keyword": k, "count": v} for k, v in word_counter.most_common(10)]

        # Daily volume — last 7 days (PostgreSQL syntax)
        cur = db_execute(
            conn,
            """
            SELECT DATE(created_at) as day, COUNT(*) as count
            FROM query_history
            WHERE created_at >= NOW() - INTERVAL '7 days'
            GROUP BY DATE(created_at)
            ORDER BY day ASC
            """,
        )
        daily_rows = cur.fetchall()
        cur.close()
        daily_volume = [dict(r) for r in daily_rows]

        cur = db_execute(conn, "SELECT COUNT(*) FROM users")
        total_users = cur.fetchone()[0]
        cur.close()

        cur = db_execute(
            conn,
            "SELECT COUNT(DISTINCT user_id) FROM query_history WHERE created_at >= NOW() - INTERVAL '1 day'",
        )
        daily_active = cur.fetchone()[0]
        cur.close()
        daily_engagement_rate = round((daily_active / total_users * 100) if total_users > 0 else 0, 1)

        cur = db_execute(
            conn,
            "SELECT COUNT(DISTINCT user_id) FROM query_history WHERE created_at >= NOW() - INTERVAL '30 days'",
        )
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
        "total_users": total_users,
    }


# --- Audit logs ---
@router.get("/audit-logs")
async def get_audit_logs(
    admin: dict = Depends(get_current_admin),
    limit: int = 50,
    offset: int = 0,
    action: str = "",
    user_id: str = "",
):
    """Paginated audit log feed (admin only).

    Supports optional filtering by ``action`` (action_type) and ``user_id``.
    """
    try:
        safe_limit = max(1, min(int(limit), 500))
    except (TypeError, ValueError):
        safe_limit = 50
    try:
        safe_offset = max(0, int(offset))
    except (TypeError, ValueError):
        safe_offset = 0

    filters: list = []
    params: list = []
    if action:
        filters.append("a.action_type = %s")
        params.append(action)
    if user_id:
        filters.append("a.user_id = %s")
        params.append(user_id)
    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    with db_session() as conn:
        cur = db_execute(
            conn,
            f"""
            SELECT a.log_id, a.user_id, a.action_type, a.entity_type, a.entity_id,
                   a.details, a.created_at, u.username as user_name
            FROM audit_logs a
            LEFT JOIN users u ON a.user_id = u.id
            {where}
            ORDER BY a.created_at DESC
            LIMIT %s OFFSET %s
            """,
            tuple(params + [safe_limit, safe_offset]),
        )
        logs = cur.fetchall()
        cur.close()

        cur2 = db_execute(
            conn,
            f"SELECT COUNT(*) FROM audit_logs a {where}",
            tuple(params),
        )
        total = cur2.fetchone()[0]
        cur2.close()

    parsed_logs = []
    for log in logs:
        d = dict(log)
        if d.get("details"):
            try:
                d["details"] = json.loads(d["details"])
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        parsed_logs.append(d)

    return {"logs": parsed_logs, "total": total}


# --- Index rebuild ---
def _do_indexing():
    """Shared indexing routine used by both sync and async paths.

    Returns (success: bool, message: str). Logs the exception details
    into audit_logs and returns the error message in case of failure so
    the caller can surface it to the operator (HF Space logs are not
    accessible via the public API).
    """
    import os
    import traceback

    import api_server
    from rag_storage import SUT_Storage_Manager, DOCX_FILE_PATH

    # Pre-flight checks — surface specific errors instead of a generic crash.
    if api_server.engine is None or getattr(api_server.engine, "embeddings_model", None) is None:
        return False, "Engine veya embeddings_model henüz başlatılmamış."

    abs_path = os.path.abspath(DOCX_FILE_PATH)
    if not os.path.exists(DOCX_FILE_PATH):
        return False, f"SUT .docx dosyası bulunamadı: {abs_path} (cwd={os.getcwd()})"

    try:
        storage = SUT_Storage_Manager(api_server.engine.embeddings_model)
        success = storage.populate_database()
        if not success:
            return False, "populate_database() False döndü — pandoc veya chunking adımı başarısız."

        new_engine = api_server.SUT_RAG_Engine()
        if new_engine.load_database():
            api_server.engine = new_engine
            return True, "Indeksleme ve motor yeniden yüklemesi tamamlandı."
        return False, "Indeksleme tamamlandı ama motor yeniden yüklenemedi."
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("[INDEXING] error")
        return False, f"{type(e).__name__}: {e}\n{tb[-800:]}"


@router.post("/rebuild-index")
async def rebuild_index(
    background_tasks: BackgroundTasks,
    admin: dict = Depends(get_current_admin),
    sync: bool = False,
):
    """Re-populate the chunks table from the on-disk SUT source.

    By default runs in a background task (returns immediately).
    Pass ``?sync=true`` to run inline and surface any error in the response —
    useful for debugging when HF Space logs are not accessible.
    """
    with db_session() as dbconn:
        log_audit(dbconn, "index_rebuild_started", user_id=admin["id"])
        dbconn.commit()

    if sync:
        ok, msg = _do_indexing()
        with db_session() as dbconn:
            log_audit(
                dbconn,
                "index_rebuild_finished" if ok else "index_rebuild_failed",
                user_id=admin["id"],
                details=msg[:500],
            )
            dbconn.commit()
        if not ok:
            raise HTTPException(status_code=500, detail=msg)
        return {"message": msg, "ok": True}

    # Background path — also log result to audit_logs for later inspection.
    def run_indexing():
        ok, msg = _do_indexing()
        try:
            with db_session() as dbconn:
                log_audit(
                    dbconn,
                    "index_rebuild_finished" if ok else "index_rebuild_failed",
                    user_id=admin["id"],
                    details=msg[:500],
                )
                dbconn.commit()
        except Exception:  # noqa: BLE001
            logger.exception("[BACKGROUND] failed to write audit log")

    background_tasks.add_task(run_indexing)
    return {
        "message": (
            "İndeksleme işlemi arka planda başlatıldı. "
            "Sonuç audit_logs tablosuna yazılacak (index_rebuild_finished/failed)."
        )
    }


# --- Announcements ---
@router.post("/announcements")
async def create_announcement(data: AnnouncementCreate, admin: dict = Depends(get_current_admin)):
    with db_session() as conn:
        db_execute(conn, "UPDATE announcements SET active = 0")
        ann_id = str(uuid.uuid4())
        db_execute(
            conn,
            "INSERT INTO announcements (id, message, created_by, active) VALUES (%s, %s, %s, 1)",
            (ann_id, data.message, admin["id"]),
        )
        log_audit(conn, "announcement_created", user_id=admin["id"], entity_type="announcement", entity_id=ann_id)
        conn.commit()
    return {"message": "Duyuru yayınlandı."}


@router.delete("/announcements/{ann_id}")
async def deactivate_announcement(ann_id: str, admin: dict = Depends(get_current_admin)):
    with db_session() as conn:
        db_execute(conn, "UPDATE announcements SET active = 0 WHERE id = %s", (ann_id,))
        log_audit(conn, "announcement_deactivated", user_id=admin["id"], entity_type="announcement", entity_id=ann_id)
        conn.commit()
    return {"message": "Duyuru kaldırıldı."}


# --- Eval results ---
@router.get("/eval/results")
async def get_eval_results(admin: dict = Depends(get_current_admin)):
    """Return the persisted retrieval evaluation results JSON."""
    eval_path = os.path.join(os.path.dirname(__file__), "..", "eval_results", "retrieval_results.json")
    eval_path = os.path.abspath(eval_path)
    if not os.path.exists(eval_path):
        raise HTTPException(status_code=404, detail="Eval results not found")
    with open(eval_path) as f:
        return json.load(f)
