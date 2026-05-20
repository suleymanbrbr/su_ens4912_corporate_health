# deps.py
# Shared FastAPI dependencies and DB helpers for the SUT Assistant backend.
#
# This module exists so individual routers in ``backend/routers/`` can import
# the same auth/DB helpers without going through the (legacy) monolithic
# ``api_server.py``.  Keeping these helpers separate avoids the circular
# imports that would otherwise occur when routers and api_server reference
# each other.

import logging
import os
import json
import uuid
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

from auth_utils import decode_access_token

logger = logging.getLogger("api_server")

# Reusable constants — kept here so routers don't all redefine them.
VALID_USER_ROLES = {"user", "admin"}
MIN_PASSWORD_LENGTH = 8                  # registration minimum (existing accounts not enforced)
MAX_PDF_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB server-side cap on PDF uploads
HISTORY_PAGE_LIMIT = 200
VALID_PROVIDERS = {"gemini", "openrouter", "local"}


# --- DB helpers ---
def get_db_conn():
    """Open a new psycopg2 connection. Callers MUST close it (use ``db_session``)."""
    return psycopg2.connect(
        os.getenv("DATABASE_URL"),
        cursor_factory=psycopg2.extras.DictCursor,
    )


@contextmanager
def db_session():
    """Context manager that yields a connection and guarantees ``.close()``.

    Always rolls back on exception so a failed query never leaves a connection
    in an inconsistent state in the pool. Callers still call ``.commit()``
    explicitly on the happy path — this preserves the existing transaction
    semantics across endpoints.
    """
    conn = get_db_conn()
    try:
        yield conn
    except Exception:
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def db_execute(conn, query, params=None):
    """Run a query and return the cursor (caller closes it)."""
    cur = conn.cursor()
    cur.execute(query, params)
    return cur


def log_audit(conn, action_type, user_id=None, entity_type=None, entity_id=None, details=None):
    """Best-effort audit logger — failures are swallowed and only logged."""
    try:
        details_json = json.dumps(details, ensure_ascii=False) if details else None
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO audit_logs (log_id, user_id, action_type, entity_type, entity_id, details) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (str(uuid.uuid4()), user_id, action_type, entity_type, entity_id, details_json),
        )
        cur.close()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to log audit: {e}")


# --- Auth dependencies ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    username: str = payload.get("sub")
    with db_session() as conn:
        cur = db_execute(conn, "SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return dict(user)


async def get_current_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return current_user
