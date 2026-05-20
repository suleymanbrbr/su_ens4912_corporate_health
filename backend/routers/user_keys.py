# routers/user_keys.py — per-user LLM API key management.
#
# Routes:
#   POST   /api/user/api-keys
#   GET    /api/user/api-keys
#   DELETE /api/user/api-keys/{provider}
#   POST   /api/user/api-keys/test
#
# Keys are stored Fernet-encrypted; we never return decrypted material.

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel

from deps import (
    VALID_PROVIDERS,
    db_session,
    get_current_user,
    log_audit,
)
from secrets_utils import encrypt_api_key, make_key_hint


logger = logging.getLogger("api_server")

router = APIRouter(prefix="/api/user/api-keys", tags=["user_keys"])


# --- Models ---
class ApiKeyCreate(BaseModel):
    provider: str  # 'gemini' | 'openrouter' | 'local'
    api_key: str
    base_url: Optional[str] = None


class ApiKeyTest(BaseModel):
    provider: str
    api_key: str
    base_url: Optional[str] = None


# --- Endpoints ---
@router.post("", status_code=201)
async def create_or_update_user_api_key(
    data: ApiKeyCreate,
    current_user: dict = Depends(get_current_user),
):
    """Store (or replace) the user's API key for a given provider."""
    if data.provider not in VALID_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Geçersiz sağlayıcı. Şunlardan biri olmalı: {sorted(VALID_PROVIDERS)}",
        )
    if not data.api_key or not data.api_key.strip():
        raise HTTPException(status_code=400, detail="API anahtarı boş olamaz.")

    encrypted = encrypt_api_key(data.api_key.strip())
    hint = make_key_hint(data.api_key.strip())

    with db_session() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        # Upsert: replace existing (user_id, provider) row.
        cur.execute(
            """
            INSERT INTO user_api_keys (user_id, provider, encrypted_key, base_url, key_hint, is_active, updated_at)
            VALUES (%s, %s, %s, %s, %s, TRUE, NOW())
            ON CONFLICT (user_id, provider) DO UPDATE
            SET encrypted_key = EXCLUDED.encrypted_key,
                base_url      = EXCLUDED.base_url,
                key_hint      = EXCLUDED.key_hint,
                is_active     = TRUE,
                updated_at    = NOW()
            RETURNING id, provider, key_hint, created_at
            """,
            (current_user["id"], data.provider, encrypted, data.base_url, hint),
        )
        row = cur.fetchone()
        log_audit(
            conn,
            "api_key_upserted",
            user_id=current_user["id"],
            entity_type="user_api_key",
            entity_id=str(row["id"]),
            details={"provider": data.provider},
        )
        conn.commit()
        cur.close()
        return {
            "id": row["id"],
            "provider": row["provider"],
            "key_hint": row["key_hint"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }


@router.get("")
async def list_user_api_keys(current_user: dict = Depends(get_current_user)):
    """List the user's stored API keys. Never returns decrypted keys."""
    with db_session() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT id, provider, key_hint, is_active, last_used_at, created_at
            FROM user_api_keys
            WHERE user_id = %s
            ORDER BY provider ASC
            """,
            (current_user["id"],),
        )
        rows = cur.fetchall()
        cur.close()
    return [
        {
            "id": r["id"],
            "provider": r["provider"],
            "key_hint": r["key_hint"],
            "is_active": r["is_active"],
            "last_used_at": r["last_used_at"].isoformat() if r["last_used_at"] else None,
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in rows
    ]


@router.delete("/{provider}", status_code=204)
async def delete_user_api_key(
    provider: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete the user's API key for a given provider. Returns 204 No Content."""
    if provider not in VALID_PROVIDERS:
        raise HTTPException(status_code=400, detail="Geçersiz sağlayıcı.")
    with db_session() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM user_api_keys WHERE user_id = %s AND provider = %s",
            (current_user["id"], provider),
        )
        log_audit(
            conn,
            "api_key_deleted",
            user_id=current_user["id"],
            entity_type="user_api_key",
            entity_id=provider,
            details={"provider": provider},
        )
        conn.commit()
        cur.close()
    return Response(status_code=204)


@router.post("/test")
async def test_user_api_key(
    data: ApiKeyTest,
    current_user: dict = Depends(get_current_user),
):
    """Perform a tiny live LLM call to validate the supplied API key.

    Returns {valid: bool, error?: str}. Errors are truncated to 200 chars to
    avoid leaking provider internals back to the UI.
    """
    if data.provider not in VALID_PROVIDERS:
        return {"valid": False, "error": "Geçersiz sağlayıcı."}
    if not data.api_key or not data.api_key.strip():
        return {"valid": False, "error": "API anahtarı boş olamaz."}

    api_key = data.api_key.strip()
    base_url = (data.base_url or "").strip() or None

    try:
        if data.provider == "gemini":
            import google.generativeai as gg
            gg.configure(api_key=api_key)
            model = gg.GenerativeModel("gemini-2.5-flash-lite")
            # Use generation_config to limit tokens — cheapest possible probe.
            model.generate_content(
                "ping",
                generation_config={"max_output_tokens": 1},
            )
            return {"valid": True}

        elif data.provider == "openrouter":
            import httpx
            r = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "google/gemma-2-9b-it",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
                timeout=15,
            )
            if r.status_code == 200:
                return {"valid": True}
            return {"valid": False, "error": f"HTTP {r.status_code}: {r.text[:160]}"}

        elif data.provider == "local":
            if not base_url:
                return {"valid": False, "error": "Yerel LLM için base_url gerekli."}
            import httpx
            r = httpx.get(
                f"{base_url.rstrip('/')}/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5,
            )
            if r.status_code == 200:
                return {"valid": True}
            return {"valid": False, "error": f"HTTP {r.status_code}"}

    except Exception as e:
        return {"valid": False, "error": str(e)[:200]}

    return {"valid": False, "error": "Bilinmeyen sağlayıcı durumu."}
