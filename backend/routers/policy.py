# routers/policy.py — full-text policy chunk search.
#
# Route:
#   GET /api/policies

import json
import logging
import re

from fastapi import APIRouter, Depends, HTTPException

from deps import (
    db_execute,
    db_session,
    get_current_user,
)


logger = logging.getLogger("api_server")

router = APIRouter(tags=["policy"])


def _chunk_row_to_result(c) -> dict:
    meta = c["metadata_json"] if isinstance(c["metadata_json"], dict) else (
        json.loads(c["metadata_json"]) if c["metadata_json"] else {}
    )
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
    """Full-text + ILIKE fallback.

    q_mode: phrase | and | or.
    If ``q`` contains a comma: OR across comma-separated groups; within each
    group, tokens (whitespace) are combined with AND (stronger matches).
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


@router.get("/api/policies")
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
                cur = db_execute(
                    conn,
                    "SELECT chunk_id, text_content, metadata_json FROM chunks LIMIT %s OFFSET %s",
                    (safe_limit, safe_offset),
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
    except Exception as e:  # noqa: BLE001
        logger.exception(f"policy search failed: {e}")
        raise HTTPException(status_code=500, detail="Arama hatası.")
    return {"results": results, "total": total, "offset": safe_offset, "limit": safe_limit}
