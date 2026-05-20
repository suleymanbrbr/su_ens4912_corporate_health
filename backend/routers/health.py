# routers/health.py — health probe.
#
# Route:
#   GET /health
#
# Used by load balancers; intentionally always returns HTTP 200 with the
# DB-status payload so monitors can inspect the body rather than relying
# on the status code (a 5xx would trigger noisy failover alerts).

from fastapi import APIRouter

from deps import db_session


router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Liveness + DB readiness probe.

    Always returns 200 with JSON body so load balancers can inspect db
    status without false-positive failover.
    """
    db_status = "down"
    try:
        with db_session() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
            cur.close()
        db_status = "ok"
    except Exception:  # noqa: BLE001
        db_status = "down"
    return {"status": "ok", "db": db_status, "version": "1.0.0"}
