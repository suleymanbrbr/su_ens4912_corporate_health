# routers/kg.py — Knowledge graph endpoints (user + admin).
#
# Routes:
#   GET  /api/kg/stats
#   GET  /api/kg/nodes
#   GET  /api/kg/node/{node_id}
#   GET  /api/kg/subgraph/{rule_id}
#   GET  /api/kg/path
#   POST /api/admin/kg/rebuild
#   GET  /api/admin/kg/stats
#   POST /api/admin/kg/benchmark

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from deps import (
    db_execute,
    db_session,
    get_current_admin,
    get_current_user,
    log_audit,
)
from kg_storage import KG_Storage_Manager


logger = logging.getLogger("api_server")

router = APIRouter(tags=["kg"])

_kg: KG_Storage_Manager | None = None


def _get_kg() -> KG_Storage_Manager:
    """Lazy-initialize KG_Storage_Manager so it connects AFTER the DB is ready."""
    global _kg
    if _kg is None:
        _kg = KG_Storage_Manager()
    return _kg


# --- User KG endpoints ---
@router.get("/api/kg/stats")
async def get_kg_stats(current_user: dict = Depends(get_current_user)):
    """Return node/edge counts by type/relation."""
    try:
        return _get_kg().get_stats()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/kg/nodes")
async def search_kg_nodes(
    q: str = "",
    type: str = "",
    limit: int = 20,
    current_user: dict = Depends(get_current_user),
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
    except Exception as e:  # noqa: BLE001
        logger.exception(f"kg/nodes failed: {e}")
        raise HTTPException(status_code=500, detail="Bilgi grafiği araması başarısız.")


@router.get("/api/kg/node/{node_id}")
async def get_kg_node(node_id: str, current_user: dict = Depends(get_current_user)):
    """Get a single node with all its neighbors."""
    node = _get_kg().get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    neighbors = _get_kg().get_neighbors(node_id, limit=20)
    return {"node": node, "neighbors": neighbors}


@router.get("/api/kg/subgraph/{rule_id}")
async def get_kg_subgraph(rule_id: str, current_user: dict = Depends(get_current_user)):
    """Get the subgraph for a RULE node (all related nodes & edges)."""
    return _get_kg().get_rule_subgraph(rule_id)


@router.get("/api/kg/path")
async def find_kg_path(
    from_id: str,
    to_id: str,
    max_hops: int = 3,
    current_user: dict = Depends(get_current_user),
):
    """Find shortest path between two nodes."""
    # Cap max_hops — BFS over the full KG can explode at higher hop counts.
    try:
        safe_hops = max(1, min(int(max_hops), 5))
    except (TypeError, ValueError):
        safe_hops = 3
    path = _get_kg().find_path(from_id, to_id, max_hops=safe_hops)
    return {"path": path, "found": len(path) > 0}


# --- Admin KG endpoints ---
@router.post("/api/admin/kg/rebuild")
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
        except Exception as e:  # noqa: BLE001
            logger.exception(f"[KG_REBUILD] Error: {e}")

    background_tasks.add_task(run_kg_build)
    with db_session() as dbconn:
        log_audit(dbconn, "kg_rebuild_started", user_id=admin["id"])
        dbconn.commit()
    return {"message": "KG yeniden oluşturma işlemi arka planda başlatıldı."}


@router.get("/api/admin/kg/stats")
async def get_admin_kg_stats(admin: dict = Depends(get_current_admin)):
    try:
        return _get_kg().get_stats()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/admin/kg/benchmark")
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
        except Exception as e:  # noqa: BLE001
            results.append({"question": item["q"], "error": str(e), "hit": False})
    return {
        "total": len(QUESTIONS),
        "hits": hits,
        "hit_rate": round(hits / len(QUESTIONS), 3),
        "results": results,
    }
