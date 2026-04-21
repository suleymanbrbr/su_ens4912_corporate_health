# kg_storage.py
# Description: Query layer for the Postgres-native Knowledge Graph.
# Used by both the agent (sut_rag_core.py) and the API (api_server.py).

import os
import json
import psycopg2
import psycopg2.extras
from typing import List, Dict, Optional
from collections import deque


def get_conn():
    return psycopg2.connect(os.getenv("DATABASE_URL"), cursor_factory=psycopg2.extras.DictCursor)


class KG_Storage_Manager:
    """
    Query interface for kg_nodes and kg_edges tables.
    All methods create their own short-lived connections — safe to use from any context.
    """

    # ─── Node Lookup ─────────────────────────────────────────────────────────

    def find_nodes_by_label(self, query: str, k: int = 5, type_filter: str = None) -> List[Dict]:
        """Exact + prefix match on node labels and IDs."""
        conn = get_conn()
        try:
            cur = conn.cursor()
            q = query.strip()
            if type_filter:
                cur.execute("""
                    SELECT node_id, label, type, text_content, atc_code, icd_code
                    FROM kg_nodes
                    WHERE (UPPER(label) LIKE UPPER(%s) OR UPPER(node_id) LIKE UPPER(%s))
                      AND type = %s
                    LIMIT %s
                """, (f"%{q}%", f"%{q}%", type_filter.upper(), k))
            else:
                cur.execute("""
                    SELECT node_id, label, type, text_content, atc_code, icd_code
                    FROM kg_nodes
                    WHERE UPPER(label) LIKE UPPER(%s) OR UPPER(node_id) LIKE UPPER(%s)
                    LIMIT %s
                """, (f"%{q}%", f"%{q}%", k))
            rows = cur.fetchall()
            cur.close()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def find_nodes_semantic(self, query: str, k: int = 5, type_filter: str = None) -> List[Dict]:
        """Semantic vector search on node label+text embeddings."""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            if not hasattr(self, '_embed_model'):
                self._embed_model = HuggingFaceEmbeddings(
                    model_name="paraphrase-multilingual-MiniLM-L12-v2",
                    model_kwargs={'device': 'cpu'}
                )
            vec = self._embed_model.embed_query(query[:512])
            vec_str = "[" + ",".join(map(str, vec)) + "]"
        except Exception as e:
            print(f"[KG_STORAGE] Embed failed: {e}")
            return []

        conn = get_conn()
        try:
            cur = conn.cursor()
            if type_filter:
                cur.execute("""
                    SELECT node_id, label, type, text_content, atc_code, icd_code,
                           1 - (embedding <=> %s) AS cosine_sim
                    FROM kg_nodes
                    WHERE embedding IS NOT NULL AND type = %s
                    ORDER BY embedding <=> %s
                    LIMIT %s
                """, (vec_str, type_filter.upper(), vec_str, k))
            else:
                cur.execute("""
                    SELECT node_id, label, type, text_content, atc_code, icd_code,
                           1 - (embedding <=> %s) AS cosine_sim
                    FROM kg_nodes
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s
                    LIMIT %s
                """, (vec_str, vec_str, k))
            rows = cur.fetchall()
            cur.close()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get a single node by its ID."""
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT node_id, label, type, text_content, atc_code, icd_code FROM kg_nodes WHERE node_id = %s",
                (node_id.strip().upper(),)
            )
            row = cur.fetchone()
            cur.close()
            return dict(row) if row else None
        finally:
            conn.close()

    # ─── Edge / Neighborhood Lookup ───────────────────────────────────────────

    def get_neighbors(self, node_id: str, relation_filter: str = None, limit: int = 40) -> List[Dict]:
        """Return direct neighbors formatted for the frontend (node, edge, direction)."""
        nid = node_id.strip().upper()
        conn = get_conn()
        try:
            cur = conn.cursor()
            if relation_filter:
                cur.execute("""
                    SELECT e.edge_id, e.source_id, e.target_id, e.relation, e.confidence, e.source_rule,
                           sn.label AS source_label, sn.type AS source_type,
                           tn.label AS target_label, tn.type AS target_type
                    FROM kg_edges e
                    JOIN kg_nodes sn ON sn.node_id = e.source_id
                    JOIN kg_nodes tn ON tn.node_id = e.target_id
                    WHERE (e.source_id = %s OR e.target_id = %s) AND e.relation = %s
                    LIMIT %s
                """, (nid, nid, relation_filter.upper(), limit))
            else:
                cur.execute("""
                    SELECT e.edge_id, e.source_id, e.target_id, e.relation, e.confidence, e.source_rule,
                           sn.label AS source_label, sn.type AS source_type,
                           tn.label AS target_label, tn.type AS target_type
                    FROM kg_edges e
                    JOIN kg_nodes sn ON sn.node_id = e.source_id
                    JOIN kg_nodes tn ON tn.node_id = e.target_id
                    WHERE e.source_id = %s OR e.target_id = %s
                    LIMIT %s
                """, (nid, nid, limit))
            rows = cur.fetchall()
            cur.close()

            results = []
            for r in rows:
                is_out = r["source_id"] == nid
                results.append({
                    "direction": "outgoing" if is_out else "incoming",
                    "edge": {
                        "edge_id": r["edge_id"],
                        "source_id": r["source_id"],
                        "target_id": r["target_id"],
                        "relation": r["relation"],
                        "confidence": r["confidence"],
                        "source_rule": r["source_rule"]
                    },
                    "node": {
                        "node_id": r["target_id"] if is_out else r["source_id"],
                        "label": r["target_label"] if is_out else r["source_label"],
                        "type": r["target_type"] if is_out else r["source_type"]
                    }
                })
            return results
        finally:
            conn.close()

    def get_rule_subgraph(self, rule_id: str) -> Dict:
        """Get a full subgraph for a RULE node (node + all neighbors)."""
        nid = rule_id.strip().upper()
        node = self.get_node(nid)
        if not node:
            return {"nodes": [], "edges": []}
        edges = self.get_neighbors(nid, limit=50)
        # Collect all involved node IDs
        node_ids = {nid}
        for e in edges:
            node_ids.add(e["source_id"])
            node_ids.add(e["target_id"])
        # Fetch all nodes
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT node_id, label, type, text_content, atc_code, icd_code FROM kg_nodes WHERE node_id = ANY(%s)",
                (list(node_ids),)
            )
            nodes = [dict(r) for r in cur.fetchall()]
            cur.close()
        finally:
            conn.close()
        return {"nodes": nodes, "edges": edges}

    # ─── Path Finding ─────────────────────────────────────────────────────────

    def find_path(self, source_id: str, target_id: str, max_hops: int = 3) -> List[List[str]]:
        """
        BFS to find shortest path between two nodes.
        Returns a list of node_ids from source to target, or [] if not found.
        """
        src = source_id.strip().upper()
        tgt = target_id.strip().upper()

        if src == tgt:
            return [[src]]

        conn = get_conn()
        try:
            # Build adjacency: node_id → list of neighbor node_ids
            cur = conn.cursor()
            cur.execute("SELECT source_id, target_id FROM kg_edges")
            edges_raw = cur.fetchall()
            cur.close()

            adj: Dict[str, List[str]] = {}
            for row in edges_raw:
                s, t = row["source_id"], row["target_id"]
                adj.setdefault(s, []).append(t)
                adj.setdefault(t, []).append(s)  # undirected for path finding

            # BFS
            queue = deque([[src]])
            visited = {src}
            while queue:
                path = queue.popleft()
                current = path[-1]
                if len(path) > max_hops + 1:
                    break
                for neighbor in adj.get(current, []):
                    if neighbor == tgt:
                        return path + [neighbor]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(path + [neighbor])
            return []
        finally:
            conn.close()

    def find_path_with_details(self, source_id: str, target_id: str, max_hops: int = 3) -> List[Dict]:
        """
        Returns path as list of {node, edge} dicts for human-readable output.
        """
        path_ids = self.find_path(source_id, target_id, max_hops)
        if not path_ids:
            return []

        conn = get_conn()
        try:
            details = []
            for i, nid in enumerate(path_ids):
                node = self.get_node(nid)
                edge_info = None
                if i < len(path_ids) - 1:
                    next_id = path_ids[i + 1]
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT relation, confidence FROM kg_edges
                        WHERE (source_id=%s AND target_id=%s) OR (source_id=%s AND target_id=%s)
                        LIMIT 1
                    """, (nid, next_id, next_id, nid))
                    edge_row = cur.fetchone()
                    cur.close()
                    if edge_row:
                        edge_info = {"relation": edge_row["relation"], "confidence": edge_row["confidence"]}
                details.append({"node": node, "edge_to_next": edge_info})
            return details
        finally:
            conn.close()

    # ─── Combined Entity Lookup (for agent tool) ──────────────────────────────

    def lookup_entity(self, query: str, k: int = 3, type_filter: str = None) -> str:
        """
        Runs both string-match and semantic search, merges results, returns
        a formatted string suitable for LLM consumption.
        """
        # String match first (fast, exact)
        exact_results = self.find_nodes_by_label(query, k=k, type_filter=type_filter)
        exact_ids = {r["node_id"] for r in exact_results}

        # Semantic search
        semantic_results = self.find_nodes_semantic(query, k=k, type_filter=type_filter)

        # Merge: exact first, then semantic (deduplicated)
        all_results = exact_results[:]
        for r in semantic_results:
            if r["node_id"] not in exact_ids:
                all_results.append(r)
        all_results = all_results[:k]

        if not all_results:
            return f"'{query}' için bilgi grafiğinde eşleşme bulunamadı."

        parts = []
        for node in all_results:
            nid = node["node_id"]
            neighbors = self.get_neighbors(nid, limit=10)

            codes = ""
            if node.get("atc_code"):
                codes += f" | ATC: {node['atc_code']}"
            if node.get("icd_code"):
                codes += f" | ICD-10: {node['icd_code']}"

            info = f"DÜĞÜM: {node['label']} (Tür: {node['type']}{codes})"
            if node.get("text_content"):
                info += f"\nİçerik: {node['text_content'][:300]}"

            edge_lines = []
            for e in neighbors:
                if e["source_id"] == nid:
                    edge_lines.append(f"  [{e['relation']}] → {e['target_label']} ({e['target_type']})")
                else:
                    edge_lines.append(f"  [{e['relation']}] ← {e['source_label']} ({e['source_type']})")

            if edge_lines:
                info += "\nİlişkiler:\n" + "\n".join(edge_lines[:8])
            parts.append(info)

        return "\n\n---\n".join(parts)

    def explore_path(self, from_entity: str, to_entity: str, max_hops: int = 3) -> str:
        """
        Finds nodes matching from/to entities and returns path description.
        """
        from_nodes = self.find_nodes_by_label(from_entity, k=1)
        to_nodes   = self.find_nodes_by_label(to_entity, k=1)

        if not from_nodes:
            from_nodes = self.find_nodes_semantic(from_entity, k=1)
        if not to_nodes:
            to_nodes = self.find_nodes_semantic(to_entity, k=1)

        if not from_nodes or not to_nodes:
            return f"Varlıklardan biri veya ikisi de bulunamadı: '{from_entity}', '{to_entity}'"

        src_id = from_nodes[0]["node_id"]
        tgt_id = to_nodes[0]["node_id"]

        path_details = self.find_path_with_details(src_id, tgt_id, max_hops=max_hops)

        if not path_details:
            return (f"'{from_nodes[0]['label']}' ile '{to_nodes[0]['label']}' arasında "
                    f"{max_hops} adım içinde bağlantı bulunamadı.")

        lines = [f"YOL: {from_nodes[0]['label']} → {to_nodes[0]['label']}"]
        for step in path_details:
            node_info = f"• {step['node']['label']} ({step['node']['type']})"
            lines.append(node_info)
            if step["edge_to_next"]:
                lines.append(f"    ↓ [{step['edge_to_next']['relation']}]")

        return "\n".join(lines)

    # ─── Admin/Stats ──────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return node/edge counts by type/relation for admin dashboard."""
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT type, COUNT(*) as count FROM kg_nodes GROUP BY type ORDER BY count DESC")
            node_types = {r["type"]: r["count"] for r in cur.fetchall()}
            cur.execute("SELECT relation, COUNT(*) as count FROM kg_edges GROUP BY relation ORDER BY count DESC")
            relation_types = {r["relation"]: r["count"] for r in cur.fetchall()}
            cur.execute("SELECT COUNT(*) FROM kg_nodes")
            total_nodes = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM kg_edges")
            total_edges = cur.fetchone()[0]
            cur.execute("SELECT COALESCE(MAX(chunks_processed), 0) FROM kg_build_log WHERE status='done'")
            chunks_covered_row = cur.fetchone()
            chunks_covered = chunks_covered_row[0] if chunks_covered_row else 0
            cur.execute("SELECT finished_at FROM kg_build_log WHERE status='done' ORDER BY finished_at DESC LIMIT 1")
            last_build_row = cur.fetchone()
            last_build = str(last_build_row["finished_at"]) if last_build_row and last_build_row["finished_at"] else None
            cur.close()
            return {
                "node_count": total_nodes,
                "edge_count": total_edges,
                "node_types": node_types,
                "relation_types": relation_types,
                "chunks_covered": chunks_covered,
                "last_build": last_build,
            }
        finally:
            conn.close()
