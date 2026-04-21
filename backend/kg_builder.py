# kg_builder.py
# Description: Full KG extraction from the chunks Postgres table via Gemini.
# Replaces the old kg.py. Stores results directly in kg_nodes and kg_edges tables.

import os
import json
import uuid
import time
import psycopg2
import psycopg2.extras
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# NOTE: Using langchain_google_genai (already pinned in requirements) to avoid
# version conflicts with direct google-generativeai. ChatGoogleGenerativeAI
# supports response_format via json_mode for structured output.
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
GEMINI_MODEL       = "gemini-2.0-flash"
BATCH_SLEEP_SEC    = 1.0          # sleep between chunks to avoid quota
MAX_RETRIES        = 3
DEDUP_COSINE_THRESH = 0.92        # merge nodes if cosine similarity >= this

# ─── Node type ontology ───────────────────────────────────────────────────────
VALID_NODE_TYPES = {
    "RULE", "DRUG", "DIAGNOSIS", "SPECIALIST",
    "CONDITION", "DOCUMENT", "DEVICE", "DOSAGE", "AGE_LIMIT", "EXCLUSION"
}

VALID_RELATIONS = {
    "COVERS", "TREATS", "ISSUED_BY", "PRESCRIBED_BY",
    "REQUIRES_CONDITION", "MUST_FAIL_FIRST", "HAS_LIMIT", "NOT_COVERED_FOR",
    "HAS_SUBRULE", "HAS_DOSAGE", "CONTRAINDICATED_FOR", "HAS_AGE_LIMIT",
    "APPROVED_BY", "REQUIRES_REPORT", "FUNDED_BY"
}

# ─── Extraction prompt ────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """
You are an expert Knowledge Graph engineer specializing in Turkish Social Security health regulations (SUT).

TASK: Extract a structured knowledge graph from the given SUT article text.

=== NODE TYPES (use EXACTLY these strings) ===
- RULE:       A SUT article/clause ID (e.g. "4.2.14.C")
- DRUG:       Active ingredient or brand name (e.g. "İnfliksimab", "Anti-TNF")
- DIAGNOSIS:  Disease name or ICD code (e.g. "Romatoid Artrit", "M05")
- SPECIALIST: Doctor speciality or board (e.g. "Romatoloji Uzmanı", "Sağlık Kurulu")
- CONDITION:  A logical prerequisite (e.g. "DAS28 > 5.1", "3 ay süreyle deneme")
- DOCUMENT:   Required paperwork (e.g. "Sağlık Kurulu Raporu", "Genetik Test Raporu")
- DEVICE:     Medical equipment (e.g. "CPAP Cihazı")
- DOSAGE:     Numeric dose rule (e.g. "Maks 10 mg/kg/gün", "Günlük 400 mg")
- AGE_LIMIT:  Patient age constraint (e.g. "18 yaş üzeri", "2-12 yaş")
- EXCLUSION:  Contraindication statement (e.g. "Aktif TB varlığında kullanılamaz")

=== EDGE RELATIONS (use EXACTLY these strings) ===
COVERS | TREATS | ISSUED_BY | PRESCRIBED_BY | REQUIRES_CONDITION |
MUST_FAIL_FIRST | HAS_LIMIT | NOT_COVERED_FOR | HAS_SUBRULE |
HAS_DOSAGE | CONTRAINDICATED_FOR | HAS_AGE_LIMIT | APPROVED_BY |
REQUIRES_REPORT | FUNDED_BY

=== RULES ===
1. Use Turkish names as they appear in the source text.
2. Always create a RULE node for the article ID.
3. Capture step-therapy with MUST_FAIL_FIRST (Drug A must fail before Drug B).
4. Capture dosage with HAS_DOSAGE edges.
5. Capture age limits with HAS_AGE_LIMIT edges.
6. Capture contraindications with CONTRAINDICATED_FOR.
7. Only include relations explicitly stated — do NOT infer.
8. Return ONLY valid JSON matching the schema below.

=== OUTPUT SCHEMA ===
{
  "nodes": [
    {"id": "<UNIQUE_UPPERCASE_ID>", "label": "<original Turkish name>", "type": "<NODE_TYPE>", "text": "<optional: relevant text excerpt>"}
  ],
  "edges": [
    {"source": "<source_node_id>", "target": "<target_node_id>", "relation": "<RELATION>", "confidence": 0.95}
  ]
}

=== INPUT ARTICLE ===
Article header: [[HEADER]]
Article content:
[[TEXT]]
"""

# ─── DB helpers ───────────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(os.getenv("DATABASE_URL"), cursor_factory=psycopg2.extras.DictCursor)


class KG_Builder:
    """
    Reads all chunks from Postgres, calls Gemini to extract KG data chunk-by-chunk,
    and writes nodes+edges directly into kg_nodes and kg_edges tables.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment.")

        self.model = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0,
        )

        # In-memory node ID cache for deduplication
        self.node_id_set: set = set()
        self.log_id: Optional[str] = None
        self._nodes_created = 0
        self._edges_created = 0
        self._chunks_processed = 0

    # ─── Embed helper (using the same model as RAG) ───────────────────────────
    def _embed_text(self, text: str) -> List[float]:
        from langchain_huggingface import HuggingFaceEmbeddings
        if not hasattr(self, '_embed_model'):
            self._embed_model = HuggingFaceEmbeddings(
                model_name="paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}
            )
        return self._embed_model.embed_query(text[:512])

    # ─── Start / finish build log ─────────────────────────────────────────────
    def _start_log(self, conn):
        self.log_id = str(uuid.uuid4())
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO kg_build_log (log_id, status) VALUES (%s, 'running')",
            (self.log_id,)
        )
        conn.commit()
        cur.close()

    def _finish_log(self, conn, status="done", error=None):
        if not self.log_id:
            return
        cur = conn.cursor()
        cur.execute(
            """UPDATE kg_build_log SET finished_at=NOW(), status=%s,
               nodes_created=%s, edges_created=%s, chunks_processed=%s, error_message=%s
               WHERE log_id=%s""",
            (status, self._nodes_created, self._edges_created,
             self._chunks_processed, error, self.log_id)
        )
        conn.commit()
        cur.close()

    # ─── Extract from one chunk ───────────────────────────────────────────────
    def _extract_chunk(self, header: str, text: str) -> Optional[Dict]:
        import re as _re
        prompt = EXTRACTION_PROMPT.replace(
            "[[HEADER]]", header or "Bilinmiyor"
        ).replace(
            "[[TEXT]]", text[:3000]
        )
        for attempt in range(MAX_RETRIES):
            try:
                response = self.model.invoke([HumanMessage(content=prompt)])
                raw = response.content.strip()

                # Fix common Gemini-2-Flash '{{' hallucination (mimicking prompt escaping)
                if "{{" in raw and "\"nodes\"" in raw:
                    raw = raw.replace("{{", "{").replace("}}", "}")

                # Strategy 1: direct JSON parse
                try:
                    data = json.loads(raw)
                    if "nodes" in data and "edges" in data:
                        return data
                except json.JSONDecodeError:
                    pass

                # Strategy 2: strip markdown fences
                if "```" in raw:
                    for part in raw.split("```"):
                        part = part.strip()
                        if part.startswith("json"):
                            part = part[4:].strip()
                        try:
                            data = json.loads(part)
                            if "nodes" in data and "edges" in data:
                                return data
                        except json.JSONDecodeError:
                            continue

                # Strategy 3: regex find JSON object
                match = _re.search(r'\{[\s\S]*?"nodes"[\s\S]*?"edges"[\s\S]*\}', raw)
                if match:
                    try:
                        data = json.loads(match.group())
                        if "nodes" in data and "edges" in data:
                            return data
                    except json.JSONDecodeError:
                        pass

                print(f"  [WARN] Could not parse JSON (len={len(raw)}). Start of response: {raw[:150]}...")
                return None
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                    wait = (attempt + 1) * 15
                    print(f"  [QUOTA] Rate limited. Waiting {wait}s... (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait)
                else:
                    print(f"  [ERROR] Extraction failed: {err[:200]}")
                    return None
        return None

    # ─── Canonicalize node ID ─────────────────────────────────────────────────
    @staticmethod
    def _canonical_id(raw_id: str) -> str:
        return str(raw_id).strip().upper()[:200]

    # ─── Insert node (with upsert and dedup) ─────────────────────────────────
    def _upsert_node(self, cur, node: Dict, source_rule: str):
        nid = self._canonical_id(node.get("id", ""))
        label = str(node.get("label", nid))[:500]
        ntype = node.get("type", "UNKNOWN")
        if ntype not in VALID_NODE_TYPES:
            ntype = "CONDITION"
        text_content = str(node.get("text", ""))[:2000]

        if nid in self.node_id_set:
            # Update text_content if it's empty in DB and we have data now
            if text_content:
                cur.execute(
                    "UPDATE kg_nodes SET text_content=%s, updated_at=NOW() WHERE node_id=%s AND (text_content='' OR text_content IS NULL)",
                    (text_content, nid)
                )
            return nid

        # Compute embedding for semantic lookup
        embed_text = f"{label} {text_content[:200]}"
        try:
            embedding = self._embed_text(embed_text)
            embed_str = "[" + ",".join(map(str, embedding)) + "]"
        except Exception:
            embed_str = None

        cur.execute("""
            INSERT INTO kg_nodes (node_id, label, type, text_content, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (node_id) DO UPDATE
            SET label=EXCLUDED.label,
                text_content=CASE WHEN EXCLUDED.text_content != '' THEN EXCLUDED.text_content ELSE kg_nodes.text_content END,
                updated_at=NOW()
        """, (nid, label, ntype, text_content, embed_str))

        self.node_id_set.add(nid)
        self._nodes_created += 1
        return nid

    # ─── Insert edge ──────────────────────────────────────────────────────────
    def _insert_edge(self, cur, source_id: str, target_id: str, relation: str,
                     confidence: float, source_rule: str):
        if relation not in VALID_RELATIONS:
            return
        if source_id not in self.node_id_set or target_id not in self.node_id_set:
            return
        edge_id = str(uuid.uuid4())
        try:
            cur.execute("""
                INSERT INTO kg_edges (edge_id, source_id, target_id, relation, confidence, source_rule)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (edge_id, source_id, target_id, relation, float(confidence), source_rule))
            self._edges_created += 1
        except Exception as e:
            print(f"  [WARN] Edge insert failed ({source_id}→{target_id}): {e}")

    # ─── Main build pipeline ──────────────────────────────────────────────────
    def build(self, clear_existing: bool = True, resume: bool = False, workers: int = 5):
        """
        Full pipeline: read all chunks from Postgres, extract KG per chunk,
        write nodes+edges to kg tables.
        """
        conn = get_conn()
        self._start_log(conn)

        try:
            if clear_existing and not resume:
                print("[KG_BUILD] Clearing existing KG data...")
                cur = conn.cursor()
                cur.execute("DELETE FROM kg_edges")
                cur.execute("DELETE FROM kg_nodes")
                conn.commit()
                cur.close()
                self.node_id_set.clear()
            elif resume:
                print("[KG_BUILD] Resume mode: loading existing node IDs from DB...")
                cur = conn.cursor()
                cur.execute("SELECT node_id FROM kg_nodes")
                self.node_id_set = {row[0] for row in cur.fetchall()}
                cur.close()

            # Fetch all chunks
            cur = conn.cursor()
            cur.execute("""
                SELECT chunk_id, text_content, header_text, metadata_json
                FROM chunks
                ORDER BY chunk_id
            """)
            chunks = cur.fetchall()
            cur.close()
            total = len(chunks)
            print(f"[KG_BUILD] Starting PARALLEL extraction (workers={workers}) over {total} chunks...")

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._build_single_chunk, i, total, chunk, resume): i
                    for i, chunk in enumerate(chunks, 1)
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"  [CRITICAL_THREAD_ERROR] {e}")

            # Build pgvector index on kg_nodes if enough nodes
            self._create_vector_index(conn)

            self._finish_log(conn, status="done")
            print(f"[KG_BUILD] ✅ Complete. Nodes: {self._nodes_created}, Edges: {self._edges_created}, Chunks: {self._chunks_processed}")

        except Exception as e:
            print(f"[KG_BUILD] ❌ Error: {e}")
            self._finish_log(conn, status="error", error=str(e))
        finally:
            conn.close()

    def _build_single_chunk(self, i: int, total: int, chunk: Dict, resume: bool):
        """Helper for parallel execution: handles one chunk extraction and DB write."""
        chunk_id   = chunk["chunk_id"]
        text       = chunk["text_content"] or ""
        header     = chunk["header_text"] or ""
        meta       = chunk["metadata_json"] or {}

        if isinstance(meta, str):
            try: meta = json.loads(meta)
            except: meta = {}

        if len(text.strip()) < 30:
            return

        # Use fresh connection for thread safety
        conn = get_conn()
        try:
            if resume:
                cur = conn.cursor()
                cur.execute(
                    "SELECT 1 FROM kg_nodes WHERE source_rule LIKE %s LIMIT 1",
                    (f"%{header[:40]}%",)
                )
                already_done = cur.fetchone()
                cur.close()
                if already_done:
                    return

            print(f"  [{i}/{total}] Processing: {header[:50]}...")
            data = self._extract_chunk(header, text)

            if data:
                cur = conn.cursor()
                source_rule = " > ".join(
                    str(v) for k, v in meta.items() if k.startswith("Header")
                ) if meta else header

                # Upsert nodes
                node_id_map: Dict[str, str] = {}
                for raw_node in data.get("nodes", []):
                    original_id = str(raw_node.get("id", "")).strip()
                    canon_id = self._upsert_node(cur, raw_node, source_rule)
                    node_id_map[original_id.upper()] = canon_id

                # Insert edges
                for raw_edge in data.get("edges", []):
                    src_raw = str(raw_edge.get("source", "")).strip().upper()
                    tgt_raw = str(raw_edge.get("target", "")).strip().upper()
                    relation = str(raw_edge.get("relation", "")).strip().upper()
                    conf     = float(raw_edge.get("confidence", 1.0))

                    src_id = node_id_map.get(src_raw, src_raw)
                    tgt_id = node_id_map.get(tgt_raw, tgt_raw)
                    self._insert_edge(cur, src_id, tgt_id, relation, conf, source_rule)

                conn.commit()
                cur.close()
                self._chunks_processed += 1
        except Exception as e:
            print(f"  [{i}/{total}] Error processing chunk: {e}")
        finally:
            conn.close()

    def _update_log(self, conn):
        if not self.log_id:
            return
        cur = conn.cursor()
        cur.execute(
            "UPDATE kg_build_log SET nodes_created=%s, edges_created=%s, chunks_processed=%s WHERE log_id=%s",
            (self._nodes_created, self._edges_created, self._chunks_processed, self.log_id)
        )
        conn.commit()
        cur.close()

    def _create_vector_index(self, conn):
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM kg_nodes WHERE embedding IS NOT NULL")
        count = cur.fetchone()[0]
        cur.close()
        if count >= 100:
            try:
                cur = conn.cursor()
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS kg_nodes_embed_idx ON kg_nodes
                    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10)
                """)
                conn.commit()
                cur.close()
                print(f"[KG_BUILD] pgvector index created on kg_nodes ({count} nodes).")
            except Exception as e:
                print(f"[KG_BUILD] Vector index creation skipped: {e}")


# ─── Enrichment pass: ATC + ICD codes ────────────────────────────────────────
class KG_Enricher:
    """
    Second-pass enrichment: adds ATC codes to DRUG nodes and ICD-10 codes
    to DIAGNOSIS nodes using a Gemini lookup.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        self.model = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0,
        )

    def _lookup_code(self, label: str, code_type: str) -> str:
        prompt = (f'For the Turkish medical term "{label}", return the standard {code_type} code. '
                  f'Return ONLY valid JSON: {{"code": "<code or empty string if unknown>"}}')
        try:
            r = self.model.invoke([HumanMessage(content=prompt)])
            raw = r.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            d = json.loads(raw)
            return d.get("code", "")
        except Exception:
            return ""

    def enrich(self):
        conn = get_conn()
        cur = conn.cursor()

        # Enrich DRUG nodes with ATC code
        cur.execute("SELECT node_id, label FROM kg_nodes WHERE type='DRUG' AND (atc_code='' OR atc_code IS NULL) LIMIT 100")
        drugs = cur.fetchall()
        cur.close()
        print(f"[KG_ENRICH] Enriching {len(drugs)} DRUG nodes with ATC codes...")
        for node in drugs:
            code = self._lookup_code(node["label"], "ATC")
            if code:
                c = conn.cursor()
                c.execute("UPDATE kg_nodes SET atc_code=%s WHERE node_id=%s", (code, node["node_id"]))
                conn.commit()
                c.close()
            time.sleep(0.5)

        # Enrich DIAGNOSIS nodes with ICD-10 code
        cur = conn.cursor()
        cur.execute("SELECT node_id, label FROM kg_nodes WHERE type='DIAGNOSIS' AND (icd_code='' OR icd_code IS NULL) LIMIT 100")
        diagnoses = cur.fetchall()
        cur.close()
        print(f"[KG_ENRICH] Enriching {len(diagnoses)} DIAGNOSIS nodes with ICD-10 codes...")
        for node in diagnoses:
            code = self._lookup_code(node["label"], "ICD-10")
            if code:
                c = conn.cursor()
                c.execute("UPDATE kg_nodes SET icd_code=%s WHERE node_id=%s", (code, node["node_id"]))
                conn.commit()
                c.close()
            time.sleep(0.5)

        conn.close()
        print("[KG_ENRICH] ✅ Enrichment complete.")


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--enrich-only", action="store_true", help="Only run enrichment pass")
    parser.add_argument("--no-clear", action="store_true", help="Don't clear existing KG before build")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint (skip already-processed chunks)")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers (default: 5)")
    args = parser.parse_args()

    if args.enrich_only:
        enricher = KG_Enricher()
        enricher.enrich()
    elif args.resume:
        # Resume: don't clear, just pick up where we left off
        print(f"[KG_BUILD] Resuming from checkpoint with {args.workers} workers...")
        builder = KG_Builder()
        builder.build(clear_existing=False, resume=True, workers=args.workers)
        enricher = KG_Enricher()
        enricher.enrich()
    else:
        builder = KG_Builder()
        builder.build(clear_existing=not args.no_clear, workers=args.workers)
        if not args.no_clear:
            enricher = KG_Enricher()
            enricher.enrich()
