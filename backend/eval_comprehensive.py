"""
eval_comprehensive.py
=====================
Kapsamlı RAG Konfigürasyon Test Süiti

Test 1: Chunking (4-level vs 6-level + strip_headers)
Test 2: Reranker (L-6 vs L-12 vs mmarco multilingual)
Test 3: HyDE (Hypothetical Document Embeddings)
Test 4: Embedding Model (MiniLM vs multilingual-e5-large)
Test 5: Sistem Prompt revizyonu etkisi

Usage:
    python eval_comprehensive.py [--tests reranker,hyde,chunking,prompt,embedding]
    python eval_comprehensive.py          # runs all tests

Results: eval_results/comprehensive_results.json
"""

import os, sys, json, csv, time, math, argparse, shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

OUT_DIR  = Path("/app/eval_results")
OUT_DIR.mkdir(exist_ok=True)
EVAL_CSV = Path("/app/sut_questions.csv")
K_VALUES = [1, 3, 5, 10]
LLM_SAMPLE = 30

sys.path.insert(0, "/app")

# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_questions(n=None):
    rows = []
    with open(EVAL_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append({
                "id": row["ID"], "category": row["Kategori"],
                "question": row["Soru"], "answer": row["Cevap"],
                "source": row["Kaynak"].strip()
            })
    return rows[:n] if n else rows

def source_in_chunk(source_ref, metadata):
    if not source_ref: return False
    header = " ".join(str(v) for v in metadata.values()).lower()
    return source_ref.lower() in header

def calc_metrics(results):
    totals = {k: {"hits": 0, "rr": 0.0, "ndcg": 0.0, "prec": 0.0} for k in K_VALUES}
    n = len(results)
    for item in results:
        src = item["source"]
        retrieved = item["retrieved_chunks"]
        for k in K_VALUES:
            top_k = retrieved[:k]
            hit, rr_val, dcg, rel = False, 0.0, 0.0, 0
            ideal_dcg = sum(1.0/math.log2(i+2) for i in range(min(1, k)))
            for rank, chunk in enumerate(top_k, start=1):
                if source_in_chunk(src, chunk.get("metadata", {})):
                    rel += 1
                    if not hit:
                        hit = True; rr_val = 1.0/rank
                    dcg += 1.0/math.log2(rank+1)
            totals[k]["hits"]  += int(hit)
            totals[k]["rr"]    += rr_val
            totals[k]["ndcg"]  += (dcg/ideal_dcg) if ideal_dcg > 0 else 0.0
            totals[k]["prec"]  += rel/k
    metrics = {}
    for k in K_VALUES:
        metrics[f"hit_rate@{k}"]  = round(totals[k]["hits"] / n, 4)
        metrics[f"mrr@{k}"]       = round(totals[k]["rr"]   / n, 4)
        metrics[f"ndcg@{k}"]      = round(totals[k]["ndcg"] / n, 4)
        metrics[f"precision@{k}"] = round(totals[k]["prec"] / n, 4)
    return metrics

def rouge_l(ref, hyp):
    rt = ref.lower().split(); ht = hyp.lower().split()
    if not rt or not ht: return 0.0
    m, n = len(rt), len(ht)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if rt[i-1]==ht[j-1] else max(dp[i-1][j],dp[i][j-1])
    lcs = dp[m][n]
    p = lcs/n if n else 0; r = lcs/m if m else 0
    return round(2*p*r/(p+r), 4) if p+r else 0.0

def eval_retrieval_with_engine(engine, questions, label=""):
    """Run retrieval eval using engine._retrieve_chunks"""
    print(f"\n[EVAL] {label} — {len(questions)} soru...")
    results, latencies = [], []
    for i, q in enumerate(questions, 1):
        if i % 10 == 0: print(f"  [{i}/{len(questions)}]")
        t0 = time.time()
        try:
            chunks = engine._retrieve_chunks(q["question"], k=max(K_VALUES))
        except Exception as e:
            print(f"    [WARN] {e}"); chunks = []
            try: engine.conn.rollback()
            except: pass
        latencies.append(time.time() - t0)
        retrieved = [{"metadata": c.get("metadata", {})} for c in chunks]
        results.append({"question": q["question"], "source": q["source"], "retrieved_chunks": retrieved})
    metrics = calc_metrics(results)
    metrics["avg_latency_sec"] = round(sum(latencies)/len(latencies), 3)
    metrics["p95_latency_sec"] = round(sorted(latencies)[int(len(latencies)*0.95)], 3)
    return metrics

def eval_retrieval_raw_pgvector(conn, embed_fn, questions, table="chunks", embed_col="embedding", text_col="text_content", meta_col="metadata_json"):
    """Run retrieval eval using raw pgvector (no reranker)"""
    print(f"\n[EVAL] Raw pgvector on '{table}' — {len(questions)} soru...")
    results, latencies = [], []
    for i, q in enumerate(questions, 1):
        if i % 10 == 0: print(f"  [{i}/{len(questions)}]")
        t0 = time.time()
        try:
            q_vec = embed_fn(q["question"])
            q_vec_str = "[" + ",".join(map(str, q_vec)) + "]"
            cur = conn.cursor()
            cur.execute(f"""
                SELECT {text_col}, {meta_col}
                FROM {table}
                ORDER BY {embed_col} <=> %s
                LIMIT %s
            """, (q_vec_str, max(K_VALUES)))
            rows = cur.fetchall()
            cur.close()
            chunks = []
            for row in rows:
                raw_meta = row[1]
                meta = raw_meta if isinstance(raw_meta, dict) else (json.loads(raw_meta) if raw_meta else {})
                chunks.append({"metadata": meta})
        except Exception as e:
            print(f"    [WARN] {e}"); chunks = []
            try: conn.rollback()
            except: pass
        latencies.append(time.time() - t0)
        results.append({"question": q["question"], "source": q["source"], "retrieved_chunks": chunks})
    metrics = calc_metrics(results)
    metrics["avg_latency_sec"] = round(sum(latencies)/len(latencies), 3)
    metrics["p95_latency_sec"] = round(sorted(latencies)[int(len(latencies)*0.95)], 3)
    return metrics

def load_baseline():
    p = OUT_DIR / "retrieval_results.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        return d.get("new_system", {}).get("metrics", {})
    return {}

def print_comparison(label_a, metrics_a, label_b, metrics_b):
    print(f"\n{'Metric':<22} {label_a:>16} {label_b:>16} {'Improvement':>12}")
    print("-"*68)
    for k in K_VALUES:
        for base in ["hit_rate", "mrr", "ndcg"]:
            key = f"{base}@{k}"
            a = metrics_a.get(key, 0); b = metrics_b.get(key, 0)
            imp = round((b-a)/a*100, 1) if a else 0
            flag = "✅" if imp > 0 else ("🔴" if imp < 0 else "=")
            print(f"  {key:<20} {a:>16.4f} {b:>16.4f} {flag} {imp:+.1f}%")
    print(f"  {'avg_latency':<20} {metrics_a.get('avg_latency_sec',0):>15.3f}s {metrics_b.get('avg_latency_sec',0):>15.3f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: RERANKER COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
def test_rerankers(questions, baseline_metrics):
    print("\n" + "="*65)
    print("TEST 1: RERANKER KARŞILAŞTIRMASI")
    print("="*65)

    from sentence_transformers import CrossEncoder
    import psycopg2

    RERANKERS = [
        ("L6-MiniLM (mevcut)", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        ("L12-MiniLM",         "cross-encoder/ms-marco-MiniLM-L-12-v2"),
        ("mMarco-Multilingual", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
    ]

    from sut_rag_core import SUT_RAG_Engine
    engine = SUT_RAG_Engine()
    engine.load_database()
    embed_fn = engine.embeddings_model.embed_query

    results_per_reranker = {}

    for name, model_id in RERANKERS:
        print(f"\n[RERANKER] Testing: {name} ({model_id})")
        try:
            reranker = CrossEncoder(model_id, device='cpu')
        except Exception as e:
            print(f"  [ERROR] Could not load {model_id}: {e}")
            continue

        # Override engine's reranker
        engine.reranker = reranker

        metrics = eval_retrieval_with_engine(engine, questions, label=name)
        results_per_reranker[name] = {"model": model_id, "metrics": metrics}
        print(f"  Hit@5: {metrics.get('hit_rate@5',0):.4f} | MRR@5: {metrics.get('mrr@5',0):.4f} | Lat: {metrics.get('avg_latency_sec',0):.3f}s")

    # Print summary
    print("\n--- RERANKER KARŞILAŞTIRMA ÖZETI ---")
    baseline_name = "L6-MiniLM (mevcut)"
    b_metrics = results_per_reranker.get(baseline_name, {}).get("metrics", baseline_metrics)
    for name, data in results_per_reranker.items():
        if name == baseline_name: continue
        print_comparison(baseline_name, b_metrics, name, data["metrics"])

    return results_per_reranker

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: HyDE (Hypothetical Document Embeddings)
# ═══════════════════════════════════════════════════════════════════════════════
def test_hyde(questions, baseline_metrics):
    print("\n" + "="*65)
    print("TEST 2: HyDE (Hypothetical Document Embeddings)")
    print("="*65)
    print("Akış: Soru → LLM(hipotez cevap) → embed(hipotez) → pgvector")

    from sut_rag_core import SUT_RAG_Engine
    from langchain_core.messages import HumanMessage

    engine = SUT_RAG_Engine()
    engine.load_database()

    HYDE_PROMPT = """Sen SUT (Sağlık Uygulama Tebliği) uzmanısın.
Aşağıdaki soruya SUT'ta bulunabilecek tipik bir cevap paragrafı yaz.
Gerçek olmak zorunda değil, sadece ilgili SUT terminolojisini kullan.
Kısa tut (2-3 cümle).

Soru: {question}

SUT'taki Tipik Cevap:"""

    results_hyde, results_direct = [], []
    latencies_hyde, latencies_direct = [], []

    for i, q in enumerate(questions, 1):
        if i % 10 == 0: print(f"  [{i}/{len(questions)}] HyDE eval...")

        # Direct embedding (baseline)
        t0 = time.time()
        try:
            direct_chunks = engine._retrieve_chunks(q["question"], k=max(K_VALUES))
        except:
            direct_chunks = []
            try: engine.conn.rollback()
            except: pass
        latencies_direct.append(time.time()-t0)
        results_direct.append({
            "question": q["question"], "source": q["source"],
            "retrieved_chunks": [{"metadata": c.get("metadata", {})} for c in direct_chunks]
        })

        # HyDE embedding
        t0 = time.time()
        try:
            # Step 1: Generate hypothetical answer
            hyde_resp = engine.llm.invoke([HumanMessage(content=HYDE_PROMPT.format(question=q["question"]))])
            hypothetical_answer = hyde_resp.content.strip()

            # Step 2: Embed hypothetical answer instead of question
            q_vec = engine.embeddings_model.embed_query(hypothetical_answer)
            q_vec_str = "[" + ",".join(map(str, q_vec)) + "]"

            # Step 3: pgvector search with hypothetical embedding
            cur = engine.conn.cursor()
            cur.execute("""
                SELECT chunk_id, text_content, metadata_json
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (q_vec_str, max(K_VALUES)*3))
            rows = cur.fetchall()
            cur.close()

            # Step 4: Rerank
            candidates = []
            for row in rows:
                meta = row[2] if isinstance(row[2], dict) else json.loads(row[2] or '{}')
                candidates.append({"text": row[1], "metadata": meta, "id": row[0]})

            if candidates:
                pairs = [[q["question"], doc['text']] for doc in candidates]
                scores = engine.reranker.predict(pairs)
                for doc, score in zip(candidates, scores):
                    doc['score'] = score
                candidates.sort(key=lambda x: x['score'], reverse=True)

            hyde_chunks = candidates[:max(K_VALUES)]
        except Exception as e:
            print(f"    [WARN] HyDE failed: {e}"); hyde_chunks = []
            try: engine.conn.rollback()
            except: pass
        latencies_hyde.append(time.time()-t0)
        results_hyde.append({
            "question": q["question"], "source": q["source"],
            "retrieved_chunks": [{"metadata": c.get("metadata", {})} for c in hyde_chunks]
        })

        time.sleep(0.3)  # Rate limit for LLM

    metrics_direct = calc_metrics(results_direct)
    metrics_direct["avg_latency_sec"] = round(sum(latencies_direct)/len(latencies_direct), 3)

    metrics_hyde = calc_metrics(results_hyde)
    metrics_hyde["avg_latency_sec"] = round(sum(latencies_hyde)/len(latencies_hyde), 3)

    print_comparison("Direct Embed", metrics_direct, "HyDE", metrics_hyde)

    return {"direct": {"metrics": metrics_direct}, "hyde": {"metrics": metrics_hyde}}

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: CHUNKING (4-level vs 6-level)
# ═══════════════════════════════════════════════════════════════════════════════
def test_chunking(questions, baseline_metrics):
    print("\n" + "="*65)
    print("TEST 3: CHUNKING KARŞİLAŞTIRMASI")
    print("4-level (mevcut, strip_headers=True) vs 6-level (strip_headers=False)")
    print("="*65)

    import psycopg2
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    import regex
    from sut_rag_core import SUT_RAG_Engine

    engine = SUT_RAG_Engine()
    engine.load_database()

    # ── Step 1: Create 6-level chunks from existing MD file ──
    DOCX_PATH = "/app/data/08.03.2025-Değişiklik Tebliği İşlenmiş Güncel 2013 SUT.docx"
    MD_PATH   = "/app/old_system/temp_cleaned_sut.md"

    # Use the existing MD if available from old_system rebuild, else create
    if not Path(MD_PATH).exists():
        print("[CHUNKING] No existing MD, converting DOCX...")
        import subprocess
        subprocess.run(["pandoc", DOCX_PATH, "-o", MD_PATH, "--wrap=none"], check=True)

    with open(MD_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # Clean
    text = regex.sub(r'~~.*?~~', '', text)
    text = regex.sub(r'►', '', text)
    def h_repl(m):
        depth = m.group(1).count('.') + 1
        hashes = '#' * min(6, depth)
        return f"{hashes} {m.group(0)}"
    text = regex.sub(r"^\*\*((\\d+\\.)+\\d+[\\.\\d\\w-]*)\\s*-*\\s*([^ \\n\\*]+.*?)\\*\\*",
        h_repl, text, flags=regex.MULTILINE)

    # 6-level split WITH strip_headers=False (old system style)
    headers_6 = [
        ("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"),
        ("####", "Header 4"), ("#####", "Header 5"), ("######", "Header 6")
    ]
    splitter_6 = MarkdownHeaderTextSplitter(headers_to_split_on=headers_6, strip_headers=False)
    chunks_6level = splitter_6.split_text(text)
    print(f"[CHUNKING] 6-level split produced: {len(chunks_6level)} chunks")

    # ── Step 2: Check if chunks_v2 table exists, create if not ──
    conn = engine.conn
    cur = conn.cursor()
    cur.execute("SELECT EXISTS(SELECT FROM pg_tables WHERE schemaname='public' AND tablename='chunks_v2')")
    exists = cur.fetchone()[0]
    cur.close()
    conn.commit()

    if not exists:
        print("[CHUNKING] Creating chunks_v2 table with 6-level chunks...")
        # Get embedding dimension from existing chunks
        cur = conn.cursor()
        cur.execute("SELECT vector_dims(embedding) FROM chunks LIMIT 1")
        dim = cur.fetchone()[0]
        cur.close()
        conn.commit()

        cur = conn.cursor()
        cur.execute(f"""
            CREATE TABLE chunks_v2 (
                chunk_id SERIAL PRIMARY KEY,
                text_content TEXT,
                header_text TEXT,
                metadata_json JSONB,
                embedding vector({dim})
            )
        """)
        conn.commit()
        cur.close()

        # Embed and insert 6-level chunks
        print(f"[CHUNKING] Embedding {len(chunks_6level)} chunks (this takes a few minutes)...")
        batch_size = 32
        inserted = 0
        for i in range(0, len(chunks_6level), batch_size):
            batch = chunks_6level[i:i+batch_size]
            texts = [c.page_content for c in batch]
            metas = [c.metadata for c in batch]

            embeddings = engine.embeddings_model.embed_documents(texts)

            cur = conn.cursor()
            for text_c, meta, emb in zip(texts, metas, embeddings):
                header_text = " > ".join([v for v in meta.values() if isinstance(v, str)])
                emb_str = "[" + ",".join(map(str, emb)) + "]"
                cur.execute("""
                    INSERT INTO chunks_v2 (text_content, header_text, metadata_json, embedding)
                    VALUES (%s, %s, %s, %s)
                """, (text_c, header_text, json.dumps(meta, ensure_ascii=False), emb_str))
                inserted += 1
            conn.commit()
            cur.close()
            print(f"  Embedded {min(i+batch_size, len(chunks_6level))}/{len(chunks_6level)}")

        print(f"[CHUNKING] Inserted {inserted} chunks into chunks_v2")
    else:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks_v2")
        cnt = cur.fetchone()[0]
        cur.close()
        print(f"[CHUNKING] chunks_v2 already exists with {cnt} chunks, using cached.")

    # ── Step 3: Eval 4-level (baseline) ──
    metrics_4level = eval_retrieval_with_engine(engine, questions, "4-level (mevcut)")

    # ── Step 4: Eval 6-level (raw pgvector, no rerank for fair test) ──
    # Also test WITH reranker using the 6-level chunks
    print("\n[CHUNKING] Evaluating 6-level chunks (no reranker, same embedding)...")
    metrics_6level_raw = eval_retrieval_raw_pgvector(
        conn, engine.embeddings_model.embed_query, questions,
        table="chunks_v2", embed_col="embedding",
        text_col="text_content", meta_col="metadata_json"
    )

    print("\n[CHUNKING] Evaluating 6-level chunks WITH reranker...")
    # Create a temporary engine-like eval with 6-level table
    results_6r, latencies_6r = [], []
    for i, q in enumerate(questions, 1):
        if i % 10 == 0: print(f"  [{i}/{len(questions)}]")
        t0 = time.time()
        try:
            q_vec = engine.embeddings_model.embed_query(q["question"])
            q_vec_str = "[" + ",".join(map(str, q_vec)) + "]"
            cur = conn.cursor()
            cur.execute("""
                SELECT text_content, metadata_json
                FROM chunks_v2
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (q_vec_str, max(K_VALUES)*3))
            rows = cur.fetchall()
            cur.close()

            candidates = []
            for row in rows:
                meta = row[1] if isinstance(row[1], dict) else json.loads(row[1] or '{}')
                candidates.append({"text": row[0], "metadata": meta})

            if candidates:
                pairs = [[q["question"], d['text']] for d in candidates]
                scores = engine.reranker.predict(pairs)
                for d, s in zip(candidates, scores): d['score'] = s
                candidates.sort(key=lambda x: x['score'], reverse=True)

            chunks_out = candidates[:max(K_VALUES)]
        except Exception as e:
            print(f"    [WARN] {e}"); chunks_out = []
            try: conn.rollback()
            except: pass
        latencies_6r.append(time.time()-t0)
        results_6r.append({
            "question": q["question"], "source": q["source"],
            "retrieved_chunks": [{"metadata": c.get("metadata", {})} for c in chunks_out]
        })

    metrics_6level_reranked = calc_metrics(results_6r)
    metrics_6level_reranked["avg_latency_sec"] = round(sum(latencies_6r)/len(latencies_6r), 3)

    print("\n--- 4-level vs 6-level (raw) ---")
    print_comparison("4-level+rerank", metrics_4level, "6-level (no rerank)", metrics_6level_raw)
    print("\n--- 4-level+rerank vs 6-level+rerank ---")
    print_comparison("4-level+rerank", metrics_4level, "6-level+rerank", metrics_6level_reranked)

    return {
        "4level_with_reranker": {"metrics": metrics_4level},
        "6level_no_reranker":   {"metrics": metrics_6level_raw},
        "6level_with_reranker": {"metrics": metrics_6level_reranked},
    }

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: EMBEDDING MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
def test_embedding_models(questions, baseline_metrics):
    print("\n" + "="*65)
    print("TEST 4: EMBEDDING MODEL KARŞILAŞTIRMASI")
    print("paraphrase-MiniLM (mevcut) vs multilingual-e5-large")
    print("="*65)

    import psycopg2
    from langchain_huggingface import HuggingFaceEmbeddings
    from sut_rag_core import SUT_RAG_Engine

    engine = SUT_RAG_Engine()
    engine.load_database()

    # Check baseline metrics (already have from retrieval_results.json)
    print("[EMBEDDING] Baseline (paraphrase-MiniLM) metrics already computed.")
    print(f"  Hit@5: {baseline_metrics.get('hit_rate@5',0):.4f} | MRR@5: {baseline_metrics.get('mrr@5',0):.4f}")

    # Load multilingual-e5-large
    print("\n[EMBEDDING] Loading intfloat/multilingual-e5-large (~560MB)...")
    try:
        e5_model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        # E5 models need query/passage prefixes
        def embed_query_e5(text):
            return e5_model.embed_query("query: " + text)
        def embed_docs_e5(texts):
            return e5_model.embed_documents(["passage: " + t for t in texts])

        print("[EMBEDDING] E5-large loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Could not load E5 model: {e}")
        return {"error": str(e)}

    conn = engine.conn

    # Check if chunks_e5 table exists
    cur = conn.cursor()
    cur.execute("SELECT EXISTS(SELECT FROM pg_tables WHERE schemaname='public' AND tablename='chunks_e5')")
    exists = cur.fetchone()[0]
    cur.close()
    conn.commit()

    if not exists:
        # Get E5 embedding dimension first
        test_emb = embed_docs_e5(["test"])[0]
        e5_dim = len(test_emb)
        print(f"[EMBEDDING] E5 embedding dimension: {e5_dim}")

        # Create chunks_e5 table
        cur = conn.cursor()
        cur.execute(f"""
            CREATE TABLE chunks_e5 (
                chunk_id SERIAL PRIMARY KEY,
                text_content TEXT,
                header_text TEXT,
                metadata_json JSONB,
                embedding vector({e5_dim})
            )
        """)
        conn.commit()
        cur.close()

        # Copy text from original chunks, re-embed with E5
        cur = conn.cursor()
        cur.execute("SELECT text_content, header_text, metadata_json FROM chunks ORDER BY chunk_id")
        orig_chunks = cur.fetchall()
        cur.close()
        print(f"[EMBEDDING] Re-embedding {len(orig_chunks)} chunks with E5-large...")

        batch_size = 16  # E5-large is bigger, smaller batches
        for i in range(0, len(orig_chunks), batch_size):
            batch = orig_chunks[i:i+batch_size]
            texts = [r[0] for r in batch]
            embeddings = embed_docs_e5(texts)
            cur = conn.cursor()
            for (text_c, header, meta), emb in zip(batch, embeddings):
                emb_str = "[" + ",".join(map(str, emb)) + "]"
                meta_str = json.dumps(meta, ensure_ascii=False) if isinstance(meta, dict) else (meta if isinstance(meta, str) else json.dumps({}))
                cur.execute("""
                    INSERT INTO chunks_e5 (text_content, header_text, metadata_json, embedding)
                    VALUES (%s, %s, %s, %s)
                """, (text_c, header, meta_str, emb_str))
            conn.commit()
            cur.close()
            print(f"  Re-embedded {min(i+batch_size, len(orig_chunks))}/{len(orig_chunks)}")

        print("[EMBEDDING] chunks_e5 table populated.")
    else:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks_e5")
        cnt = cur.fetchone()[0]
        cur.close()
        print(f"[EMBEDDING] chunks_e5 already exists with {cnt} chunks.")

    # Eval E5 without reranker
    print("\n[EMBEDDING] Evaluating E5-large (no reranker)...")
    metrics_e5_raw = eval_retrieval_raw_pgvector(
        conn, embed_query_e5, questions,
        table="chunks_e5", embed_col="embedding",
        text_col="text_content", meta_col="metadata_json"
    )

    # Eval E5 with reranker
    print("\n[EMBEDDING] Evaluating E5-large WITH reranker...")
    results_e5r, latencies_e5r = [], []
    for i, q in enumerate(questions, 1):
        if i % 10 == 0: print(f"  [{i}/{len(questions)}]")
        t0 = time.time()
        try:
            q_vec = embed_query_e5(q["question"])
            q_vec_str = "[" + ",".join(map(str, q_vec)) + "]"
            cur = conn.cursor()
            cur.execute("""
                SELECT text_content, metadata_json FROM chunks_e5
                ORDER BY embedding <=> %s LIMIT %s
            """, (q_vec_str, max(K_VALUES)*3))
            rows = cur.fetchall()
            cur.close()
            candidates = []
            for row in rows:
                meta = row[1] if isinstance(row[1], dict) else json.loads(row[1] or '{}')
                candidates.append({"text": row[0], "metadata": meta})
            if candidates:
                pairs = [[q["question"], d['text']] for d in candidates]
                scores = engine.reranker.predict(pairs)
                for d, s in zip(candidates, scores): d['score'] = s
                candidates.sort(key=lambda x: x['score'], reverse=True)
            chunks_out = candidates[:max(K_VALUES)]
        except Exception as e:
            print(f"    [WARN] {e}"); chunks_out = []
            try: conn.rollback()
            except: pass
        latencies_e5r.append(time.time()-t0)
        results_e5r.append({
            "question": q["question"], "source": q["source"],
            "retrieved_chunks": [{"metadata": c.get("metadata",{})} for c in chunks_out]
        })

    metrics_e5_reranked = calc_metrics(results_e5r)
    metrics_e5_reranked["avg_latency_sec"] = round(sum(latencies_e5r)/len(latencies_e5r), 3)

    print_comparison("MiniLM+rerank (baseline)", baseline_metrics, "E5-large (no rerank)", metrics_e5_raw)
    print_comparison("MiniLM+rerank (baseline)", baseline_metrics, "E5-large+rerank", metrics_e5_reranked)

    return {
        "minilm_baseline":  {"model": "paraphrase-multilingual-MiniLM-L12-v2", "metrics": baseline_metrics},
        "e5_large_no_rerank": {"model": "multilingual-e5-large", "metrics": metrics_e5_raw},
        "e5_large_reranked":  {"model": "multilingual-e5-large + reranker", "metrics": metrics_e5_reranked},
    }

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: SYSTEM PROMPT REVISION
# ═══════════════════════════════════════════════════════════════════════════════
def test_system_prompt(baseline_metrics):
    print("\n" + "="*65)
    print("TEST 5: SİSTEM PROMPT REVİZYONU")
    print("Mevcut agentic prompt vs. geliştirilmiş (madde numarası + kısa cevap)")
    print("="*65)

    from sut_rag_core import SUT_RAG_Engine
    from langchain_core.messages import HumanMessage, SystemMessage

    engine = SUT_RAG_Engine()
    engine.load_database()

    questions = load_questions(LLM_SAMPLE)

    IMPROVED_TOOL_SCHEMA_ADDITION = """
CEVAP FORMATI (finish tool için):
- İLK CÜMLE: Soruya doğrudan, net, kısa cevap ver (evet/hayır + kısa açıklama).
- KAYNAK: İlgili SUT madde numarasını [Madde X.X.X] olarak belirt.
- DETAY: Gerekirse 1-2 cümle ek bilgi. Fazladan açıklama yapma.

ÖRNEK:
S: "Diyaliz hastası haftada kaç seans hemodiyalize girebilir?"
C: "Haftada en fazla 3 seans hemodiyaliz yapılabilir. [Madde 5.3.2.A]
    İstisnai durumlarda bu sayı artırılabilir."
"""

    rouge_old, rouge_new = 0.0, 0.0
    faithful_old, faithful_new = 0.0, 0.0
    judge_n = 0

    per_question_results = []

    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{LLM_SAMPLE}] {q['question'][:55]}...")

        # Get context chunks
        try:
            chunks = engine._retrieve_chunks(q["question"], k=5)
            ctx = "\n\n".join([c.get("text","") for c in chunks[:3]])
        except:
            ctx = ""; engine.conn.rollback()

        # OLD prompt (current TOOL_REACTION_PROMPT style — simulate a finish answer)
        old_sys = f"""Sen SUT uzmanısın. Aşağıdaki bağlamı kullanarak soruyu Türkçe cevapla.

BAĞLAM:
{ctx[:2000]}"""
        # NEW prompt with improved format guidance
        new_sys = f"""Sen SUT uzmanısın. Aşağıdaki bağlamı kullanarak soruyu Türkçe cevapla.

BAĞLAM:
{ctx[:2000]}

{IMPROVED_TOOL_SCHEMA_ADDITION}"""

        ans_old = ans_new = ""
        try:
            r = engine.llm.invoke([SystemMessage(content=old_sys), HumanMessage(content=q["question"])])
            ans_old = r.content
        except Exception as e:
            ans_old = f"[ERROR] {e}"

        time.sleep(0.5)

        try:
            r = engine.llm.invoke([SystemMessage(content=new_sys), HumanMessage(content=q["question"])])
            ans_new = r.content
        except Exception as e:
            ans_new = f"[ERROR] {e}"

        rl_old = rouge_l(q["answer"], ans_old)
        rl_new = rouge_l(q["answer"], ans_new)
        rouge_old += rl_old
        rouge_new += rl_new

        # LLM judge faithfulness
        try:
            import re as _re
            judge = engine.llm.invoke([HumanMessage(content=f"""Rate faithfulness 0.0-1.0:
Q: {q['question']}
Reference: {q['answer']}
Answer A (old prompt): {ans_old[:300]}
Answer B (new prompt): {ans_new[:300]}
Respond ONLY JSON: {{"faithful_a": 0.8, "faithful_b": 0.9}}""")])
            m2 = _re.search(r'\{.*?\}', judge.content, _re.DOTALL)
            if m2:
                jd = json.loads(m2.group())
                faithful_old += jd.get("faithful_a", 0)
                faithful_new += jd.get("faithful_b", 0)
                judge_n += 1
        except: pass

        per_question_results.append({
            "question": q["question"], "reference": q["answer"],
            "ans_old": ans_old, "ans_new": ans_new,
            "rouge_old": rl_old, "rouge_new": rl_new
        })
        time.sleep(1)

    summary = {
        "num_questions": LLM_SAMPLE,
        "old_prompt": {
            "avg_rouge_l": round(rouge_old/LLM_SAMPLE, 4),
            "avg_faithfulness": round(faithful_old/max(judge_n,1), 4),
            "description": "Current agentic finish prompt (no format guidance)"
        },
        "new_prompt": {
            "avg_rouge_l": round(rouge_new/LLM_SAMPLE, 4),
            "avg_faithfulness": round(faithful_new/max(judge_n,1), 4),
            "description": "Improved prompt with direct answer + [Madde X.X.X] citation format"
        },
        "rouge_l_gain_%": round((rouge_new-rouge_old)/max(rouge_old,0.001)*100, 1),
        "faithfulness_gain_%": round((faithful_new-faithful_old)/max(faithful_old,0.001)*100, 1),
    }

    print(f"\n  ROUGE-L old: {summary['old_prompt']['avg_rouge_l']:.4f}")
    print(f"  ROUGE-L new: {summary['new_prompt']['avg_rouge_l']:.4f} ({summary['rouge_l_gain_%']:+.1f}%)")
    print(f"  Faithfulness old: {summary['old_prompt']['avg_faithfulness']:.4f}")
    print(f"  Faithfulness new: {summary['new_prompt']['avg_faithfulness']:.4f} ({summary['faithfulness_gain_%']:+.1f}%)")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", default="reranker,hyde,chunking,prompt,embedding",
                        help="Comma-separated list of tests to run")
    args = parser.parse_args()
    tests_to_run = set(args.tests.split(","))

    questions = load_questions()
    baseline_metrics = load_baseline()

    print("="*65)
    print("KAPSAMLI RAG KONFIGÜRASYON TEST SÜİTİ")
    print(f"Soru sayısı: {len(questions)} | Baseline Hit@5: {baseline_metrics.get('hit_rate@5',0):.4f}")
    print("="*65)

    all_results = {
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_questions": len(questions),
        "baseline": baseline_metrics,
        "tests": {}
    }

    if "reranker" in tests_to_run:
        all_results["tests"]["reranker_comparison"] = test_rerankers(questions, baseline_metrics)

    if "hyde" in tests_to_run:
        all_results["tests"]["hyde_comparison"] = test_hyde(questions[:100], baseline_metrics)  # 100q for speed

    if "chunking" in tests_to_run:
        all_results["tests"]["chunking_comparison"] = test_chunking(questions, baseline_metrics)

    if "prompt" in tests_to_run:
        all_results["tests"]["prompt_comparison"] = test_system_prompt(baseline_metrics)

    if "embedding" in tests_to_run:
        all_results["tests"]["embedding_comparison"] = test_embedding_models(questions, baseline_metrics)

    # Save
    out_path = OUT_DIR / "comprehensive_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*65}")
    print(f"[SAVED] Tüm sonuçlar: {out_path}")
    print("="*65)

if __name__ == "__main__":
    main()
