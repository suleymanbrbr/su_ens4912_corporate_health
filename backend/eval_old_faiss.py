"""
eval_old_faiss.py
=================
Evaluates the OLD FAISS + SQLite system (no reranker) against 200 questions.
Uses the actual original sut_rag_core.py code from git commit 2b38285.

Steps:
1. Imports OLD sut_rag_core code from sut_rag_core_old_faiss.py
2. Calls populate_database() to rebuild the FAISS index + SQLite DB
3. Runs the same 200-question retrieval evaluation
4. Saves results to eval_results/old_system_real_results.json

Run inside docker container:
    python eval_old_faiss.py
"""

import os, sys, json, csv, time, math
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OUT_DIR  = Path("/app/eval_results")
OUT_DIR.mkdir(exist_ok=True)
EVAL_CSV = Path("/app/sut_questions.csv")
K_VALUES = [1, 3, 5, 10]

# Paths the old system uses (running from /app/old_system/)
OLD_DIR = Path("/app/old_system")
OLD_DIR.mkdir(exist_ok=True)

# ─── Load Questions ───────────────────────────────────────────────────────────
def load_questions():
    questions = []
    with open(EVAL_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                "id": row["ID"], "category": row["Kategori"],
                "question": row["Soru"], "answer": row["Cevap"],
                "source": row["Kaynak"].strip(),
            })
    return questions

# ─── Source Matching (same logic as eval_retrieval.py) ───────────────────────
def source_in_chunk(source_ref, chunk_metadata):
    if not source_ref:
        return False
    full_header = " ".join(str(v) for v in chunk_metadata.values()).lower()
    return source_ref.lower() in full_header

def calc_metrics(results):
    totals = {k: {"hits": 0, "rr": 0.0, "ndcg": 0.0, "prec": 0.0} for k in K_VALUES}
    n = len(results)
    for item in results:
        src = item["source"]
        retrieved = item["retrieved_chunks"]
        for k in K_VALUES:
            top_k = retrieved[:k]
            hit, rr_val, dcg, relevant = False, 0.0, 0.0, 0
            ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(1, k)))
            for rank, chunk in enumerate(top_k, start=1):
                if source_in_chunk(src, chunk.get("metadata", {})):
                    relevant += 1
                    if not hit:
                        hit = True
                        rr_val = 1.0 / rank
                    dcg += 1.0 / math.log2(rank + 1)
            ndcg = (dcg / ideal_dcg) if ideal_dcg > 0 else 0.0
            totals[k]["hits"]  += int(hit)
            totals[k]["rr"]    += rr_val
            totals[k]["ndcg"]  += ndcg
            totals[k]["prec"]  += relevant / k

    metrics = {}
    for k in K_VALUES:
        metrics[f"hit_rate@{k}"]  = round(totals[k]["hits"] / n, 4)
        metrics[f"mrr@{k}"]       = round(totals[k]["rr"]   / n, 4)
        metrics[f"ndcg@{k}"]      = round(totals[k]["ndcg"] / n, 4)
        metrics[f"precision@{k}"] = round(totals[k]["prec"] / n, 4)
    return metrics

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("OLD SYSTEM (FAISS + SQLite) — Real Evaluation")
    print("=" * 60)

    # Step 1: Import the old FAISS module
    # Set working dir to old_system so it creates its DB/index files there
    os.chdir(str(OLD_DIR))
    sys.path.insert(0, str(OLD_DIR))

    # Fix the DOCX path in old code — it references relative path
    DOCX_SRC = "/app/data/08.03.2025-Değişiklik Tebliği İşlenmiş Güncel 2013 SUT.docx"
    DOCX_DEST = str(OLD_DIR / "08.03.2025-Değişiklik Tebliği İşlenmiş Güncel 2013 SUT.docx")
    import shutil
    if not Path(DOCX_DEST).exists():
        shutil.copy2(DOCX_SRC, DOCX_DEST)
        print(f"[SETUP] Copied DOCX to old_system/")

    import importlib.util
    spec = importlib.util.spec_from_file_location("sut_rag_core_old", OLD_DIR / "sut_rag_core_old_faiss.py")
    old_core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_core)
    print("[SETUP] Old FAISS module loaded.")

    # Step 2: Build the old FAISS index + SQLite DB
    db_path   = OLD_DIR / "sut_knowledge_base.db"
    idx_path  = OLD_DIR / "sut_faiss.index"

    engine_old = old_core.SUT_RAG_Engine(llm_provider="google", model_name="gemini-2.5-flash")

    if not idx_path.exists():
        print("\n[BUILD] Building FAISS index + SQLite DB from DOCX...")
        print("        (This takes ~5-10 minutes for embedding generation)")
        engine_old.populate_database()
        print("[BUILD] Done.")
    else:
        print("[SETUP] Existing FAISS index found, loading...")
        engine_old.load_database()

    if not engine_old.faiss_index:
        print("[ERROR] FAISS index not loaded. Aborting.")
        sys.exit(1)

    print(f"[VERIFY] FAISS index has {engine_old.faiss_index.ntotal} vectors.")

    # Step 3: Run eval on all 200 questions
    print(f"\n[EVAL] Evaluating {len(load_questions())} questions with old FAISS system...")
    questions = load_questions()
    results = []
    latencies = []

    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['question'][:65]}...")
        t0 = time.time()
        try:
            chunks = engine_old._retrieve_chunks(q["question"], k=max(K_VALUES))
        except Exception as e:
            print(f"    [WARN] {e}")
            chunks = []
        elapsed = time.time() - t0
        latencies.append(elapsed)

        retrieved = [{"metadata": c.get("metadata", {}), "score": 0.0} for c in chunks]
        results.append({"question": q["question"], "source": q["source"], "retrieved_chunks": retrieved})

    metrics = calc_metrics(results)
    metrics["avg_latency_sec"] = round(sum(latencies) / len(latencies), 3)
    metrics["p95_latency_sec"] = round(sorted(latencies)[int(len(latencies) * 0.95)], 3)

    # Step 4: Save results
    output = {
        "evaluation_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "system": "FAISS IndexFlatL2 + SQLite, no reranker, same embedding model (paraphrase-multilingual-MiniLM-L12-v2)",
        "git_commit": "2b38285",
        "num_questions": len(questions),
        "k_values": K_VALUES,
        "metrics": metrics,
        "note": "REAL empirical measurement on old FAISS system rebuilt from original code (git commit 2b38285) and same DOCX file."
    }

    out_path = OUT_DIR / "old_system_real_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Step 5: Print comparison summary
    new_path = OUT_DIR / "retrieval_results.json"
    if new_path.exists():
        with open(new_path) as f:
            new_data = json.load(f)
        new_m = new_data["new_system"]["metrics"]

        print("\n" + "=" * 65)
        print("REAL COMPARISON: Old FAISS vs New pgvector + Cross-Encoder")
        print("=" * 65)
        print(f"\n{'Metric':<23} {'Old FAISS':>12} {'New pgvector':>14} {'Improvement':>13}")
        print("-" * 65)
        for k in K_VALUES:
            for base in ["hit_rate", "mrr", "ndcg", "precision"]:
                key = f"{base}@{k}"
                ov = metrics.get(key, 0)
                nv = new_m.get(key, 0)
                imp = round((nv - ov) / ov * 100, 1) if ov > 0 else 0
                flag = "✅" if imp > 0 else "🔴"
                print(f"  {key:<21} {ov:>12.4f} {nv:>14.4f} {flag} +{imp}%")
        print(f"\n  {'avg_latency_sec':<21} {metrics.get('avg_latency_sec',0):>12.3f}s {new_m.get('avg_latency_sec',0):>13.3f}s")
        print(f"  {'p95_latency_sec':<21} {metrics.get('p95_latency_sec',0):>12.3f}s {new_m.get('p95_latency_sec',0):>13.3f}s")

    print(f"\n[SAVED] {out_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
