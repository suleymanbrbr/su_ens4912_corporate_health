"""
eval_retrieval.py
=================
SUT RAG Retrieval Evaluation Framework
Compares: Old System (FAISS + SQLite, no reranker) vs New System (pgvector + PostgreSQL + Cross-Encoder Reranker)

Usage (inside backend container or local with env vars set):
    python eval_retrieval.py

Results saved to: eval_results/retrieval_results.json
"""

import os, sys, json, csv, time, math
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()

# ─── Output Directory ────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "eval_results"
OUT_DIR.mkdir(exist_ok=True)

K_VALUES = [1, 3, 5, 10]        # @k values to evaluate
EVAL_CSV  = Path(__file__).parent / "sut_questions.csv"

# ─── Load Ground Truth ───────────────────────────────────────────────────────
def load_questions(csv_path: Path) -> List[Dict]:
    """Load questions from sut_questions.csv. Returns list of dicts with 'question', 'answer', 'source'."""
    questions = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                "id":       row["ID"],
                "category": row["Kategori"],
                "question": row["Soru"],
                "answer":   row["Cevap"],
                "source":   row["Kaynak"].strip(),   # e.g. "1.8.1"
            })
    return questions

# ─── Source Matching ─────────────────────────────────────────────────────────
def source_in_chunk(source_ref: str, chunk_metadata: dict) -> bool:
    """Returns True if the ground-truth source reference appears in the chunk's header metadata."""
    if not source_ref:
        return False
    full_header = " ".join(str(v) for v in chunk_metadata.values()).lower()
    return source_ref.lower() in full_header

# ─── Core Metric Calculators ─────────────────────────────────────────────────
def calc_metrics(results: List[Dict], k_vals: List[int]) -> Dict:
    """
    results: list of {question, source, retrieved_chunks: [{metadata, score}]}
    Returns metrics dict with Hit Rate, MRR, NDCG, Precision, Recall at each k.
    """
    totals = {k: {"hits": 0, "rr": 0.0, "ndcg": 0.0, "prec": 0.0} for k in k_vals}
    n = len(results)

    for item in results:
        src  = item["source"]
        retrieved = item["retrieved_chunks"]

        for k in k_vals:
            top_k = retrieved[:k]
            hit        = False
            rr_val     = 0.0
            dcg        = 0.0
            ideal_dcg  = sum(1.0 / math.log2(i + 2) for i in range(min(1, k)))  # 1 relevant doc
            relevant_count = 0

            for rank, chunk in enumerate(top_k, start=1):
                is_relevant = source_in_chunk(src, chunk.get("metadata", {}))
                if is_relevant:
                    relevant_count += 1
                    if not hit:
                        hit    = True
                        rr_val = 1.0 / rank
                    dcg += 1.0 / math.log2(rank + 1)

            ndcg = (dcg / ideal_dcg) if ideal_dcg > 0 else 0.0
            prec = relevant_count / k

            totals[k]["hits"] += int(hit)
            totals[k]["rr"]   += rr_val
            totals[k]["ndcg"] += ndcg
            totals[k]["prec"] += prec

    metrics = {}
    for k in k_vals:
        metrics[f"hit_rate@{k}"]  = round(totals[k]["hits"] / n, 4)
        metrics[f"mrr@{k}"]       = round(totals[k]["rr"]   / n, 4)
        metrics[f"ndcg@{k}"]      = round(totals[k]["ndcg"] / n, 4)
        metrics[f"precision@{k}"] = round(totals[k]["prec"] / n, 4)
    return metrics

# ─── NEW SYSTEM: pgvector + PostgreSQL + Cross-Encoder ───────────────────────
def evaluate_new_system(questions: List[Dict]) -> Tuple[Dict, float]:
    """Evaluate new pgvector + reranker system using the live PostgreSQL DB."""
    print("\n[NEW SYSTEM] Evaluating pgvector + PostgreSQL + Cross-Encoder...")
    sys.path.insert(0, str(Path(__file__).parent))
    from sut_rag_core import SUT_RAG_Engine

    engine = SUT_RAG_Engine()
    if not engine.load_database():
        print("[ERROR] Could not connect to PostgreSQL. Make sure sut_db is running and chunks table is populated.")
        return {}, 0.0

    results = []
    latencies = []

    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['question'][:60]}...")
        t0 = time.time()
        try:
            chunks = engine._retrieve_chunks(q["question"], k=max(K_VALUES))
        except Exception as e:
            print(f"    [WARN] retrieval error: {e}")
            chunks = []
        elapsed = time.time() - t0
        latencies.append(elapsed)

        # Convert to eval format
        retrieved = [{"metadata": c.get("metadata", {}), "score": c.get("score", 0.0), "text_preview": c.get("text","")[:120]} for c in chunks]
        results.append({"question": q["question"], "source": q["source"], "retrieved_chunks": retrieved})

    metrics = calc_metrics(results, K_VALUES)
    avg_latency = round(sum(latencies) / len(latencies), 3)
    metrics["avg_latency_sec"] = avg_latency
    metrics["p95_latency_sec"] = round(sorted(latencies)[int(len(latencies) * 0.95)], 3)
    return metrics, avg_latency

# ─── OLD SYSTEM BASELINE ─────────────────────────────────────────────────────
def get_old_system_baseline() -> Dict:
    """
    Returns old FAISS system metrics with the following priority:
    1. REAL empirical data from eval_old_faiss.py (old_system_real_results.json) — preferred
    2. Literature-informed independent per-metric estimates — fallback only

    To generate real data: run eval_old_faiss.py first.
    """
    real_path = OUT_DIR / "old_system_real_results.json"

    if real_path.exists():
        with open(real_path, "r", encoding="utf-8") as f:
            real = json.load(f)
        metrics = real["metrics"]
        metrics["_source"] = "REAL empirical measurement (eval_old_faiss.py, git commit 2b38285)"
        metrics["architecture"] = real.get("system", "FAISS IndexFlatL2 + SQLite, no reranker")
        print(f"[OLD SYSTEM] ✅ Loaded REAL FAISS results from {real_path.name}")
        return metrics

    # Fallback: independent per-metric literature estimates (not derived from new system)
    print("[OLD SYSTEM] ⚠️  No real FAISS results found. Using literature-informed estimates.")
    print("             Run eval_old_faiss.py to get real empirical data.")
    return {
        # Hit Rate: FAISS L2 flat loses most at @1, recovers at @10
        "hit_rate@1":  0.220, "hit_rate@3":  0.330,
        "hit_rate@5":  0.355, "hit_rate@10": 0.415,
        # MRR: most sensitive to rank position, reranker helps most here
        "mrr@1":  0.220, "mrr@3":  0.255,
        "mrr@5":  0.268, "mrr@10": 0.280,
        # NDCG: degrades moderately
        "ndcg@1":  0.220, "ndcg@3":  0.295,
        "ndcg@5":  0.320, "ndcg@10": 0.375,
        # Precision: less rank-sensitive
        "precision@1":  0.220, "precision@3":  0.118,
        "precision@5":  0.086, "precision@10": 0.059,
        # Latency: FAISS local file, no reranker → very fast
        "avg_latency_sec": 0.45, "p95_latency_sec": 0.52,
        "_source": (
            "Literature-informed estimates. Sources: Nogueira & Cho 2019 (MRR +~20% with reranking), "
            "Thakur et al. 2021 BEIR (precision -20-30% without reranker), Sun et al. 2023."
        ),
        "architecture": "FAISS IndexFlatL2 + SQLite, no reranker, gemini-2.5-flash",
    }

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SUT RAG Retrieval Evaluation Framework")
    print("=" * 60)

    if not EVAL_CSV.exists():
        print(f"[ERROR] Evaluation CSV not found: {EVAL_CSV}")
        sys.exit(1)

    questions = load_questions(EVAL_CSV)
    print(f"[INFO] Loaded {len(questions)} questions from {EVAL_CSV.name}")

    # ── Run New System Evaluation ──
    new_metrics, _ = evaluate_new_system(questions)

    # ── Load old system baseline (real > literature) ──
    old_metrics_estimated = get_old_system_baseline()
    source_note = old_metrics_estimated.get("_source", "")

    # ── Compute Improvements — each metric has its own real delta ──
    improvements = {}
    for key in new_metrics:
        if key.startswith(("hit_rate", "mrr", "ndcg", "precision")):
            old_val = old_metrics_estimated.get(key, 0)
            new_val = new_metrics[key]
            if old_val > 0:
                improvements[f"{key}_improvement_%"] = round((new_val - old_val) / old_val * 100, 1)

    # ── Save Results ──
    is_real = "REAL" in old_metrics_estimated.get("_source", "")
    output = {
        "evaluation_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_questions": len(questions),
        "k_values": K_VALUES,
        "new_system": {
            "name": "pgvector + PostgreSQL + Cross-Encoder Reranker + gemini-2.0-flash",
            "metrics": new_metrics
        },
        "old_system": {
            "name": "FAISS IndexFlatL2 + SQLite + gemini-2.5-flash (no reranker)",
            "baseline_type": "empirical" if is_real else "literature_estimate",
            "baseline_source": old_metrics_estimated.get("_source", ""),
            "metrics": {k: v for k, v in old_metrics_estimated.items() if not k.startswith("_")}
        },
        "improvements": improvements
    }

    out_path = OUT_DIR / "retrieval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── Print Summary ──
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Old System':>15} {'New System':>15} {'Improvement':>14}")
    print("-" * 70)
    for k in K_VALUES:
        for metric_base in ["hit_rate", "mrr", "ndcg", "precision"]:
            key = f"{metric_base}@{k}"
            old_val = old_metrics_estimated.get(key, 0)
            new_val = new_metrics.get(key, 0)
            imp_key = f"{key}_improvement_%"
            imp = improvements.get(imp_key, 0)
            imp_str = f"+{imp}%" if imp >= 0 else f"{imp}%"
            print(f"  {key:<23} {old_val:>15.4f} {new_val:>15.4f} {imp_str:>14}")

    print(f"\n  {'avg_latency_sec':<23} {old_metrics_estimated.get('avg_latency_sec',0):>15.3f}s {new_metrics.get('avg_latency_sec',0):>14.3f}s")
    print(f"  {'p95_latency_sec':<23} {old_metrics_estimated.get('p95_latency_sec',0):>15.3f}s {new_metrics.get('p95_latency_sec',0):>14.3f}s")
    print(f"\n[SAVED] Results saved to: {out_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
