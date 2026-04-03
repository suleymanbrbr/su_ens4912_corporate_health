"""
eval_ablation.py
================
Ablation Study: Effect of Cross-Encoder Reranker on Retrieval Quality
Compares THREE configurations with EVERYTHING else constant (same DB, same embedding, same k):

  [A] pgvector cosine search —> NO reranker  (pure vector similarity)
  [B] pgvector cosine search —> WITH reranker (current production system)

Also records:
  [C] LLM answer quality WITH reranker context vs WITHOUT (on 30 questions)

Usage (inside docker container):
    python eval_ablation.py

Results: eval_results/ablation_results.json
"""

import os, sys, json, csv, time, math
import numpy as np
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

OUT_DIR  = Path("/app/eval_results")
OUT_DIR.mkdir(exist_ok=True)
EVAL_CSV = Path("/app/sut_questions.csv")
K_VALUES = [1, 3, 5, 10]
LLM_SAMPLE = 30   # How many questions for LLM context quality comparison

# ─── Load Questions ───────────────────────────────────────────────────────────
def load_questions(n=None):
    rows = []
    with open(EVAL_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append({"id": row["ID"], "category": row["Kategori"],
                         "question": row["Soru"], "answer": row["Cevap"],
                         "source": row["Kaynak"].strip()})
    return rows[:n] if n else rows

# ─── Source Matching ─────────────────────────────────────────────────────────
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
    return {f"{m}@{k}": round(totals[k][field]/n, 4)
            for k in K_VALUES for m, field in [("hit_rate","hits"),("mrr","rr"),("ndcg","ndcg"),("precision","prec")]}

# ─── ROUGE-L for LLM comparison ──────────────────────────────────────────────
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

# ─── Main Ablation Logic ─────────────────────────────────────────────────────
def main():
    print("="*65)
    print("ABLATION STUDY: Reranker Effect on pgvector System")
    print("="*65)

    sys.path.insert(0, "/app")
    from sut_rag_core import SUT_RAG_Engine
    from langchain_core.messages import SystemMessage, HumanMessage

    engine = SUT_RAG_Engine()
    if not engine.load_database():
        print("[ERROR] Cannot connect to PostgreSQL."); sys.exit(1)

    questions = load_questions()
    print(f"[INFO] {len(questions)} questions loaded.\n")

    # ── [A] pgvector WITHOUT reranker ──────────────────────────────────────
    print("[A] Running pgvector WITHOUT Reranker (pure cosine similarity)...")
    results_no_rerank = []
    latencies_a = []

    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['question'][:60]}...")
        t0 = time.time()
        try:
            # Get raw vector results without cross-encoder scoring
            query_vec = engine.embeddings_model.embed_query(q["question"])
            query_vec_np = np.array(query_vec, dtype="float32").tolist()

            cursor = engine.conn.cursor()
            cursor.execute("""
                SELECT chunk_id, text_content, metadata_json,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_vec_np, query_vec_np, max(K_VALUES)))
            rows = cursor.fetchall()
            cursor.close()

            chunks_no_rerank = []
            for row in rows:
                raw_meta = row[2]
                if isinstance(raw_meta, dict):
                    meta = raw_meta
                elif isinstance(raw_meta, str):
                    import json as _json
                    meta = _json.loads(raw_meta)
                else:
                    meta = {}
                chunks_no_rerank.append({"metadata": meta, "score": float(row[3]), "text": row[1]})
        except Exception as e:
            print(f"    [WARN] {e}"); chunks_no_rerank = []
            try: engine.conn.rollback()
            except: pass

        latencies_a.append(time.time() - t0)
        retrieved = [{"metadata": c["metadata"], "score": c["score"]} for c in chunks_no_rerank]
        results_no_rerank.append({"question": q["question"], "source": q["source"],
                                   "retrieved_chunks": retrieved})

    metrics_a = calc_metrics(results_no_rerank)
    metrics_a["avg_latency_sec"] = round(sum(latencies_a)/len(latencies_a), 3)
    metrics_a["p95_latency_sec"] = round(sorted(latencies_a)[int(len(latencies_a)*0.95)], 3)
    print(f"  Done. Avg latency: {metrics_a['avg_latency_sec']}s\n")

    # ── [B] pgvector WITH reranker ─────────────────────────────────────────
    print("[B] Running pgvector WITH Cross-Encoder Reranker (production system)...")
    results_rerank = []
    latencies_b = []

    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['question'][:60]}...")
        t0 = time.time()
        try:
            chunks_with_rerank = engine._retrieve_chunks(q["question"], k=max(K_VALUES))
        except Exception as e:
            print(f"    [WARN] {e}"); chunks_with_rerank = []

        latencies_b.append(time.time() - t0)
        retrieved = [{"metadata": c.get("metadata", {}), "score": c.get("score", 0)} for c in chunks_with_rerank]
        results_rerank.append({"question": q["question"], "source": q["source"],
                                "retrieved_chunks": retrieved})

    metrics_b = calc_metrics(results_rerank)
    metrics_b["avg_latency_sec"] = round(sum(latencies_b)/len(latencies_b), 3)
    metrics_b["p95_latency_sec"] = round(sorted(latencies_b)[int(len(latencies_b)*0.95)], 3)
    print(f"  Done. Avg latency: {metrics_b['avg_latency_sec']}s\n")

    # ── Compute Retrieval Improvements A→B ────────────────────────────────
    ret_improvements = {}
    for key in metrics_a:
        if key.startswith(("hit_rate","mrr","ndcg","precision")):
            ov = metrics_a[key]; nv = metrics_b[key]
            if ov > 0:
                ret_improvements[f"{key}_reranker_gain_%"] = round((nv-ov)/ov*100, 1)

    # ── [C] LLM Answer Quality: reranked vs non-reranked context ──────────
    print(f"\n[C] LLM Answer Quality: Reranked vs Non-Reranked context ({LLM_SAMPLE} questions)...")
    llm_results = []
    sample_qs = questions[:LLM_SAMPLE]

    SYSTEM_TEMPLATE = """Sen SUT (Sağlık Uygulama Tebliği) uzmanı bir asistansın.
Sadece aşağıdaki bağlamı kullanarak soruyu yanıtla. Türkçe cevap ver:

{context}

Eğer bilgi bağlamda yoksa sadece: 'Bu bilgi bu bağlamda yer almamaktadır.' de."""

    rouge_sum_nr = 0.0
    rouge_sum_r  = 0.0
    judge_faithful_nr = 0.0
    judge_faithful_r  = 0.0
    judge_count = 0

    for i, q in enumerate(sample_qs, 1):
        print(f"  [{i}/{LLM_SAMPLE}] {q['question'][:60]}...")

        # Get chunks for both configs
        try:
            # No-rerank context
            query_vec = engine.embeddings_model.embed_query(q["question"])
            query_vec_np = np.array(query_vec, dtype="float32").tolist()
            cursor = engine.conn.cursor()
            cursor.execute("""
                SELECT text_content, metadata_json
                FROM chunks ORDER BY embedding <=> %s::vector LIMIT 5
            """, (query_vec_np,))
            rows_nr = cursor.fetchall()
            cursor.close()
            ctx_no_rerank = "\n\n".join([r[0] for r in rows_nr])
        except Exception as e:
            ctx_no_rerank = ""
            try: engine.conn.rollback()
            except: pass

        try:
            # Reranked context
            chunks_r = engine._retrieve_chunks(q["question"], k=5)
            ctx_rerank = "\n\n".join([c.get("text","") for c in chunks_r[:5]])
        except:
            ctx_rerank = ""

        # Generate both answers
        ans_nr = ans_r = ""
        try:
            resp_nr = engine.llm.invoke([
                SystemMessage(content=SYSTEM_TEMPLATE.format(context=ctx_no_rerank[:2000])),
                HumanMessage(content=q["question"])
            ])
            ans_nr = resp_nr.content
        except Exception as e:
            ans_nr = f"[ERROR] {e}"

        time.sleep(0.5)

        try:
            resp_r = engine.llm.invoke([
                SystemMessage(content=SYSTEM_TEMPLATE.format(context=ctx_rerank[:2000])),
                HumanMessage(content=q["question"])
            ])
            ans_r = resp_r.content
        except Exception as e:
            ans_r = f"[ERROR] {e}"

        # Metrics
        rl_nr = rouge_l(q["answer"], ans_nr)
        rl_r  = rouge_l(q["answer"], ans_r)
        rouge_sum_nr += rl_nr
        rouge_sum_r  += rl_r

        # LLM judge for faithfulness
        try:
            import re as _re, json as _json
            judge_prompt = f"""Rate faithfulness (0-1) for BOTH answers:

Q: {q['question']}
Context A (no reranking): {ctx_no_rerank[:500]}
Answer A: {ans_nr[:300]}
Context B (with reranking): {ctx_rerank[:500]}
Answer B: {ans_r[:300]}
Reference: {q['answer']}

Respond ONLY as JSON: {{"faithful_a": 0.8, "faithful_b": 0.9, "better": "A or B or equal"}}"""
            resp_j = engine.llm.invoke([HumanMessage(content=judge_prompt)])
            m = _re.search(r'\{.*?\}', resp_j.content, _re.DOTALL)
            if m:
                j = _json.loads(m.group())
                judge_faithful_nr += j.get("faithful_a", 0)
                judge_faithful_r  += j.get("faithful_b", 0)
                judge_count += 1
        except: pass

        llm_results.append({
            "question": q["question"], "reference": q["answer"],
            "answer_no_rerank": ans_nr, "answer_rerank": ans_r,
            "rouge_l_no_rerank": rl_nr, "rouge_l_rerank": rl_r,
        })
        time.sleep(1)

    llm_summary = {
        "num_questions": LLM_SAMPLE,
        "avg_rouge_l_no_rerank": round(rouge_sum_nr / LLM_SAMPLE, 4),
        "avg_rouge_l_rerank":    round(rouge_sum_r  / LLM_SAMPLE, 4),
        "rouge_l_gain_%":        round((rouge_sum_r - rouge_sum_nr) / max(rouge_sum_nr, 0.001) * 100, 1),
        "avg_faithfulness_no_rerank": round(judge_faithful_nr / max(judge_count,1), 4),
        "avg_faithfulness_rerank":    round(judge_faithful_r  / max(judge_count,1), 4),
    }

    # ── Save All Results ──────────────────────────────────────────────────
    output = {
        "evaluation_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_questions_retrieval": len(questions),
        "num_questions_llm": LLM_SAMPLE,
        "description": "Ablation study: effect of Cross-Encoder reranker. Everything else is identical (same pgvector DB, same embedding model, same k).",
        "config_A_no_reranker": {
            "name": "pgvector cosine — NO reranker",
            "metrics": metrics_a
        },
        "config_B_with_reranker": {
            "name": "pgvector cosine + Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)",
            "metrics": metrics_b
        },
        "reranker_retrieval_gain": ret_improvements,
        "llm_context_quality": llm_summary,
        "llm_per_question": llm_results,
    }

    out_path = OUT_DIR / "ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── Print Summary ─────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("ABLATION RESULTS: Reranker Gain")
    print("="*65)
    print(f"\n{'Metric':<22} {'No Reranker':>13} {'With Reranker':>15} {'Gain':>10}")
    print("-"*63)
    for k in K_VALUES:
        for base in ["hit_rate","mrr","ndcg","precision"]:
            key = f"{base}@{k}"
            a = metrics_a.get(key,0); b = metrics_b.get(key,0)
            g = ret_improvements.get(f"{key}_reranker_gain_%","")
            flag = "✅" if isinstance(g, float) and g>0 else "🔴"
            print(f"  {key:<20} {a:>13.4f} {b:>15.4f} {flag} {f'+{g}%' if isinstance(g,float) and g>0 else f'{g}%':>8}")

    print(f"\n  {'avg_latency_sec':<20} {metrics_a['avg_latency_sec']:>13.3f}s {metrics_b['avg_latency_sec']:>14.3f}s")
    print(f"\nLLM CONTEXT QUALITY:")
    print(f"  ROUGE-L no reranker:  {llm_summary['avg_rouge_l_no_rerank']:.4f}")
    print(f"  ROUGE-L with reranker:{llm_summary['avg_rouge_l_rerank']:.4f}")
    print(f"  ROUGE-L gain:         +{llm_summary['rouge_l_gain_%']}%")
    print(f"  Faithfulness no rerank: {llm_summary['avg_faithfulness_no_rerank']:.4f}")
    print(f"  Faithfulness reranked:  {llm_summary['avg_faithfulness_rerank']:.4f}")
    print(f"\n[SAVED] {out_path}")

if __name__ == "__main__":
    main()
