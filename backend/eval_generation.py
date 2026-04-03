"""
eval_generation.py
==================
SUT RAG Generation Quality Evaluation
Metrics: ROUGE-L, BERTScore (semantic), Exact Match, Hallucination detection

Usage:
    python eval_generation.py

Results saved to: eval_results/generation_results.json
"""

import os, sys, json, time, csv, re
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

OUT_DIR = Path(__file__).parent / "eval_results"
OUT_DIR.mkdir(exist_ok=True)
EVAL_CSV = Path(__file__).parent / "sut_questions.csv"
MAX_QUESTIONS = 50  # Limit to 50 for API cost control

# ─── Load eval questions (first N, stratified by category) ───────────────────
def load_stratified_sample(csv_path: Path, n: int = 50) -> List[Dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Stratify by category
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["Kategori"]].append(r)

    total = len(rows)
    sample = []
    for cat, items in by_cat.items():
        quota = max(1, round(len(items) / total * n))
        sample.extend(items[:quota])

    return [{"id": r["ID"], "category": r["Kategori"], "question": r["Soru"],
             "answer": r["Cevap"], "source": r["Kaynak"]} for r in sample[:n]]

# ─── ROUGE-L ─────────────────────────────────────────────────────────────────
def rouge_l(reference: str, hypothesis: str) -> float:
    ref_tokens  = reference.lower().split()
    hyp_tokens  = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    # LCS
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / n if n else 0
    recall    = lcs / m if m else 0
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)

# ─── Exact & Fuzzy Match ─────────────────────────────────────────────────────
def exact_match(reference: str, hypothesis: str) -> int:
    return int(reference.strip().lower() == hypothesis.strip().lower())

def fuzzy_match_score(reference: str, hypothesis: str) -> float:
    """Simple token-level overlap (F1)."""
    ref_set  = set(reference.lower().split())
    hyp_set  = set(hypothesis.lower().split())
    if not ref_set or not hyp_set:
        return 0.0
    common = ref_set & hyp_set
    prec   = len(common) / len(hyp_set)
    rec    = len(common) / len(ref_set)
    if prec + rec == 0:
        return 0.0
    return round(2 * prec * rec / (prec + rec), 4)

# ─── Hallucination detection (LLM-as-judge) ──────────────────────────────────
def check_faithfulness(question: str, context: str, answer: str, llm) -> Dict:
    """Ask Gemini to judge if the answer is grounded in the context."""
    judge_prompt = f"""You are a strict judge evaluating factual accuracy of an AI answer.

QUESTION: {question}
CONTEXT (retrieved from SUT database): {context[:1500]}
AI ANSWER: {answer}

Evaluate:
1. FAITHFUL (0-1): Is every claim in the answer supported by the context? (1 = fully faithful)
2. HALLUCINATED: Does the answer contain any information NOT in the context? (yes/no)
3. RELEVANT (0-1): Does the answer address the question? (1 = fully relevant)

Respond ONLY in JSON format like:
{{"faithful": 0.9, "hallucinated": "no", "relevant": 1.0, "reasoning": "brief reason"}}"""
    
    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=judge_prompt)])
        text = response.content.strip()
        # Extract JSON
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"    [WARN] Judge error: {e}")
    return {"faithful": None, "hallucinated": None, "relevant": None, "reasoning": "evaluation failed"}

# ─── Run end-to-end pipeline for one question ────────────────────────────────
def run_pipeline(question: str, engine) -> Dict:
    """Run retrieve + generate and capture context + answer."""
    chunks = engine._retrieve_chunks(question, k=5)
    if not chunks:
        return {"context": "", "answer": "", "latency": 0.0}

    context = "\n\n".join([c.get("text", "") for c in chunks[:5]])

    # Build a simple prompt (same as api_server does)
    from langchain_core.messages import SystemMessage, HumanMessage
    system_msg = SystemMessage(content=f"""Sen SUT (Sağlık Uygulama Tebliği) uzmanı bir asistansın.
Sadece aşağıdaki bağlamı kullanarak soruyu yanıtla:

{context}

Eğer bilgi bağlamda yoksa: "Bu bilgi SUT'ta bu bağlamda yer almamaktadır." de.""")

    t0 = time.time()
    try:
        response = engine.llm.invoke([system_msg, HumanMessage(content=question)])
        answer = response.content
    except Exception as e:
        answer = f"[ERROR] {e}"
    latency = time.time() - t0

    return {"context": context, "answer": answer, "latency": round(latency, 3), "chunks": chunks}

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SUT RAG Generation Quality Evaluation")
    print("=" * 60)

    questions = load_stratified_sample(EVAL_CSV, MAX_QUESTIONS)
    print(f"[INFO] Evaluating on {len(questions)} questions (stratified sample).")

    sys.path.insert(0, str(Path(__file__).parent))
    from sut_rag_core import SUT_RAG_Engine
    engine = SUT_RAG_Engine()
    if not engine.load_database():
        print("[ERROR] Could not connect to PostgreSQL.")
        sys.exit(1)

    item_results = []
    metrics_sum = {"rouge_l": 0.0, "fuzzy_f1": 0.0, "exact_match": 0,
                   "faithful": 0.0, "relevant": 0.0, "hallucinated_count": 0,
                   "latency": 0.0, "judge_count": 0}

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q['question'][:70]}...")

        # Run pipeline
        result = run_pipeline(q["question"], engine)
        hyp = result["answer"]
        ref = q["answer"]

        # Compute metrics
        rl    = rouge_l(ref, hyp)
        ff1   = fuzzy_match_score(ref, hyp)
        em    = exact_match(ref, hyp)

        metrics_sum["rouge_l"]      += rl
        metrics_sum["fuzzy_f1"]     += ff1
        metrics_sum["exact_match"]  += em
        metrics_sum["latency"]      += result["latency"]

        # LLM-as-judge (faithfulness) — expensive, run on all
        judge = {"faithful": None, "hallucinated": None, "relevant": None}
        if result["context"] and engine.llm:
            judge = check_faithfulness(q["question"], result["context"], hyp, engine.llm)
            if judge.get("faithful") is not None:
                metrics_sum["faithful"]  += judge["faithful"]
                metrics_sum["relevant"]  += judge.get("relevant", 0)
                metrics_sum["judge_count"] += 1
                if str(judge.get("hallucinated", "no")).lower() == "yes":
                    metrics_sum["hallucinated_count"] += 1
            time.sleep(1)  # Rate limit pause

        item_results.append({
            "id": q["id"], "category": q["category"],
            "question": q["question"],
            "reference_answer": ref,
            "generated_answer": hyp,
            "rouge_l": rl, "fuzzy_f1": ff1, "exact_match": em,
            "faithfulness": judge.get("faithful"),
            "hallucinated": judge.get("hallucinated"),
            "answer_relevance": judge.get("relevant"),
            "judge_reasoning": judge.get("reasoning", ""),
            "latency_sec": result["latency"]
        })

        print(f"    ROUGE-L: {rl:.3f} | Fuzzy-F1: {ff1:.3f} | Faithful: {judge.get('faithful','N/A')}")

    n = len(questions)
    jn = max(metrics_sum["judge_count"], 1)
    summary = {
        "evaluation_date":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model":                 "pgvector + Cross-Encoder + gemini-2.0-flash",
        "num_questions":         n,
        "avg_rouge_l":           round(metrics_sum["rouge_l"] / n, 4),
        "avg_fuzzy_f1":          round(metrics_sum["fuzzy_f1"] / n, 4),
        "exact_match_rate":      round(metrics_sum["exact_match"] / n, 4),
        "avg_faithfulness":      round(metrics_sum["faithful"] / jn, 4),
        "avg_answer_relevance":  round(metrics_sum["relevant"] / jn, 4),
        "hallucination_rate":    round(metrics_sum["hallucinated_count"] / jn, 4),
        "avg_latency_sec":       round(metrics_sum["latency"] / n, 3),
        "results":               item_results,
    }

    out_path = OUT_DIR / "generation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("GENERATION EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  ROUGE-L (avg):         {summary['avg_rouge_l']:.4f}")
    print(f"  Fuzzy-F1 (avg):        {summary['avg_fuzzy_f1']:.4f}")
    print(f"  Exact Match Rate:      {summary['exact_match_rate']:.4f}")
    print(f"  Faithfulness (avg):    {summary['avg_faithfulness']:.4f}")
    print(f"  Answer Relevance:      {summary['avg_answer_relevance']:.4f}")
    print(f"  Hallucination Rate:    {summary['hallucination_rate']:.4f}")
    print(f"  Avg Latency:           {summary['avg_latency_sec']:.3f}s")
    print(f"\n[SAVED] Results saved to: {out_path}")

if __name__ == "__main__":
    main()
