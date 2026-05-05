# eval_llm_kg.py
# Description: Comprehensive AI Evaluation Suite comparing Gemma, Qwen, and Gemini
# With and Without Knowledge Graph support (GraphRAG vs Standard Agentic RAG).
# Evaluates using MAP, NDCG, and LLM-as-a-Judge Faithfulness metrics.

import os
import time
import json
import math
import csv
import re
from typing import List, Dict
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

# We import the core engine to test it
from langchain_google_genai import ChatGoogleGenerativeAI
from sut_rag_core import SUT_RAG_Engine
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

EVAL_CSV = "sut_questions_v2.csv"
OUT_DIR = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)

# ─── Config ──────────────────────────────────────────────────────────────────
# Adjust to test smaller slices for speed
Num_Questions_To_Test = 100

MODELS_TO_TEST = [
    # --- GEMMA (Open Weights) ---
    {"provider": "google", "name": "gemma-3-1b-it"},
    {"provider": "google", "name": "gemma-3-4b-it"},
    {"provider": "google", "name": "gemma-3-12b-it"},
    {"provider": "google", "name": "gemma-4-26b-a4b-it"},
    {"provider": "google", "name": "gemma-4-31b-it"},
    
    # --- GEMINI (Proprietary) ---
    {"provider": "google", "name": "gemini-2.5-flash-lite"},
    {"provider": "google", "name": "gemini-2.5-flash"},
    {"provider": "google", "name": "gemini-2.5-pro"},
    {"provider": "google", "name": "gemini-3.1-pro-preview"}
]

# ─── Metrics Calculation ─────────────────────────────────────────────────────

def calculate_map(retrieved_chunks: List[Dict], source_ref: str) -> float:
    """Mean Average Precision"""
    if not source_ref or not retrieved_chunks: return 0.0
    
    hits = 0
    sum_precisions = 0.0
    
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        meta = chunk.get("metadata", {})
        header = " ".join(str(v) for v in meta.values()).lower()
        if source_ref.lower() in header:
            hits += 1
            sum_precisions += hits / rank
            
    # If there are no relevant documents retrieved, MAP is 0
    return round(sum_precisions / (hits if hits > 0 else 1.0), 4)

def calculate_ndcg(retrieved_chunks: List[Dict], source_ref: str, k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain"""
    if not source_ref or not retrieved_chunks: return 0.0
    
    dcg = 0.0
    hits = 0
    for rank, chunk in enumerate(retrieved_chunks[:k], start=1):
        meta = chunk.get("metadata", {})
        header = " ".join(str(v) for v in meta.values()).lower()
        if source_ref.lower() in header:
            dcg += 1.0 / math.log2(rank + 1)
            hits += 1
            
    # Calculate ideal DCG based on number of relevant hits found
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(hits))
    return round(dcg / ideal_dcg, 4) if ideal_dcg > 0 else 0.0

# ─── LLM Judge ───────────────────────────────────────────────────────────────

def evaluate_llm_judge(question: str, reference: str, answer: str, judge_model, model_name: str = "", q_index: int = 0) -> (float, float):
    """
    Uses an LLM to evaluate:
    1. Faithfulness: Does the AI answer hallucinate or contradict the reference?
    2. Answer Relevance: Does the AI answer directly address the user question?
    Returns (faithfulness_score, relevance_score)
    """
    if not answer or "[ERROR]" in answer or "Hata:" in answer:
        print(f"    [JUDGE] [{model_name} | Q{q_index}] Response contains error or is empty. Scoring 0.0.")
        return 0.0, 0.0
    
    prompt = f"""
Sen tarafsız bir değerlendirici yapay zekasın. RAG (Retrieval-Augmented Generation) sisteminin çıktısını iki kritere göre değerlendireceksin.

Soru: {question}
Gerçek/Beklenen Cevap (Referans): {reference}
Yapay Zeka (RAG) Cevabı: {answer[:1500]}

KRİTER 1: FAITHFULNESS (Sadakat)
- RAG Cevabı, Referans bilgisiyle çelişiyor mu veya Referansta olmayan bir bilgiyi uyduruyor mu?
- 0.0 = Tamamen çelişiyor veya uyduruyor (Hallucination)
- 0.5 = Kısmen doğru ama eksik/fazla bilgi var
- 1.0 = Tamamen referansa sadık (Farklı ifade edilmiş olsa bile anlam aynıysa 1.0 ver)

KRİTER 2: RELEVANCE (İlgililik)
- RAG Cevabı, Soruyu doğrudan ve net bir şekilde yanıtlıyor mu?
- 0.0 = Soruyla alakasız veya soruyu yanıtlamıyor
- 0.5 = Sorunun etrafında dolanıyor ama tam yanıt vermiyor
- 1.0 = Soruyu doğrudan ve net olarak yanıtlıyor

Çıktıyı SADECE aşağıdaki JSON formatında ver, başka hiçbir metin (markdown dahil) ekleme:
{{
  "faithfulness": 1.0,
  "relevance": 1.0
}}
"""
    try:
        raw_resp = judge_model.invoke([HumanMessage(content=prompt)]).content
        if isinstance(raw_resp, list):
            parts = []
            for part in raw_resp:
                if isinstance(part, dict):
                    parts.append(part.get("text", str(part)))
                else:
                    parts.append(str(part))
            resp = "".join(parts)
        else:
            resp = str(raw_resp)
            
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        if m:
            raw_json = m.group()
            try:
                data = json.loads(raw_json)
            except json.JSONDecodeError:
                data = json.loads(raw_json.replace("'", '"'))
            
            f_score = float(data.get("faithfulness", 0.0))
            r_score = float(data.get("relevance", 0.0))
            
            print(f"\n    [JUDGE DEBUG] [{model_name} | Q{q_index}]")
            print(f"    FAITHFULNESS: {f_score:.2f} | RELEVANCE: {r_score:.2f}")
            return f_score, r_score
        else:
            print(f"    [JUDGE] Could not parse JSON from response.")
    except Exception as e:
        print(f"    [JUDGE ERROR] {type(e).__name__}: {e}")
    return 0.5, 0.5

# ─── Schema Helper ───────────────────────────────────────────────────────────

def strip_kg_tools(schema: str) -> str:
    """Robustly removes KG tools from the TOOL_SCHEMA for 'no_kg' testing."""
    # Remove tool definitions 3 and 4
    schema = re.sub(r'3\. lookup_kg_entity.*?\n\n', '', schema, flags=re.DOTALL)
    schema = re.sub(r'4\. explore_kg_path.*?\n\n', '', schema, flags=re.DOTALL)
    # Remove strategy rules mentions of KG
    schema = re.sub(r'- TEŞHİS/İLAÇ/MADDE isimleri için ÖNCE mutlaka \'lookup_kg_entity\' kullan.*?\n', '', schema)
    schema = re.sub(r'- "X yaş altı", "Y raporu", "Z uzmanı" gibi spesifik kısıtlamaları bulmak için explore_kg_path aracını zorla\..*?\n', '', schema)
    schema = re.sub(r'- Bilgi Grafiği \(KG\) sonuçları SUT metninden daha önceliklidir, çelişki varsa KG\'deki yapısal ilişkiyi baz al\..*?\n', '', schema)
    return schema

# ─── Evaluator Core ──────────────────────────────────────────────────────────

def evaluate_single_model(model_cfg, test_subset, judge_model, OUT_DIR, MODELS_DIR):
    provider = model_cfg["provider"]
    model_name = model_cfg["name"]
    print(f"\n[STARTING] {model_name}")
    
    current_model_data = {
        "model_name": model_name,
        "provider": provider,
        "timestamp": datetime.now().isoformat(),
        "results": {}
    }
    
    # We test two modes: 'no_kg' (Baseline RAG) and 'with_kg' (GraphRAG)
    for mode in ["no_kg", "with_kg"]:
        try:
            engine = SUT_RAG_Engine(llm_provider=provider, model_name=model_name)
            engine.load_database()
            if mode == "no_kg":
                engine.TOOL_SCHEMA = strip_kg_tools(engine.TOOL_SCHEMA)
        except Exception as e:
            print(f"Failed to load engine for {model_name}: {e}")
            continue
            
        mode_results = []
        latencies = []
        map_scores = []
        ndcg_scores = []
        recall_scores = []
        faith_scores = []
        relevance_scores = []
        tool_counts = []
        tool_success = 0
        
        for i, q in enumerate(test_subset, 1):
            t0 = time.time()
            final_answer = ""
            agent_steps = []
            retrieved_chunks = []
            
            try:
                for chunk in engine.query_agentic_rag_stream(q["question"]):
                    if "agent_step" in chunk:
                        step = chunk["agent_step"]
                        agent_steps.append(step)
                        if step.get("tool") == "search_sut_chunks" and isinstance(step.get("result"), str):
                            try:
                                c_res = engine._retrieve_chunks(q["question"], k=5)
                                retrieved_chunks.extend(c_res)
                            except: pass
                    if "final_answer" in chunk:
                        final_answer = chunk["final_answer"]
                        tool_success += 1
            except Exception as e:
                final_answer = f"[ERROR] {str(e)}"
                
            latency = time.time() - t0
            
            # Metrics
            q_map = calculate_map(retrieved_chunks, q["source"])
            q_ndcg = calculate_ndcg(retrieved_chunks, q["source"], k=5)
            q_recall = 1.0 if q_map > 0 else 0.0
            
            # Judge Eval
            f_score, r_score = evaluate_llm_judge(q["question"], q["answer"], final_answer, judge_model, model_name, i)
            
            t_count = len(agent_steps)
            
            latencies.append(latency)
            map_scores.append(q_map)
            ndcg_scores.append(q_ndcg)
            recall_scores.append(q_recall)
            faith_scores.append(f_score)
            relevance_scores.append(r_score)
            tool_counts.append(t_count)
            
            mode_results.append({
                "question": q["question"],
                "expected": q["answer"],
                "source": q["source"],
                "ai_answer": final_answer,
                "metrics": {"map": q_map, "ndcg": q_ndcg, "recall": q_recall, "faithfulness": f_score, "relevance": r_score, "latency": latency, "tool_steps": t_count}
            })
            time.sleep(1)
        
        # Aggregate
        avg_lat = sum(latencies)/len(latencies) if latencies else 0
        avg_map = sum(map_scores)/len(map_scores) if map_scores else 0
        avg_ndcg = sum(ndcg_scores)/len(ndcg_scores) if ndcg_scores else 0
        avg_recall = sum(recall_scores)/len(recall_scores) if recall_scores else 0
        avg_faith = sum(faith_scores)/len(faith_scores) if faith_scores else 0
        avg_rel = sum(relevance_scores)/len(relevance_scores) if relevance_scores else 0
        avg_tools = sum(tool_counts)/len(tool_counts) if tool_counts else 0
        
        mode_results_summary = {
            "avg_map": round(avg_map, 4),
            "avg_ndcg": round(avg_ndcg, 4),
            "context_recall": round(avg_recall, 4),
            "faithfulness": round(avg_faith, 4),
            "answer_relevance": round(avg_rel, 4),
            "avg_tool_steps": round(avg_tools, 2),
            "avg_latency_s": round(avg_lat, 2)
        }
        
        current_model_data["results"][mode] = {
            "summary": mode_results_summary,
            "details": mode_results
        }
        
    # Save individual JSON
    safe_model_name = model_name.replace("/", "_")
    indiv_file = MODELS_DIR / f"{safe_model_name}_results.json"
    with open(indiv_file, "w", encoding="utf-8") as f:
        json.dump(current_model_data, f, ensure_ascii=False, indent=2)
    print(f"[FINISHED] {model_name} -> {indiv_file}")
    
    return model_name, current_model_data["results"]

def run_evaluation():
    # Load Questions
    rows = []
    try:
        with open(EVAL_CSV, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "id": row["ID"], "category": row["Kategori"],
                    "question": row["Soru"], "answer": row["Cevap"],
                    "source": row["Kaynak"].strip()
                })
    except Exception as e:
        print(f"Error loading {EVAL_CSV}: {e}")
        return

    test_subset = rows[:Num_Questions_To_Test]
    results_summary = {}
    
    # Ensure directories exist
    OUT_DIR.mkdir(exist_ok=True)
    MODELS_DIR = OUT_DIR / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Initialize Judge
    judge_model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0)

    # Run in parallel
    print(f"\n🚀 Starting parallel evaluation for {len(MODELS_TO_TEST)} models...")
    with ThreadPoolExecutor(max_workers=len(MODELS_TO_TEST)) as executor:
        future_to_model = {executor.submit(evaluate_single_model, cfg, test_subset, judge_model, OUT_DIR, MODELS_DIR): cfg for cfg in MODELS_TO_TEST}
        for future in as_completed(future_to_model):
            try:
                m_name, m_results = future.result()
                results_summary[m_name] = m_results
            except Exception as e:
                print(f"Model thread generated an exception: {e}")
    
    # Save overall 
    final_output = {
        "timestamp": time.time(),
        "num_questions": len(test_subset),
        "results": results_summary
    }
    
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_DIR / "llm_kg_benchmark.json", "w", encoding="utf-8") as f:
        # Save a clean version without the massive details for the summary
        summary_only = {
            "timestamp": final_output["timestamp"],
            "num_questions": final_output["num_questions"],
            "results": {}
        }
        for m, mdata in results_summary.items():
            summary_only["results"][m] = {k: v for k, v in mdata.items() if not k.endswith("_details")}
        json.dump(summary_only, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "detailed_eval_log.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ Benchmarking completed.")
    print(f"📊 Summary: {OUT_DIR / 'llm_kg_benchmark.json'}")
    print(f"📝 Detailed Logs: {OUT_DIR / 'detailed_eval_log.json'}")

if __name__ == "__main__":
    run_evaluation()
