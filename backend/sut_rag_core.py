# sut_rag_core.py
# Description: SUT Agentic RAG Engine — ReAct-style Tool-Calling Loop, PostgreSQL Edition

import os
import json
import math
import psycopg2
from typing import List, Dict, Generator
from sentence_transformers import CrossEncoder

# LangChain & AI Libraries
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ─── Tool Icon Map ───────────────────────────────────────────────────────────
TOOL_ICONS = {
    "search_sut_chunks":    "🔍",
    "search_sut_fulltext":  "📄",
    "lookup_knowledge_graph": "🕸️",
    "calculate":            "🔢",
    "finish":               "✅",
}

MAX_AGENT_ITERATIONS = 6   # safety hard-stop

class SUT_RAG_Engine:
    def __init__(self, llm_provider: str = "google", model_name: str = "gemini-2.0-flash"):
        self.embeddings_model = self._initialize_embeddings()
        print("[INIT] Loading Reranker Model...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
        self.conn = None
        self.cursor = None
        self.llm = None
        self.provider = llm_provider
        self.knowledge_graph = self._load_knowledge_graph()

        if llm_provider == "google":
            self._init_google_llm(model_name)
        elif llm_provider == "openrouter":
            self._init_openrouter_llm(model_name)
        else:
            self._init_google_llm("gemini-1.5-flash")

        print(f"[INIT] SUT Engine Initialized. Provider: '{llm_provider}', Model: '{model_name}'")

    # ─── LLM Init ────────────────────────────────────────────────────────────

    def _init_google_llm(self, model_name: str):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.1)

    def _init_openrouter_llm(self, model_name: str):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return
        self.llm = ChatOpenAI(
            model=model_name, openai_api_key=api_key, openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1
        )

    def _initialize_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )

    def _load_knowledge_graph(self) -> Dict:
        kg_path = "sut_knowledge_graph.json"
        if not os.path.exists(kg_path):
            print("[WARN] Knowledge graph JSON not found.")
            return {"nodes": [], "edges": []}
        try:
            with open(kg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load knowledge graph: {e}")
            return {"nodes": [], "edges": []}

    def load_database(self) -> bool:
        try:
            self.conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            cur = self.conn.cursor()
            cur.execute("SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname='public' AND tablename='chunks')")
            exists = cur.fetchone()[0]
            cur.close()
            if not exists:
                print("[WARN] 'chunks' table not found. Re-index required from Admin Panel.")
            return True
        except Exception as e:
            print(f"[WARN] Postgres database connection failed: {e}")
            return False

    # ─── Tool Definitions (schema injected into system prompt) ───────────────

    TOOL_SCHEMA = """
You have access to the following tools. Call them by outputting ONLY a JSON object — no extra text, no markdown fences.

AVAILABLE TOOLS:
1. search_sut_chunks
   Description: Semantic vector search in the SUT database. Use this for general, conceptual questions.
   Input: {"tool": "search_sut_chunks", "query": "<natural language query>", "k": <number 3-8>}

2. search_sut_fulltext
   Description: Full-text keyword search. Best for specific article IDs (e.g. "4.2.63") or exact drug/diagnosis names.
   Input: {"tool": "search_sut_fulltext", "query": "<keywords or article id>", "k": <number 3-8>}

3. lookup_knowledge_graph
   Description: Looks up a drug, diagnosis, condition, or rule in the SUT knowledge graph to discover structured relationships (e.g., what drugs treat a disease, what conditions a rule requires).
   Input: {"tool": "lookup_knowledge_graph", "entity": "<entity name or ID>"}

4. calculate
   Description: Performs a safe numeric calculation. Useful for dosage, age limits, duration calculations.
   Input: {"tool": "calculate", "expression": "<math expression e.g. 65*0.15>"}

5. finish
   Description: Call this when you have gathered sufficient information to give a complete and accurate answer. Do NOT call finish before searching at least once. The argument becomes the final answer shown to the user.
   Input: {"tool": "finish", "answer": "<your final comprehensive answer in Turkish>"}

RULES:
- Respond with ONLY ONE tool call JSON per turn, nothing else.
- Always search at least once before calling finish.
- Cite sources in your final answer using [Kaynak N] notation.
- Write the final answer in TURKISH.
- If search results contain a <KAYNAKLAR> block, include it at the end of your finish answer.
"""

    TOOL_REACTION_PROMPT = """
You are SUT Uzmanı (Sağlık Uygulama Tebliği Expert), an expert AI assistant for Turkish Social Security health regulations.

{tool_schema}

--- CONVERSATION HISTORY ---
{history}

--- TOOL RESULTS SO FAR ---
{observations}

--- USER QUESTION ---
{user_query}

Now decide your next action. Output ONLY a single JSON tool call.
"""

    # ─── Main Agentic Stream ─────────────────────────────────────────────────

    def query_agentic_rag_stream(
        self,
        user_query: str,
        chat_history: List[Dict] = None,
        k: int = 5
    ) -> Generator[Dict, None, None]:

        if chat_history is None:
            chat_history = []
        if self.llm is None:
            yield {"error": "LLM not initialized."}
            return

        history_str = "\n".join(
            f"{'Kullanıcı' if m['role'] == 'user' else 'Asistan'}: {m['content'][:300]}"
            for m in chat_history[-6:]
        )

        observations: List[Dict] = []   # list of {tool, args, result}
        agent_steps: List[Dict] = []    # for frontend trace

        yield {"status": "Sorgu analiz ediliyor..."}

        for iteration in range(MAX_AGENT_ITERATIONS):
            obs_str = self._format_observations(observations)

            prompt = self.TOOL_REACTION_PROMPT.format(
                tool_schema=self.TOOL_SCHEMA,
                history=history_str or "(Yeni konuşma)",
                observations=obs_str or "(Henüz araç kullanılmadı.)",
                user_query=user_query,
            )

            try:
                decision_msg = self.llm.invoke([HumanMessage(content=prompt)])
                raw = decision_msg.content.strip()

                # Strip markdown fences if the LLM wraps in ```json
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:].strip()

                tool_call = json.loads(raw)
            except Exception as e:
                yield {"agent_step": {
                    "iteration": iteration + 1,
                    "tool": "error",
                    "icon": "⚠️",
                    "args": {},
                    "result": f"Karar ayrıştırılamadı: {str(e)} | Raw: {raw[:200]}"
                }}
                break

            tool_name = tool_call.get("tool", "unknown")
            icon = TOOL_ICONS.get(tool_name, "🔧")

            # ── finish tool ──────────────────────────────────────────────────
            if tool_name == "finish":
                final_answer = tool_call.get("answer", "")
                agent_steps.append({
                    "iteration": iteration + 1,
                    "tool": "finish",
                    "icon": icon,
                    "args": {},
                    "result": "Yanıt tamamlandı."
                })
                yield {"agent_step": agent_steps[-1]}
                yield {"agent_steps_complete": agent_steps}
                # Stream the final answer token-by-token style
                yield {"final_answer": final_answer}
                return

            # ── other tools: stream the thinking step ────────────────────────
            yield {"status": f"{icon} {tool_name} çalıştırılıyor..."}

            result = self._run_tool(tool_name, tool_call, k)
            observations.append({"tool": tool_name, "args": tool_call, "result": result})

            step = {
                "iteration": iteration + 1,
                "tool": tool_name,
                "icon": icon,
                "args": {k2: v for k2, v in tool_call.items() if k2 != "tool"},
                "result": result[:500] if isinstance(result, str) else str(result)[:500]
            }
            agent_steps.append(step)
            yield {"agent_step": step}

        # ── Hard stop fallback ───────────────────────────────────────────────
        yield {"status": "Yanıt oluşturuluyor..."}
        fallback_answer = self._generate_fallback_answer(user_query, observations, chat_history)
        yield {"agent_steps_complete": agent_steps}
        yield {"final_answer": fallback_answer}

    # ─── Tool Runner ─────────────────────────────────────────────────────────

    def _run_tool(self, tool_name: str, args: Dict, default_k: int) -> str:
        try:
            if tool_name == "search_sut_chunks":
                query = args.get("query", "")
                k = int(args.get("k", default_k))
                chunks = self._retrieve_chunks(query, k)
                return self._format_chunks_result(chunks)

            elif tool_name == "search_sut_fulltext":
                query = args.get("query", "")
                k = int(args.get("k", default_k))
                chunks = self._fulltext_search(query, k)
                return self._format_chunks_result(chunks)

            elif tool_name == "lookup_knowledge_graph":
                entity = args.get("entity", "")
                return self._lookup_kg(entity)

            elif tool_name == "calculate":
                expression = args.get("expression", "")
                return self._safe_calculate(expression)

            else:
                return f"Bilinmeyen araç: {tool_name}"
        except Exception as e:
            return f"[ARAÇ HATASI] {tool_name}: {str(e)}"

    # ─── Tool Implementations ─────────────────────────────────────────────────

    def _retrieve_chunks(self, query: str, k: int) -> List[Dict]:
        if not self.conn:
            return []
        initial_k = k * 3
        try:
            q_vec = self.embeddings_model.embed_query(query)
            q_vec_str = "[" + ",".join(map(str, q_vec)) + "]"

            cur = self.conn.cursor()
            cur.execute("""
                SELECT chunk_id, text_content, metadata_json
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (q_vec_str, initial_k))

            candidates = []
            for row in cur.fetchall():
                meta = row[2] if isinstance(row[2], dict) else json.loads(row[2])
                candidates.append({"id": row[0], "text": row[1], "metadata": meta})
            cur.close()

            if candidates:
                pairs = [[query, doc['text']] for doc in candidates]
                scores = self.reranker.predict(pairs)
                for doc, score in zip(candidates, scores):
                    doc['score'] = score
                candidates.sort(key=lambda x: x['score'], reverse=True)

            return candidates[:k]
        except Exception as e:
            print(f"[ERROR] Chunk retrieval failed: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass
            return []

    def _fulltext_search(self, query: str, k: int) -> List[Dict]:
        """PostgreSQL full-text search using Turkish dictionary."""
        if not self.conn:
            return []
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT chunk_id, text_content, metadata_json,
                       ts_rank(to_tsvector('turkish', COALESCE(header_text,'') || ' ' || text_content),
                               websearch_to_tsquery('turkish', %s)) AS rank
                FROM chunks
                WHERE to_tsvector('turkish', COALESCE(header_text,'') || ' ' || text_content)
                      @@ websearch_to_tsquery('turkish', %s)
                ORDER BY rank DESC
                LIMIT %s
            """, (query, query, k))
            results = []
            for row in cur.fetchall():
                meta = row[2] if isinstance(row[2], dict) else json.loads(row[2])
                results.append({"id": row[0], "text": row[1], "metadata": meta, "score": float(row[3])})
            cur.close()
            return results
        except Exception as e:
            print(f"[ERROR] Full-text search failed: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass
            return []

    def _lookup_kg(self, entity: str) -> str:
        """Search the knowledge graph for an entity and return its relationships."""
        entity_upper = entity.strip().upper()
        nodes = self.knowledge_graph.get("nodes", [])
        edges = self.knowledge_graph.get("edges", [])

        # Find matching nodes (exact or partial)
        matched_nodes = [
            n for n in nodes
            if entity_upper in n.get("id", "").upper() or
               entity_upper in n.get("label", "").upper()
        ]

        if not matched_nodes:
            return f"'{entity}' için bilgi grafiğinde eşleşme bulunamadı."

        results = []
        for node in matched_nodes[:3]:
            nid = node["id"]
            related_edges = [
                e for e in edges
                if e.get("source", "").upper() == nid.upper() or
                   e.get("target", "").upper() == nid.upper()
            ]

            node_info = f"DÜĞÜM: {node['label']} (Tür: {node['type']})"
            if node.get("text"):
                node_info += f"\nMetin: {node['text'][:200]}"

            edge_info = []
            for e in related_edges[:8]:
                src_label = next((n['label'] for n in nodes if n['id'].upper() == e['source'].upper()), e['source'])
                tgt_label = next((n['label'] for n in nodes if n['id'].upper() == e['target'].upper()), e['target'])
                edge_info.append(f"  [{e['relation']}]: {src_label} → {tgt_label}")

            results.append(node_info + "\nİlişkiler:\n" + "\n".join(edge_info) if edge_info else node_info)

        return "\n\n".join(results)

    def _safe_calculate(self, expression: str) -> str:
        """Safely evaluate a numeric math expression."""
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "pow": pow, "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor,
        }
        try:
            # Only allow safe characters
            safe_expr = "".join(c for c in expression if c in "0123456789+-*/.() ")
            result = eval(safe_expr, {"__builtins__": {}}, allowed_names)
            return f"Hesaplama: {expression} = {result}"
        except Exception as e:
            return f"Hesaplama hatası: {str(e)}"

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _format_chunks_result(self, chunks: List[Dict]) -> str:
        if not chunks:
            return "Araştırma sonucu bulunamadı. Farklı anahtar kelimeler deneyin."
        parts = []
        for i, c in enumerate(chunks):
            headers = [v for key, v in c['metadata'].items() if key.startswith("Header")]
            breadcrumb = " > ".join(headers) if headers else "Bölüm"
            score_str = f" (skor: {c.get('score', 0):.2f})" if 'score' in c else ""
            parts.append(f"--- KAYNAK {i+1}{score_str} ---\nBAŞLIK: {breadcrumb}\nİÇERİK:\n{c['text'][:800]}\n")
        return "\n".join(parts)

    def _format_observations(self, observations: List[Dict]) -> str:
        if not observations:
            return ""
        parts = []
        for i, o in enumerate(observations):
            tool = o.get("tool", "?")
            args_str = json.dumps({k: v for k, v in o.get("args", {}).items() if k != "tool"}, ensure_ascii=False)
            result_preview = str(o.get("result", ""))[:600]
            parts.append(f"[Adım {i+1}] Araç: {tool} | Girdi: {args_str}\nSonuç:\n{result_preview}")
        return "\n\n".join(parts)

    def _generate_fallback_answer(
        self,
        user_query: str,
        observations: List[Dict],
        chat_history: List[Dict]
    ) -> str:
        """Called when max iterations hit — asks LLM to synthesize from what we have."""
        obs_str = self._format_observations(observations)

        system_prompt = f"""Sen SUT (Sağlık Uygulama Tebliği) uzmanısın.
Aşağıdaki araştırma sonuçlarını kullanarak kullanıcının sorusunu Türkçe olarak kapsamlı şekilde cevapla.
Bilgi bulunamadıysa açıkça belirt.

ARAŞTIRMA SONUÇLARI:
{obs_str or '(Araştırma sonucu yok)'}
"""
        try:
            messages = [SystemMessage(content=system_prompt)]
            for msg in chat_history[-4:]:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content", "")))
            messages.append(HumanMessage(content=user_query))

            full_response = ""
            for chunk in self.llm.stream(messages):
                content = chunk.content if hasattr(chunk, 'content') else ""
                if content:
                    full_response += content
            return full_response
        except Exception as e:
            return f"Yanıt oluşturulurken hata oluştu: {str(e)}"