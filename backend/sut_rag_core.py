# sut_rag_core.py
# Description: SUT Agentic RAG Engine — ReAct-style Tool-Calling Loop, PostgreSQL Edition

import os
import json
import math
import uuid
import psycopg2
from typing import List, Dict, Generator
from sentence_transformers import CrossEncoder

# LangChain & AI Libraries
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# KG Storage
from kg_storage import KG_Storage_Manager

# ─── Tool Icon Map ───────────────────────────────────────────────────────────
TOOL_ICONS = {
    "search_sut_chunks":    "🔍",
    "search_sut_fulltext":  "📄",
    "lookup_kg_entity":     "🕸️",
    "explore_kg_path":      "🗺️",
    "calculate":            "🔢",
    "finish":               "✅",
}

MAX_AGENT_ITERATIONS = 8   # safety hard-stop
MIN_SEARCHES_BEFORE_FINISH = 1  # agent must call at least 1 search tool

class SUT_RAG_Engine:
    def __init__(self, llm_provider: str = "google", model_name: str = "gemini-2.0-flash"):
        self.embeddings_model = self._initialize_embeddings()
        print("[INIT] Loading Reranker Model...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
        
        self.conn = None
        self.cursor = None
        self.llm = None
        self.provider = llm_provider
        self.kg = KG_Storage_Manager()

        if llm_provider == "google":
            self._init_google_llm(model_name)
        elif llm_provider == "openrouter":
            self._init_openrouter_llm(model_name)
        else:
            self._init_google_llm("gemini-2.0-flash")

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
Sen bir SUT Uzmanı yapay zeka asistanısın. Görevin: kullanıcının sorusunu Türkçe olarak eksiksiz, doğru ve kaynaklı biçimde yanıtlamak.

KULLANILABILECEK ARAÇLAR (Her turda yalnızca 1 araç çağır. Çıktının yalnızca JSON olması gerekiyor):

1. search_sut_chunks — Semantik vektör arama. Genel, kavramsal sorular için kullan.
   {"tool": "search_sut_chunks", "query": "<doğal dil sorgu>", "k": <3-8>}

2. search_sut_fulltext — Tam metin arama. Belirli madde numaraları ("4.2.63"), ilaç adları veya ICD kodları için kullan.
   {"tool": "search_sut_fulltext", "query": "<anahtar kelimeler>", "k": <3-8>}

3. lookup_kg_entity — Bilgi Grafiği varlık araması. Bir ilaç, teşhis, uzman veya madde hakkında yapısal bilgi almak için kullan.
   {"tool": "lookup_kg_entity", "entity": "<varlık adı>", "type_filter": "<opsiyonel: DRUG|DIAGNOSIS|RULE|SPECIALIST|CONDITION|DOCUMENT>"}

4. explore_kg_path — Bilgi Grafiği yolu keşfetme. "X ilacı Y teşhisi için ödeniyor mu?" gibi çok aşamalı sorular için kullan.
   {"tool": "explore_kg_path", "from_entity": "<kaynak>", "to_entity": "<hedef>", "max_hops": <1-3>}

5. calculate — Güvenli matematiksel hesaplama. Doz, yaş sınırı, süre hesaplamalarında kullan.
   {"tool": "calculate", "expression": "<mat. ifade>"}

6. finish — Son yanıtı oluştur. En az 1 arama yaptıktan sonra çağır.
   {"tool": "finish", "answer": "<Türkçe kapsamlı yanıt>"}

STRATEJİ VE DERİNLİK KURALLARI:
- ASLA finish çağırmadan önce en az 2 arama aracı kullan (örn: KG lookup + Vektör search).
- TEŞHİS/İLAÇ/MADDE isimleri için ÖNCE mutlaka 'lookup_kg_entity' kullan (Knowledge Graph en kesin veridir).
- Eğer ilk araman 'bulunamadı' veya yetersiz dönerse pes etme; farklı anahtar kelimelerle 'search_sut_chunks' dene.
- "X yaş altı", "Y raporu", "Z uzmanı" gibi spesifik kısıtlamaları bulmak için explore_kg_path aracını zorla.
- Bilgi Grafiği (KG) sonuçları SUT metninden daha önceliklidir, çelişki varsa KG'deki yapısal ilişkiyi baz al.
- Her turda sadece 1 JSON tool call yaz, başka hiçbir metin çıktısı üretme.

SONUÇ YAZMA SÜRECİ (finish çağırırken):
1. Sorunun doğrudan yanıtıyla başla (evet/hayır + kısa açıklama).
2. Elde edilen bilgileri madde madde açıkla.
3. Her maddeyi [Madde X.X.X] veya [Kaynak N] ile kaynak göster.
4. Yanıtın sonuna kaynakları aşağıdaki XML formatında bir blok olarak ekle:
   <KAYNAKLAR>
   <KAYNAK baslik="Madde X.X.X (veya Kısa Başlık)">Kaynak metni burada...</KAYNAK>
   </KAYNAKLAR>
5. Tüm yanıtı Türkçe yaz.
"""

    TOOL_REACTION_PROMPT = """
== SUT UZMAN ASİSTAN (ROL: {role}) ==
Rol: {role_description}

{tool_schema}

--- SON KONUŞMA GEÇMİŞİ ---
{history}

--- YAPILAN ARAÇ ÇAĞRILARI VE SONUÇLARI ---
{observations}

--- KULLANICI SORUSU ---
{user_query}

Şimdi tek bir JSON araç çağrısı üret. Başka hiçbir metin yazma.
"""

    CRITIC_PROMPT = """
Sen bir SUT Denetçisisin (Critic). Aşağıdaki yanıtı, sağlanan literatür ve SUT kaynaklarıyla karşılaştırarak doğrula.

SORU: {user_query}
RETRIEVED CONTEXT / OBSERVATIONS:
{observations}

ADAY YANIT:
{final_answer}

GÖREVİN:
1. Yanıtın içindeki hiçbir bilgi SUT kaynaklarıyla çelişmemeli.
2. Yanıtın içindeki her kısıtlama (yaş, doz, rapor türü) kaynaklarda geçmeli.
3. Yanıtın içindeki madde numaraları [Madde X.X.X] doğru olmalı.

Eğer yanıt %100 doğruysa sadece "TAMAM" yaz.
Eğer hata varsa, hatayı açıklayan kısa bir geri bildirim yaz ve asistanın düzeltmesini iste.
"""

    # ─── Main Agentic Stream ─────────────────────────────────────────────────

    def query_agentic_rag_stream(
        self,
        user_query: str,
        chat_history: List[Dict] = None,
        k: int = 5,
        role: str = "PATIENT"
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

        role_meta = {
            "DOCTOR": {
                "name": "Uzman Doktor",
                "desc": "Tıbbi terimlere, teknik ICD-10 ve ATC kodlarına hakim bir tıp doktoru. SUT annex tablolarını ve klinik detayları ön plana çıkar."
            },
            "ADMIN": {
                "name": "SGK Denetçisi / Yönetici",
                "desc": "Maliyet, bütçe, bürokratik onay süreçleri ve fatura kontrolü odaklı yönetici. Kurumsal dil kullan."
            },
            "PATIENT": {
                "name": "Vatandaş / Hasta",
                "desc": "Tıbbi ve hukuki terimleri anlamayabilecek bir vatandaş. Sade Türkçe kullan, 'Ödenir mi?', 'Ne kadar ödenir?' sorularına net odaklan."
            }
        }.get(role.upper(), {"name": "SUT Uzmanı", "desc": "Genel SUT uzmanı."})

        for iteration in range(MAX_AGENT_ITERATIONS):
            obs_str = self._format_observations(observations)

            prompt = self.TOOL_REACTION_PROMPT.format(
                role=role_meta["name"],
                role_description=role_meta["desc"],
                tool_schema=self.TOOL_SCHEMA,
                history=history_str or "(Yeni konuşma)",
                observations=obs_str or "(Henüz araç kullanılmadı.)",
                user_query=user_query,
            )

            try:
                decision_msg = self.llm.invoke([HumanMessage(content=prompt)])
                raw = decision_msg.content.strip()

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

            if tool_name == "finish":
                final_answer = tool_call.get("answer", "")
                yield {"status": "🕵️ Critic denetimi yapılıyor..."}
                critic_feedback = self._verify_with_critic(user_query, final_answer, obs_str)
                
                if critic_feedback.strip().upper() == "TAMAM":
                    agent_steps.append({
                        "iteration": iteration + 1,
                        "tool": "finish",
                        "icon": icon,
                        "args": {},
                        "result": "Yanıt critic tarafından onaylandı."
                    })
                    yield {"agent_step": agent_steps[-1]}
                    yield {"agent_steps_complete": agent_steps}
                    yield {"final_answer": final_answer}
                    return
                else:
                    observations.append({
                        "tool": "critic_feedback",
                        "args": {"feedback": critic_feedback},
                        "result": f"DÜZELTME GEREKLİ: {critic_feedback}"
                    })
                    yield {"agent_step": {
                        "iteration": iteration + 1,
                        "tool": "critic",
                        "icon": "🧐",
                        "args": {},
                        "result": f"Düzeltme isteniyor: {critic_feedback}"
                    }}
                    continue

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

        yield {"status": "Yanıt oluşturuluyor..."}
        fallback_answer = self._generate_fallback_answer(user_query, observations, chat_history)
        yield {"agent_steps_complete": agent_steps}
        yield {"final_answer": fallback_answer}

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
            elif tool_name == "lookup_kg_entity":
                entity = args.get("entity", "")
                type_filter = args.get("type_filter", None)
                return self.kg.lookup_entity(entity, k=3, type_filter=type_filter)
            elif tool_name == "explore_kg_path":
                from_entity = args.get("from_entity", "")
                to_entity   = args.get("to_entity", "")
                max_hops    = int(args.get("max_hops", 3))
                return self.kg.explore_path(from_entity, to_entity, max_hops=max_hops)
            elif tool_name == "calculate":
                expression = args.get("expression", "")
                return self._safe_calculate(expression)
            elif tool_name == "lookup_knowledge_graph":
                entity = args.get("entity", "")
                return self.kg.lookup_entity(entity, k=3)
            else:
                return f"Bilinmeyen araç: {tool_name}"
        except Exception as e:
            return f"[ARAÇ HATASI] {tool_name}: {str(e)}"

    def _retrieve_chunks(self, query: str, k: int) -> List[Dict]:
        if not self.conn: return []
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
            if self.conn: self.conn.rollback()
            return []

    def _fulltext_search(self, query: str, k: int) -> List[Dict]:
        if not self.conn: return []
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
            if self.conn: self.conn.rollback()
            return []

    def _safe_calculate(self, expression: str) -> str:
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow, "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor}
        try:
            safe_expr = "".join(c for c in expression if c in "0123456789+-*/.() ")
            result = eval(safe_expr, {"__builtins__": {}}, allowed_names)
            return f"Hesaplama: {expression} = {result}"
        except Exception as e:
            return f"Hesaplama hatası: {str(e)}"

    def _format_chunks_result(self, chunks: List[Dict]) -> str:
        if not chunks: return "Araştırma sonucu bulunamadı. Farklı anahtar kelimeler deneyin."
        parts = []
        for i, c in enumerate(chunks):
            headers = [v for key, v in c['metadata'].items() if key.startswith("Header")]
            breadcrumb = " > ".join(headers) if headers else "Bölüm"
            score_str = f" (skor: {c.get('score', 0):.2f})" if 'score' in c else ""
            parts.append(f"--- KAYNAK {i+1}{score_str} ---\nBAŞLIK: {breadcrumb}\nİÇERİK:\n{c['text'][:800]}\n")
        return "\n".join(parts)

    def _format_observations(self, observations: List[Dict]) -> str:
        if not observations: return ""
        parts = []
        for i, o in enumerate(observations):
            tool = o.get("tool", "?")
            args_str = json.dumps({k: v for k, v in o.get("args", {}).items() if k != "tool"}, ensure_ascii=False)
            result_preview = str(o.get("result", ""))[:600]
            parts.append(f"[Adım {i+1}] Araç: {tool} | Girdi: {args_str}\nSonuç:\n{result_preview}")
        return "\n\n".join(parts)

    def _verify_with_critic(self, user_query: str, final_answer: str, observations: str) -> str:
        """Second-pass verification by a separate LLM call."""
        critic_prompt = self.CRITIC_PROMPT.format(
            user_query=user_query,
            observations=observations,
            final_answer=final_answer
        )
        try:
            # Use same LLM for verification but with higher precision focus
            response = self.llm.invoke([HumanMessage(content=critic_prompt)])
            return response.content.strip()
        except Exception as e:
            return "TAMAM" # Fallback if critic fails, don't block user

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
                if content: full_response += content
            return full_response
        except Exception as e: return f"Hata: {str(e)}"
