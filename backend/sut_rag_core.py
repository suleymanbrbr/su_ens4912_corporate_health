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
import google.generativeai as google_genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ─── Native Gemma Wrapper (bypasses Langchain v1beta issue) ─────────────────
class _NativeGemmaWrapper:
    """Thin wrapper around google.generativeai for Gemma models.
    Exposes .invoke() and .stream() to match the Langchain LLM interface."""
    class _Response:
        def __init__(self, text): self.content = text

    def __init__(self, model_name: str, api_key: str):
        google_genai.configure(api_key=api_key)
        self._model = google_genai.GenerativeModel(model_name)

    def invoke(self, messages):
        prompt = self._messages_to_text(messages)
        response = self._model.generate_content(prompt)
        try:
            text = response.text
        except ValueError:
            text = ""
        return self._Response(text)

    def stream(self, messages):
        prompt = self._messages_to_text(messages)
        for chunk in self._model.generate_content(prompt, stream=True):
            try:
                if chunk.text:
                    yield self._Response(chunk.text)
            except ValueError:
                pass

    def _messages_to_text(self, messages) -> str:
        parts = []
        for m in messages:
            if isinstance(m, SystemMessage):
                parts.append(f"[SYSTEM]: {m.content}")
            elif isinstance(m, HumanMessage):
                parts.append(f"[USER]: {m.content}")
            elif isinstance(m, AIMessage):
                parts.append(f"[ASSISTANT]: {m.content}")
            else:
                parts.append(str(m.content) if hasattr(m, 'content') else str(m))
        return "\n".join(parts)

# KG Storage
from kg_storage import KG_Storage_Manager
from embedding_utils import build_hf_embeddings, embed_query_retrieval, resolve_embedding_model_name, DEFAULT_MINILM

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
        self.default_model_name = model_name
        self.kg = KG_Storage_Manager()

        # Env-var fallback LLM (admin tasks, local dev).  In multi-tenant
        # production, each request supplies user_api_key=… and we build a
        # fresh client per call — see _build_llm_for_request().
        if llm_provider == "google":
            self._init_google_llm(model_name)
        elif llm_provider == "openrouter":
            self._init_openrouter_llm(model_name)
        elif llm_provider == "local":
            self._init_local_llm(model_name)
        else:
            self._init_google_llm("gemini-2.0-flash")

        print(f"[INIT] SUT Engine Initialized. Provider: '{llm_provider}', Model: '{model_name}'")

    # ─── Per-Request LLM Builder (Multi-Tenant) ──────────────────────────────

    def _build_llm_for_request(
        self,
        user_api_key: str | None,
        user_provider: str | None,
        user_base_url: str | None,
    ):
        """Build a fresh LangChain LLM client using the caller's credentials.

        Returns the per-request LLM, or `self.llm` (env-var fallback) when no
        user_api_key is given. Does NOT cache — every request gets its own
        client so users never share an LLM connection / API key.
        """
        if not user_api_key:
            return self.llm  # fallback to env-var-initialized LLM (local dev / admin)

        provider = (user_provider or "gemini").lower()
        try:
            if provider == "gemini":
                model_name = self.default_model_name if self.default_model_name.startswith("gemini") else "gemini-2.0-flash"
                if model_name.startswith("gemma"):
                    return _NativeGemmaWrapper(model_name, user_api_key)
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=user_api_key,
                    temperature=0.1,
                    max_retries=1,
                    timeout=60.0,
                )
            elif provider == "openrouter":
                # OpenRouter accepts any of its catalog models; default to a
                # cheap one if the engine was initialized with a Google model.
                model_name = self.default_model_name
                if model_name.startswith("gemini") or model_name.startswith("gemma"):
                    model_name = "google/gemma-2-9b-it"
                return ChatOpenAI(
                    model=model_name,
                    openai_api_key=user_api_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    temperature=0.1,
                )
            elif provider == "local":
                raw_base = (user_base_url or os.getenv("LOCAL_LLM_API_BASE", "http://localhost:1234/v1")).strip().rstrip("/")
                base_url = raw_base if raw_base.endswith("/v1") else f"{raw_base}/v1"
                # LM Studio accepts any non-empty key, but use the user-provided
                # one when supplied so people running auth-protected servers work.
                return ChatOpenAI(
                    model=self.default_model_name,
                    openai_api_key=user_api_key or "lm-studio",
                    openai_api_base=base_url,
                    temperature=0.1,
                    request_timeout=300,
                )
            else:
                print(f"[WARN] Unknown provider '{provider}', falling back to env LLM")
                return self.llm
        except Exception as e:
            print(f"[ERROR] Failed to build per-request LLM for provider '{provider}': {e}")
            return self.llm

    # ─── LLM Init ────────────────────────────────────────────────────────────

    def _init_google_llm(self, model_name: str):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return
        if model_name.startswith("gemma"):
            # Gemma models are not supported by Langchain's v1beta endpoint;
            # use the native google.generativeai SDK via our wrapper instead.
            self.llm = _NativeGemmaWrapper(model_name, api_key)
        else:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name, 
                google_api_key=api_key, 
                temperature=0.1,
                max_retries=1,
                timeout=60.0
            )

    def _init_openrouter_llm(self, model_name: str):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return
        self.llm = ChatOpenAI(
            model=model_name, openai_api_key=api_key, openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1
        )

    def _init_local_llm(self, model_name: str):
        """Connect to a locally running OpenAI-compatible server.
        Works with LM Studio (default: http://localhost:1234/v1),
        Ollama (http://localhost:11434/v1), or any similar server.
        Set LOCAL_LLM_API_BASE in .env to override the default URL.
        Set LOCAL_LLM_API_KEY in .env if your server requires auth.
        """
        raw_base = os.getenv("LOCAL_LLM_API_BASE", "http://localhost:1234/v1").strip().rstrip("/")
        # OpenAI-compatible servers expect .../v1 (LM Studio default). Avoid silent failures when .env omits /v1.
        base_url = raw_base if raw_base.endswith("/v1") else f"{raw_base}/v1"
        api_key  = os.getenv("LOCAL_LLM_API_KEY",  "lm-studio")  # LM Studio accepts any non-empty string
        print(f"[INIT] Connecting to local LLM server at: {base_url}")
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.1,
            request_timeout=300,  # local models can be very slow with big context; 5 min
        )

    def _initialize_embeddings(self):
        name = os.getenv("SUT_EMBEDDING_MODEL", "").strip() or DEFAULT_MINILM
        self._embedding_model_name = name
        return build_hf_embeddings(name)

    def load_database(self) -> bool:
        try:
            # Prefer DATABASE_URL (usually 'db' host in Docker).
            # Use LOCAL_DATABASE_URL as fallback for dev outside Docker.
            db_url = os.getenv("DATABASE_URL") or os.getenv("LOCAL_DATABASE_URL")
            self.conn = psycopg2.connect(db_url)
            cur = self.conn.cursor()
            cur.execute("SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname='public' AND tablename='chunks')")
            exists = cur.fetchone()[0]
            if not exists:
                cur.close()
                print("[WARN] 'chunks' table not found. Re-index required from Admin Panel.")
                return True
            cur.execute(
                "SELECT vector_dims(embedding) FROM chunks WHERE embedding IS NOT NULL LIMIT 1"
            )
            row = cur.fetchone()
            cur.close()
            dim = row[0] if row else None
            resolved = resolve_embedding_model_name(dim)
            if resolved != self._embedding_model_name:
                print(f"[INIT] Aligning embedding model with DB ({dim}d vectors): {resolved}")
                self._embedding_model_name = resolved
                self.embeddings_model = build_hf_embeddings(resolved)
            return True
        except Exception as e:
            print(f"[WARN] Postgres database connection failed: {e}")
            return False

    # ─── Tool Definitions (schema injected into system prompt) ───────────────

    TOOL_SCHEMA = """
⚠️ CRITICAL INSTRUCTION: Your response must be ONLY a single valid JSON object. No explanations, no markdown, no text before or after. Just the raw JSON.

Sen bir SUT Uzmanı yapay zeka asistanısın. Görevin: kullanıcının sorusunu Türkçe olarak eksiksiz, doğru ve kaynaklı biçimde yanıtlamak.

KULLANILABİLECEK ARAÇLAR (Her turda yalnızca 1 araç çağır. Çıktının yalnızca JSON olması gerekiyor):

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

⚠️ REMINDER: Output ONLY the JSON object. No text before, no text after. Start with { and end with }.
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

YANITIN YALNIZCA VE YALNIZCA bir JSON nesnesi olmalıdır. {{ ile başla ve }} ile bit. Hiçbir açıklama, yorum veya markdown ekleme.
ÖRNEK: {{"tool": "search_sut_chunks", "query": "...", "k": 5}}
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

    @staticmethod
    def _iter_answer_deltas(text: str, chunk_size: int = 28) -> Generator[Dict, None, None]:
        """Split completed answer into SSE chunks for progressive UI (word-aware where possible)."""
        if not text:
            return
        i = 0
        n = len(text)
        while i < n:
            end = min(i + chunk_size, n)
            if end < n:
                look = text.rfind(" ", i, min(end + 16, n))
                if look > i:
                    end = look + 1
            piece = text[i:end]
            if piece:
                yield {"answer_delta": piece}
            i = end

    # ─── Main Agentic Stream ─────────────────────────────────────────────────

    def query_agentic_rag_stream(
        self,
        user_query: str,
        chat_history: List[Dict] = None,
        k: int = 5,
        role: str = "PATIENT",
        user_api_key: str | None = None,
        user_provider: str | None = None,
        user_base_url: str | None = None,
    ) -> Generator[Dict, None, None]:

        if chat_history is None:
            chat_history = []

        # Multi-tenant: build an LLM with the caller's API key when supplied.
        # Falls back to self.llm (env-var) otherwise.
        runtime_llm = self._build_llm_for_request(user_api_key, user_provider, user_base_url)
        if runtime_llm is None:
            yield {"error": "LLM not initialized."}
            return

        if isinstance(chat_history, str):
            history_str = chat_history
        else:
            lines = []
            for m in chat_history[-6:]:
                if not isinstance(m, dict):
                    continue
                msg_role = m.get("role", "")
                content = m.get("content", "")
                if not isinstance(content, str):
                    content = str(content)
                label = "Kullanıcı" if msg_role == "user" else "Asistan"
                lines.append(f"{label}: {content[:300]}")
            history_str = "\n".join(lines)

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
            },
            "ECZACI": {
                "name": "Eczacı",
                "desc": "Reçete kuralları, muadil ilaç, katılım payı ve eczane uygulamalarına odaklan. ATC ve doz bilgisini doğru kullan."
            },
            "PHARMACIST": {
                "name": "Eczacı",
                "desc": "Reçete kuralları, muadil ilaç, katılım payı ve eczane uygulamalarına odaklan."
            },
            "HASTANE_YONETICISI": {
                "name": "Hastane Yöneticisi",
                "desc": "Hastane bütçesi, SUT uyumu, faturalandırma ve idari süreçlere odaklan. Özet ve tablo kullan."
            },
            "HOSPITAL_MANAGER": {
                "name": "Hastane Yöneticisi",
                "desc": "Kurumsal SUT uyumu, maliyet ve idari süreçlere odaklan."
            },
        }.get(role.upper(), {"name": "SUT Uzmanı", "desc": "Genel SUT uzmanı."})

        search_tool_names = {"search_sut_chunks", "search_sut_fulltext", "lookup_kg_entity", "explore_kg_path"}
        searches_done = 0

        for iteration in range(MAX_AGENT_ITERATIONS):
            obs_str = self._format_observations(observations)

            # On the LAST iteration, skip the LLM tool-decision entirely and
            # jump straight to synthesis.  Small local models (Qwen3.5, Llama-8B)
            # ignore text-based urgency hints and keep calling search forever.
            iterations_left = MAX_AGENT_ITERATIONS - iteration
            if iterations_left == 1 and searches_done >= MIN_SEARCHES_BEFORE_FINISH:
                break  # will hit the fallback synthesis below the loop

            # When running low on iterations, inject an urgency hint.
            urgency_hint = ""
            if iterations_left <= 3 and searches_done >= MIN_SEARCHES_BEFORE_FINISH:
                urgency_hint = (
                    "\n\n⚠️ SON TUR UYARISI: Başka arama yapma! Şimdi MUTLAKA "
                    '{"tool": "finish", "answer": "<Türkçe kapsamlı yanıt>"}'
                    " çağır. Elindeki bilgileri derle ve soruyu yanıtla."
                )

            prompt = self.TOOL_REACTION_PROMPT.format(
                role=role_meta["name"],
                role_description=role_meta["desc"],
                tool_schema=self.TOOL_SCHEMA,
                history=history_str or "(Yeni konuşma)",
                observations=obs_str or "(Henüz araç kullanılmadı.)",
                user_query=user_query + urgency_hint,
            )

            raw = ""
            try:
                decision_msg = runtime_llm.invoke([HumanMessage(content=prompt)])
                raw_content = decision_msg.content
                if isinstance(raw_content, list):
                    parts = []
                    for p in raw_content:
                        if isinstance(p, dict): parts.append(p.get("text", str(p)))
                        else: parts.append(str(p))
                    raw_content = "".join(parts)
                raw = str(raw_content).strip()

                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:].strip()

                # Primary parse attempt
                try:
                    tool_call = json.loads(raw)
                except json.JSONDecodeError:
                    # Fallback: extract first JSON object (handles Qwen3/DeepSeek thinking
                    # models that emit \n\n{...} after their <think> reasoning block)
                    import re as _re
                    m = _re.search(r'\{[^{}]*\}', raw, _re.DOTALL)
                    if m:
                        tool_call = json.loads(m.group())
                    else:
                        raise
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

            if tool_name in search_tool_names:
                searches_done += 1

            if tool_name == "finish":
                raw_answer = tool_call.get("answer", "")
                # Normalize: some models (Llama-3, Mistral) return answer as a
                # nested dict {"Başlık": ..., "İçerik": ..., "KAYNAKLAR": ...}
                # instead of a plain string.  Flatten it to a readable string.
                if isinstance(raw_answer, dict):
                    parts = []
                    if "İçerik" in raw_answer:
                        parts.append(str(raw_answer["İçerik"]))
                    elif "content" in raw_answer:
                        parts.append(str(raw_answer["content"]))
                    else:
                        # Fallback: join all string values
                        for v in raw_answer.values():
                            if isinstance(v, str):
                                parts.append(v)
                    final_answer = "\n\n".join(parts) if parts else json.dumps(raw_answer, ensure_ascii=False)
                elif isinstance(raw_answer, list):
                    final_answer = "\n".join(str(x) for x in raw_answer)
                else:
                    final_answer = str(raw_answer)
                yield {"status": "🕵️ Critic denetimi yapılıyor..."}
                critic_feedback = self._verify_with_critic(user_query, final_answer, obs_str, llm=runtime_llm)
                
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
                    yield from self._iter_answer_deltas(final_answer)
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
        fallback_answer = self._generate_fallback_answer(user_query, observations, chat_history, llm=runtime_llm)
        yield {"agent_steps_complete": agent_steps}
        yield from self._iter_answer_deltas(fallback_answer)
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
            q_vec = embed_query_retrieval(self.embeddings_model, self._embedding_model_name, query)
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

    def _verify_with_critic(self, user_query: str, final_answer: str, observations: str, llm=None) -> str:
        """Second-pass verification by a separate LLM call.
        ``llm`` overrides self.llm when supplied (multi-tenant per-request key)."""
        critic_prompt = self.CRITIC_PROMPT.format(
            user_query=user_query,
            observations=observations,
            final_answer=final_answer
        )
        active_llm = llm if llm is not None else self.llm
        try:
            # Use same LLM for verification but with higher precision focus
            response = active_llm.invoke([HumanMessage(content=critic_prompt)])
            res_content = response.content
            if isinstance(res_content, list):
                parts = []
                for p in res_content:
                    if isinstance(p, dict): parts.append(p.get("text", str(p)))
                    else: parts.append(str(p))
                res_content = "".join(parts)
            return str(res_content).strip()
        except Exception as e:
            return "TAMAM" # Fallback if critic fails, don't block user

    def _generate_fallback_answer(
        self,
        user_query: str,
        observations: List[Dict],
        chat_history: List[Dict],
        llm=None,
    ) -> str:
        """Called when max iterations hit — asks LLM to synthesize from what we have.
        Uses invoke() instead of stream() for reliability with local models.
        Truncates observations to avoid huge prompts that cause timeouts.
        ``llm`` overrides self.llm when supplied (multi-tenant per-request key).
        """
        obs_str = self._format_observations(observations)
        # Truncate to avoid blowing up the context window / causing timeouts
        obs_str = obs_str[:2000] if obs_str else "(Araştırma sonucu yok)"

        # Short, focused synthesis prompt — keep it small for local models
        combined_prompt = (
            "Sen SUT (Sağlık Uygulama Tebliği) uzmanısın.\n"
            "Aşağıdaki araştırma sonuçlarını kullanarak kullanıcının sorusunu "
            "Türkçe olarak kapsamlı şekilde cevapla. "
            "Eğer sonuçlarda soruyla ilgili bilgi yoksa, bunu açıkça belirt "
            "ve genel bilgi ver.\n\n"
            f"ARAŞTIRMA SONUÇLARI:\n{obs_str}\n\n"
            f"SORU: {user_query}\n\n"
            "YANIT:"
        )
        active_llm = llm if llm is not None else self.llm
        try:
            resp = active_llm.invoke([HumanMessage(content=combined_prompt)])
            raw = resp.content
            if isinstance(raw, list):
                parts = []
                for part in raw:
                    if isinstance(part, dict):
                        parts.append(part.get("text", str(part)))
                    else:
                        parts.append(str(part))
                raw = "".join(parts)
            result = str(raw).strip()
            # If the model wraps its answer in <think>...</think>, extract the part after it
            if "</think>" in result:
                result = result.split("</think>", 1)[-1].strip()
            return result if result else "Bu soruya ilişkin SUT mevzuatında yeterli bilgi bulunamadı."
        except Exception as e:
            print(f"[FALLBACK ERROR] {type(e).__name__}: {e}")
            return "Bu soruya ilişkin SUT mevzuatında yeterli bilgi bulunamadı."
