# sut_rag_core.py
# Description: Pure SUT RAG Engine (Retrieval & Agentic Loop only), PostgreSQL Edition

import os
import json
import psycopg2
from typing import List, Dict, Generator
from sentence_transformers import CrossEncoder 

# LangChain & AI Libraries
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class SUT_RAG_Engine:
    def __init__(self, llm_provider: str = "google", model_name: str = "gemini-2.0-flash"):
        self.embeddings_model = self._initialize_embeddings()
        print("[INIT] Loading Reranker Model...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
        self.conn = None
        self.cursor = None
        self.llm = None
        self.provider = llm_provider

        if llm_provider == "google":
            self._init_google_llm(model_name)
        elif llm_provider == "openrouter":
            self._init_openrouter_llm(model_name)
        else:
            self._init_google_llm("gemini-1.5-flash") # Default

        print(f"[INIT] SUT Engine Initialized. Provider: '{llm_provider}', Model: '{model_name}'")

    def _init_google_llm(self, model_name: str):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.1)

    def _init_openrouter_llm(self, model_name: str):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key: return
        self.llm = ChatOpenAI(
            model=model_name, openai_api_key=api_key, openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1
        )

    def _initialize_embeddings(self):
        return HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2", model_kwargs={'device': 'cpu'})

    def load_database(self) -> bool:
        try:
            self.conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            print(f"[WARN] Postgres database connection failed: {e}")
            return False

    def query_agentic_rag_stream(self, user_query: str, chat_history: List[Dict] = None, k: int = 5) -> Generator[Dict, None, None]:
        if chat_history is None:
            chat_history = []
        if self.llm is None:
            yield {"error": "LLM not initialized."}
            return

        yield {"status": "Sorgu genişletiliyor..."}
        expanded_query = self._expand_query(user_query)

        yield {"status": "SUT veritabanı taranıyor..."}
        chunks = self._retrieve_chunks(expanded_query, k)
        
        if not chunks:
            yield {"final_answer": "SUT içinde ilgili bilgi bulunamadı."}
            return

        context_str = ""
        for i, c in enumerate(chunks):
            headers = [v for key, v in c['metadata'].items() if key.startswith("Header")]
            breadcrumb = " > ".join(headers) if headers else "Bölüm"
            chunk_text = c['text']
            formatted = f"\n--- KAYNAK {i+1} ---\nBAŞLIK: {breadcrumb}\nİÇERİK:\n{chunk_text}\n"
            context_str += formatted

        system_prompt = f"""Role: SUT Uzmanı (Sağlık Uygulama Tebliği).
CONTEXT:
{context_str}

KURALLAR:
1. Öncelikle yukarıdaki bağlamı (SUT metinlerini) kullanarak cevap ver.
2. Eğer sorulan soru bağlamda yer almıyorsa, kendi tıbbi ve mevzuat bilgini kullanarak cevaplayabilirsin ancak cümlenin başında "SUT metninde doğrudan yer almamakla birlikte..." veya "Mevcut SUT bağlamında bulunmamasına rağmen genel uygulamada..." gibi bir uyarı EKLEMEK ZORUNDASIN.
2. Metin içinde bilgi aldığın kaynaklara atıf yap (örn: [Kaynak 1]).
3. Cevabının EN SONUNA, SADECE kullandığın kaynakların başlık ve içeriklerini aşağıdaki formatta ekle:
<KAYNAKLAR>
<KAYNAK baslik="Kaynak 1: [BAŞLIK BURAYA]">
[KAYNAĞIN TAM İÇERİĞİ BURAYA]
</KAYNAK>
</KAYNAKLAR>
Eğer hiçbir kaynak kullanmadıysan bu bölümü ekleme.
"""
        
        try:
            messages = [SystemMessage(content=system_prompt)]
            for msg in chat_history:
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
                    yield {"final_answer": full_response}
        except Exception as e:
            yield {"error": f"LLM Generation Error: {str(e)}"}

    def _retrieve_chunks(self, query: str, k: int) -> List[Dict]:
        if not self.conn: return []
        initial_k = k * 3
        q_vec = self.embeddings_model.embed_query(query)
        q_vec_str = "[" + ",".join(map(str, q_vec)) + "]"
        
        self.cursor.execute("""
            SELECT chunk_id, text_content, metadata_json 
            FROM chunks 
            ORDER BY embedding <=> %s 
            LIMIT %s
        """, (q_vec_str, initial_k))
        
        candidates = []
        for row in self.cursor.fetchall():
            meta = row[2] if isinstance(row[2], dict) else json.loads(row[2])
            candidates.append({"id": row[0], "text": row[1], "metadata": meta})

        if candidates:
            pairs = [[query, doc['text']] for doc in candidates]
            scores = self.reranker.predict(pairs)
            for doc, score in zip(candidates, scores): doc['score'] = score
            candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates[:k]

    def _expand_query(self, user_query: str) -> str:
        # Simple expansion to avoid logic bloat
        return user_query