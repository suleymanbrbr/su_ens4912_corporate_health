# sut_rag_core.py
# Description: Pure SUT RAG Engine (Retrieval & Agentic Loop only)

import os
import json
import sqlite3
import numpy as np
import faiss
from typing import List, Dict, Generator
from sentence_transformers import CrossEncoder 

# LangChain & AI Libraries
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from rag_storage import DB_PATH, FAISS_INDEX_PATH, FAISS_MAPPING_PATH

class SUT_RAG_Engine:
    def __init__(self, llm_provider: str = "google", model_name: str = "gemini-2.0-flash"):
        self.embeddings_model = self._initialize_embeddings()
        print("[INIT] Loading Reranker Model...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
        self.conn = None
        self.cursor = None
        self.faiss_index = None
        self.id_mapping = None
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
        if not os.path.exists(FAISS_INDEX_PATH): return False
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
            self.id_mapping = json.load(f)
        return True

    def query_agentic_rag_stream(self, user_query: str, k: int = 5) -> Generator[Dict, None, None]:
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
            headers = [v for k, v in c['metadata'].items() if k.startswith("Header")]
            breadcrumb = " > ".join(headers) if headers else "Bölüm"
            chunk_text = c['text']
            formatted = f"\n--- KAYNAK {i+1} ---\nBAŞLIK: {breadcrumb}\nİÇERİK:\n{chunk_text}\n"
            context_str += formatted
            yield {"source": {"title": breadcrumb, "content": chunk_text}}

        system_prompt = f"Role: SUT Expert. Answer using context only.\nContext:\n{context_str}\nRules: Turkish only. Cite articles."
        
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]
            full_response = ""
            for chunk in self.llm.stream(messages):
                content = chunk.content if hasattr(chunk, 'content') else ""
                if content:
                    full_response += content
                    yield {"final_answer": full_response}
        except Exception as e:
            yield {"error": f"LLM Generation Error: {str(e)}"}

    def _retrieve_chunks(self, query: str, k: int) -> List[Dict]:
        if not self.faiss_index: return []
        initial_k = k * 3
        q_vec = np.array([self.embeddings_model.embed_query(query)]).astype('float32')
        _, indices = self.faiss_index.search(q_vec, initial_k)
        
        combined_ids = []
        for idx in indices[0]:
            if idx != -1: combined_ids.append(self.id_mapping[idx])

        candidates = []
        for cid in combined_ids:
            self.cursor.execute("SELECT chunk_id, text_content, metadata_json FROM chunks WHERE chunk_id=?", (cid,))
            row = self.cursor.fetchone()
            if row:
                candidates.append({"id": row[0], "text": row[1], "metadata": json.loads(row[2])})

        if candidates:
            pairs = [[query, doc['text']] for doc in candidates]
            scores = self.reranker.predict(pairs)
            for doc, score in zip(candidates, scores): doc['score'] = score
            candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates[:k]

    def _expand_query(self, user_query: str) -> str:
        # Simple expansion to avoid logic bloat
        return user_query