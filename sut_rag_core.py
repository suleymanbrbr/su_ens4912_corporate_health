import pypandoc
import sqlite3
import json
import faiss
import numpy as np
import uuid
import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI


from typing import List, Dict, Tuple

# --- LangGraph Agent Imports ---
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage, HumanMessage

# --- Configuration ---
DOCX_FILE_PATH = "08.03.2025-Değişiklik Tebliği İşlenmiş Güncel 2013 SUT.docx"
MARKDOWN_FILE_PATH = "sut_converted.md"
DB_PATH = "sut_knowledge_base.db"
FAISS_INDEX_PATH = "sut_faiss.index"
FAISS_MAPPING_PATH = "sut_faiss.index.mapping"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class SUT_RAG_Engine:
    def __init__(self):
        self.embeddings_model = self._initialize_embeddings()
        self.conn = None
        self.cursor = None
        self.faiss_index = None
        self.id_mapping = None

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[WARNING] GEMINI_API_KEY ortam değişkeni ayarlanmamış.")
            self.llm = None
        else:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    api_key=api_key,
                    temperature=0.2,
                    max_output_tokens=2048
                )
                print("[INIT] Gemini LLM başarıyla başlatıldı (LangChain Agentic RAG).")
            except Exception as e:
                print(f"[ERROR] Gemini LLM başlatılırken hata oluştu: {e}")
                self.llm = None




    def _initialize_embeddings(self):
        """Loads the multilingual Sentence Transformer embedding model."""
        print(f"[INIT] Loading embedding model: '{EMBEDDING_MODEL}'...")
        model_kwargs = {'device': 'cpu'} 
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=model_kwargs
        )
        return embeddings

    def _convert_docx_to_markdown(self) -> str or None:
        """Converts DOCX to Markdown using pandoc."""
        if not os.path.exists(DOCX_FILE_PATH):
            print(f"[ERROR] Source file not found at {DOCX_FILE_PATH}")
            return None
        print(f"[PREP] Converting '{DOCX_FILE_PATH}' to Markdown...")
        try:
            pypandoc.convert_file(DOCX_FILE_PATH, 'md', outputfile=MARKDOWN_FILE_PATH)
            return MARKDOWN_FILE_PATH
        except Exception as e:
            print(f"[ERROR] during DOCX conversion: {e}")
            return None

    def _get_markdown_chunks(self, md_path) -> List:
        """Chunks the Markdown file based on SUT headers."""
        print("[PREP] Chunking Markdown file based on headers...")
        headers_to_split_on = [("#", "Header 1"),("##", "Header 2"),("###", "Header 3"),("####", "Header 4"),]
        with open(md_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        chunks = markdown_splitter.split_text(markdown_text)
        print(f"[PREP] Document split into {len(chunks)} chunks.")
        if os.path.exists(md_path):
            os.remove(md_path)
        return chunks

    def _setup_database(self):
        """Creates the SQLite database (Document Store) and clears old FAISS files."""
        if os.path.exists(DB_PATH): os.remove(DB_PATH)
        if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
        if os.path.exists(FAISS_MAPPING_PATH): os.remove(FAISS_MAPPING_PATH)
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
        self.cursor.execute("CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, text_content TEXT NOT NULL, metadata_json TEXT);")
        self.conn.commit()
        print(f"[DB] SQLite database '{DB_PATH}' setup complete.")

    def populate_database(self):
        """Runs the entire ingestion and indexing pipeline (SQLite + FAISS)."""
        md_file = self._convert_docx_to_markdown()
        if not md_file: return
        chunks = self._get_markdown_chunks(md_file)
        if not chunks: return
        self._setup_database()
        print("[DB] Populating SQLite and preparing for FAISS...")
        texts_to_embed, string_ids = [], []
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            metadata_json = json.dumps(chunk.metadata)
            self.cursor.execute("INSERT INTO chunks (chunk_id, text_content, metadata_json) VALUES (?, ?, ?)", (chunk_id, chunk.page_content, metadata_json))
            texts_to_embed.append(chunk.page_content)
            string_ids.append(chunk_id)
        self.conn.commit()
        print("[DB] All chunks saved to SQLite.")
        print("[DB] Embedding texts for FAISS...")
        vectors = self.embeddings_model.embed_documents(texts_to_embed)
        d = len(vectors[0])
        vector_array = np.array(vectors).astype('float32')
        index = faiss.IndexFlatL2(d)
        index.add(vector_array)
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
            json.dump(string_ids, f)
        self.faiss_index, self.id_mapping = index, string_ids
        print(f"[DB] FAISS index created and saved with {index.ntotal} vectors.")
        
    def _retrieve_chunks(self, query: str, k: int) -> List[Dict]:
        """Helper function to perform vector search and retrieve chunk data from SQLite."""
        if not self.faiss_index:
            return []
        query_vector = self.embeddings_model.embed_query(query)
        query_vector_np = np.array([query_vector]).astype('float32')
        _, indices = self.faiss_index.search(query_vector_np, k)
        retrieved_chunks = []
        for faiss_id in indices[0]:
            if faiss_id == -1: continue
            chunk_id = self.id_mapping[faiss_id]
            self.cursor.execute("SELECT chunk_id, text_content, metadata_json FROM chunks WHERE chunk_id = ?", (chunk_id,))
            result = self.cursor.fetchone()
            if result:
                retrieved_chunk_id, text, metadata_str = result
                retrieved_chunks.append({"id": retrieved_chunk_id, "text": text, "metadata": json.loads(metadata_str)})
        return retrieved_chunks

    def get_chunk_content_by_id(self, chunk_id: str) -> str:
        """Internal method to retrieve the full text content of a specific chunk by its ID."""
        print(f"[AGENT TOOL USED] Fetching content for chunk_id: {chunk_id}")
        self.cursor.execute("SELECT text_content FROM chunks WHERE chunk_id = ?", (chunk_id,))
        result = self.cursor.fetchone()
        return result[0] if result else "Hata: Belirtilen ID ile içerik bulunamadı."

    def query_agentic_rag(self, query: str, k: int = 5) -> Tuple[str, List[Dict]]:
        """Executes a multi-step agentic RAG query using LangGraph."""
        if self.llm is None:
            return "[LLM DEVRE DIŞI] Lütfen GEMINI_API_KEY'i ayarlayın.", []

        candidate_chunks = self._retrieve_chunks(query, k)
        if not candidate_chunks:
            return "Üzgünüm, bu soruya SUT içerisinde alakalı bir bilgi bulamadım.", []

        tools = [
            Tool(
                name="get_sut_section_content",
                func=self.get_chunk_content_by_id,
                description="SUT'un belirli bir bölümünün tam metnini almak için bu aracı kullan. Yalnızca sana sağlanan listedeki 'ID'lerden birini kullanmalısın."
            )
        ]
        
        agent_executor = create_react_agent(self.llm, tools)

        context_summary = ""
        for i, chunk in enumerate(candidate_chunks):
            section_info = '; '.join([v for k, v in chunk['metadata'].items() if k.startswith('Header') and v])
            context_summary += f"[{i+1}] ID: '{chunk['id']}'\n    Başlık: {section_info}\n    Önizleme: {chunk['text'][:150].strip()}...\n\n"

        system_message = """SEN BİR SAĞLIK UZMANI ASİSTANISIN. Görevin, SUT (Sağlık Uygulama Tebliği) hakkındaki soruları cevaplamaktır.

GÖREVİN:
1.  **Analiz Et:** Sağlanan bölüm özetlerini dikkatlice incele. Soruyu cevaplamak için yeterli olup olmadıklarına karar ver.
2.  **Araç Kullan:** Eğer özetler yetersizse veya daha fazla detaya ihtiyacın varsa, en alakalı görünen bölümün tam metnini almak için `get_sut_section_content` aracını KULLANMALISIN. Aracın `chunk_id` parametresi olarak sana verilen ID'yi kullan.
3.  **Cevapla:** Gerekli tüm bilgileri topladıktan sonra, kullanıcı sorusunu net ve Türkçe olarak yanıtla. Cevabında kullandığın SUT maddesini açıkça belirt. Eğer bilgi SUT'ta yoksa, "SUT'ta bu konuyla ilgili spesifik bir bilgi bulunmamaktadır." de."""

        human_input = f"Kullanıcı Sorusu: {query}\n\nİlgili Olabilecek SUT Bölümleri:\n---\n{context_summary}"
        
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_input)
        ]
        
        try:
            # LangGraph ajanını çalıştırıyoruz — tool kullanabilir
            result = agent_executor.invoke({"messages": messages})

            # Bazı modellerde 'output' yerine 'final_output' döner
            if isinstance(result, dict):
                final_answer = result.get("output") or result.get("final_output") or str(result)
            else:
                final_answer = str(result)

        except Exception as e:
            final_answer = f"[AGENT ERROR] Ajan çalıştırılırken hata: {e}"

        return final_answer, candidate_chunks

        
    def load_database(self):
        """Loads an existing SQLite and FAISS index from disk."""
        if not all([os.path.exists(DB_PATH), os.path.exists(FAISS_INDEX_PATH), os.path.exists(FAISS_MAPPING_PATH)]):
            return False
        print(f"[DB] Loading SQLite from: '{DB_PATH}' and FAISS from: '{FAISS_INDEX_PATH}'")
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False) 
        self.cursor = self.conn.cursor()
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
            self.id_mapping = json.load(f)
        print(f"[DB] Veritabanı yüklendi. (FAISS {self.faiss_index.ntotal} belge içeriyor)")
        return True

    def __del__(self):
        """Closes the database connection on object deletion."""
        if self.conn:
            self.conn.close()

            # sut_rag_core.py dosyasının içine EKLENECEK YENİ FONKSİYON

# sut_rag_core.py dosyasındaki get_section_by_title fonksiyonunun TAMAMI

    def get_section_by_title(self, section_title: str) -> str:
        """
        Retrieves the full text content of a section by its title/header using a flexible search.
        """
        print(f"[AGENT TOOL USED] Searching for section by title: '{section_title}'")
        
        # --- DÜZELTME BAŞLANGICI ---
        # 1. Gelen başlık metnindeki olası boşlukları ve tırnak işaretlerini temizle.
        #    Bu, modelin metinden çıkardığı başlığın daha tutarlı olmasını sağlar.
        cleaned_title = section_title.strip().strip("'").strip('"')

        # 2. SQL sorgusunu daha esnek hale getir. Sadece temizlenmiş başlığı arayacak
        #    ve etrafındaki JSON formatına veya boşluklara takılmayacak.
        #    Örneğin, hem "4.2.13.1" hem de "4.2.13.1 -" aramalarını yakalayabilir.
        like_query = f'%{cleaned_title}%'
        # --- DÜZELTME SONU ---

        self.cursor.execute("SELECT text_content FROM chunks WHERE metadata_json LIKE ?", (like_query,))
        result = self.cursor.fetchone()

        if result:
            print(f"[AGENT TOOL] Found section '{section_title}'.")
            return result[0]
        else:
            print(f"[AGENT TOOL] Section '{section_title}' not found in database.")
            return f"'{section_title}' başlıklı bölüm veritabanında bulunamadı."