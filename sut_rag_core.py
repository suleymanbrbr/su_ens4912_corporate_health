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

# --- Configuration ---
DOCX_FILE_PATH = "08.03.2025-Değişiklik Tebliği İşlenmiş Güncel 2013 SUT.docx"
MARKDOWN_FILE_PATH = "sut_converted.md"
DB_PATH = "sut_knowledge_base.db"
FAISS_INDEX_PATH = "sut_faiss.index"
FAISS_MAPPING_PATH = "sut_faiss.index.mapping" # Maps FAISS int ID to UUID string
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2" 

class SUT_RAG_Engine:
    def __init__(self):
        self.embeddings_model = self._initialize_embeddings()
        self.conn = None
        self.cursor = None
        self.faiss_index = None
        self.id_mapping = None
        
        # Gemini LLM initialization - Anahtar kontrolü ve direkt geçiş
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[WARNING] GEMINI_API_KEY ortam değişkeni ayarlanmamış. LLM yanıtı yerine prompt metni döndürülecek.")
            self.llm = None
        else:
            try:
                # ChatGoogleGenerativeAI yazım hatası düzeltildi ve api_key direkt geçirildi.
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    api_key=api_key 
                ) 
                print("[INIT] Gemini LLM başarıyla başlatıldı.")
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
            print("Please ensure 'pandoc' is installed and accessible.")
            return None

    def _get_markdown_chunks(self, md_path) -> List:
        """Chunks the Markdown file based on SUT headers."""
        print("[PREP] Chunking Markdown file based on headers...")
        
        headers_to_split_on = [
            ("#", "Header 1"), ("##", "Header 2"),
            ("###", "Header 3"), ("####", "Header 4"),
        ]

        with open(md_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )
        
        chunks = markdown_splitter.split_text(markdown_text)
        print(f"[PREP] Document split into {len(chunks)} chunks.")
        
        os.remove(md_path)
        return chunks

    def _setup_database(self):
        """Creates the SQLite database (Document Store) and clears old FAISS files."""
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)
        if os.path.exists(FAISS_MAPPING_PATH):
            os.remove(FAISS_MAPPING_PATH)

        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
        
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            text_content TEXT NOT NULL,
            metadata_json TEXT
        );
        """)
        self.conn.commit()
        print(f"[DB] SQLite database '{DB_PATH}' setup complete.")

    def populate_database(self):
        """Phase 1A: Runs the entire ingestion and indexing pipeline (SQLite + FAISS)."""
        
        # 1. Prepare Chunks
        md_file = self._convert_docx_to_markdown()
        if not md_file: return
        chunks = self._get_markdown_chunks(md_file)
        if not chunks: return

        # 2. Setup SQLite
        self._setup_database()
        
        print("[DB] Populating SQLite and preparing for FAISS...")
        
        texts_to_embed = []
        string_ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            
            # Store in SQLite (Document Store)
            metadata_json = json.dumps(chunk.metadata)
            self.cursor.execute(
                "INSERT INTO chunks (chunk_id, text_content, metadata_json) VALUES (?, ?, ?)",
                (chunk_id, chunk.page_content, metadata_json)
            )
            
            texts_to_embed.append(chunk.page_content)
            string_ids.append(chunk_id)
        
        self.conn.commit()
        print("[DB] All chunks saved to SQLite.")

        # 3. Create FAISS Index (Vector Store)
        print("[DB] Embedding texts for FAISS...")
        vectors = self.embeddings_model.embed_documents(texts_to_embed)
        
        d = len(vectors[0])
        vector_array = np.array(vectors).astype('float32')
        
        index = faiss.IndexFlatL2(d)
        index.add(vector_array)
        
        # Save FAISS index and the mapping list (integer ID -> UUID string ID)
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
            json.dump(string_ids, f)
        
        self.faiss_index = index
        self.id_mapping = string_ids
        
        print(f"[DB] FAISS index created and saved with {index.ntotal} vectors.")
        
    def _build_prompt(self, query: str, retrieved_contexts: List[Dict]) -> str:
        """Constructs the augmented prompt for the LLM."""
        context_str = ""
        for i, context in enumerate(retrieved_contexts):
            # Format the metadata for citation
            section_info = []
            if context['metadata'].get('Header 1'):
                section_info.append(context['metadata']['Header 1'])
            if context['metadata'].get('Header 3'):
                 section_info.append(context['metadata']['Header 3'])
            
            citation = f"Source {i+1} [Citation: SUT, {'; '.join(section_info)}]"
            
            context_str += f"{citation}\n---\n{context['text']}\n===\n"
            
        prompt = f"""
SEN BİR SAĞLIK UZMANI ASİSTANISIN. Yalnızca aşağıda sağlanan SUT (Sağlık Uygulama Tebliği) bağlamını kullanarak kullanıcının sorusunu yanıtla.
Cevabını doğrudan, net ve Türkçe olarak ver.
Cevabında, kullandığın SUT maddesini açıkça belirterek alıntı yap. Örneğin: "Evet, KMY ölçümü gereklidir. (Kaynak: DÖRDÜNCÜ BÖLÜM; Madde 4.2.17)"

Kullanıcı Sorusu:
{query}

Bağlam (Context) - SUT'tan Alınan İlgili Bilgiler:
---
{context_str}
---

Yanıt:
"""
        return prompt

    def query_rag(self, query: str, k: int = 3) -> Tuple[str, List[Dict]]:
        """Phase 1B: Executes the RAG query pipeline (Retrieve, Augment, Generate)."""
        if self.llm is None:
            # LLM başlatılmadıysa, uyarı mesajını döndür.
            return "[LLM DEVRE DIŞI] Lütfen GEMINI_API_KEY'i ayarlayın.", []
        
        if not self.faiss_index:
            return "Veritabanı başlatılamadı.", []

        print(f"[QUERY] Searching FAISS for top {k} results for: '{query}'")
        
        # 1. Retrieve (FAISS)
        query_vector = self.embeddings_model.embed_query(query)
        query_vector_np = np.array([query_vector]).astype('float32')
        
        # D = distances, I = indices (FAISS internal integer IDs)
        distances, indices = self.faiss_index.search(query_vector_np, k)
        
        retrieved_contexts = []
        
        # 2. Augment (SQLite)
        print("[QUERY] Retrieving full context from SQLite...")
        for faiss_id in indices[0]:
            if faiss_id == -1: continue 
            
            # Convert FAISS int ID to UUID string ID using the mapping
            chunk_id = self.id_mapping[faiss_id]
            
            # Fetch the full chunk details from SQLite
            self.cursor.execute("SELECT text_content, metadata_json FROM chunks WHERE chunk_id = ?", (chunk_id,))
            result = self.cursor.fetchone()
            
            if result:
                text, metadata_str = result
                retrieved_contexts.append({
                    "text": text,
                    "metadata": json.loads(metadata_str)
                })

        if not retrieved_contexts:
            return "Üzgünüm, bu soruya SUT içerisinde alakalı bir bilgi bulamadım.", []

        # 3. Generate (LLM)
        prompt = self._build_prompt(query, retrieved_contexts)
        
        try:
            llm_response = self.llm.invoke(prompt)
            final_answer = llm_response.content
        except Exception as e:
            # Handle potential API errors during generation
            final_answer = f"[LLM ERROR] Yanıt oluşturulurken bir hata oluştu: {e}"
            
        return final_answer, retrieved_contexts

    def load_database(self):
        """Loads an existing SQLite and FAISS index from disk."""
        if not os.path.exists(DB_PATH) or not os.path.exists(FAISS_INDEX_PATH):
            return False
            
        print(f"[DB] Loading SQLite from: '{DB_PATH}' and FAISS from: '{FAISS_INDEX_PATH}'")
        
        # 1. Load SQLite
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()

        # 2. Load FAISS and Mapping
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
            self.id_mapping = json.load(f)
        
        print(f"[DB] Veritabanı yüklendi. (FAISS {self.faiss_index.ntotal} belge içeriyor)")
        return True

    def __del__(self):
        """Closes the database connection on object deletion."""
        if self.conn:
            self.conn.close()