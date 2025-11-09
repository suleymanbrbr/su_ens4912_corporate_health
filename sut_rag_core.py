# sut_rag_core.py dosyasının TAMAMI

import pypandoc
import sqlite3
import json
import faiss
import numpy as np
import uuid
import os
import regex as re
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict

# --- Configuration ---
DOCX_FILE_PATH = "08.03.2025-Değişiklik Tebliği İşlenmiş Güncel 2013 SUT.docx"
MARKDOWN_FILE_PATH = "sut_converted_temp.md"
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
                print("[INIT] Gemini LLM başarıyla başlatıldı.")
            except Exception as e:
                print(f"[ERROR] Gemini LLM başlatılırken hata oluştu: {e}")
                self.llm = None

    def _initialize_embeddings(self):
        print(f"[INIT] Loading embedding model: '{EMBEDDING_MODEL}'...")
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})

    # --- GÜNCELLENMİŞ FONKSİYON 1 ---
    def _remove_strikethrough_and_save_temp(self, input_path: str) -> str:
        try:
            from docx import Document
        except ImportError:
            print("[ERROR] 'python-docx' library is not installed. Please run 'pip install python-docx'.")
            return None

        print(f"[PREP] Aggressively removing strikethrough elements from '{input_path}'...")
        doc = Document(input_path)
        temp_output_path = "temp_cleaned_sut.docx"
        
        runs_removed_count = 0

        def process_paragraph_runs(paragraph):
            nonlocal runs_removed_count
            # Paragraftaki 'run'ları (metin parçalarını) geriye doğru tarayarak silme işlemi yaparız.
            # Bu, döngü sırasında listeden eleman silmenin en güvenli yoludur.
            for i in range(len(paragraph.runs) - 1, -1, -1):
                run = paragraph.runs[i]
                is_strike = run.font.strike
                is_double_strike = False
                if run._r.rPr is not None and run._r.rPr.dstrike is not None:
                    is_double_strike = True

                if is_strike or is_double_strike:
                    # Sadece metni boşaltmak yerine, 'run' elementini paragraftan tamamen siliyoruz.
                    p = paragraph._p
                    p.remove(run._r)
                    runs_removed_count += 1

        # Belgedeki tüm paragrafları işle (tablolar dahil)
        for paragraph in doc.paragraphs:
            process_paragraph_runs(paragraph)
        
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        process_paragraph_runs(paragraph)
        
        print(f"[PREP] Total of {runs_removed_count} strikethrough 'run' elements were completely removed.")
        
        doc.save(temp_output_path)
        print(f"[PREP] Cleaned document saved to temporary file: '{temp_output_path}'")
        return temp_output_path

    # --- GÜNCELLENMİŞ FONKSİYON 2 ---
    def _get_markdown_chunks(self, md_path) -> List:
        print("[PREP] Post-processing and chunking Markdown file...")

        with open(md_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()

        # --- YENİ TEMİZLİK ADIMI BAŞLANGICI ---
        print("[PREP] Cleaning Markdown text from artifacts...")
        # Pandoc'un dönüştürdüğü strikethrough formatını (~~metin~~) temizle
        cleaned_text = re.sub(r'~~.*?~~', '', markdown_text)
        # ► gibi liste kalıntılarını temizle
        cleaned_text = re.sub(r'►', '', cleaned_text)
        # Sadece boşluk içeren veya tamamen boş satırları temizle
        cleaned_text = "\n".join(line for line in cleaned_text.splitlines() if line.strip())
        print("[PREP] Markdown artifacts cleaned.")
        # --- YENİ TEMİZLİK ADIMI SONU ---

        # Post-Pandoc fix: Find bolded numeric headers that were missed
        header_pattern = re.compile(r"^\*\*((\d+\.)+\d+[\.\d\w-]*)\s*-*\s*([^ \n\*]+.*?)\*\*", re.MULTILINE)
        
        def replace_with_markdown_header(match):
            header_num = match.group(1).strip()
            level = header_num.count('.') + 1
            if level > 6: level = 6
            return f"{'#' * level} {match.group(0)}"

        processed_text = header_pattern.sub(replace_with_markdown_header, cleaned_text)

        headers_to_split_on = [
            ("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"),
            ("####", "Header 4"), ("#####", "Header 5"), ("######", "Header 6"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        chunks = markdown_splitter.split_text(processed_text)
        print(f"[PREP] Document split into {len(chunks)} chunks using the hybrid method.")
        return chunks
    def _setup_database(self):
        if os.path.exists(DB_PATH): os.remove(DB_PATH)
        if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
        if os.path.exists(FAISS_MAPPING_PATH): os.remove(FAISS_MAPPING_PATH)

        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)  # 🔹 EKLENDİ
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
        # NOT: Orijinal DOCX'inizde "Değişiklikleri İzle" özelliğini kapattığınızdan emin olun.
        cleaned_docx_path = self._remove_strikethrough_and_save_temp(DOCX_FILE_PATH)
        if not cleaned_docx_path: return

        print(f"[PREP] Converting '{cleaned_docx_path}' to Markdown...")
        try:
            pypandoc.convert_file(cleaned_docx_path, 'md', outputfile=MARKDOWN_FILE_PATH)
        except Exception as e:
            print(f"[ERROR] during DOCX to Markdown conversion: {e}")
            if os.path.exists(cleaned_docx_path): os.remove(cleaned_docx_path)
            return
        
        chunks = self._get_markdown_chunks(MARKDOWN_FILE_PATH)
        if not chunks: 
            if os.path.exists(cleaned_docx_path): os.remove(cleaned_docx_path)
            if os.path.exists(MARKDOWN_FILE_PATH): os.remove(MARKDOWN_FILE_PATH)
            return
            
        self._setup_database()
        print("[DB] Populating SQLite and preparing for FAISS...")
        texts_to_embed, string_ids = [], []

        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            metadata_json = json.dumps(chunk.metadata, ensure_ascii=False)
            page_content = chunk.page_content
            self.cursor.execute("INSERT INTO chunks (chunk_id, text_content, metadata_json) VALUES (?, ?, ?)", (chunk_id, page_content, metadata_json))
            
            combined_header = " ".join(chunk.metadata.values())
            full_text_for_embedding = f"{combined_header}\n\n{page_content}"
            texts_to_embed.append(full_text_for_embedding)
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
        
        if os.path.exists(cleaned_docx_path): os.remove(cleaned_docx_path)
        if os.path.exists(MARKDOWN_FILE_PATH): os.remove(MARKDOWN_FILE_PATH)
        print("[PREP] Temporary files cleaned up.")

    def _retrieve_chunks(self, query: str, k: int) -> List[Dict]:
        if not self.faiss_index: return []
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
        print(f"[AGENT TOOL USED] Fetching content for chunk_id: {chunk_id}")
        self.cursor.execute("SELECT text_content FROM chunks WHERE chunk_id = ?", (chunk_id,))
        result = self.cursor.fetchone()
        return result[0] if result else "Hata: Belirtilen ID ile içerik bulunamadı."

    def get_section_by_title(self, section_title: str) -> str:
        try:
            from thefuzz import fuzz
        except ImportError:
            print("[ERROR] 'thefuzz' library is not installed. Please run 'pip install thefuzz python-Levenshtein'.")
            return "Fuzzy search kütüphanesi yüklü değil."

        print(f"[AGENT TOOL USED] Fuzzy searching for section by title: '{section_title}'")
        
        def clean_string(text):
            return re.sub(r'[\.\-\s]', '', text).lower()

        cleaned_query_title = clean_string(section_title)
        
        self.cursor.execute("SELECT chunk_id, metadata_json FROM chunks")
        all_metadata = self.cursor.fetchall()

        best_match = {"score": 0, "chunk_id": None, "title": ""}
        
        for chunk_id, metadata_json in all_metadata:
            try:
                metadata = json.loads(metadata_json)
                full_header = ' '.join(metadata.values())
                
                cleaned_db_title = clean_string(full_header)
                score = fuzz.ratio(cleaned_query_title, cleaned_db_title)
                
                if score > best_match["score"]:
                    best_match["score"] = score
                    best_match["chunk_id"] = chunk_id
                    best_match["title"] = full_header
            except json.JSONDecodeError:
                continue
        
        if best_match["score"] > 90:
            print(f"[AGENT TOOL] Found best match '{best_match['title']}' with score {best_match['score']}. Fetching content for chunk_id: {best_match['chunk_id']}.")
            return self.get_chunk_content_by_id(best_match['chunk_id'])
        else:
            print(f"[AGENT TOOL] No sufficiently similar section found for '{section_title}'. Best score was {best_match.get('score', 0)}.")
            return f"'{section_title}' başlığına yeterince benzeyen bir bölüm bulunamadı."

    def search_for_related_sections(self, query: str) -> str:
        print(f"[AGENT TOOL USED] Performing keyword search for: '{query}'")
        like_query = f'%{query}%'
        
        self.cursor.execute("SELECT metadata_json, text_content FROM chunks WHERE text_content LIKE ?", (like_query,))
        results = self.cursor.fetchall()

        if not results:
            return f"'{query}' anahtar kelimesini içeren bir bölüm bulunamadı."
            
        summary = f"'{query}' anahtar kelimesi için bulunan ilgili bölümlerin özeti:\n\n"
        for i, (metadata_json, text_content) in enumerate(results[:5]): # Return max 5 results
            metadata = json.loads(metadata_json)
            header = " ".join(metadata.values())
            try:
                start_index = text_content.lower().index(query.lower())
                preview = text_content[max(0, start_index-50) : start_index+100]
            except ValueError:
                preview = text_content[:150]

            summary += f"[{i+1}] Başlık: {header}\n    Önizleme: ...{preview.strip()}...\n\n"
            
        return summary
        
    def load_database(self):
        if not all([os.path.exists(DB_PATH), os.path.exists(FAISS_INDEX_PATH), os.path.exists(FAISS_MAPPING_PATH)]):
            return False
        print(f"[DB] Loading SQLite from: '{DB_PATH}' and FAISS from: '{FAISS_INDEX_PATH}'")
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)  # 🔹 EKLENDİ
        self.cursor = self.conn.cursor()
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
            self.id_mapping = json.load(f)
        print(f"[DB] Veritabanı yüklendi. (FAISS {self.faiss_index.ntotal} belge içeriyor)")
        return True


    def __del__(self):
        if self.conn:
            self.conn.close()