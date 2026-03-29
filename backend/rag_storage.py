# rag_storage.py
# Description: Handles SUT Data Extraction, Chunking, and SQLite/FAISS Storage.

import os
import re as regex
import json
import uuid
import sqlite3
import numpy as np
import faiss
import pypandoc
from typing import List, Dict

try:
    from docx import Document
except ImportError:
    print("[WARN] 'python-docx' library not found.")

# --- Configuration ---
DOCX_FILE_PATH = "data/08.03.2025-Değişiklik Tebliği İşlenmiş Güncel 2013 SUT.docx"
MARKDOWN_FILE_PATH = "data/sut_converted_temp.md"
DB_PATH = "data/sut_knowledge_base.db"
FAISS_INDEX_PATH = "data/sut_faiss.index"
FAISS_MAPPING_PATH = "data/sut_faiss.index.mapping"

class SUT_Storage_Manager:
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.conn = None
        self.cursor = None

    def populate_database(self):
        """Orchestrates the full database creation pipeline."""
        print("[PREP] Starting database population...")
        
        cleaned_path = self._remove_strikethrough_and_save_temp(DOCX_FILE_PATH)
        if not cleaned_path:
            print("[ERROR] Failed to clean DOCX. Aborting.")
            return False

        print("[PREP] Converting to Markdown (Pandoc)...")
        try:
            pypandoc.convert_file(cleaned_path, 'md', outputfile=MARKDOWN_FILE_PATH)
        except Exception as e:
            print(f"[ERROR] Pandoc conversion failed: {e}")
            return False

        print("[PREP] Splitting text into semantic chunks...")
        chunks = self._get_markdown_chunks(MARKDOWN_FILE_PATH)
        if not chunks: return False

        self._setup_database()

        print("[DB] Inserting data into SQLite (Standard + FTS) and FAISS...")
        texts_to_embed, string_ids = [], []

        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            metadata_json = json.dumps(chunk.metadata, ensure_ascii=False)
            page_content = chunk.page_content
            header_text = " ".join([v for k, v in chunk.metadata.items() if k.startswith("Header")])

            self.cursor.execute(
                "INSERT INTO chunks (chunk_id, text_content, metadata_json) VALUES (?, ?, ?)",
                (chunk_id, page_content, metadata_json)
            )

            try:
                self.cursor.execute(
                    "INSERT INTO title_search (chunk_id, header_text) VALUES (?, ?)",
                    (chunk_id, header_text)
                )
            except:
                pass

            full_text_for_embed = f"{header_text}\n\n{page_content}"
            texts_to_embed.append(full_text_for_embed)
            string_ids.append(chunk_id)

        self.conn.commit()
        self._create_and_save_faiss_index(texts_to_embed, string_ids)
        
        if os.path.exists(cleaned_path): os.remove(cleaned_path)
        if os.path.exists(MARKDOWN_FILE_PATH): os.remove(MARKDOWN_FILE_PATH)
        print("[SUCCESS] Database population complete.")
        return True

    def _remove_strikethrough_and_save_temp(self, input_path):
        if not os.path.exists(input_path): return None
        try:
            doc = Document(input_path)
            temp_output = "temp_cleaned_sut.docx"
            def clean_p(paragraph):
                for i in range(len(paragraph.runs)-1, -1, -1):
                    run = paragraph.runs[i]
                    if run.font.strike:
                        paragraph._p.remove(run._r)
            for p in doc.paragraphs: clean_p(p)
            for t in doc.tables:
                for r in t.rows:
                    for c in r.cells:
                        for p in c.paragraphs: clean_p(p)
            doc.save(temp_output)
            return temp_output
        except Exception as e:
            print(f"[ERROR] DOCX processing failed: {e}")
            return None

    def _get_markdown_chunks(self, md_path):
        from langchain_text_splitters import MarkdownHeaderTextSplitter
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                text = f.read()
            text = regex.sub(r'~~.*?~~', '', text)
            text = regex.sub(r'►', '', text)
            def h_repl(m):
                depth = m.group(1).count('.') + 1
                hashes = '#' * min(6, depth)
                return f"{hashes} {m.group(0)}"
            text = regex.sub(r"^\*\*((\d+\.)+\d+[\.\d\w-]*)\s*-*\s*([^ \n\*]+.*?)\*\*", h_repl, text, flags=regex.MULTILINE)
            headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
            splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            return splitter.split_text(text)
        except Exception as e:
            print(f"[ERROR] Chunking failed: {e}")
            return []

    def _setup_database(self):
        # DO NOT remove DB_PATH - it contains users and history
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Purge Knowledge Base Chunks ONLY
        self.cursor.execute("DROP TABLE IF EXISTS chunks")
        self.cursor.execute("DROP TABLE IF EXISTS title_search")

        # RAG Chunks Table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                text_content TEXT NOT NULL,
                metadata_json TEXT
            )
        """)
        
        # User Accounts Table (Should already exist, but for safety)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE,
                email TEXT UNIQUE,
                hashed_password TEXT,
                role TEXT DEFAULT 'user',
                is_approved INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        try:
            self.cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS title_search USING fts5(chunk_id, header_text)")
        except:
            pass
        self.conn.commit()

    def _create_and_save_faiss_index(self, texts, ids):
        vectors = self.embeddings_model.embed_documents(texts)
        index = faiss.IndexFlatL2(len(vectors[0]))
        index.add(np.array(vectors).astype('float32'))
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
            json.dump(ids, f)
