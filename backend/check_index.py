import os
import sys
import json
import sqlite3
import faiss
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_storage import DB_PATH, FAISS_INDEX_PATH, FAISS_MAPPING_PATH
from sut_rag_core import SUT_RAG_Engine

def verify_indices():
    print("--- SUT Index Verification ---")
    load_dotenv()
    
    # Check if files exist
    if not os.path.exists(DB_PATH):
        print("[ERROR] Database file not found!")
        return
    if not os.path.exists(FAISS_INDEX_PATH):
        print("[ERROR] FAISS index not found!")
        return
    if not os.path.exists(FAISS_MAPPING_PATH):
        print("[ERROR] FAISS mapping not found!")
        return

    # Count SQLite Chunks
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chunks")
    db_count = cursor.fetchone()[0]
    print(f"[VERIFY] SQLite 'chunks' table count: {db_count}")
    
    # Read FAISS Index Length
    index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"[VERIFY] FAISS Index Total Vectors:  {index.ntotal}")
    
    # Read FAISS Mapping Length
    with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    print(f"[VERIFY] FAISS Mapping Items:      {len(mapping)}")
    
    if db_count == index.ntotal == len(mapping):
        print("[SUCCESS] All counts match perfectly.")
    else:
        print("[WARN] Mismatch detected. Please rebuild the index.")

    print("\n[VERIFY] Testing Retrieval Engine...")
    engine = SUT_RAG_Engine()
    engine.load_database()
    
    test_query = "Ayakta tedavide kullanılan ilaçlar"
    print(f"[TEST] Retrieving data for query: '{test_query}'")
    
    chunks = engine._retrieve_chunks(test_query, k=3)
    
    if chunks:
        print(f"[SUCCESS] Retrieved {len(chunks)} relevant chunks!")
        for i, doc in enumerate(chunks):
            print(f"\n--- Result {i+1} ---")
            print(f"ID: {doc['id']}")
            print(f"Content Outline: {doc['text'][:150]}...")
            print(f"Rerank Score: {doc.get('score', 0):.4f}")
    else:
        print("[ERROR] No chunks retrieved. Something is wrong with the embeddings/index.")

if __name__ == "__main__":
    verify_indices()
