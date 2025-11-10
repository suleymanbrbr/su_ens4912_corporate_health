# sut_rag_core.py
# Açıklama: Bu dosya, Sağlık Uygulama Tebliği (SUT) dokümanını işleyerek
# bir Vektör Veritabanı (FAISS) ve bir SQLite veritabanı oluşturan,
# ve bu veritabanı üzerinde anlamsal arama ve sorgulama yapılmasını
# sağlayan RAG (Retrieval-Augmented Generation) motorunu içerir.

# --- Gerekli Kütüphaneler ---
import os
import re as regex
import json
import uuid
import sqlite3
import numpy as np
import faiss
import pypandoc
from typing import List, Dict

# LangChain ve Dış Kütüphaneler
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Opsiyonel Kütüphaneler (Hata mesajları ile kontrol ediliyor)
try:
    from docx import Document
except ImportError:
    print("[UYARI] 'python-docx' kütüphanesi yüklü değil. Belge temizleme işlemi için gereklidir. 'pip install python-docx' ile yükleyebilirsiniz.")
try:
    from thefuzz import fuzz
except ImportError:
    print("[UYARI] 'thefuzz' kütüphanesi yüklü değil. Başlık arama özelliği için gereklidir. 'pip install thefuzz python-Levenshtein' ile yükleyebilirsiniz.")

# --- Yapılandırma Ayarları ---
DOCX_FILE_PATH = "08.03.2025-Değişiklik Tebliği İşlenmiş Güncel 2013 SUT.docx"
MARKDOWN_FILE_PATH = "sut_converted_temp.md"
DB_PATH = "sut_knowledge_base.db"
FAISS_INDEX_PATH = "sut_faiss.index"
FAISS_MAPPING_PATH = "sut_faiss.index.mapping"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class SUT_RAG_Engine:
    """
    Sağlık Uygulama Tebliği (SUT) dokümanları için bir RAG motoru.

    Bu sınıf, bir DOCX dosyasını işler, metni parçalara (chunks) ayırır,
    bu parçaları gömme (embedding) vektörlerine dönüştürür ve hem SQLite
    veritabanında hem de FAISS vektör dizininde saklar. Ardından, bu yapı
    üzerinden anlamsal arama ve LLM destekli sorgu yanıtlama yetenekleri sunar.
    """
    # =========================================================================
    # 1. BAŞLATMA (INITIALIZATION)
    # =========================================================================
    def __init__(self, llm_provider: str = "google", model_name: str = "gemini-2.5-flash"):
        """
        SUT_RAG_Engine sınıfını başlatır.

        Args:
            llm_provider (str): Kullanılacak LLM sağlayıcısı ('google', 'openrouter', 'lmstudio').
            model_name (str): LLM sağlayıcısı tarafından kullanılacak modelin adı.
        """
        self.embeddings_model = self._initialize_embeddings()
        self.conn = None
        self.cursor = None
        self.faiss_index = None
        self.id_mapping = None
        self.llm = None

        print(f"[INIT] LLM başlatılıyor. Sağlayıcı: '{llm_provider}', Model: '{model_name}'")

        if llm_provider == "google":
            self._init_google_llm(model_name)
        elif llm_provider == "openrouter":
            self._init_openrouter_llm(model_name)
        elif llm_provider == "lmstudio":
            self._init_lmstudio_llm(model_name)
        else:
            print(f"[ERROR] Desteklenmeyen LLM sağlayıcısı: {llm_provider}")

    def _init_google_llm(self, model_name: str):
        """Google Gemini LLM'ini başlatır."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[WARNING] GEMINI_API_KEY .env dosyasında bulunamadı veya ayarlanmamış.")
            return
        try:
            self.llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key, temperature=0.2, max_output_tokens=2048)
            print("[INIT] Google Gemini LLM başarıyla başlatıldı.")
        except Exception as e:
            print(f"[ERROR] Google Gemini LLM başlatılırken hata oluştu: {e}")

    def _init_openrouter_llm(self, model_name: str):
        """OpenRouter LLM'ini başlatır."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("[WARNING] OPENROUTER_API_KEY .env dosyasında bulunamadı veya ayarlanmamış.")
            return
        try:
            site_url = os.getenv("YOUR_SITE_URL", "http://localhost")
            site_name = os.getenv("YOUR_SITE_NAME", "SUT RAG Assistant")
            self.llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.2,
                max_tokens=2048,
                default_headers={"HTTP-Referer": site_url, "X-Title": site_name}
            )
            print("[INIT] OpenRouter LLM başarıyla başlatıldı.")
        except Exception as e:
            print(f"[ERROR] OpenRouter LLM başlatılırken hata oluştu: {e}")

    def _init_lmstudio_llm(self, model_name: str):
        """LM Studio üzerinden yerel bir LLM başlatır."""
        try:
            self.llm = ChatOpenAI(
                model=model_name,
                openai_api_key="not-needed",
                openai_api_base="http://localhost:1234/v1",
                temperature=0.2,
                max_tokens=2048
            )
            print("[INIT] LM Studio (Yerel LLM) başarıyla başlatıldı.")
            print(">>> Lütfen LM Studio uygulamasında sunucunun çalıştığından emin olun. <<<")
        except Exception as e:
            print(f"[ERROR] LM Studio LLM başlatılırken hata oluştu: {e}")
            print(">>> Hata Sebebi: LM Studio'daki yerel sunucunun çalışır durumda olduğundan emin olun. <<<")

    def _initialize_embeddings(self):
        """Gömme (embedding) modelini yükler ve başlatır."""
        print(f"[INIT] Embedding modeli yükleniyor: '{EMBEDDING_MODEL}'...")
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})

    def __del__(self):
        """Nesne yok edilirken veritabanı bağlantısını kapatır."""
        if self.conn:
            self.conn.close()
            print("[DB] Veritabanı bağlantısı kapatıldı.")

    # =========================================================================
    # 2. VERİ HAZIRLAMA VE VERİTABANI OLUŞTURMA PİPELINE'I
    # =========================================================================

    def populate_database(self):
        """
        Tüm veri işleme ve veritabanı oluşturma sürecini yönetir.
        DOCX dosyasını temizler, Markdown'a çevirir, parçalara ayırır ve
        hem SQLite hem de FAISS veritabanlarını doldurur.
        """
        # Adım 1: DOCX dosyasındaki üstü çizili metinleri temizle
        cleaned_docx_path = self._remove_strikethrough_and_save_temp(DOCX_FILE_PATH)
        if not cleaned_docx_path:
            return

        # Adım 2: Temizlenmiş DOCX'i Markdown'a dönüştür
        print(f"[PREP] '{cleaned_docx_path}' dosyası Markdown formatına dönüştürülüyor...")
        try:
            pypandoc.convert_file(cleaned_docx_path, 'md', outputfile=MARKDOWN_FILE_PATH)
        except Exception as e:
            print(f"[ERROR] DOCX'ten Markdown'a dönüştürme sırasında hata: {e}")
            if os.path.exists(cleaned_docx_path): os.remove(cleaned_docx_path)
            return

        # Adım 3: Markdown metnini anlamsal parçalara (chunk) ayır
        chunks = self._get_markdown_chunks(MARKDOWN_FILE_PATH)
        if not chunks:
            if os.path.exists(cleaned_docx_path): os.remove(cleaned_docx_path)
            if os.path.exists(MARKDOWN_FILE_PATH): os.remove(MARKDOWN_FILE_PATH)
            return

        # Adım 4: Veritabanlarını (SQLite, FAISS) sıfırdan kur
        self._setup_database()

        # Adım 5: Parçaları veritabanlarına işle ve vektör dizinini oluştur
        print("[DB] SQLite dolduruluyor ve FAISS için hazırlık yapılıyor...")
        texts_to_embed, string_ids = [], []
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            metadata_json = json.dumps(chunk.metadata, ensure_ascii=False)
            page_content = chunk.page_content

            self.cursor.execute("INSERT INTO chunks (chunk_id, text_content, metadata_json) VALUES (?, ?, ?)",
                                (chunk_id, page_content, metadata_json))

            combined_header = " ".join(chunk.metadata.values())
            full_text_for_embedding = f"{combined_header}\n\n{page_content}"
            texts_to_embed.append(full_text_for_embedding)
            string_ids.append(chunk_id)

        self.conn.commit()
        print("[DB] Tüm metin parçaları SQLite'a kaydedildi.")

        # Adım 6: FAISS vektör dizinini oluştur ve kaydet
        self._create_and_save_faiss_index(texts_to_embed, string_ids)

        # Adım 7: Geçici dosyaları temizle
        if os.path.exists(cleaned_docx_path): os.remove(cleaned_docx_path)
        if os.path.exists(MARKDOWN_FILE_PATH): os.remove(MARKDOWN_FILE_PATH)
        print("[PREP] Geçici dosyalar temizlendi.")

    def _remove_strikethrough_and_save_temp(self, input_path: str) -> str:
        """
        Bir DOCX dosyasındaki tüm üstü çizili (strikethrough) metinleri kaldırır
        ve sonucu geçici bir dosyaya kaydeder.
        """
        try:
            doc = Document(input_path)
        except NameError:
            print("[ERROR] 'python-docx' kütüphanesi bulunamadı. Lütfen 'pip install python-docx' ile yükleyin.")
            return None
        except Exception as e:
            print(f"[ERROR] DOCX dosyası okunurken bir hata oluştu: {e}")
            return None
            
        print(f"[PREP] '{input_path}' dosyasındaki üstü çizili metinler temizleniyor...")
        temp_output_path = "temp_cleaned_sut.docx"
        runs_removed_count = 0

        def process_paragraph_runs(paragraph):
            nonlocal runs_removed_count
            # Paragraftaki 'run'ları tersten dolaşarak silme işlemini güvenli hale getiriyoruz.
            for i in range(len(paragraph.runs) - 1, -1, -1):
                run = paragraph.runs[i]
                is_strike = run.font.strike
                is_double_strike = run._r.rPr is not None and run._r.rPr.dstrike is not None

                if is_strike or is_double_strike:
                    p = paragraph._p
                    p.remove(run._r)
                    runs_removed_count += 1

        for paragraph in doc.paragraphs:
            process_paragraph_runs(paragraph)

        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        process_paragraph_runs(paragraph)

        print(f"[PREP] Toplam {runs_removed_count} adet üstü çizili 'run' öğesi kaldırıldı.")
        doc.save(temp_output_path)
        print(f"[PREP] Temizlenmiş belge geçici olarak kaydedildi: '{temp_output_path}'")
        return temp_output_path

    def _get_markdown_chunks(self, md_path: str) -> List:
        """
        Markdown dosyasını okur, temizler ve başlık yapısına göre parçalara ayırır.
        """
        print("[PREP] Markdown dosyası işleniyor ve parçalara ayrılıyor...")
        with open(md_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()

        print("[PREP] Markdown metnindeki artıklar (artifacts) temizleniyor...")
        # Üstü çizili metin kalıntılarını (~~metin~~) temizle
        cleaned_text = regex.sub(r'~~.*?~~', '', markdown_text)
        # Özel karakterleri (►) temizle
        cleaned_text = regex.sub(r'►', '', cleaned_text)
        # Boş satırları kaldır
        cleaned_text = "\n".join(line for line in cleaned_text.splitlines() if line.strip())
        print("[PREP] Markdown artıkları temizlendi.")
        
        # Özel başlık formatını (**1.2.3. Başlık**) standart Markdown başlığına (# Başlık) dönüştür
        header_pattern = regex.compile(r"^\*\*((\d+\.)+\d+[\.\d\w-]*)\s*-*\s*([^ \n\*]+.*?)\*\*", regex.MULTILINE)
        def replace_with_markdown_header(match):
            header_num = match.group(1).strip()
            level = header_num.count('.') + 1
            if level > 6: level = 6  # Markdown'da maksimum başlık seviyesi 6'dır
            return f"{'#' * level} {match.group(0)}"

        processed_text = header_pattern.sub(replace_with_markdown_header, cleaned_text)

        headers_to_split_on = [
            ("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"),
            ("####", "Header 4"), ("#####", "Header 5"), ("######", "Header 6")
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        chunks = markdown_splitter.split_text(processed_text)

        print(f"[PREP] Belge, başlık yapısına göre {len(chunks)} parçaya ayrıldı.")
        return chunks

    def _setup_database(self):
        """Mevcut veritabanı ve dizin dosyalarını siler ve yenilerini oluşturur."""
        if os.path.exists(DB_PATH): os.remove(DB_PATH)
        if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
        if os.path.exists(FAISS_MAPPING_PATH): os.remove(FAISS_MAPPING_PATH)

        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                text_content TEXT NOT NULL,
                metadata_json TEXT
            );
        """)
        self.conn.commit()
        print(f"[DB] SQLite veritabanı '{DB_PATH}' başarıyla kuruldu.")

    def _create_and_save_faiss_index(self, texts_to_embed: List[str], string_ids: List[str]):
        """
        Verilen metinleri vektörlere dönüştürür, bir FAISS dizini oluşturur ve kaydeder.
        """
        print("[DB] Metinler FAISS için vektörlere dönüştürülüyor...")
        vectors = self.embeddings_model.embed_documents(texts_to_embed)
        d = len(vectors[0])  # Vektör boyutu
        vector_array = np.array(vectors).astype('float32')

        index = faiss.IndexFlatL2(d)
        index.add(vector_array)

        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
            json.dump(string_ids, f)

        self.faiss_index = index
        self.id_mapping = string_ids
        print(f"[DB] FAISS dizini {index.ntotal} vektör ile oluşturuldu ve kaydedildi.")


    # =========================================================================
    # 3. VERİ ERİŞİMİ VE ARAMA ARAÇLARI (AGENT TOOLS)
    # =========================================================================

    def _retrieve_chunks(self, query: str, k: int) -> List[Dict]:
        """
        Verilen bir sorgu için FAISS'te anlamsal arama yapar ve en benzer k sonucu döndürür.
        Bu fonksiyon, doğrudan ajan aracı olarak değil, diğer fonksiyonlar için bir yardımcıdır.
        """
        if not self.faiss_index:
            print("[ERROR] FAISS dizini yüklenmemiş. Arama yapılamıyor.")
            return []

        query_vector = self.embeddings_model.embed_query(query)
        query_vector_np = np.array([query_vector]).astype('float32')

        # FAISS'te arama yap
        _, indices = self.faiss_index.search(query_vector_np, k)

        retrieved_chunks = []
        for faiss_id in indices[0]:
            if faiss_id == -1: continue # Geçersiz ID
            chunk_id = self.id_mapping[faiss_id]
            self.cursor.execute("SELECT chunk_id, text_content, metadata_json FROM chunks WHERE chunk_id = ?", (chunk_id,))
            result = self.cursor.fetchone()
            if result:
                retrieved_chunk_id, text, metadata_str = result
                retrieved_chunks.append({
                    "id": retrieved_chunk_id,
                    "text": text,
                    "metadata": json.loads(metadata_str)
                })
        return retrieved_chunks
    
    def get_chunk_content_by_id(self, chunk_id: str) -> str:
        """
        Belirtilen chunk ID'sine sahip metin parçasının içeriğini döndürür.
        """
        print(f"[AGENT TOOL USED] İçerik getiriliyor (chunk_id: {chunk_id})")
        self.cursor.execute("SELECT text_content FROM chunks WHERE chunk_id = ?", (chunk_id,))
        result = self.cursor.fetchone()
        return result[0] if result else "Hata: Belirtilen ID ile içerik bulunamadı."

    def get_section_by_title(self, section_title: str) -> str:
        """
        Verilen başlığa en çok benzeyen bölümü bulanık arama (fuzzy search) ile bulur ve içeriğini döndürür.
        """
        try:
            from thefuzz import fuzz
        except ImportError:
            return "[ERROR] 'thefuzz' kütüphanesi yüklü değil. 'pip install thefuzz python-Levenshtein' ile yükleyin."

        print(f"[AGENT TOOL USED] Başlığa göre bulanık arama yapılıyor: '{section_title}'")
        
        def clean_string(text):
            return regex.sub(r'[\.\-\s]', '', text).lower()

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
                    best_match.update({
                        "score": score,
                        "chunk_id": chunk_id,
                        "title": full_header
                    })
            except json.JSONDecodeError:
                continue

        # Eşleşme skorunun yeterince yüksek olup olmadığını kontrol et
        if best_match["score"] > 90:
            print(f"[AGENT TOOL] En iyi eşleşme bulundu: '{best_match['title']}' (Skor: {best_match['score']}). İçerik getiriliyor...")
            return self.get_chunk_content_by_id(best_match['chunk_id'])
        else:
            print(f"[AGENT TOOL] '{section_title}' için yeterli benzerlikte bir bölüm bulunamadı. En iyi skor: {best_match.get('score', 0)}.")
            return f"'{section_title}' başlığına yeterince benzeyen bir bölüm bulunamadı."

    def search_for_related_sections(self, query: str) -> str:
        """
        Metin içeriklerinde anahtar kelime araması yapar ve ilgili bölümlerin bir özetini döndürür.
        """
        print(f"[AGENT TOOL USED] Anahtar kelime araması yapılıyor: '{query}'")
        like_query = f'%{query}%'
        self.cursor.execute("SELECT metadata_json, text_content FROM chunks WHERE text_content LIKE ?", (like_query,))
        results = self.cursor.fetchall()

        if not results:
            return f"'{query}' anahtar kelimesini içeren bir bölüm bulunamadı."

        summary = f"'{query}' anahtar kelimesi için bulunan ilgili bölümlerin özeti:\n\n"
        # Sonuçları ilk 5 ile sınırla
        for i, (metadata_json, text_content) in enumerate(results[:5]):
            metadata = json.loads(metadata_json)
            header = " ".join(metadata.values())
            try:
                # Anahtar kelimenin geçtiği yerden bir önizleme al
                start_index = text_content.lower().index(query.lower())
                preview = text_content[max(0, start_index-50) : start_index+100]
            except ValueError:
                preview = text_content[:150] # Bulunamazsa baştan al

            summary += f"[{i+1}] Başlık: {header}\n    Önizleme: ...{preview.strip()}...\n\n"

        return summary

    # =========================================================================
    # 4. VERİTABANI YÖNETİMİ
    # =========================================================================
    def load_database(self) -> bool:
        """
        Daha önce oluşturulmuş olan SQLite ve FAISS veritabanı dosyalarını yükler.
        """
        if not all([os.path.exists(DB_PATH), os.path.exists(FAISS_INDEX_PATH), os.path.exists(FAISS_MAPPING_PATH)]):
            print("[DB] Gerekli veritabanı dosyaları bulunamadı. 'populate_database()' fonksiyonunu çalıştırmanız gerekebilir.")
            return False

        print(f"[DB] SQLite yükleniyor: '{DB_PATH}'")
        print(f"[DB] FAISS dizini yükleniyor: '{FAISS_INDEX_PATH}'")

        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        
        with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
            self.id_mapping = json.load(f)

        print(f"[DB] Veritabanı başarıyla yüklendi. (FAISS {self.faiss_index.ntotal} belge içeriyor)")
        return True