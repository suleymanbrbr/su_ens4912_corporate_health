from sut_rag_core import SUT_RAG_Engine, DB_PATH, FAISS_INDEX_PATH, FAISS_MAPPING_PATH
import os

def main():
    """Main execution flow for the RAG project."""
    print("--- Graduation Project RAG Pipeline (SUT Document) ---")
    
    engine = SUT_RAG_Engine()
    
    # 1. Veritabanını Kontrol Et/Oluştur
    # Gerekli tüm dosyaların varlığını kontrol eder (SQLite DB, FAISS Index, ve Mapping)
    if os.path.exists(DB_PATH) and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
        if not engine.load_database():
            return
    else:
        print("\n--- VERİTABANI KURULUMU BAŞLADI ---")
        engine.populate_database()
        print("--- VERİTABANI KURULUMU TAMAMLANDI ---")
        
    # 2. Test Sorgusunu Çalıştır
    test_query = "Osteoporoz tedavisi için KMY ölçümü gerekir mi?"
    print(f"\n[USER] Soru: {test_query}")
    
    # K=3 ile en alakalı 3 parçayı alıyoruz
    answer, contexts = engine.query_rag(test_query, k=3)
    
    print("\n---------------------------------------------------")
    print(f"Final Cevap: {answer}")
    print("---------------------------------------------------")

    # 3. Kaynakları Doğrulama İçin Göster
    if contexts:
        print("\n--- Kullanılan Kaynak Parçaları (Contexts) ---")
        for i, context in enumerate(contexts):
            # Sadece Bölüm ve Madde başlıklarını alıyoruz
            section_info = [
                v for k, v in context['metadata'].items() if k.startswith('Header') and v
            ]
            print(f"\n[Kaynak {i+1}]")
            print(f"  Başlıklar: {'; '.join(section_info)}")
            print(f"  Metin: {context['text'][:300]}...")
    else:
        print("Kaynak bulunamadı.")


if __name__ == "__main__":
    main()