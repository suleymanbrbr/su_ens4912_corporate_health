import streamlit as st
import os
import time
from sut_rag_core import SUT_RAG_Engine, DB_PATH, FAISS_INDEX_PATH, FAISS_MAPPING_PATH

# Streamlit sayfa konfigürasyonu
st.set_page_config(
    page_title="SUT Retrieval-Augmented Generation (RAG) Asistanı",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# ARAYÜZ FONKSİYONLARI
# ----------------------------------------------------------------------

def display_sidebar(engine):
    """Kenar çubuğunu (Sidebar) oluşturur ve veritabanı durumunu gösterir."""
    st.sidebar.header("Proje Durumu")

    # Kontrol: API Anahtarı
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        st.sidebar.success("Gemini API Anahtarı yüklü.")
    else:
        st.sidebar.error("GEMINI_API_KEY ortam değişkeni ayarlanmamış.")
        st.sidebar.markdown("Lütfen terminalinizde `export GEMINI_API_KEY='anahtarınız'` komutunu kullanın.")
        return False

    # Kontrol: Veritabanı Durumu
    db_exists = (os.path.exists(DB_PATH) and 
                 os.path.exists(FAISS_INDEX_PATH) and 
                 os.path.exists(FAISS_MAPPING_PATH))

    if db_exists:
        st.sidebar.success("Veritabanı (SQLite/FAISS) yüklü ve hazır.")
        # Veritabanını silme seçeneği ekleyelim
        if st.sidebar.button("Mevcut Veritabanını Sil", type="secondary"):
            os.remove(DB_PATH)
            os.remove(FAISS_INDEX_PATH)
            os.remove(FAISS_MAPPING_PATH)
            st.rerun() # Sayfayı yeniden yükle
            
    else:
        st.sidebar.warning("Veritabanı bulunamadı. Lütfen oluşturun.")
        if st.sidebar.button("Veritabanını Oluştur / Yeniden Başlat", type="primary"):
            with st.spinner("Pandoc ile DOCX dönüştürülüyor ve Vektörler oluşturuluyor... Bu işlem birkaç dakika sürebilir."):
                engine.populate_database()
            st.rerun()
            
    return api_key and db_exists

def display_chat_interface(engine):
    """Ana sohbet arayüzünü oluşturur ve sorguları işler."""
    st.title("SUT RAG Asistanı")
    st.caption("Sosyal Güvenlik Kurumu Sağlık Uygulama Tebliği (SUT) ile desteklenen yapay zeka asistanı.")

    # Sohbet geçmişini başlatma
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Geçmiş mesajları görüntüleme
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Yeni kullanıcı girişi
    if prompt := st.chat_input("SUT ile ilgili bir soru sorun..."):
        # Kullanıcı mesajını ekle
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Asistan yanıtını üret
        with st.chat_message("assistant"):
            st.markdown("Cevap hazırlanıyor...")
            start_time = time.time()
            
            # RAG Motoru Sorgusu
            answer, contexts = engine.query_rag(prompt, k=4)
            
            elapsed_time = time.time() - start_time
            
            # Yanıtı ekle
            st.markdown(answer)
            
            # Kaynakları ve süre bilgisini göster
            st.caption(f"Sorgu süresi: {elapsed_time:.2f} saniye")
            
            if contexts:
                with st.expander("Kullanılan Kaynaklar (Detaylı Alıntı)"):
                    for i, context in enumerate(contexts):
                        section_info = [
                            v for k, v in context['metadata'].items() if k.startswith('Header') and v
                        ]
                        st.markdown(f"**Kaynak {i+1}** (Alaka Düzeyi: En yüksekten düşüğe)")
                        st.markdown(f"**Başlıklar:** `{'; '.join(section_info)}`")
                        st.text_area(f"Metin {i+1}", context['text'], height=150, disabled=True, label_visibility="collapsed")
                        st.markdown("---")

        # Asistan yanıtını geçmişe ekle
        st.session_state.messages.append({"role": "assistant", "content": answer})


# ----------------------------------------------------------------------
# UYGULAMA BAŞLANGICI
# ----------------------------------------------------------------------

def main_app():
    # RAG Motorunu başlat
    try:
        engine = SUT_RAG_Engine()
    except Exception as e:
        st.error(f"Uygulama başlatılırken kritik bir hata oluştu: {e}")
        st.stop()
        
    # Kenar çubuğunu kontrol et ve uygulama durumunu al
    is_ready = display_sidebar(engine)
    
    # Eğer API ve DB hazırsa sohbet arayüzünü göster
    if is_ready:
        # Veritabanı yüklü değilse yükle
        if not engine.faiss_index:
            if not engine.load_database():
                st.error("Veritabanı dosyaları mevcut ancak yüklenemedi. Yeniden oluşturmayı deneyin.")
                return

        # Sohbet arayüzünü göster
        display_chat_interface(engine)
    else:
        st.info("Lütfen kenar çubuğundaki (sidebar) adımları tamamlayarak API anahtarını ayarlayın ve veritabanını oluşturun.")

if __name__ == "__main__":
    main_app()