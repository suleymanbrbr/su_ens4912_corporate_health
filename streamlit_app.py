# streamlit_app.py
# Açıklama: Bu dosya, SUT RAG motorunu kullanarak bir Streamlit arayüzü oluşturur.
# Kullanıcıların bir yapay zeka modeli seçmesine, veritabanını yönetmesine ve
# Sağlık Uygulama Tebliği (SUT) hakkında ajan destekli sorular sormasına olanak tanır.

# --- Gerekli Kütüphaneler ---
import streamlit as st
import os
import time
import json
from dotenv import load_dotenv

# Yerel Modüller ve Sınıflar
from sut_rag_core import SUT_RAG_Engine, DB_PATH, FAISS_INDEX_PATH, FAISS_MAPPING_PATH

# LangChain Kütüphaneleri
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent

# --- Uygulama Yapılandırması (Configuration) ---

# Kullanılabilir LLM (Büyük Dil Modeli) seçenekleri ve yapılandırmaları
AVAILABLE_MODELS = {
    "LM Studio: GPT-OSS 20B (Yerel)": {
        "provider": "lmstudio",
        "model_name": "local-model",
        "api_key_name": None  # Yerel model için API anahtarı gerekmez
    },
        "LM Studio: llama 3.1 8B": {
        "provider": "lmstudio",
        "model_name": "meta-llama-3.1-8b-instruct",
        "api_key_name": None  # Yerel model için API anahtarı gerekmez
    },
        "LM Studio: qwen3 8B": {
        "provider": "lmstudio",
        "model_name": "qwen/qwen3-8b",
        "api_key_name": None  # Yerel model için API anahtarı gerekmez
    },
    "Google Gemini 2.5 Flash": {
        "provider": "google",
        "model_name": "gemini-2.5-flash",
        "api_key_name": "GEMINI_API_KEY"
    },
    "OpenRouter: Qwen/Qwen2 7B (Free)": {
        "provider": "openrouter",
        "model_name": "qwen/qwen2-7b-instruct:free",
        "api_key_name": "OPENROUTER_API_KEY"
    }
}


# =============================================================================
# ARAYÜZ OLUŞTURMA FONKSİYONLARI (UI FUNCTIONS)
# =============================================================================

def display_sidebar(engine_class):
    """
    Streamlit kenar çubuğunu (sidebar) oluşturur ve yönetir.

    Bu fonksiyon, model seçimi, API anahtarı kontrolü ve veritabanı
    yönetimi gibi ayarları kullanıcıya sunar.

    Returns:
        tuple: (seçilen modelin yapılandırması, uygulamanın çalışmaya hazır olup olmadığı)
    """
    st.sidebar.header("Ayarlar")

    # --- Model Seçimi ---
    selected_model_display_name = st.sidebar.selectbox(
        "Kullanılacak Yapay Zeka Modelini Seçin:",
        options=list(AVAILABLE_MODELS.keys())
    )
    selected_model_config = AVAILABLE_MODELS[selected_model_display_name]
    api_key_name = selected_model_config["api_key_name"]

    st.sidebar.header("Sistem Durumu")

    # --- API Anahtarı Kontrolü ---
    is_api_key_ok = False
    if api_key_name:
        api_key = os.getenv(api_key_name)
        if api_key:
            st.sidebar.success(f"{api_key_name} yüklü.")
            is_api_key_ok = True
        else:
            st.sidebar.error(f"{api_key_name} ortam değişkeni ayarlanmamış.")
            st.sidebar.info(f"Lütfen '{api_key_name}' anahtarını .env dosyasına ekleyin.")
            is_api_key_ok = False
    else:
        st.sidebar.success("Yerel model için API anahtarı gerekmez.")
        is_api_key_ok = True

    # --- Veritabanı Kontrolü ---
    db_exists = all([
        os.path.exists(DB_PATH),
        os.path.exists(FAISS_INDEX_PATH),
        os.path.exists(FAISS_MAPPING_PATH)
    ])

    if db_exists:
        st.sidebar.success("Veritabanı (SQLite/FAISS) yüklü ve hazır.")
        if st.sidebar.button("Mevcut Veritabanını Sil", type="secondary"):
            if os.path.exists(DB_PATH): os.remove(DB_PATH)
            if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
            if os.path.exists(FAISS_MAPPING_PATH): os.remove(FAISS_MAPPING_PATH)
            st.rerun()
    else:
        st.sidebar.warning("Veritabanı bulunamadı. Lütfen oluşturun.")
        if st.sidebar.button("Veritabanını Oluştur", type="primary"):
            temp_engine = engine_class()
            with st.spinner("DOCX dönüştürülüyor ve vektör veritabanı oluşturuluyor..."):
                temp_engine.populate_database()
            st.rerun()

    is_ready = is_api_key_ok and db_exists
    return selected_model_config, is_ready


def display_chat_interface(engine):
    """
    Ana sohbet arayüzünü oluşturur ve kullanıcı etkileşimini yönetir.
    """
    st.title("SUT Ajan Asistanı")
    st.caption("Sosyal Güvenlik Kurumu Sağlık Uygulama Tebliği (SUT) destekli yapay zeka asistanı.")

    # Oturum durumunda (session state) mesaj geçmişini başlat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mesaj geçmişini ekrana yazdır
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            # Eğer mesajda kaynaklar varsa, bunları bir expander içinde göster
            if "sources" in msg and msg["sources"]:
                for source in msg["sources"]:
                    with st.expander(f"İncelenen Kaynak: {source['title']}"):
                        cleaned_content = source['content'].replace('~~', '')
                        st.markdown(cleaned_content, unsafe_allow_html=True)

    # Kullanıcıdan yeni bir girdi al
    if prompt := st.chat_input("SUT ile ilgili bir soru sorun..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Asistanın cevabını oluştur ve akış (stream) olarak göster
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_text, final_sources = "", []
            start_time = time.time()

            try:
                # Ajan motorundan gelen adımları (yield) tek tek işle
                for step in engine.query_agentic_rag_stream(prompt, k=10):
                    current_display_text = full_response_text

                    if "status" in step:
                        current_display_text += f"*{step['status']}*\n\n"
                    elif "tool_call" in step:
                        header = step['tool_call'].get('header', 'Başlık bilgisi yok')
                        args = step['tool_call'].get('args', '{}')
                        current_display_text += f"⚙️ **Araç Kullanımı:** Model, **'{header}'** başlıklı bölümü (`{args}`) incelemek için bir araç kullanıyor.\n\n"
                    elif "tool_output" in step:
                        current_display_text += f"**Araç Çıktısı Alındı ve Modele Geri Gönderiliyor...**\n\n"
                    elif "final_answer" in step:
                        full_response_text = step['final_answer']
                        final_sources = step.get("used_sources", [])
                        current_display_text = full_response_text

                    # Arayüzü güncel metinle anlık olarak güncelle
                    message_placeholder.markdown(current_display_text + "▌")

                # Akış bittiğinde, son metni imleç olmadan göster
                message_placeholder.markdown(full_response_text)

                # Cevapta kullanılan kaynakları göster
                if final_sources:
                    for source in final_sources:
                        with st.expander(f"İncelenen Kaynak: {source['title']}"):
                            cleaned_content = source['content'].replace('~~', '')
                            st.markdown(cleaned_content, unsafe_allow_html=True)

                # Son cevabı mesaj geçmişine ekle
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response_text,
                    "sources": final_sources
                })

            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")

            elapsed_time = time.time() - start_time
            st.caption(f"Sorgu süresi: {elapsed_time:.2f} saniye")


# =============================================================================
# MOTOR VE AJAN KURULUMU (ENGINE & AGENT SETUP)
# =============================================================================

@st.cache_resource
def get_rag_engine(provider: str, model_name: str):
    """
    SUT_RAG_Engine'i başlatır ve ajan sorgulama yeteneğini ekler.

    Streamlit'in @st.cache_resource dekoratörü sayesinde, bu fonksiyon sadece
    argümanları değiştiğinde çalışır ve motor nesnesini hafızada tutar.
    Bu, her etkileşimde motorun yeniden yüklenmesini engeller.
    """
    print(f"Yeni RAG motoru oluşturuluyor: Sağlayıcı='{provider}', Model='{model_name}'")
    engine = SUT_RAG_Engine(llm_provider=provider, model_name=model_name)

    def query_agentic_rag_stream(self, query: str, k: int = 5):
        """
        Bir sorguyu ajan tabanlı bir RAG akışıyla işler.

        Bu fonksiyon, RAG motoru nesnesine sonradan eklenir (monkey-patching).
        Sorguyla ilgili SUT bölümlerini bulur, bir LLM ajanını araçlarla donatır
        ve cevabı adımlar halinde (stream) döndürür.
        """
        # --- 1. Başlangıç Kontrolleri ---
        if self.llm is None:
            yield {"error": "[LLM DEVRE DIŞI] Lütfen ilgili API anahtarını ayarlayın veya LM Studio sunucusunu başlatın."}
            return

        yield {"status": "İlgili SUT bölümleri aranıyor..."}
        candidate_chunks = self._retrieve_chunks(query, k)

        if not candidate_chunks:
            yield {"final_answer": "Üzgünüm, bu soruya SUT içerisinde alakalı bir bilgi bulamadım.", "used_sources": []}
            return

        chunk_id_to_metadata_map = {chunk['id']: chunk['metadata'] for chunk in candidate_chunks}
        yield {"status": "Potansiyel olarak ilgili bölümler bulundu. LLM'e gönderiliyor..."}

        # --- 2. Ajan Araçlarının Tanımlanması ---
        class ChunkIdInput(BaseModel):
            chunk_id: str = Field(description="İçeriği alınacak bölümün benzersiz kimliği (ID).")
        class SectionTitleInput(BaseModel):
            section_title: str = Field(description="İçeriği aranacak bölümün başlığı veya numarası (örn: '2.5.2.A').")

        tools = [
            StructuredTool.from_function(
                name="get_sut_section_content_by_id",
                func=self.get_chunk_content_by_id,
                description="İlk başta sana sunulan aday listeden bir bölümün tam metnini almak için bu aracı kullan.",
                args_schema=ChunkIdInput
            ),
            StructuredTool.from_function(
                name="get_sut_section_by_title",
                func=self.get_section_by_title,
                description="Okuduğun bir metin içinde '2.5.2.A maddesine bakınız' gibi spesifik bir SUT bölüm numarasına metin bazlı bir referans görürsen, o referans verilen bölümün tam metnini almak için bu aracı kullan.",
                args_schema=SectionTitleInput
            )
        ]

        # --- 3. Ajanın Hazırlanması (Sistem Mesajı ve Prompt) ---
        agent_executor = create_react_agent(self.llm, tools)

        # Başlangıç bağlamını (context) oluştur
        context_summary = ""
        for i, chunk in enumerate(candidate_chunks):
            section_info = ' > '.join([v for k, v in chunk['metadata'].items() if k.startswith('Header') and v])
            context_summary += f"[{i+1}] ID: '{chunk['id']}'\n    Başlık: {section_info}\n    Önizleme: {chunk['text'][:200].strip()}...\n\n"
        
        # --- DEĞİŞİKLİK: AJANIN DAVRANIŞINI KONTROL EDEN SİSTEM MESAJI ---
        # Bu sistem mesajı, ajanın sonsuz döngüye girmesini engellemek ve
        # görevini doğru bir şekilde yerine getirmesini sağlamak için kritik öneme sahiptir.
        system_message = """SEN BİR SAĞLIK UZMANI ASİSTANISIN. Görevin, SUT (Sağlık Uygulama Tebliği) hakkındaki soruları cevaplamaktır. Sadece sana verilen metinlere ve araçlara dayanarak cevap ver. Yorum yapma, sadece metinlerdeki bilgiyi özetle.

**EN ÖNEMLİ KURAL: Aynı aracı aynı chunk_id için kullanma. Kullanıcının sorusunu cevaplamak için yeterli bilgiyi topladığına inandığında, KESİNLİKLE başka bir araç KULLANMA. Amacın, araçları kullandıktan sonra nihai bir cevap oluşturmaktır. Aynı aracı aynı argümanla tekrar tekrar çağırma. Bilgiyi topladıktan sonra, düşüncelerini özetleyip kullanıcıya cevap ver.**

GÖREVİN:
1.  **Analiz Et:** Sağlanan bölüm özetlerini dikkatlice incele. Kullanıcının sorusunu doğrudan cevaplayan bir bilgi içerip içermediklerini kontrol et.
2.  **Araç Kullan (Gerekirse):**
    *   Eğer özetler soruyu cevaplamak için yetersizse, en alakalı görünen bölümün tam metnini almak için `get_sut_section_content_by_id` aracını KULLAN.
    *   EĞER OKUDUĞUN BİR METNİN İÇİNDE başka bir SUT bölümüne atıf yapıldığını görürsen, o referans verilen bölümün içeriğini getirmek için `get_sut_section_by_title` aracını KULLAN.
3.  **Cevapla:** Gerekli tüm bilgileri topladıktan sonra, kullanıcı sorusunu net ve kapsamlı bir şekilde Türkçe olarak yanıtla. Cevabın kesinlikle bilgi topladığın metinlerden türetilmelidir. Eğer metinlerde cevap yoksa, "Sağlanan SUT metinlerinde bu soruya doğrudan bir cevap bulunmamaktadır." de.
4.  **Kaynak Belirt:** Cevabının sonuna `\n\n**Kullanılan Kaynaklar:**` başlığı ekle. Bu başlığın altına, cevabını dayandırdığın SUT bölümlerinin başlıklarını madde imleri kullanarak listele."""
        
        human_input = f"Kullanıcı Sorusu: {query}\n\nİlgili Olabilecek SUT Bölümleri:\n---\n{context_summary}"
        messages = [SystemMessage(content=system_message), HumanMessage(content=human_input)]
        
        # --- 4. Ajanın Çalıştırılması ve Yanıtın İşlenmesi ---
        used_sources, last_called_header, final_answer = [], "Bilinmeyen Kaynak", "Cevap oluşturulamadı."
        
        print("\n" + "="*50 + " YENİ SORGU İÇİN AJAN BAŞLATILIYOR " + "="*50)
        print(f"\n[DEBUG] AJANA GÖNDERİLEN PROMPT:\n--- SYSTEM MESSAGE ---\n{system_message}\n--- HUMAN MESSAGE ---\n{human_input}\n" + "-"*120)

        try:
            # Not: recursion_limit'i artırmak bir çözüm değildir, asıl çözüm prompt'u düzeltmektir.
            response = agent_executor.invoke({"messages": messages})
            
            print(f"\n[DEBUG] AJANDAN GELEN HAM YANIT (RAW RESPONSE):\n{response}\n" + "="*120 + "\n")

            output_messages = response.get("messages", [])
            for msg in output_messages:
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        # Model bir araç kullanmaya karar verdi
                        for tool_call in msg.tool_calls:
                            args_dict = tool_call.get('args', {})
                            header_text, tool_name = "Bilinmeyen Kaynak", tool_call['name']

                            # Kullanılan araca göre kaynak başlığını belirle
                            if tool_name == 'get_sut_section_content_by_id' and (chunk_id := args_dict.get('chunk_id')):
                                if metadata := chunk_id_to_metadata_map.get(chunk_id, {}):
                                    section_titles = [v for k, v in metadata.items() if k.startswith("Header") and v]
                                    if section_titles: header_text = ' > '.join(section_titles)
                            elif tool_name == 'get_sut_section_by_title' and (title_arg := args_dict.get('section_title')):
                                header_text = title_arg

                            last_called_header = header_text
                            yield {"tool_call": {"name": tool_name, "header": header_text, "args": json.dumps(args_dict)}}
                    else:
                        # Model nihai cevabını verdi
                        final_answer_content = msg.content if isinstance(msg.content, str) else "".join(part.get('text', '') for part in msg.content if isinstance(part, dict))
                        final_answer = final_answer_content.strip() if final_answer_content.strip() else "Anlaşılır bir cevap üretilemedi."
                        print(f"\n[DEBUG] PARSE EDİLEN NİHAİ CEVAP: '{final_answer}'\n")

                elif isinstance(msg, ToolMessage):
                    # Aracın çıktısı geldi, bunu kaynak olarak ekle
                    used_sources.append({"title": last_called_header, "content": msg.content})
                    yield {"tool_output": msg.content}

            yield {"final_answer": final_answer, "used_sources": used_sources}

        except Exception as e:
            print(f"\n[DEBUG] AJAN ÇALIŞTIRILIRKEN KRİTİK HATA: {e}\n")
            yield {"error": f"[AGENT ERROR] Ajan çalıştırılırken hata: {e}"}

    # Oluşturulan `query_agentic_rag_stream` fonksiyonunu engine nesnesine bir metot olarak ekle
    engine.query_agentic_rag_stream = query_agentic_rag_stream.__get__(engine, SUT_RAG_Engine)
    return engine


# =============================================================================
# ANA UYGULAMA AKIŞI (MAIN APPLICATION FLOW)
# =============================================================================

def main_app():
    """
    Uygulamanın ana giriş noktası.
    """
    load_dotenv()  # .env dosyasındaki ortam değişkenlerini yükle

    # Kenar çubuğunu göster ve ayarları al
    selected_model_config, is_ready = display_sidebar(SUT_RAG_Engine)

    if is_ready:
        try:
            # RAG motorunu başlat (cache'den al)
            engine = get_rag_engine(
                provider=selected_model_config["provider"],
                model_name=selected_model_config["model_name"]
            )
        except Exception as e:
            st.error(f"Uygulama başlatılırken kritik bir hata oluştu: {e}")
            st.stop()

        # Motorun veritabanını yüklediğinden emin ol
        if not engine.faiss_index:
            if not engine.load_database():
                st.error("Veritabanı dosyaları mevcut ancak yüklenemedi...")
                return

        # Sohbet arayüzünü göster
        display_chat_interface(engine)
    else:
        # Sistem hazır değilse kullanıcıyı bilgilendir
        st.info("Lütfen kenar çubuğundaki (sidebar) adımları tamamlayarak devam edin.")


if __name__ == "__main__":
    main_app()