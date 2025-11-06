import streamlit as st
import os
import time
import json
from sut_rag_core import SUT_RAG_Engine, DB_PATH, FAISS_INDEX_PATH, FAISS_MAPPING_PATH
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent

# ----------------------------------------------------------------------
# ARAYÜZ FONKSİYONLARI (MODIFIED)
# ----------------------------------------------------------------------

def display_sidebar(engine):
    """Kenar çubuğunu (Sidebar) oluşturur ve veritabanı durumunu gösterir."""
    st.sidebar.header("Sistem Durumu")
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        st.sidebar.success("Gemini API Anahtarı yüklü.")
    else:
        st.sidebar.error("GEMINI_API_KEY ortam değişkeni ayarlanmamış.")
        return False

    db_exists = (os.path.exists(DB_PATH) and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH))
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
            with st.spinner("DOCX dönüştürülüyor ve vektör veritabanı oluşturuluyor..."):
                engine.populate_database()
            st.rerun()
    return api_key and db_exists

def display_chat_interface(engine):
    """
    Handles the entire chat UI, including message history with expandable sources
    and real-time streaming of the agent's response.
    """
    st.title("SUT Ajan Asistanı")
    st.caption("Sosyal Güvenlik Kurumu Sağlık Uygulama Tebliği (SUT) destekli yapay zeka asistanı.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages from session state
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            # If the message has associated sources, display them as expanders
            if "sources" in msg and msg["sources"]:
                for source in msg["sources"]:
                    with st.expander(f"İncelenen Kaynak: {source['title']}"):
                        st.text_area("Kaynak Metni", value=source['content'], height=200, disabled=True, key=f"source_{msg['id']}_{source['title']}")

    # Handle new chat input
    if prompt := st.chat_input("SUT ile ilgili bir soru sorun..."):
        # Add user message to state and display it
        user_msg_id = f"user_{time.time()}"
        st.session_state.messages.append({"role": "user", "content": prompt, "id": user_msg_id})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Start displaying agent's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_text = ""
            final_sources = []
            start_time = time.time()
            
            try:
                # Stream the agent's thought process and final answer
                for step in engine.query_agentic_rag_stream(prompt, k=5):
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
                        current_display_text = full_response_text # Only show the final text at the end
                    
                    message_placeholder.markdown(current_display_text + "▌")

                # After streaming is complete, render the final state permanently
                message_placeholder.markdown(full_response_text)
                if final_sources:
                    for source in final_sources:
                        with st.expander(f"İncelenen Kaynak: {source['title']}"):
                            st.text_area("Kaynak Metni", value=source['content'], height=200, disabled=True, key=f"source_current_{source['title']}")
                
                # Save the complete response (text + sources) to session state for history
                assistant_msg_id = f"asst_{time.time()}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response_text,
                    "sources": final_sources,
                    "id": assistant_msg_id
                })

            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
            
            elapsed_time = time.time() - start_time
            st.caption(f"Sorgu süresi: {elapsed_time:.2f} saniye")


@st.cache_resource
def get_rag_engine():
    """
    Initializes the SUT_RAG_Engine and monkey-patches the streaming query method onto it.
    This function is cached to act as a singleton, ensuring the engine is created only once.
    """
    engine = SUT_RAG_Engine()
    
    # Define the core agentic logic as a function that will be bound to the engine instance.
    def query_agentic_rag_stream(self, query: str, k: int = 5):
        """
        A generator function that executes the agentic RAG process and yields
        the state of each step for real-time display in the UI.
        """
        # --- 1. Initial Setup and Retrieval ---
        if self.llm is None:
            yield {"error": "[LLM DEVRE DIŞI] Lütfen GEMINI_API_KEY'i ayarlayın."}
            return

        yield {"status": "İlgili SUT bölümleri aranıyor..."}
        candidate_chunks = self._retrieve_chunks(query, k)
        
        if not candidate_chunks:
            yield {"final_answer": "Üzgünüm, bu soruya SUT içerisinde alakalı bir bilgi bulamadım.", "used_sources": []}
            return
        
        chunk_id_to_metadata_map = {chunk['id']: chunk['metadata'] for chunk in candidate_chunks}
        yield {"status": "Potansiyel olarak ilgili bölümler bulundu. LLM'e gönderiliyor..."}

        # --- 2. Define Agent's Tools ---
        tools = [
            Tool(
                name="get_sut_section_content_by_id",
                func=self.get_chunk_content_by_id,
                description="İlk başta sana sunulan aday listeden bir bölümün tam metnini almak için bu aracı kullan. Argüman olarak bölümün 'chunk_id' değerini ver."
            ),
            Tool(
                name="get_sut_section_by_title",
                func=self.get_section_by_title,
                description="Okuduğun bir metin içinde '2.5.2.A maddesine bakınız' gibi spesifik bir SUT bölüm numarasına metin bazlı bir referans görürsen, o referans verilen bölümün tam metnini almak için bu aracı kullan. Argüman olarak bölüm başlığını (örneğin '2.5.2.A') ver."
            )
        ]
        
        agent_executor = create_react_agent(self.llm, tools)
        
        # --- 3. Construct the Prompt for the Agent ---
        context_summary = ""
        for i, chunk in enumerate(candidate_chunks):
            section_info = '; '.join([v for k, v in chunk['metadata'].items() if k.startswith('Header') and v])
            context_summary += f"[{i+1}] ID: '{chunk['id']}'\n    Başlık: {section_info}\n    Önizleme: {chunk['text'][:150].strip()}...\n\n"

        system_message = """SEN BİR SAĞLIK UZMANI ASİSTANISIN. Görevin, SUT (Sağlık Uygulama Tebliği) hakkındaki soruları cevaplamaktır.

GÖREVİN:
1.  **Analiz Et:** Sağlanan bölüm özetlerini dikkatlice incele. Cevap için yeterli olup olmadıklarına karar ver.
2.  **Araç Kullan (Gerekirse):**
    *   Eğer özetler yetersizse, en alakalı görünen bölümün tam metnini almak için `get_sut_section_content_by_id` aracını KULLAN.
    *   EĞER OKUDUĞUN BİR METNİN İÇİNDE başka bir SUT bölümüne atıf yapıldığını (örneğin '2.5.3.A maddesine bakınız') görürsen, o referans verilen bölümün içeriğini getirmek için `get_sut_section_by_title` aracını KULLAN. Bu, cevabı zenginleştirmek için çok önemlidir.
3.  **Cevapla:** Gerekli tüm bilgileri topladıktan sonra, kullanıcı sorusunu net ve kapsamlı bir şekilde Türkçe olarak yanıtla.
4.  **Kaynak Belirt:** Cevabının sonuna `\n\n**Kullanılan Kaynaklar:**` başlığı ekle. Bu başlığın altına, cevabını dayandırdığın SUT bölümlerinin başlıklarını (örneğin, '2.5.3.A - Yurt dışı tedavi') madde imleri kullanarak listele."""
        
        human_input = f"Kullanıcı Sorusu: {query}\n\nİlgili Olabilecek SUT Bölümleri:\n---\n{context_summary}"
        messages = [SystemMessage(content=system_message), HumanMessage(content=human_input)]
        
        # --- 4. Execute Agent and Stream Results ---
        used_sources = []
        last_called_header = "Bilinmeyen Kaynak"
        final_answer = "Cevap oluşturulamadı."

        try:
            response = agent_executor.invoke({"messages": messages})
            output_messages = response.get("messages", [])

            for msg in output_messages:
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            args_raw = tool_call.get('args', {})
                            args_dict = {}
                            if isinstance(args_raw, str):
                                try: args_dict = json.loads(args_raw)
                                except json.JSONDecodeError: args_dict = {"raw_string": args_raw}
                            elif isinstance(args_raw, dict):
                                args_dict = args_raw
                            
                            header_text = "Bölüm Başlığı"
                            chunk_id = args_dict.get('chunk_id')
                            if chunk_id:
                                metadata = chunk_id_to_metadata_map.get(chunk_id, {})
                                section_titles = [v for k, v in metadata.items() if k.startswith("Header") and v]
                                if section_titles: header_text = ', '.join(section_titles)
                            
                            last_called_header = header_text
                            yield {"tool_call": {"name": tool_call['name'], "header": header_text, "args": json.dumps(args_dict)}}
                    else:
                        final_answer_content = ""
                        if isinstance(msg.content, str):
                            final_answer_content = msg.content
                        elif isinstance(msg.content, list):
                            for part in msg.content:
                                if isinstance(part, dict) and 'text' in part:
                                    final_answer_content += part['text']
                        final_answer = final_answer_content.strip() if final_answer_content else "Anlaşılır bir cevap üretilemedi."
                
                elif isinstance(msg, ToolMessage):
                    used_sources.append({"title": last_called_header, "content": msg.content})
                    yield {"tool_output": msg.content}

            yield {"final_answer": final_answer, "used_sources": used_sources}
        except Exception as e:
            yield {"error": f"[AGENT ERROR] Ajan çalıştırılırken hata: {e}"}

    SUT_RAG_Engine.query_agentic_rag_stream = query_agentic_rag_stream
    return engine

def main_app():
    try:
        engine = get_rag_engine()
    except Exception as e:
        st.error(f"Uygulama başlatılırken kritik bir hata oluştu: {e}")
        st.stop()
        
    is_ready = display_sidebar(engine)
    if is_ready:
        if not engine.faiss_index:
            if not engine.load_database():
                st.error("Veritabanı dosyaları mevcut ancak yüklenemedi...")
                return
        display_chat_interface(engine)
    else:
        st.info("Lütfen kenar çubuğundaki (sidebar) adımları tamamlayarak devam edin.")

if __name__ == "__main__":
    main_app()