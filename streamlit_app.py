# streamlit_app.py
# Description: Streamlit UI for SUT RAG.
# Compatible with: Golden Agent Loop (Direct Chain)

import streamlit as st
import os
import time
from dotenv import load_dotenv
from sut_rag_core import SUT_RAG_Engine, DB_PATH

# --- Configuration ---
AVAILABLE_MODELS = {
    "LM Studio: GPT-OSS (Harmony/Local)": {
        "provider": "lmstudio",
        "model_name": "local-model",
        "api_key_name": None
    },
    "Google Gemini 2.5 Flash": {
        "provider": "google",
        "model_name": "gemini-2.5-flash",
        "api_key_name": "GEMINI_API_KEY"
    },
    "OpenRouter (Free Tier)": {
        "provider": "openrouter",
        "model_name": "qwen/qwen2-7b-instruct:free",
        "api_key_name": "OPENROUTER_API_KEY"
    }
}

def display_sidebar():
    st.sidebar.header("⚙️ Ayarlar")
    
    # 1. Model Selection
    selected_name = st.sidebar.selectbox("Yapay Zeka Modeli", list(AVAILABLE_MODELS.keys()))
    config = AVAILABLE_MODELS[selected_name]
    
    # 2. API Key Check
    is_ready = False
    if config["api_key_name"]:
        if os.getenv(config["api_key_name"]):
            st.sidebar.success(f"✅ {config['api_key_name']} yüklü.")
            is_ready = True
        else:
            st.sidebar.error(f"❌ {config['api_key_name']} eksik!")
    else:
        st.sidebar.success("✅ Yerel Sunucu Modu")
        is_ready = True

    st.sidebar.divider()

    # 3. Database Management
    db_exists = os.path.exists(DB_PATH)
    if db_exists:
        st.sidebar.success("📂 Veritabanı Hazır")
        if st.sidebar.button("Veritabanını Sıfırla"):
            if os.path.exists(DB_PATH): os.remove(DB_PATH)
            st.rerun()
    else:
        st.sidebar.warning("⚠️ Veritabanı Yok")
        if st.sidebar.button("Veritabanı Oluştur (SUT)"):
            engine = SUT_RAG_Engine()
            with st.spinner("Belgeler işleniyor, indeksleniyor..."):
                engine.populate_database()
            st.success("Tamamlandı!")
            time.sleep(1)
            st.rerun()
            
    return config, (is_ready and db_exists)

def display_chat_interface(engine):
    st.title("🏥 SUT Mevzuat Asistanı")
    st.caption("Sağlık Uygulama Tebliği (SUT) üzerinde hibrit arama ve yapay zeka destekli analiz.")

    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # Display Analysis (CoT) if present
            if "analysis" in msg and msg["analysis"]:
                with st.expander("🧠 Yapay Zeka Düşünce Süreci", expanded=False):
                    st.markdown(msg["analysis"])
            st.markdown(msg["content"])

    # User Input
    if prompt := st.chat_input("Örn: Hangi kanser ilaçları ödenir?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant Response
        with st.chat_message("assistant"):
            # UI Placeholders
            analysis_expander = st.expander("🧠 Analiz Süreci (CoT)", expanded=True)
            analysis_placeholder = analysis_expander.empty()
            
            final_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Variables for accumulation
            full_response = ""
            full_analysis = ""
            used_sources = []
            
            # Timing
            start_time = time.time()
            
            try:
                # Iterate through the generator
                for step in engine.query_agentic_rag_stream(prompt):
                    
                    # 1. Status Update
                    if "status" in step:
                        status_placeholder.info(f"⚙️ {step['status']}")
                    
                    # 2. Source Metadata
                    elif "source" in step:
                        used_sources.append(step["source"])
                    
                    # 3. Harmony Analysis (Overwrite mode for clean updates)
                    elif "analysis_content" in step:
                        full_analysis = step['analysis_content']
                        analysis_placeholder.markdown(full_analysis)
                    
                    # 4. Final Answer (Delta or Block)
                    elif "final_answer" in step:
                        content = step['final_answer']
                        
                        # If using Local Harmony, it sends full blocks (Overwrite)
                        if engine.provider in ["lmstudio", "local"]:
                            full_response = content # Harmony sends full buffer usually
                        # If using Cloud/Gemini, it sends deltas (Append)
                        else:
                            full_response += content
                            
                        final_placeholder.markdown(full_response + "▌")
                    
                    # 5. Error
                    elif "error" in step:
                        st.error(step["error"])
                        full_response = "Hata oluştu."

            except Exception as e:
                st.error(f"Beklenmeyen Sistem Hatası: {e}")

            # Finalize UI
            status_placeholder.empty()
            final_placeholder.markdown(full_response) # Remove cursor
            
            elapsed_time = time.time() - start_time
            st.caption(f"Yanıt süresi: {elapsed_time:.2f} saniye")

            # Show Sources
            if used_sources:
                with st.expander("📚 Yararlanılan SUT Maddeleri"):
                    for i, src in enumerate(used_sources):
                        st.markdown(f"**{i+1}. {src.get('title', 'Bölüm')}**")
                        preview = src.get('content', '')
                        if len(preview) > 250: preview = preview[:250] + "..."
                        st.text(preview)

            # Save to History
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "analysis": full_analysis
            })

def main():
    load_dotenv()
    
    # Setup Engine Class
    config, is_ready = display_sidebar()
    
    if is_ready:
        try:
            # Initialize Engine
            engine = SUT_RAG_Engine(
                llm_provider=config["provider"],
                model_name=config["model_name"]
            )
            
            # Load Data
            if engine.load_database():
                display_chat_interface(engine)
            else:
                st.warning("Veritabanı oluşturulmadı. Lütfen yan menüden oluşturun.")
                
        except Exception as e:
            st.error(f"Başlatma Hatası: {e}")
    else:
        st.info("Sistemi başlatmak için lütfen sol menüyü kullanın.")

if __name__ == "__main__":
    main()