import os
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load API Key
# It looks for .env file or uses the hardcoded string if not found
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# If you want to paste your key directly for a quick test, uncomment below:
# API_KEY = "YOUR_PASTED_API_KEY_HERE"

if not API_KEY:
    print("❌ Error: API Key not found. Please set GEMINI_API_KEY in .env or script.")
    exit()

genai.configure(api_key=API_KEY)

def list_available_models():
    print(f"🔍 Checking available models for your API Key...\n")
    
    try:
        # Fetch all models
        models = list(genai.list_models())
        
        chat_models = []
        embedding_models = []
        other_models = []

        for m in models:
            # Check capabilities
            if 'generateContent' in m.supported_generation_methods:
                chat_models.append(m)
            elif 'embedContent' in m.supported_generation_methods:
                embedding_models.append(m)
            else:
                other_models.append(m)

        # --- DISPLAY CHAT/GENERATION MODELS ---
        print(f"🤖 --- GENERATION MODELS (Chat/Text) ---")
        print(f"{'Model Name':<30} | {'Input Limit':<12} | {'Output Limit':<12}")
        print("-" * 60)
        for m in chat_models:
            # Clean up the name (remove 'models/')
            name = m.name.replace('models/', '')
            in_limit = m.input_token_limit
            out_limit = m.output_token_limit
            print(f"{name:<30} | {in_limit:<12} | {out_limit:<12}")

        # --- DISPLAY EMBEDDING MODELS ---
        print(f"\n🧠 --- EMBEDDING MODELS (Vector Search) ---")
        for m in embedding_models:
            name = m.name.replace('models/', '')
            print(f" • {name}")

    except Exception as e:
        print(f"❌ Error fetching models: {e}")

if __name__ == "__main__":
    list_available_models()