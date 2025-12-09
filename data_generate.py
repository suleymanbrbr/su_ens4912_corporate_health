import pandas as pd
import json
import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Import your RAG engine
try:
    from sut_rag_core import SUT_RAG_Engine
    RAG_AVAILABLE = True
except ImportError:
    print("❌ ERROR: sut_rag_core.py not found.")
    exit()

# --- CONFIGURATION ---
INPUT_CSV = "sut_questions.csv"
OUTPUT_JSONL = "sut_agentic_harmony_finetune_long.jsonl"
MODEL_NAME = "gemini-2.5-flash" 

# --- AGENT PROMPT ---
AGENT_SIMULATION_PROMPT = """
You are an expert AI creating a dataset for "Tool-Use Reasoning".
You need to write the internal monologue of an AI Agent solving a problem.

INPUT DATA:
- User Question: "{question}"
- Correct Answer: "{answer}"
- Tool To Use: SUT Database Search
- Real Tool Output (Context): "{context}"

TASK:
Write the response in OpenAI Harmony format with two channels.

1. 'analysis' Channel:
   - Start by understanding the user's intent.
   - Explicitly state: "I need to search the SUT for section {source_ref}."
   - SIMULATE the tool output. Write: "Tool Output: [Insert the Real Tool Output provided above]".
   - Reason about that text. Does it support the answer?
   - Conclude.

2. 'final' Channel:
   - Provide the polite, accurate answer based on the analysis.

REQUIRED FORMAT (Strict Text):
<|start|>role=assistant|channel=analysis<|message|>
User is asking about...
I will check the SUT database for section {source_ref}.
> Tool Call: get_section_by_title("{source_ref}")
> Tool Output: "{context}"
Reviewing the text... The section states that...
Therefore...
<|end|>
<|start|>role=assistant|channel=final<|message|>
{answer}
<|end|>
"""

class AgenticDatasetGenerator:
    def __init__(self):
        load_dotenv()
        
        # 1. Load Keys List
        self.api_keys = self._load_api_keys()
        if not self.api_keys:
            print("❌ No API Keys found in GEMINI_API_KEYS variable.")
            exit()
            
        self.current_key_index = 0
        print(f"🔑 Loaded {len(self.api_keys)} API Keys. Starting with Key #1.")
        
        # 2. Initialize First LLM
        self._init_llm()
        
        print("🔌 Initializing SUT Engine to fetch REAL Contexts...")
        self.engine = SUT_RAG_Engine()
        self.engine.load_database()

    def _load_api_keys(self):
        """Robustly parses keys from .env whether they are JSON list or comma-separated"""
        keys_env = os.getenv("GEMINI_API_KEYS")
        if not keys_env: return []
        
        try:
            # Try parsing as JSON list '["a", "b"]'
            return json.loads(keys_env)
        except:
            # Fallback to comma separation "a,b,c"
            return [k.strip() for k in keys_env.split(",") if k.strip()]

    def _init_llm(self):
        """Initializes the LLM with the CURRENT key"""
        current_key = self.api_keys[self.current_key_index]
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            api_key=current_key,
            temperature=1.0, 
        )

    def rotate_key(self):
        """Switches to the next key in the list"""
        old_index = self.current_key_index
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        print(f"🔄 Switching API Key: From Key #{old_index+1} -> Key #{self.current_key_index+1}")
        self._init_llm()

    def load_progress(self):
        finished_ids = set()
        if os.path.exists(OUTPUT_JSONL):
            with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "custom_id" in data:
                            finished_ids.add(str(data["custom_id"]))
                    except: pass
        return finished_ids

    def get_real_context(self, source_ref):
        query = str(source_ref)
        chunks = self.engine._retrieve_chunks(query, k=1)
        if chunks:
            return chunks[0]['text']
        else:
            return None

    def run(self):
        if not os.path.exists(INPUT_CSV):
            print(f"❌ {INPUT_CSV} not found.")
            return

        df = pd.read_csv(INPUT_CSV)
        df['ID'] = df['ID'].astype(str)
        finished_ids = self.load_progress()
        
        df_todo = df[~df['ID'].isin(finished_ids)]
        print(f"🚀 Processing {len(df_todo)} items with Multi-Key Rotation...")

        for index, row in df_todo.iterrows():
            row_id = row['ID']
            question = row['Soru']
            target_answer = row['Cevap']
            source_ref = row['Kaynak']
            
            print(f"ID {row_id} | Ref: {source_ref}...", end="", flush=True)

            # --- KEY ROTATION & RETRY LOGIC ---
            attempts = 0
            max_attempts = len(self.api_keys) + 1 # Try all keys + 1 retry of the first one
            success = False

            while attempts < max_attempts:
                try:
                    # 1. Fetch Context (Only need to do this once, but it's fast)
                    real_context = self.get_real_context(source_ref)
                    if not real_context:
                        print(f" ⚠️ Context missing. Skipping.")
                        success = True # Mark as "handled" so we move to next row
                        break

                    if len(real_context) >6000:
                        real_context = real_context[:6000] + "... (truncated)"
                    
                    real_context_flat = real_context.replace("\n", " ")

                    # 2. Generate
                    prompt = AGENT_SIMULATION_PROMPT.format(
                        question=question,
                        answer=target_answer,
                        source_ref=source_ref,
                        context=real_context_flat
                    )

                    response = self.llm.invoke([HumanMessage(content=prompt)])
                    harmony_output = response.content.strip()
                    
                    # Clean up
                    harmony_output = harmony_output.replace("```xml", "").replace("```", "").strip()

                    # 3. Save
                    entry = {
                        "custom_id": row_id,
                        "instruction": "You are a SUT Expert. Use the Harmony format to analyze and answer.",
                        "input": question,
                        "output": harmony_output
                    }

                    with open(OUTPUT_JSONL, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

                    print(f" Saved (Key #{self.current_key_index+1}). ✅")
                    time.sleep(1) 
                    success = True
                    break # Break the retry loop, move to next row

                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # CHECK FOR QUOTA / RATE LIMITS
                    if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                        print(f"\n   ⚠️ Key #{self.current_key_index+1} Exhausted. Rotating...")
                        self.rotate_key()
                        attempts += 1
                        time.sleep(1) # Brief pause before retry
                    
                    # CHECK FOR MODEL NOT FOUND
                    elif "404" in error_msg and "not found" in error_msg:
                        print(f"\n   ❌ Model '{MODEL_NAME}' not found/supported by this key.")
                        self.rotate_key()
                        attempts += 1
                    
                    else:
                        print(f"\n   ❌ Non-Quota Error: {e}")
                        # If it's a content error, skipping the row is safer than retrying forever
                        success = True 
                        break
            
            # If we exited the while loop and success is False, it means we ran out of keys
            if not success:
                print(f"\n🚨 All {len(self.api_keys)} API keys exhausted (tried full cycle). Stopping script.")
                break

if __name__ == "__main__":
    gen = AgenticDatasetGenerator()
    gen.run()