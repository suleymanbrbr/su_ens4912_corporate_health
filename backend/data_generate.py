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
        
        # 1. Load Single Key
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("❌ No GEMINI_API_KEY found in environment.")
            exit()
            
        print("🔑 Loaded Gemini API Key.")
        
        # 2. Initialize LLM
        self._init_llm()
        
        print("🔌 Initializing SUT Engine to fetch REAL Contexts...")
        self.engine = SUT_RAG_Engine()
        self.engine.load_database()

    def _init_llm(self):
        """Initializes the LLM with the API key"""
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            api_key=self.api_key,
            temperature=1.0, 
        )

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

            try:
                # 1. Fetch Context
                real_context = self.get_real_context(source_ref)
                if not real_context:
                    print(f" ⚠️ Context missing. Skipping.")
                    continue

                if len(real_context) > 6000:
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

                print(f" Saved. ✅")
                time.sleep(1) 

            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "quota" in error_msg:
                    print(f"\n   ⚠️ Quota Exhausted. Waiting 60s...")
                    time.sleep(60)
                else:
                    print(f"\n   ❌ Error: {e}")
                    continue

if __name__ == "__main__":
    gen = AgenticDatasetGenerator()
    gen.run()