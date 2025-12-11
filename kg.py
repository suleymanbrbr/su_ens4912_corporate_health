import os
import json
import re
import time
import google.generativeai as genai
from docx import Document
from dotenv import load_dotenv
from tqdm import tqdm
import networkx as nx
from pyvis.network import Network

# --- CONFIGURATION ---
INPUT_DOCX = "08.03.2025-Değişiklik Tebliği İşlenmiş Güncel 2013 SUT.docx"
JSON_OUTPUT = "sut_knowledge_graph.json"
HTML_OUTPUT = "sut_knowledge_graph.html"
MODEL_NAME = "gemini-2.5-flash" 

# --- CLASS: API KEY MANAGER ---
class KeyManager:
    def __init__(self):
        load_dotenv()
        keys_str = os.getenv("GEMINI_API_KEYS", "[]")
        try:
            self.keys = json.loads(keys_str)
        except json.JSONDecodeError:
            print("❌ Error: GEMINI_API_KEYS in .env is not a valid JSON list.")
            self.keys = []
        
        if not self.keys:
            raise ValueError("No API Keys found in .env file!")
        
        self.current_index = 0
        print(f"🔑 Loaded {len(self.keys)} API Keys.")

    def get_current_key(self):
        return self.keys[self.current_index]

    def rotate_key(self):
        self.current_index = (self.current_index + 1) % len(self.keys)
        new_key = self.keys[self.current_index]
        print(f"🔄 Rotating to API Key #{self.current_index + 1}...")
        return new_key

# --- CLASS: SUT CHUNKER ---
class SUTChunker:
    def __init__(self, filepath):
        self.filepath = filepath
        self.header_pattern = re.compile(r"^(\d+(\.\d+)+(\.[A-Za-z])?|EK-[\d\w/]+)\s")

    def _get_clean_text(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File {self.filepath} not found.")
        
        doc = Document(self.filepath)
        clean_lines = []

        for p in doc.paragraphs:
            para_text = ""
            for run in p.runs:
                if not run.font.strike:
                    para_text += run.text
            
            cleaned = para_text.strip()
            if cleaned:
                clean_lines.append(cleaned)
        return clean_lines

    def chunk_document(self):
        lines = self._get_clean_text()
        chunks = []
        current_chunk = {"id": "ROOT", "title": "General", "text": [], "parent": "ROOT"}

        print(f"📄 Processing {len(lines)} clean text lines...")

        for line in lines:
            match = self.header_pattern.match(line)
            if match:
                if current_chunk["text"]:
                    chunks.append(self._finalize(current_chunk))

                code = match.group(1)
                parent = "ROOT"
                if "." in code:
                    parts = code.rsplit(".", 1)
                    if len(parts) > 0:
                        parent = parts[0]

                current_chunk = {
                    "id": code,
                    "title": line[:100],
                    "text": [line],
                    "parent": parent
                }
            else:
                current_chunk["text"].append(line)
        
        if current_chunk["text"]:
            chunks.append(self._finalize(current_chunk))
            
        return chunks

    def _finalize(self, data):
        return {
            "id": data["id"],
            "parent_id": data["parent"],
            "full_text": "\n".join(data["text"])
        }

# --- CLASS: GRAPH BUILDER ---
class GraphBuilder:
    def __init__(self, key_manager):
        self.key_manager = key_manager
        self.nodes = {} 
        self.edges = []
        self._configure_model()

    def _configure_model(self):
        """Re-configures Gemini with the current active key and UNSAFE settings."""
        genai.configure(api_key=self.key_manager.get_current_key())
        
        # Disable safety filters to prevent empty responses on medical text
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        self.model = genai.GenerativeModel(
            MODEL_NAME,
            generation_config={"response_mime_type": "application/json"},
            safety_settings=safety_settings
        )

    def _get_prompt(self, chunk):
        return f"""
        You are an expert Data Engineer specializing in the Turkish Social Security Institution (SGK) Health Implementation Notification (SUT).
        
        GOAL: Extract a highly structured Knowledge Graph from the provided SUT legal text.
        
        --- ONTOLOGY (STRICTLY FOLLOW THESE TYPES) ---
        
        1. **NODES** (Entities):
           - "RULE": The SUT Article ID (e.g., "4.2.14.C"). The text property holds the content.
           - "DRUG": Active ingredient or brand name (e.g., "Infliximab", "Anti-TNF").
           - "DIAGNOSIS": Disease name or ICD code (e.g., "Romatoid Artrit", "G80").
           - "SPECIALIST": Doctor branch or board (e.g., "Nöroloji Uzmanı", "3. Basamak Sağlık Kurulu").
           - "CONDITION": A logical requirement (e.g., "DAS28 > 5.1", "3 ay süreyle", "Metotreksat direnci").
           - "DOCUMENT": Required paperwork (e.g., "Genetik Test Raporu", "Patoloji Raporu").
           - "DEVICE": Medical equipment (e.g., "CPAP Cihazı").

        2. **EDGES** (Relationships):
           - "COVERS": Rule -> Drug/Device (The rule talks about this item).
           - "TREATS": Drug -> Diagnosis (Indication).
           - "ISSUED_BY": Document -> Specialist (Who writes the report).
           - "PRESCRIBED_BY": Drug -> Specialist (Who writes the prescription).
           - "REQUIRES_CONDITION": Drug/Rule -> Condition (Logic gate).
           - "MUST_FAIL_FIRST": Drug -> Drug (Step therapy: Must fail Drug A to get Drug B).
           - "HAS_LIMIT": Drug -> Condition (Max dose, quantity limits).
           - "NOT_COVERED_FOR": Drug -> Diagnosis (Negative rule/Exclusion).

        --- FEW-SHOT EXAMPLE (LEARN FROM THIS) ---
        
        **Input Text:**
        "4.2.1.C - Romatoid Artrit:
        (1) Metotreksat tedavisine yanıt alınamayan (DAS28 > 5.1) hastalarda, 3 ay süreli sağlık kurulu raporuna dayanılarak Anti-TNF tedavisine başlanır. Raporu Romatoloji uzmanı düzenler."

        **Output JSON:**
        {{
            "nodes": [
                {{"id": "4.2.1.C", "label": "Madde 4.2.1.C", "type": "RULE"}},
                {{"id": "Romatoid Artrit", "label": "Romatoid Artrit", "type": "DIAGNOSIS"}},
                {{"id": "Metotreksat", "label": "Metotreksat", "type": "DRUG"}},
                {{"id": "Anti-TNF", "label": "Anti-TNF", "type": "DRUG"}},
                {{"id": "DAS28 > 5.1", "label": "DAS28 > 5.1", "type": "CONDITION"}},
                {{"id": "3 Ay Süreli Rapor", "label": "3 Ay Süreli Rapor", "type": "DOCUMENT"}},
                {{"id": "Romatoloji Uzmanı", "label": "Romatoloji Uzmanı", "type": "SPECIALIST"}}
            ],
            "edges": [
                {{"source": "4.2.1.C", "target": "Anti-TNF", "relation": "COVERS"}},
                {{"source": "Anti-TNF", "target": "Romatoid Artrit", "relation": "TREATS"}},
                {{"source": "Anti-TNF", "target": "Metotreksat", "relation": "MUST_FAIL_FIRST"}},
                {{"source": "Anti-TNF", "target": "DAS28 > 5.1", "relation": "REQUIRES_CONDITION"}},
                {{"source": "Anti-TNF", "target": "3 Ay Süreli Rapor", "relation": "REQUIRES_REPORT"}},
                {{"source": "3 Ay Süreli Rapor", "target": "Romatoloji Uzmanı", "relation": "ISSUED_BY"}}
            ]
        }}

        --- REAL TASK ---
        
        **INPUT TEXT (Article {chunk['id']}):**
        {chunk['full_text']}

        **INSTRUCTIONS:**
        1. Extract entities in their original **TURKISH** names.
        2. Ensure the RULE node is always created with ID "{chunk['id']}".
        3. Capture "Step Therapy" logic using "MUST_FAIL_FIRST" if mentioned.
        4. Capture limits (quantity, age) using "HAS_LIMIT".
        5. Return ONLY valid JSON.
        """

    def process_chunk(self, chunk):
        retries = 3
        for attempt in range(retries):
            try:
                prompt = self._get_prompt(chunk)
                response = self.model.generate_content(prompt)
                data = json.loads(response.text)
                return data

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    print(f"⚠️ Quota exceeded on Key #{self.key_manager.current_index + 1}. Rotating...")
                    self.key_manager.rotate_key()
                    self._configure_model()
                    time.sleep(2)
                    continue 
                else:
                    print(f"❌ Non-Quota Error processing {chunk['id']}: {e}")
                    return None
        return None

    def add_data(self, data, chunk_meta):
        if not data: return

        for n in data.get("nodes", []):
            n_id = str(n["id"]).strip().upper()
            
            # Ensure RULE nodes have the text property
            if n["type"] == "RULE" and "text" not in n:
                n["text"] = chunk_meta["full_text"]

            if n_id not in self.nodes:
                self.nodes[n_id] = {
                    "id": n_id,
                    "label": n.get("label", n_id),
                    "type": n.get("type", "UNKNOWN"),
                    "text": n.get("text", "")
                }
            elif n["type"] == "RULE": 
                # Update text if missing on existing node
                if not self.nodes[n_id].get("text") and n.get("text"):
                    self.nodes[n_id]["text"] = n["text"]

        for e in data.get("edges", []):
            self.edges.append({
                "source": str(e["source"]).strip().upper(),
                "target": str(e["target"]).strip().upper(),
                "relation": e["relation"]
            })
        
        # Link to Parent Rule if applicable
        if chunk_meta['parent_id'] != "ROOT":
            self.edges.append({
                "source": str(chunk_meta['parent_id']).strip().upper(),
                "target": str(chunk_meta['id']).strip().upper(),
                "relation": "HAS_SUBRULE"
            })

    def save_state(self):
        output_data = {
            "nodes": list(self.nodes.values()),
            "edges": self.edges
        }
        with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

# --- VISUALIZATION ---
def generate_html_graph():
    if not os.path.exists(JSON_OUTPUT): 
        print("No JSON file found to visualize.")
        return

    print("🎨 Generating HTML Visualization...")
    with open(JSON_OUTPUT, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", filter_menu=True)
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=95)

    color_map = {
        "RULE": "#ff6961", "DRUG": "#77dd77", "DIAGNOSIS": "#fdfd96",
        "SPECIALIST": "#84b6f4", "CONDITION": "#fdcae1", "DEVICE": "#b19cd9", "DOCUMENT": "#ffb347"
    }

    for n in data["nodes"]:
        c = color_map.get(n.get("type"), "#cccccc")
        tooltip = n.get("text", "")[:300] + "..." if len(n.get("text","")) > 0 else n["label"]
        net.add_node(n["id"], label=n["label"], title=tooltip, color=c, group=n.get("type"))

    for e in data["edges"]:
        try:
            net.add_edge(e["source"], e["target"], title=e["relation"])
        except:
            pass 

    net.save_graph(HTML_OUTPUT)
    print(f"✨ Graph saved to {HTML_OUTPUT}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        key_mgr = KeyManager()
    except ValueError as e:
        print(f"❌ Setup Error: {e}")
        exit()
    
    if not os.path.exists(INPUT_DOCX):
        print(f"❌ Error: {INPUT_DOCX} not found.")
        exit()

    chunker = SUTChunker(INPUT_DOCX)
    chunks = chunker.chunk_document()
    print(f"✅ Document split into {len(chunks)} logical chunks.")

    builder = GraphBuilder(key_mgr)
    print("🚀 Starting Extraction...")
    
    for i, chunk in enumerate(tqdm(chunks)):
        result = builder.process_chunk(chunk)
        
        # --- VERBOSE LOGGING ---
        if result and "nodes" in result:
            node_count = len(result["nodes"])
            # print(f"   🔹 Chunk {i}: Found {node_count} nodes.") # Uncomment for detail
        else:
            print(f"   ⚠️ Chunk {i}: No data extracted.")
        # -----------------------

        builder.add_data(result, chunk)
        
        # SAVE EVERY STEP
        builder.save_state()
        
        # Tiny sleep to ensure write operation completes
        time.sleep(0.5) 
            
    builder.save_state()
    print("✅ Extraction Complete.")
    generate_html_graph()