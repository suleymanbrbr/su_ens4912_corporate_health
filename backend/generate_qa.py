import os
import csv
import json
import random
import psycopg2
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# We need a new file to save the questions
OUTPUT_CSV = "sut_questions_v2.csv"
NUM_QUESTIONS_NEEDED = 100

def get_random_chunks_from_db(limit=50):
    """Fetch random chunks that have enough text to generate good questions."""
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cur = conn.cursor()
    # Get chunks that are decently long but not massive
    cur.execute("""
        SELECT chunk_id, header_text, text_content 
        FROM chunks 
        WHERE LENGTH(text_content) > 150 AND LENGTH(text_content) < 1500
        ORDER BY RANDOM() 
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def generate_qa_from_chunk(llm, chunk_data):
    """Uses LLM to generate 2-3 questions from a single chunk."""
    chunk_id, header_text, text_content = chunk_data
    
    # Try to extract a section number (like 1.8.1) from the header
    import re
    m = re.search(r'(\d+\.\d+(\.\d+[A-Z-\d]*)?)', str(header_text))
    kaynak = m.group(1) if m else "Bilinmiyor"
    
    prompt = f"""
Sen uzman bir Sağlık Uygulama Tebliği (SUT) veri seti oluşturucususun.
Aşağıda veritabanımızdan alınmış GERÇEK bir SUT metin parçası (chunk) ve başlığı bulunuyor.

Başlık: {header_text}
Metin: {text_content}

Görev: Bu metne dayanarak Türkçe 2 adet Soru-Cevap çifti oluştur. Biri "Kolay/Doğrudan", diğeri "Zor/Dolaylı" olsun.
Kurallar:
1. Soru ve Cevap SADECE bu metne dayanmalıdır. Metinde olmayan bir bilgiyi uydurma.
2. Cevap kısa, net ve doğrudan olmalıdır (örn: "4 yıldır.", "Evet, alınır.", "%30'dur.").
3. Eğer metin soru çıkarmak için anlamsız veya çok kısaysa, boş liste döndür.

Çıktıyı SADECE aşağıdaki JSON formatında ver, başka hiçbir metin ekleme:
[
  {{"Kategori": "Zorluk Derecesi (Kolay/Zor)", "Soru": "...", "Cevap": "..."}},
  {{"Kategori": "Zorluk Derecesi (Kolay/Zor)", "Soru": "...", "Cevap": "..."}}
]
"""
    try:
        resp = llm.invoke([HumanMessage(content=prompt)]).content
        # Extract JSON
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', resp, re.DOTALL)
        if json_match:
            qa_list = json.loads(json_match.group(0))
            # Add Kaynak
            for qa in qa_list:
                qa['Kaynak'] = kaynak
            return qa_list
    except Exception as e:
        print(f"Hata: {e}")
    return []

def main():
    print("Veritabanından rastgele metinler çekiliyor...")
    chunks = get_random_chunks_from_db(limit=80) # Fetch 80 chunks, assuming ~1.5 questions per chunk
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    
    all_qa = []
    
    print("Soru-Cevap üretimi başlıyor (Paralel İşlem)...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_chunk = {executor.submit(generate_qa_from_chunk, llm, chunk): chunk for chunk in chunks}
        
        for future in as_completed(future_to_chunk):
            qa_list = future.result()
            if qa_list:
                all_qa.extend(qa_list)
                print(f"Toplam üretilen soru: {len(all_qa)}/{NUM_QUESTIONS_NEEDED}")
            
            if len(all_qa) >= NUM_QUESTIONS_NEEDED:
                break
                
    # Trim to exactly NUM_QUESTIONS_NEEDED
    all_qa = all_qa[:NUM_QUESTIONS_NEEDED]
    
    print(f"\n{len(all_qa)} adet soru üretildi. CSV dosyasına kaydediliyor...")
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Kategori', 'Soru', 'Cevap', 'Kaynak'])
        for i, qa in enumerate(all_qa, start=1):
            kategori = qa.get('Kategori', 'Genel').replace(',', '')
            soru = qa.get('Soru', '').replace('\n', ' ')
            cevap = qa.get('Cevap', '').replace('\n', ' ')
            kaynak = qa.get('Kaynak', 'Bilinmiyor')
            writer.writerow([i, kategori, soru, cevap, kaynak])
            
    print(f"✅ Başarılı! Sorular {OUTPUT_CSV} dosyasına kaydedildi.")

if __name__ == "__main__":
    main()
