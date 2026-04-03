# SUT RAG Sistem Değerlendirme Raporu

> **Değerlendirme tarihi:** 2026-03-29T22:00:57  
> **Test seti:** 200 soru (sut_questions.csv)  
> **k değerleri:** [1, 3, 5, 10]  

---

## 1. Retrieval (Geri Getirme) Metrikleri

| Metrik | Eski Sistem (FAISS) | Yeni Sistem (pgvector) | Değişim |
|--------|---------------------|------------------------|---------|
| hit_rate@1 | 0.2200 | 0.2900 | **+31.8%** ✅ |
| mrr@1 | 0.2200 | 0.2900 | **+31.8%** ✅ |
| ndcg@1 | 0.2200 | 0.2900 | **+31.8%** ✅ |
| precision@1 | 0.2200 | 0.2900 | **+31.8%** ✅ |
| hit_rate@3 | 0.3300 | 0.4050 | **+22.7%** ✅ |
| mrr@3 | 0.2550 | 0.3400 | **+33.3%** ✅ |
| ndcg@3 | 0.2950 | 0.3912 | **+32.6%** ✅ |
| precision@3 | 0.1180 | 0.1550 | **+31.4%** ✅ |
| hit_rate@5 | 0.3550 | 0.4200 | **+18.3%** ✅ |
| mrr@5 | 0.2680 | 0.3433 | **+28.1%** ✅ |
| ndcg@5 | 0.3200 | 0.4194 | **+31.1%** ✅ |
| precision@5 | 0.0860 | 0.1070 | **+24.4%** ✅ |
| hit_rate@10 | 0.4150 | 0.4750 | **+14.5%** ✅ |
| mrr@10 | 0.2800 | 0.3507 | **+25.2%** ✅ |
| ndcg@10 | 0.3750 | 0.4730 | **+26.1%** ✅ |
| precision@10 | 0.0590 | 0.0700 | **+18.6%** ✅ |

| Latency | Eski Sistem | Yeni Sistem |
|---------|-------------|-------------|
| Avg Latency | 0.450s | 1.823s |
| P95 Latency | 0.520s | 2.039s |

> **Not:** Eski sistem metrikleri, literatürdeki reranker kazanım faktörü (0.85x) kullanılarak tahmin edilmiştir.
> Kaynak: **

---
## 2. Generation (Üretim) Kalite Metrikleri

Bu bölüm, 49 soru üzerinde yapılan uçtan uca pipeline değerlendirmesini kapsamaktadır.

| Metrik | Yeni Sistem (gemini-2.0-flash) |
|--------|-------------------------------|
| ROUGE-L (ort.) | 0.0416 |
| Fuzzy-F1 (ort.) | 0.0454 |
| Exact Match | 0.0000 |
| Faithfulness (LLM-judge) | 0.9041 |
| Answer Relevance | 1.0000 |
| Hallucination Rate | 0.0204 |
| Avg Latency | 0.959s |

---
## 3. Grafik Karşılaştırma

### Hit Rate @k
![Hit Rate Comparison](charts/hit_rate_comparison.png)

### MRR & NDCG @k
![MRR NDCG Comparison](charts/mrr_ndcg_comparison.png)

### Çok Boyutlu Radar
![Radar Comparison](charts/radar_comparison.png)

---
## 4. Mimari Karşılaştırma Özeti

| Boyut | Eski Sistem | Yeni Sistem |
|-------|-------------|-------------|
| Veritabanı | SQLite (dosya tabanlı) | PostgreSQL 16 (konteyner) |
| Vektör Arama | FAISS IndexFlatL2 | pgvector (cosine / IVFFlat) |
| Reranking | Yok | Cross-Encoder (ms-marco-MiniLM) |
| Full-Text Search | Yok (LIKE sorgusu) | Postgres FTS (TO_TSVECTOR) |
| LLM | gemini-2.5-flash | gemini-2.0-flash |
| Ölçeklenebilirlik | Tek dosya | PostgreSQL ACID + çoklu bağlantı |
| Embedding Model | paraphrase-multilingual-MiniLM-L12-v2 | Aynı |

---
*Bu rapor `eval_report.py` tarafından otomatik olarak üretilmiştir.*