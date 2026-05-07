# SUT ASİSTANI: ABLASYON TEST RAPORU
## Yapılan Deneyler ve Karşılaştırmalı Sonuçlar

Bu rapor, SUT RAG sistemi üzerinde yapılan 5 farklı ablasyon (bileşen bazlı test) çalışmasının sonuçlarını karşılaştırmalı olarak sunar.

---

### TEST 1: Embedding Model Karşılaştırması
Sistemin temel anlama kabiliyetini ölçmek için kullanılan embedding modelleri test edilmiştir.

| Model Parametresi | MiniLM-L12 (Baseline) | **Multilingual-E5-Large** | Değişim |
|---|---|---|---|
| Hit Rate@1 | 0.2900 | **0.4900** | **+%69.0** |
| Hit Rate@5 | 0.4200 | **0.6850** | **+%63.1** |
| MRR@5 | 0.3433 | **0.5676** | **+%65.3** |
| Ortalama Gecikme | 1.823s | **0.181s** | -90.1% |

**Sonuç**: E5-Large modeli hem doğruluk hem de hız açısından MiniLM'i her metrikte geride bırakmıştır.

---

### TEST 2: Reranker Model Karşılaştırması
Aramada dönen adayların önceliklendirilmesi (ranking) kalitesi ölçülmüştür.

| Reranker Modeli | L6-MiniLM (Mevcut) | **mMarco-Multilingual** | Değişim |
|---|---|---|---|
| Hit Rate@1 | 0.2900 | **0.3950** | **+%36.2** |
| Hit Rate@5 | 0.4200 | **0.5100** | **+%21.4** |
| MRR@5 | 0.3433 | **0.4382** | **+%27.6** |

**Sonuç**: Türkçe dil desteği olan mMarco modeli, İngilizce odaklı L6 modeline göre anlamlı bir doğruluk artışı sağlamıştır.

---

### TEST 3: Chunking (Veri Bölme) Stratejisi
Verinin ne kadar granüler bölündüğünün sistem başarısına etkisi ölçülmüştür.

| Strateji | 4-Level (Hiyerarşik) | 6-Level (FAISS Stili) | Değişim |
|---|---|---|---|
| Hit Rate@1 | **0.2900** | 0.2000 | -31.0% |
| Hit Rate@5 | **0.4200** | 0.3200 | -23.8% |
| MRR@5 | **0.3433** | 0.2525 | -26.4% |

**Sonuç**: 4-seviyeli hiyerarşik bölme, çok derin split'lere göre bağlamın (context) daha iyi korunmasını sağlamıştır.

---

### TEST 4: HyDE (Hypothetical Document Embeddings)
Soru yerine üretilen hipotez cevabın aranması yöntemi test edilmiştir.

| Metot | Direct Embedding | **HyDE (Hypothetical)** | Değişim |
|---|---|---|---|
| Hit Rate@1 | 0.3700 | **0.3900** | **+%5.4** |
| Hit Rate@10 | **0.5600** | 0.5500 | -1.8% |
| Gecikme | **2.474s** | 4.106s | +66.0% |

**Sonuç**: HyDE yöntemi kısıtlı bir kazanç sağlamasına karşın arama süresini ciddi oranda artırmıştır.

---

### TEST 5: Prompt Mühendisliği ve Güvenilirlik
Agentic RAG sisteminin yanıt sadakati (faithfulness) LLM-as-a-judge yöntemiyle ölçülmüştür.

| Metot | Mevcut Prompt | **Optimize Edilmiş Prompt** | Değişim |
|---|---|---|---|
| Faithfulness (Sadakat) | 0.75 | **0.89** | **+%18.7** |
| Kaynak Gösterme (Citation) | Tutarsız | **Zorunlu ve Doğru** | - |

**Sonuç**: Sisteme eklenen format kuralları, LLM'in uydurmada (hallucination) bulunma riskini düşürerek daha güvenilir cevaplar üretmesini sağlamıştır.
