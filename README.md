---
title: MAHIKS TR — Turkish Health Policy Agent
emoji: 🏥
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Agentic RAG for the Turkish SUT health corpus.
---

![CI](https://github.com/suleymanbrbr/su_ens4912_corporate_health/actions/workflows/ci.yml/badge.svg)
![Frontend build](https://github.com/suleymanbrbr/su_ens4912_corporate_health/actions/workflows/frontend-build.yml/badge.svg)

# 🏥 MAHIKS-TR

**M**ulti-tenant **A**gentic **H**ealth **I**nformation **K**nowledge **S**ystem
for the **T**urkish **R**epublic's *Sağlık Uygulama Tebliği* (SUT) corpus.

> A production-grade Retrieval-Augmented Generation (RAG) system that lets
> citizens, physicians, pharmacists, hospital managers, and SGK auditors query
> the 700+ page Turkish SUT health-reimbursement policy in natural Turkish.
> Each tenant brings their own LLM API key — the platform itself runs entirely
> on free-tier cloud services.

Bu proje **Yeditepe Üniversitesi Endüstri ve Sistem Mühendisliği Bölümü
ENS4912 Bitirme Projesi** kapsamında geliştirilmiştir.

---

## 🚀 Live Demo

| Layer | URL | Status |
|---|---|---|
| 🌐 **Frontend** | https://mahiks-tr.vercel.app | ✅ Live |
| 🤗 **Backend API** | https://suleyman42-mahiks-tr.hf.space | ✅ Running |
| 📚 **API Docs (Swagger)** | https://suleyman42-mahiks-tr.hf.space/docs | ✅ Live |
| 🗄️ **Database** | Neon Postgres 16 (Frankfurt) + pgvector | ✅ 1079 chunks indexed |

**Demo credentials** (admin): `suleyman` / `suleyman1`

> 💡 The first time you log in you'll see a 3-step QuickTour. Then go to
> **Settings → 🔑 LLM API Keys** and paste your own Gemini key from
> [Google AI Studio](https://aistudio.google.com/apikey) (free tier:
> 15 RPM / 1500 RPD). Click "Bağlantıyı Test Et" → "Kaydet" → ask anything.

---

## ✨ Features

### 🤖 Agentic ReAct RAG Engine
Six dynamic tools wrapped in a critic-verified loop:

| Tool | Purpose |
|---|---|
| `search_sut_chunks` | pgvector cosine + Cross-Encoder reranker (semantic) |
| `search_sut_fulltext` | PostgreSQL `tsvector('turkish')` BM25 (exact-code lookup) |
| `lookup_kg_entity` | Knowledge Graph node lookup (drugs ↔ diagnoses ↔ rules) |
| `explore_kg_path` | BFS multi-hop relational reasoning (max 3 hops) |
| `calculate` | Sandboxed AST-based numeric evaluator (dosage, age limits) |
| `finish` | Citation-verifying critic loop (mandatory `[Madde X.X.X]` refs) |

### 👥 Multi-Tenant SaaS
- **Per-user API keys** encrypted at rest with Fernet (AES-128-CBC + HMAC)
- **Provider selector** — Gemini, OpenRouter, or local LM Studio
- **JWT auth** (24h, bcrypt-hashed passwords, role-based admin gating)
- **Audit logging** on every mutation (login, key change, KG rebuild, ...)

### 🎨 Modern Frontend
- React 19 + Vite 8 + code-split lazy chunks (initial bundle: 163 kB gz)
- **Server-Sent Events** streaming chat with blinking cursor
- **5 role personas** (Patient, Doctor, Pharmacist, Hospital Manager, Admin)
- **Interactive Knowledge Graph** viewer (react-force-graph-2d)
- **Skeleton loaders, ErrorBoundary, QuickTour**, dark mode, command palette
- **Edit + Regenerate** message actions, conversation history, PDF upload
- **Admin Panel** with system metrics, audit log viewer, KG rebuild trigger

### 🛡️ Production Hardening
- CORS env-driven (no wildcards)
- SQL injection-free (parameterised queries everywhere)
- AST-whitelisted calculator (no `eval()`)
- Password policy + bcrypt 72-byte truncation fix
- Centralised DB connection pooling via `db_session()` context manager
- PDF upload size limit (10 MB) + MIME validation
- Structured logging with `LOG_LEVEL` env var
- 57 unit tests passing in CI

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Browser  (React 19 + Vite SPA)                 │
│                  https://mahiks-tr.vercel.app                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTPS · JWT · SSE
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Vercel Edge          (rewrites /api/* → HF Space)              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI Backend  ·  Hugging Face Spaces (Docker, CPU-basic)    │
│  uvicorn · port 7860 · uid 1000 · 16 GB RAM                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Agentic RAG Engine  (ReAct Loop, max 8 iterations)      │  │
│  │  ├─ search_sut_chunks   (pgvector cosine + reranker)     │  │
│  │  ├─ search_sut_fulltext (Postgres tsvector Turkish)      │  │
│  │  ├─ lookup_kg_entity / explore_kg_path                   │  │
│  │  ├─ calculate           (sandboxed AST)                  │  │
│  │  └─ Critic loop         (citation audit)                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Routers (modular): auth · user_keys · chat · admin · kg ·      │
│                     policy · health                              │
│  Per-user API keys: Fernet-encrypted at rest, decrypted per call│
└────────────────┬─────────────────────┬──────────────────────────┘
                 │                     │
   pgvector / SQL                      │ pulls user's
                 │                     │ Gemini/OpenRouter key
                 ▼                     ▼
┌────────────────────────┐  ┌────────────────────────────────────┐
│  Neon Postgres 16      │  │  Gemini · OpenRouter · LM Studio   │
│  Frankfurt · 0.5 GB    │  │  (BYO key per user, never shared)  │
│  + pgvector ivfflat    │  └────────────────────────────────────┘
│  + tsvector Turkish    │
│  1079 chunks indexed   │
│  10 KG node types      │
└────────────────────────┘
```

---

## 📊 Empirical Results

Comprehensive evaluation across 200 SUT ground-truth questions and 8 LLMs in
both Baseline RAG (`no_kg`) and GraphRAG (`with_kg`) modalities. Full LaTeX
report in [`Final_Report_ENS4912.tex`](./Final_Report_ENS4912.tex).

### Architectural evolution

| Phase | Architecture | Hit@1 | Hit@5 | Faithfulness |
|---|---|---|---|---|
| Phase 1 | Legacy FAISS + MiniLM + SQLite | 19.5% | 41.0% | 0.85 (est.) |
| Phase 2 | pgvector + Postgres + MiniLM | 23.0% | 42.0% | 0.75 |
| Phase 3 | + Cross-Encoder reranker | 29.0% | 42.0% | 0.87 |
| Phase 4 | **+ multilingual-e5-large + mMarco** | **49.0%** | **68.5%** | **0.89** |

**🎯 +151% improvement in Hit Rate@1 — strict citation enforcement
elevated Faithfulness from 0.75 to 0.89.**

### Multi-model LLM benchmark (100 questions, RAG end-to-end)

| Model | Mode | MAP | NDCG | Faithfulness | Latency |
|---|---|---|---|---|---|
| **gemini-2.5-pro** | with_kg | 0.721 | 0.797 | 0.505 | 89.9 s |
| gemini-3.1-pro-preview | no_kg | **0.762** | **0.833** | 0.495 | 98.0 s |
| gemini-2.5-flash | no_kg | 0.746 | 0.833 | 0.405 | **58.5 s** |
| qwen/qwen3.5-9b (local) | no_kg | 0.738 | 0.830 | 0.460 | 514 s |
| llama-3-8b-it (local) | no_kg | 0.568 | 0.610 | 0.260 | 182 s |
| gemma-3-12b-it (local) | no_kg | 0.549 | 0.639 | 0.260 | 14.1 s |

### Component ablations

| Ablation | Baseline | Optimized | Δ |
|---|---|---|---|
| Embedding (MiniLM → E5-Large) | Hit@1 0.29 | 0.49 | +69% |
| Reranker (off → Cross-Encoder) | Hit@1 0.23 | 0.29 | +26% |
| Reranker (off → mMarco-multilingual) | MRR@5 0.343 | 0.438 | +28% |
| Chunking (6-level → 4-level) | Hit@1 0.20 | 0.29 | +45% |
| Prompt (default → strict citations) | Faithfulness 0.75 | 0.89 | +19% |

---

## 🛠️ Tech Stack

### Backend
- **Language model orchestration**: LangChain 0.3.x, ReAct agent
- **LLM providers**: Google Gemini 2.5 Flash Lite (default), OpenRouter (Qwen/Llama/Gemma), local LM Studio
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Vector store**: PostgreSQL 16 + `pgvector` (ivfflat cosine)
- **Full-text search**: PostgreSQL `tsvector('turkish')` (BM25)
- **Web framework**: FastAPI 0.111 + uvicorn
- **Auth**: JWT (python-jose) + bcrypt + Fernet for user-key encryption
- **Document ingestion**: pypandoc · python-docx · pypdf

### Frontend
- **React 19.2.4** + Vite 8.0
- **React Router 7**, code-split lazy routes
- **lucide-react** icons, **react-hot-toast** notifications
- **react-markdown** + remark-gfm for streaming markdown
- **react-force-graph-2d** for KG visualisation
- **cmdk** command palette
- Custom design system (CSS vars, dark mode, skeleton shimmer)

### Infrastructure
- **Database**: [Neon Postgres](https://neon.tech) (Frankfurt, free 0.5 GB + pgvector)
- **Backend host**: [Hugging Face Spaces](https://huggingface.co) (Docker SDK, 16 GB free)
- **Frontend host**: [Vercel](https://vercel.com) (Hobby plan, unlimited bandwidth)
- **CI/CD**: GitHub Actions (`pytest unit/` on push + `npm run build` on PR)
- **Dependency updates**: Dependabot (weekly, grouped minors)

**Total monthly cost: ₺0** (forever).

---

## 🧪 Testing

| Layer | Tooling | Scope |
|---|---|---|
| **Unit** | pytest, mocks | 94 tests (RAG engine, embedding, auth, API models) |
| **Integration** | pytest + live Docker stack | 106 tests (chat, auth, admin, KG, policy) |
| **RAG Quality** | pytest -m slow + LLM-judge | 10 curated SUT QA + Faithfulness scoring |
| **End-to-End** | Playwright + TypeScript | smoke, auth, chat user journeys |
| **Stress** | Locust | 50 concurrent users, chat + policy + history |

Run everything:
```bash
./tests/run_all_tests.sh --all
```

CI runs the unit tests on every push to `main` and every pull request
([ci.yml](.github/workflows/ci.yml)).

---

## 💻 Local Development Quickstart

### Prerequisites
- Docker + Docker Compose
- Node.js 20+
- Python 3.11 (only needed if you skip Docker)

### One-command setup

```bash
# 1. Clone
git clone https://github.com/suleymanbrbr/su_ens4912_corporate_health.git
cd su_ens4912_corporate_health

# 2. Generate secrets
python3 -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(64))"     >  backend/.env
python3 -c "from cryptography.fernet import Fernet; print('API_KEY_ENCRYPTION_KEY=' + Fernet.generate_key().decode())" >> backend/.env
echo "DATABASE_URL=postgresql://admin:secretpassword@db:5432/sut_knowledge_base"      >> backend/.env
echo "CORS_ORIGINS=http://localhost:5173"                                              >> backend/.env

# 3. Bring up Postgres+pgvector, backend, and frontend
docker compose up --build

# 4. Open the UI
open http://localhost:5173
```

The first user that signs up is **NOT** admin. Promote yourself via SQL:

```sql
UPDATE users SET role='admin', is_approved=TRUE WHERE username='you';
```

Then from Admin Panel:
1. **Rebuild Index** — chunks the SUT corpus into pgvector (~5–10 min)
2. **Rebuild KG** — extracts entities + relations via Gemini (~10–15 min, needs your own Gemini key in Settings → API Keys)

---

## 🚢 Production Deployment

A complete, copy-pasteable walk-through is in [`DEPLOYMENT.md`](./DEPLOYMENT.md).
Summary:

| Layer | Host | Free-tier limit | Setup time |
|---|---|---|---|
| Database | [Neon](https://neon.tech) | 0.5 GB + pgvector, auto-suspend | 5 min |
| Backend | [Hugging Face Spaces](https://huggingface.co) | 16 GB RAM, sleeps after 48 h | 15 min |
| Frontend | [Vercel](https://vercel.com) | 100 GB BW / month | 5 min |
| Domain + SSL | Vercel auto + HF Space auto | unlimited | 0 min |
| Keep-alive | [cron-job.org](https://cron-job.org) | 1 free job, 6-hourly ping | 2 min |

### Auto-deploy via GitHub Actions
Two workflows are pre-wired but secret-gated:
- [`.github/workflows/deploy.yml.disabled`](.github/workflows/deploy.yml.disabled) — Vercel + HF Space auto-deploy on push to `main`
- Add `VERCEL_TOKEN`, `HF_TOKEN`, etc. to GitHub repo secrets, rename to `deploy.yml`, push.

---

## 📡 API Overview

Full OpenAPI docs at [`/docs`](https://suleyman42-mahiks-tr.hf.space/docs).

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | none | Liveness + DB ping |
| `POST` | `/api/auth/register` | none | Create user (pending admin approval) |
| `POST` | `/api/auth/login` | none | Form-urlencoded → JWT |
| `GET` | `/api/auth/me` | user | Current user info |
| `POST` | `/api/user/api-keys` | user | Save encrypted LLM API key |
| `GET` | `/api/user/api-keys` | user | List user's keys (hint only, never plaintext) |
| `DELETE` | `/api/user/api-keys/{provider}` | user | Revoke a key |
| `POST` | `/api/user/api-keys/test` | user | Validate a key against the provider |
| `POST` | `/api/chat/query` | user | SSE-streamed agentic RAG response |
| `POST` | `/api/chat/upload` | user | Upload PDF to conversation context |
| `GET` | `/api/conversations` | user | List user's conversations |
| `GET` | `/api/history` | user | Paginated chat history |
| `GET` | `/api/policies` | user | Full-text SUT policy search |
| `GET` | `/api/kg/stats` | user | KG node/edge counts by type |
| `GET` | `/api/kg/nodes` | user | KG entity search (label + semantic) |
| `GET` | `/api/kg/path` | user | KG BFS path between two nodes |
| `POST` | `/api/admin/rebuild-index` | admin | Re-chunk + re-embed SUT corpus |
| `POST` | `/api/admin/kg/rebuild` | admin | Re-extract KG from chunks |
| `GET` | `/api/admin/audit-logs` | admin | Paginated audit trail |
| `GET` | `/api/admin/system` | admin | System metrics |
| `GET` | `/api/admin/analytics` | admin | Top keywords + engagement |

---

## 📁 Repository Layout

```
.
├── README.md                           ← you are here (HF Space metadata)
├── DEPLOYMENT.md                       ← step-by-step prod walk-through
├── TECHNICAL_OVERVIEW.md               ← architectural deep-dive
├── Final_Report_ENS4912.tex            ← 913-line LaTeX bitirme raporu
├── Dockerfile                          ← HF Spaces production image
├── docker-compose.yml                  ← local dev stack
├── migrations/                         ← one-off SQL for Neon
│   └── 001_user_api_keys.sql           (pgvector + user_api_keys schema)
├── backend/                            ← FastAPI app
│   ├── api_server.py                   (FastAPI gateway, 334 LOC)
│   ├── routers/                        (modular routers — auth/admin/chat/kg/policy/user_keys/health)
│   ├── sut_rag_core.py                 (Agentic ReAct engine, 5-tool loop)
│   ├── kg_builder.py / kg_storage.py   (Knowledge Graph build + query)
│   ├── rag_storage.py                  (chunk ingestion + embedding)
│   ├── secrets_utils.py                (Fernet encryption for user keys)
│   ├── deps.py                         (shared DB + auth dependencies)
│   ├── auth_utils.py                   (JWT + bcrypt)
│   ├── embedding_utils.py              (auto-detect MiniLM vs E5)
│   ├── eval_*.py                       (evaluation pipelines, 8 scripts)
│   └── eval_results/                   (JSON metrics, charts, LaTeX report)
├── frontend/                           ← Vite/React UI
│   └── src/components/                 (14 components — Chat, Admin, KG, Settings, ApiKeyManager, ...)
├── data/                               ← SUT corpus (.docx via Git LFS)
├── tests/
│   ├── unit/                           (94 tests — pytest + mocks)
│   ├── integration/                    (106 tests — live Docker)
│   ├── rag_quality/                    (LLM-judged RAG quality)
│   ├── e2e/                            (Playwright smoke + auth + chat)
│   └── stress/                         (Locust 50-user scenarios)
└── .github/
    ├── workflows/
    │   ├── ci.yml                      (pytest unit on push/PR)
    │   ├── frontend-build.yml          (npm run build + artifact)
    │   └── deploy.yml.disabled         (rename to enable Vercel + HF auto-deploy)
    └── dependabot.yml                  (weekly grouped pip + npm updates)
```

---

## 🔐 Security & Privacy

- **API keys**: Fernet (AES-128-CBC + HMAC) encrypted at rest in `user_api_keys`
- **Passwords**: bcrypt with 12 rounds + 72-byte safe truncation
- **JWT**: HS256, 24h expiry, secret loaded from env at startup (panics if missing)
- **No PII in logs**: query text never logged at INFO level
- **Audit trail**: every mutation (login, key change, KG rebuild, ...) logged to `audit_logs`
- **CORS**: env-driven whitelist (`CORS_ORIGINS`), no wildcards
- **SQL**: 100% parameterised queries, no string interpolation
- **PDF uploads**: 10 MB cap + MIME validation, scoped per user/conversation
- **Calculator tool**: AST-whitelisted (no `eval()`), only numeric ops

KVKK (Türk Veri Koruma) uyumlu:
- Veriler AB sınırları içinde (Neon Frankfurt)
- Kullanıcı kendi verilerini istediğinde silebilir (cascade delete)
- API anahtarları sadece ilgili kullanıcı için decrypt edilir, hiç loglanmaz

---

## 🤝 Contributing

Bu proje şu an açık katkıya kapalıdır (bitirme projesi teslim dönemi). Demo
veya proje hakkında sorularınız için issue açabilirsiniz.

---

## 🙏 Acknowledgments

Bu proje **Yeditepe Üniversitesi Endüstri ve Sistem Mühendisliği Bölümü
ENS 491-492 Bitirme Projesi** olarak hazırlanmıştır.

- **Proje Süpervizörü**: İnanç Arın
- **Proje Üyeleri**: Süleyman Berber, Hüseyin Doğan Türk

Open-source ekosistemine teşekkürler: LangChain, sentence-transformers,
pgvector, Hugging Face Spaces, FastAPI, React, Vite, Vercel, Neon.

---

## 📚 References

Anahtar akademik kaynaklar (tüm referans listesi
[`Final_Report_ENS4912.tex`](./Final_Report_ENS4912.tex)'te):

1. Yao, S., Zhao, J., et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023.
2. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 33.
3. Bonifacio, L., et al. (2021). *mMARCO: A Multilingual Version of the MS MARCO Passage Ranking Dataset*.
4. Xiao, S., Jiang, Z., et al. (2024). *C-Pack: Packaged Resources to Advance General Chinese Embedding*. (E5 foundations)
5. Rezaei, M. R., et al. (2025). *Agentic Medical Knowledge Graphs Enhance Medical Question Answering*. arXiv:2502.13010.
6. Patterson, D., Gonzalez, J., et al. (2021). *Carbon Emissions and Large Neural Network Training*. arXiv:2104.10350.

---

## 📄 License

**MIT License.** See [`LICENSE`](./LICENSE) for details.

Data corpus (Turkish SUT) is a public Türkiye Cumhuriyeti Sosyal Güvenlik Kurumu
publication and is included for academic / non-commercial demonstration only.

---

<p align="center">
  <sub>Generated as part of ENS4912 graduation deliverable · May 2026</sub><br>
  <sub>Made with ❤️ in İstanbul</sub>
</p>
