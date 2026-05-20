---
title: MAHIKS TR — Turkish Health Policy Agent
emoji: 🏥
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Multi-tenant Agentic RAG for the Turkish SUT health-policy corpus.
---

[![CI](https://github.com/suleymanbrbr/su_ens4912_corporate_health/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/suleymanbrbr/su_ens4912_corporate_health/actions/workflows/ci.yml)
[![Frontend build](https://github.com/suleymanbrbr/su_ens4912_corporate_health/actions/workflows/frontend-build.yml/badge.svg)](https://github.com/suleymanbrbr/su_ens4912_corporate_health/actions/workflows/frontend-build.yml)

# MAHIKS-TR

**M**ulti-tenant **A**gentic **H**ealth **I**nformation **K**nowledge **S**ystem
for the **T**urkish **R**epublic's *Sağlık Uygulama Tebliği* (SUT) corpus.

MAHIKS-TR is a production-grade Retrieval-Augmented Generation system that lets
citizens, physicians, and administrators query the 700+ page Turkish SUT
health-reimbursement policy in natural Turkish. Each tenant brings their own
LLM API key (Gemini, OpenRouter, or local) — the platform itself is free to
operate on the Hugging Face / Neon / Vercel free tiers.

The reasoning core is an **agentic RAG** engine that combines `pgvector`
semantic search, PostgreSQL `tsvector('turkish')` BM25, and a
schema-extracted Knowledge Graph (drugs ↔ diagnoses ↔ rules ↔ specialists)
behind a ReAct tool-use loop. See [`TECHNICAL_OVERVIEW.md`](./TECHNICAL_OVERVIEW.md)
for the deep dive.

---

## Architecture

```
                          ┌────────────────────────────┐
                          │ Frontend  ·  Vercel (Hobby)│
                          │  React + Vite (SSE chat)   │
                          └──────────────┬─────────────┘
                                         │  HTTPS / JWT
                                         ▼
              ┌──────────────────────────────────────────────────┐
              │  Backend  ·  Hugging Face Spaces (Docker, CPU)   │
              │  FastAPI  ·  uvicorn  ·  port 7860               │
              │  ┌─────────────────────────────────────────────┐ │
              │  │ Agentic RAG (ReAct)                         │ │
              │  │  ├─ search_sut_chunks   (pgvector cosine)   │ │
              │  │  ├─ search_sut_fulltext (tsvector Turkish)  │ │
              │  │  ├─ lookup_kg_entity / explore_kg_path      │ │
              │  │  ├─ read_user_report / calculate            │ │
              │  │  └─ Critic loop (citation audit)            │ │
              │  └─────────────────────────────────────────────┘ │
              │  Per-user API keys (Fernet-encrypted at rest)    │
              └────────────────┬─────────────────┬───────────────┘
                               │                 │
                  pgvector / SQL                  │ pulls user's
                               │                 │ Gemini/OpenRouter key
                               ▼                 ▼
                ┌────────────────────────┐  ┌────────────────────┐
                │  Neon Postgres 16      │  │  Gemini · OpenRouter│
                │  (3 GB free, eu-c-1)   │  │  (BYO key per user) │
                │  + pgvector extension  │  └────────────────────┘
                └────────────────────────┘
```

---

## Local development quickstart

```bash
# 1. Clone
git clone https://github.com/<you>/su_ens4912_corporate_health.git
cd su_ens4912_corporate_health

# 2. Bring up Postgres+pgvector, backend, and frontend
cp backend/.env.example backend/.env
# edit backend/.env: set JWT_SECRET_KEY and (for local dev only) optionally
# a server-side GEMINI_API_KEY. In production each user supplies their own.
docker compose up --build

# 3. Open the UI
open http://localhost:5173
```

The first signed-up user becomes the admin (or run
`UPDATE users SET role='admin', is_approved=TRUE WHERE username='you';`
in the local Postgres).

---

## Production deployment

A complete, copy-pasteable walk-through is in [`DEPLOYMENT.md`](./DEPLOYMENT.md).
Summary of the stack we target:

| Layer       | Host                      | Free-tier limit                         |
|-------------|---------------------------|-----------------------------------------|
| Frontend    | Vercel (Hobby)            | 100 GB bandwidth / unlimited builds     |
| Backend     | Hugging Face Spaces       | CPU basic, 16 GB RAM, sleeps after 48 h |
| Database    | Neon Postgres 16          | 3 GB storage, pgvector enabled          |
| Keep-alive  | cron-job.org              | 1 free job, 6-hourly ping               |

---

## Tech stack

- **Language model orchestration**: LangChain 0.3.x, ReAct agent
- **LLM providers**: Google Gemini 2.0 Flash, OpenRouter (Qwen/Llama), local
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Vector store**: PostgreSQL 16 + `pgvector` (ivfflat)
- **Full-text search**: PostgreSQL `tsvector('turkish')` (BM25)
- **API**: FastAPI 0.111 + uvicorn, JWT auth, Fernet-encrypted user keys
- **Frontend**: React 19 + Vite 8 + react-force-graph-2d + react-markdown
- **Document ingestion**: pypandoc · python-docx · pypdf

---

## Repository layout

```
.
├── Dockerfile                   ← HF Spaces production image (this dir)
├── README.md                    ← you are here (HF Space metadata header)
├── DEPLOYMENT.md                ← step-by-step prod walk-through
├── TECHNICAL_OVERVIEW.md        ← architectural deep-dive
├── docker-compose.yml           ← local dev stack
├── migrations/                  ← one-off SQL for Neon
├── backend/                     ← FastAPI app, agent, RAG core, eval scripts
├── frontend/                    ← Vite/React UI
├── data/                        ← SUT corpus (.docx); pgdata is .gitignored
└── tests/                       ← unit / integration / e2e / stress
```

---

## Acknowledgments

This project was produced as the ENS4912 capstone deliverable at **Yeditepe
University, Department of Industrial & Systems Engineering**, advised by the
project supervisor and graded jurors. Special thanks to the open-source
maintainers of LangChain, sentence-transformers, pgvector, and HF Spaces.

## License

MIT. See the SPDX header in source files; data corpus (Turkish SUT) is a
public Turkish Government publication and is included for academic
demonstration only.
