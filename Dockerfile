# ──────────────────────────────────────────────────────────────────────────────
# MAHIKS-TR — Production Dockerfile for Hugging Face Spaces (SDK: docker)
#
# This image is built and run by HF Spaces on the free CPU-basic tier.
# Key requirements satisfied here:
#   • Non-root user `user` with UID 1000 (HF Spaces requirement).
#   • Listens on port 7860 (HF Spaces standard).
#   • Models pre-warmed at build time so first request is fast.
#   • Persistent data lives under /data (HF Spaces persistent storage).
#
# Local development still uses backend/Dockerfile via docker-compose.yml.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Environment ──────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/data/hf_cache \
    TRANSFORMERS_OFFLINE=0 \
    SUT_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        libgomp1 \
        pandoc \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (HF Spaces mandates uid 1000) ──────────────────────────────
RUN useradd -m -u 1000 user \
    && mkdir -p /app /data/hf_cache \
    && chown -R user:user /app /data

WORKDIR /app

# ── Python dependencies (cache layer) ────────────────────────────────────────
COPY --chown=user:user backend/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

# ── Pre-warm embedding + reranker caches at build time ───────────────────────
#  This downloads ~500 MB of weights into /data/hf_cache so cold-starts on
#  the deployed Space are near-instant. Running as root here is fine because
#  we chowned /data to user:user above and the download writes there.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', cache_folder='/data/hf_cache'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', cache_folder='/data/hf_cache')" \
    && chown -R user:user /data

# ── Application code ─────────────────────────────────────────────────────────
COPY --chown=user:user backend/ /app/

# ── Ship the SUT corpus into the image (≈1 MB) ───────────────────────────────
#  The legacy faiss.index and sqlite db are intentionally excluded via
#  .dockerignore — production uses Neon Postgres + pgvector instead.
RUN mkdir -p /app/data
COPY --chown=user:user "data/08.03.2025-Değişiklik Tebliği İşlenmiş Güncel 2013 SUT.docx" /app/data/

# ── Drop privileges & expose HF Spaces port ──────────────────────────────────
USER user
EXPOSE 7860

# Healthcheck so HF Spaces' router knows when the app is live.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"]
