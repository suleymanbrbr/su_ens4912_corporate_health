# embedding_utils.py
# Shared retrieval embedding config: must match how `chunks` / `kg_nodes` were indexed.

import os
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_E5 = "intfloat/multilingual-e5-large"


def resolve_embedding_model_name(db_vector_dim: Optional[int]) -> str:
    """
    Pick HF model to match pgvector columns. Override anytime with SUT_EMBEDDING_MODEL in .env.
    """
    env = os.getenv("SUT_EMBEDDING_MODEL", "").strip()
    if env:
        return env
    if db_vector_dim == 1024:
        return DEFAULT_E5
    if db_vector_dim == 384:
        return DEFAULT_MINILM
    if db_vector_dim not in (None,):
        print(f"[WARN] Unknown embedding dimension in DB ({db_vector_dim}); using {DEFAULT_MINILM}")
    return DEFAULT_MINILM


def build_hf_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    kw = dict(model_name=model_name, model_kwargs={"device": "cpu"})
    if "e5" in model_name.lower() or model_name.startswith("intfloat/"):
        kw["encode_kwargs"] = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(**kw)


def embed_query_retrieval(embeddings: HuggingFaceEmbeddings, model_name: str, text: str):
    """Queries must use the E5 'query: ' prefix when the indexed passages used 'passage: '."""
    if "e5" in model_name.lower() or model_name.startswith("intfloat/"):
        return embeddings.embed_query(f"query: {text}")
    return embeddings.embed_query(text)
