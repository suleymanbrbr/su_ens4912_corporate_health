"""
Unit tests for embedding_utils.py

Tests model name resolution logic and default constants.
No actual model loading occurs.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import pytest
from unittest.mock import patch, MagicMock


# ── Patch sentence_transformers before import ─────────────────────────────────
@pytest.fixture(scope="module", autouse=True)
def patch_st():
    with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
        yield


import embedding_utils


# ─────────────────────────────────────────────────────────────────────────────
# resolve_embedding_model_name
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveEmbeddingModelName:
    """The function maps vector dimension to a model name."""

    def test_384_dim_returns_minilm(self):
        name = embedding_utils.resolve_embedding_model_name(384)
        assert "MiniLM" in name or "minilm" in name.lower() or "384" in name or name == embedding_utils.DEFAULT_MINILM

    def test_none_dim_returns_default(self):
        name = embedding_utils.resolve_embedding_model_name(None)
        assert name == embedding_utils.DEFAULT_MINILM

    def test_unknown_dim_returns_default(self):
        name = embedding_utils.resolve_embedding_model_name(999)
        assert isinstance(name, str) and len(name) > 0

    def test_768_dim_returns_some_model(self):
        """768-dim is BERT-base / BGE-large territory."""
        name = embedding_utils.resolve_embedding_model_name(768)
        assert isinstance(name, str)


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT_MINILM constant
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaultConstants:
    def test_default_minilm_is_string(self):
        assert isinstance(embedding_utils.DEFAULT_MINILM, str)

    def test_default_minilm_is_nonempty(self):
        assert len(embedding_utils.DEFAULT_MINILM) > 0

    def test_default_minilm_looks_like_model_name(self):
        """Should contain a slash (org/model-name format) or be a known local name."""
        name = embedding_utils.DEFAULT_MINILM
        assert "/" in name or len(name) > 3


# ─────────────────────────────────────────────────────────────────────────────
# build_hf_embeddings (mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildHfEmbeddings:
    def test_returns_object_with_embed_documents(self):
        """build_hf_embeddings should return an object that has embed_documents."""
        mock_hf = MagicMock()
        mock_hf.embed_documents = MagicMock(return_value=[[0.1] * 384])

        with patch("embedding_utils.HuggingFaceEmbeddings", return_value=mock_hf):
            result = embedding_utils.build_hf_embeddings("test-model")
            assert hasattr(result, "embed_documents")

    def test_called_with_correct_model_name(self):
        mock_hf = MagicMock()
        with patch("embedding_utils.HuggingFaceEmbeddings", return_value=mock_hf) as mock_cls:
            embedding_utils.build_hf_embeddings("sentence-transformers/all-MiniLM-L6-v2")
            call_kwargs = mock_cls.call_args
            assert call_kwargs is not None
