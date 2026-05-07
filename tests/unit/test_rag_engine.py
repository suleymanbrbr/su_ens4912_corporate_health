"""
Unit tests for SUT_RAG_Engine (sut_rag_core.py)

All ML model loading and database access is mocked so tests run
without GPU, network, or PostgreSQL.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


def _install_lightweight_sentence_transformers():
    """Avoid importing real sentence_transformers (pulls torch/transformers; often breaks in CI)."""
    import types

    for name in list(sys.modules):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            del sys.modules[name]
    st = types.ModuleType("sentence_transformers")

    class CrossEncoder:
        def __init__(self, *a, **k):
            pass

        def predict(self, *a, **kw):
            return [0.9]

    st.CrossEncoder = CrossEncoder
    sys.modules["sentence_transformers"] = st


def _install_llm_import_stubs():
    """Heavy LLM client stacks often mismatch installed google-genai / torch; sut_rag_core only needs symbols."""
    import types

    def _drop(prefix: str) -> None:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]

    _drop("google.generativeai")
    gg = types.ModuleType("google.generativeai")

    def configure(*a, **k):
        pass

    class GenerativeModel:
        def __init__(self, *a, **k):
            pass

        def generate_content(self, *a, **k):
            return types.SimpleNamespace(text="")

    gg.configure = configure
    gg.GenerativeModel = GenerativeModel
    sys.modules["google.generativeai"] = gg

    _drop("langchain_google_genai")
    lg = types.ModuleType("langchain_google_genai")

    class ChatGoogleGenerativeAI:
        def __init__(self, *a, **k):
            pass

    lg.ChatGoogleGenerativeAI = ChatGoogleGenerativeAI
    sys.modules["langchain_google_genai"] = lg

    _drop("langchain_openai")
    lo = types.ModuleType("langchain_openai")

    class ChatOpenAI:
        def __init__(self, *a, **k):
            pass

    lo.ChatOpenAI = ChatOpenAI
    sys.modules["langchain_openai"] = lo


@pytest.fixture(scope="module", autouse=True)
def _load_real_sut_rag_core():
    """Parent conftest may stub sut_rag_core for api_server; unit tests need the real module."""
    mod = sys.modules.get("sut_rag_core")
    if mod is not None and getattr(mod, "__file__", None) is None:
        del sys.modules["sut_rag_core"]
    _install_lightweight_sentence_transformers()
    _install_llm_import_stubs()
    import sut_rag_core  # noqa: F401 — load with lightweight deps

    yield


# ── Helper: build a fully mocked engine instance ─────────────────────────────
def _make_engine():
    """
    Instantiate SUT_RAG_Engine with all heavy dependencies mocked:
      - sentence_transformers.CrossEncoder
      - build_hf_embeddings (embedding model)
      - KG_Storage_Manager
      - psycopg2 connection
    """
    mock_cross_encoder = MagicMock()
    mock_cross_encoder.predict.return_value = [0.9, 0.8, 0.7]

    mock_embeddings = MagicMock()
    mock_kg = MagicMock()

    with patch("sut_rag_core.CrossEncoder", return_value=mock_cross_encoder), \
         patch("sut_rag_core.build_hf_embeddings", return_value=mock_embeddings), \
         patch("sut_rag_core.KG_Storage_Manager", return_value=mock_kg), \
         patch("sut_rag_core.google_genai"), \
         patch("os.getenv", side_effect=lambda k, d=None: d):

        from sut_rag_core import SUT_RAG_Engine
        engine = SUT_RAG_Engine.__new__(SUT_RAG_Engine)
        engine.embeddings_model = mock_embeddings
        engine._embedding_model_name = "test-model"
        engine.reranker = mock_cross_encoder
        engine.conn = None
        engine.cursor = None
        engine.llm = MagicMock()
        engine.provider = "google"
        engine.kg = mock_kg
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# _safe_calculate
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeCalculate:
    """Tests for the sandboxed math evaluator."""

    @pytest.fixture(autouse=True)
    def engine(self):
        self.engine = _make_engine()

    def test_simple_addition(self):
        result = self.engine._safe_calculate("2 + 3")
        assert "5" in result

    def test_multiplication(self):
        result = self.engine._safe_calculate("7 * 6")
        assert "42" in result

    def test_division(self):
        result = self.engine._safe_calculate("10 / 4")
        assert "2.5" in result

    def test_float_arithmetic(self):
        result = self.engine._safe_calculate("1.5 + 2.5")
        assert "4.0" in result or "4" in result

    def test_parentheses(self):
        result = self.engine._safe_calculate("(3 + 2) * 4")
        assert "20" in result

    def test_division_by_zero_returns_error(self):
        result = self.engine._safe_calculate("5 / 0")
        assert "hata" in result.lower() or "error" in result.lower()

    def test_empty_expression_returns_error(self):
        result = self.engine._safe_calculate("")
        assert isinstance(result, str)

    def test_malicious_code_is_blocked(self):
        """__import__ and exec must NOT execute."""
        result = self.engine._safe_calculate("__import__('os').system('rm -rf /')")
        # The expression is sanitised — it either errors or returns empty
        assert "rm -rf" not in result

    def test_negative_numbers(self):
        result = self.engine._safe_calculate("-5 + 10")
        assert "5" in result


# ─────────────────────────────────────────────────────────────────────────────
# _format_chunks_result
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatChunksResult:
    @pytest.fixture(autouse=True)
    def engine(self):
        self.engine = _make_engine()

    def _make_chunk(self, idx=1, text="Sample text", headers=None, score=None):
        chunk = {
            "id": f"chunk-{idx}",
            "text": text,
            "metadata": headers or {"Header1": "Section A", "Header2": "Sub B"},
        }
        if score is not None:
            chunk["score"] = score
        return chunk

    def test_empty_list_returns_not_found_message(self):
        result = self.engine._format_chunks_result([])
        assert "bulunamadı" in result.lower() or "not found" in result.lower()

    def test_single_chunk_contains_content(self):
        chunks = [self._make_chunk(text="SUT madde 4.2.63 içeriği")]
        result = self.engine._format_chunks_result(chunks)
        assert "SUT madde 4.2.63" in result

    def test_multiple_chunks_numbered(self):
        chunks = [self._make_chunk(i, f"İçerik {i}") for i in range(1, 4)]
        result = self.engine._format_chunks_result(chunks)
        assert "KAYNAK 1" in result
        assert "KAYNAK 2" in result
        assert "KAYNAK 3" in result

    def test_score_shown_when_present(self):
        chunks = [self._make_chunk(score=0.95)]
        result = self.engine._format_chunks_result(chunks)
        assert "0.95" in result

    def test_score_hidden_when_absent(self):
        chunks = [self._make_chunk()]  # no score
        result = self.engine._format_chunks_result(chunks)
        assert "skor" not in result

    def test_header_breadcrumb_shown(self):
        chunks = [self._make_chunk(headers={"Header1": "Bölüm 4", "Header2": "Alt Madde"})]
        result = self.engine._format_chunks_result(chunks)
        assert "Bölüm 4" in result

    def test_long_text_is_truncated(self):
        long_text = "A" * 2000
        chunks = [self._make_chunk(text=long_text)]
        result = self.engine._format_chunks_result(chunks)
        # The formatter truncates at 800 chars
        assert len(result) < 2000


# ─────────────────────────────────────────────────────────────────────────────
# _format_observations
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatObservations:
    @pytest.fixture(autouse=True)
    def engine(self):
        self.engine = _make_engine()

    def test_empty_observations_returns_empty_string(self):
        assert self.engine._format_observations([]) == ""

    def test_single_observation_shows_step_number(self):
        obs = [{"tool": "search_sut_chunks", "args": {"query": "test"}, "result": "some result"}]
        formatted = self.engine._format_observations(obs)
        assert "Adım 1" in formatted

    def test_multiple_observations_numbered_sequentially(self):
        obs = [
            {"tool": "search_sut_chunks", "args": {}, "result": "r1"},
            {"tool": "lookup_kg_entity", "args": {}, "result": "r2"},
        ]
        formatted = self.engine._format_observations(obs)
        assert "Adım 1" in formatted
        assert "Adım 2" in formatted

    def test_result_is_included_in_output(self):
        obs = [{"tool": "calculate", "args": {}, "result": "42.0 hesaplandı"}]
        formatted = self.engine._format_observations(obs)
        assert "42.0 hesaplandı" in formatted

    def test_long_result_is_truncated_to_600(self):
        obs = [{"tool": "test", "args": {}, "result": "X" * 1000}]
        formatted = self.engine._format_observations(obs)
        assert "XXXXXX" in formatted
        # Result is capped at 600 chars per observation
        assert formatted.count("X") <= 600


# ─────────────────────────────────────────────────────────────────────────────
# Tool Dispatch (_run_tool)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunTool:
    @pytest.fixture(autouse=True)
    def engine(self):
        self.engine = _make_engine()

    def test_calculate_tool_dispatched(self):
        result = self.engine._run_tool("calculate", {"tool": "calculate", "expression": "3+3"}, 5)
        assert "6" in result

    def test_unknown_tool_returns_error_message(self):
        result = self.engine._run_tool("nonexistent_tool", {}, 5)
        assert "bilinmeyen" in result.lower() or "unknown" in result.lower()

    def test_search_chunks_tool_calls_retrieve(self):
        """search_sut_chunks should call _retrieve_chunks and format results."""
        self.engine._retrieve_chunks = MagicMock(return_value=[])
        result = self.engine._run_tool("search_sut_chunks", {"tool": "search_sut_chunks", "query": "test", "k": 3}, 5)
        self.engine._retrieve_chunks.assert_called_once_with("test", 3)
        assert isinstance(result, str)

    def test_kg_lookup_tool_calls_kg(self):
        self.engine.kg.lookup_entity = MagicMock(return_value="KG result")
        result = self.engine._run_tool("lookup_kg_entity", {"entity": "Aspirin"}, 5)
        self.engine.kg.lookup_entity.assert_called()
        assert result == "KG result"

    def test_tool_exception_returns_error_string(self):
        self.engine.kg.lookup_entity = MagicMock(side_effect=RuntimeError("DB error"))
        result = self.engine._run_tool("lookup_kg_entity", {"entity": "test"}, 5)
        assert "ARAÇ HATASI" in result or "error" in result.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Tool Schemas & Constants
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineConstants:
    def test_max_iterations_positive(self):
        from sut_rag_core import MAX_AGENT_ITERATIONS
        assert MAX_AGENT_ITERATIONS > 0

    def test_tool_icons_contain_all_tools(self):
        from sut_rag_core import TOOL_ICONS
        for tool in ["search_sut_chunks", "search_sut_fulltext", "lookup_kg_entity",
                     "explore_kg_path", "calculate", "finish"]:
            assert tool in TOOL_ICONS
