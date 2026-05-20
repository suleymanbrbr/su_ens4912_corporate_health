"""
conftest.py — Shared pytest fixtures for all test suites.

Fixtures provided:
  - test_client       : FastAPI TestClient with DB mocked
  - admin_token       : JWT for an admin user
  - user_token        : JWT for a regular approved user
  - db_conn           : psycopg2 connection to the test DB
  - admin_headers     : Authorization header dict for admin
  - user_headers      : Authorization header dict for regular user
"""

import os
import sys
import types
import uuid
import pytest
from contextlib import asynccontextmanager

# ── Make the backend importable ───────────────────────────────────────────────
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

# ── Point to the real running DB by default (integration mode) ───────────────
# Override DATABASE_URL to a dedicated test schema if you want full isolation.
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://admin:secretpassword@localhost:5432/sut_knowledge_base"
)
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-pytest-only")
os.environ.setdefault("GEMINI_API_KEY", "dummy-key-for-unit-tests")
# API_KEY_ENCRYPTION_KEY is now required by api_server at import time.
# Use a deterministic Fernet key so test data stays decryptable across runs.
os.environ.setdefault(
    "API_KEY_ENCRYPTION_KEY",
    "vJ2vP3y4Z6v5sQ7mY8u9N0pL1tW2k3J4f5G6h7K8d9o=",
)

# ── Lazy import so unit tests (no DB) don't crash ────────────────────────────
@pytest.fixture(scope="session")
def app():
    """
    Return the FastAPI app with the lifespan skipped for speed.
    We patch the global `engine` with a MagicMock so chat endpoints
    don't try to load actual ML models.

    Starlette's TestClient runs app lifespan on startup; the production
    lifespan assigns ``engine = SUT_RAG_Engine()`` and would replace our
    mock. We swap ``router.lifespan_context`` for a test lifespan that
    only initializes tables and keeps ``mock_engine``.
    """
    # Avoid importing sentence_transformers / torch when loading api_server in CI.
    if "sut_rag_core" not in sys.modules:
        _stub = types.ModuleType("sut_rag_core")

        class SUT_RAG_Engine:
            pass

        _stub.SUT_RAG_Engine = SUT_RAG_Engine
        sys.modules["sut_rag_core"] = _stub

    from unittest.mock import MagicMock, patch

    mock_engine = MagicMock()
    mock_engine.load_database.return_value = True
    # New iterator per chat request (iter(...) exhausts after one consumption).
    _chunks = [
        {"status": "Sorgu analiz ediliyor..."},
        {"agent_step": {"iteration": 1, "tool": "finish", "icon": "✅", "args": {}, "result": "ok"}},
        {"agent_steps_complete": []},
        {"answer_delta": "Test "},
        {"answer_delta": "yanıtı: "},
        {"final_answer": "Test yanıtı: SUT kapsamında ödenir."},
    ]
    mock_engine.query_agentic_rag_stream.side_effect = lambda *a, **kw: iter(_chunks)

    with patch("api_server.SUT_RAG_Engine", return_value=mock_engine):
        import api_server

        @asynccontextmanager
        async def test_lifespan(app):
            api_server.init_system_tables()
            api_server.engine = mock_engine
            yield

        api_server.app.router.lifespan_context = test_lifespan
        api_server.engine = mock_engine
        yield api_server.app


@pytest.fixture(scope="session")
def test_client(app):
    from fastapi.testclient import TestClient
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


# ── Database connection (integration tests only) ──────────────────────────────
@pytest.fixture(scope="session")
def db_conn():
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        conn = psycopg2.connect(os.environ["DATABASE_URL"], cursor_factory=RealDictCursor)
        yield conn
        conn.close()
    except Exception:
        pytest.skip("Database not available — skipping integration test")


# ── Create a fresh admin user for the test session ────────────────────────────
@pytest.fixture(scope="session")
def admin_token(test_client):
    """Register the first user (auto-approved as admin) and return a JWT."""
    username = f"test_admin_{uuid.uuid4().hex[:6]}"
    email = f"{username}@example.com"
    password = "Admin@1234!"

    r = test_client.post("/api/auth/register", json={
        "username": username,
        "email": email,
        "password": password,
        "role": "admin",
    })
    # First user is auto-approved; if already exists just login
    if r.status_code not in (200, 400):
        pytest.fail(f"Register failed: {r.text}")

    r_login = test_client.post(
        "/api/auth/login",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if r_login.status_code != 200:
        # Fallback: create token directly
        from auth_utils import create_access_token
        from datetime import timedelta
        return create_access_token(
            {"sub": username, "role": "admin"},
            expires_delta=timedelta(hours=1)
        )
    return r_login.json()["access_token"]


@pytest.fixture(scope="session")
def admin_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture(scope="session")
def user_token(test_client, admin_headers):
    """Register a regular user, approve them via admin, then login."""
    username = f"test_user_{uuid.uuid4().hex[:6]}"
    email = f"{username}@example.com"
    password = "User@1234!"

    r = test_client.post("/api/auth/register", json={
        "username": username,
        "email": email,
        "password": password,
        "role": "user",
    })
    if r.status_code not in (200, 400):
        pytest.fail(f"Register failed: {r.text}")

    user_id = r.json().get("id") if r.status_code == 200 else None

    # Approve via admin
    if user_id:
        test_client.put(f"/api/admin/users/{user_id}/approve", headers=admin_headers)

    r_login = test_client.post(
        "/api/auth/login",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if r_login.status_code != 200:
        from auth_utils import create_access_token
        from datetime import timedelta
        return create_access_token(
            {"sub": username, "role": "user"},
            expires_delta=timedelta(hours=1)
        )
    return r_login.json()["access_token"]


@pytest.fixture(scope="session")
def user_headers(user_token):
    return {"Authorization": f"Bearer {user_token}"}
