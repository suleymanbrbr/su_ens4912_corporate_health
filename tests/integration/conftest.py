"""
conftest.py — Black-box integration fixtures (httpx against a running API).

These fixtures are prefixed with ``live_`` so they do not override the parent
``tests/conftest.py`` fixtures (``test_client``, ``user_headers``, …) used for
in-process tests with a mocked RAG engine.

Strategy:
  - Tries to register a fresh admin. If the DB already has users, the first
    registered user is NOT auto-approved, so we fall back to creating a token
    directly with auth_utils (using the TEST JWT_SECRET_KEY).
  - Set TEST_ADMIN_USER / TEST_ADMIN_PASS env vars to use a real existing account.
"""
import os
import sys
import uuid
import pytest
import httpx

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "../../backend")
sys.path.insert(0, BACKEND_DIR)

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Use real credentials from env, or auto-generate ──────────────────────────
_ADMIN_USER = os.getenv("TEST_ADMIN_USER", "")
_ADMIN_PASS = os.getenv("TEST_ADMIN_PASS", "")
_USER_USER = os.getenv("TEST_USER", "")
_USER_PASS = os.getenv("TEST_USER_PASS", "")


def _post_login(client, username, password):
    return client.post(
        "/api/auth/login",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )


def _make_jwt(username: str, role: str) -> str:
    """Generate a JWT directly — used as fallback when the DB rejects auto-register."""
    from auth_utils import create_access_token
    from datetime import timedelta
    # Use the same secret the running backend uses
    secret = os.getenv("JWT_SECRET_KEY", "")
    if not secret:
        # Read from .env file
        env_file = os.path.join(BACKEND_DIR, ".env")
        for line in open(env_file).readlines():
            if line.startswith("JWT_SECRET_KEY="):
                secret = line.split("=", 1)[1].strip()
                break
    import auth_utils as au
    orig = au.SECRET_KEY
    if secret:
        au.SECRET_KEY = secret
    token = create_access_token({"sub": username, "role": role}, timedelta(hours=2))
    au.SECRET_KEY = orig
    return token


# ─────────────────────────────────────────────────────────────────────────────
# Session HTTP client (live server)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def http_client():
    import time
    max_retries = 10
    ready = False
    for i in range(max_retries):
        try:
            httpx.get(f"{BASE_URL}/docs", timeout=2)
            ready = True
            break
        except Exception:
            time.sleep(2)

    if not ready:
        pytest.skip(f"Backend not reachable at {BASE_URL} after {max_retries} retries")

    with httpx.Client(base_url=BASE_URL, timeout=60) as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# Admin token (live server)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def live_admin_token(http_client):
    # Option 1: use env var credentials for an existing approved admin
    if _ADMIN_USER and _ADMIN_PASS:
        r = _post_login(http_client, _ADMIN_USER, _ADMIN_PASS)
        if r.status_code == 200:
            return r.json()["access_token"]

    # Option 2: register a fresh admin (works only if DB is empty → auto-approve)
    username = f"adm_{uuid.uuid4().hex[:8]}"
    email = f"{username}@testmail.com"
    password = "Admin@1234!"

    http_client.post("/api/auth/register", json={
        "username": username, "email": email,
        "password": password, "role": "admin",
    })

    r = _post_login(http_client, username, password)
    if r.status_code == 200:
        return r.json()["access_token"]

    # Option 3: DB has existing users → auto-approve didn't fire.
    # Approve this new user via DB directly (psycopg2), then login.
    try:
        import psycopg2
        db_url = "postgresql://admin:secretpassword@localhost:5432/sut_knowledge_base"
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("UPDATE users SET is_approved=1, role='admin' WHERE username=%s", (username,))
        conn.commit()
        cur.close()
        conn.close()

        r2 = _post_login(http_client, username, password)
        if r2.status_code == 200:
            return r2.json()["access_token"]
    except Exception:
        pass

    # Option 4: fabricate a JWT directly (last resort — works only if JWT_SECRET matches)
    return _make_jwt(username, "admin")


@pytest.fixture(scope="session")
def live_admin_headers(live_admin_token):
    return {"Authorization": f"Bearer {live_admin_token}"}


# ─────────────────────────────────────────────────────────────────────────────
# Regular user token (live server)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def live_user_token(http_client, live_admin_headers):
    if _USER_USER and _USER_PASS:
        r = _post_login(http_client, _USER_USER, _USER_PASS)
        if r.status_code == 200:
            return r.json()["access_token"]

    username = f"usr_{uuid.uuid4().hex[:8]}"
    email = f"{username}@testmail.com"
    password = "User@1234!"

    r = http_client.post("/api/auth/register", json={
        "username": username, "email": email,
        "password": password, "role": "user",
    })
    user_id = r.json().get("id") if r.status_code == 200 else None

    # Approve via admin endpoint
    if user_id:
        http_client.put(f"/api/admin/users/{user_id}/approve", headers=live_admin_headers)

    r2 = _post_login(http_client, username, password)
    if r2.status_code == 200:
        return r2.json()["access_token"]

    # Fallback: approve via DB
    try:
        import psycopg2
        db_url = "postgresql://admin:secretpassword@localhost:5432/sut_knowledge_base"
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("UPDATE users SET is_approved=1 WHERE username=%s", (username,))
        conn.commit()
        cur.close()
        conn.close()
        r3 = _post_login(http_client, username, password)
        if r3.status_code == 200:
            return r3.json()["access_token"]
    except Exception:
        pass

    return _make_jwt(username, "user")


@pytest.fixture(scope="session")
def live_user_headers(live_user_token):
    return {"Authorization": f"Bearer {live_user_token}"}
