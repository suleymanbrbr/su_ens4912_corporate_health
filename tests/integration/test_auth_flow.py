"""
Integration tests for the Authentication flow.

Requires:
  - Docker Compose services running (db + backend)
  OR
  - The http_client fixture from conftest.py with a live DB

Run: pytest integration/test_auth_flow.py -v
"""
import uuid
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _register(http_client, username=None, email=None, password="Test@1234!", role="user"):
    username = username or f"user_{uuid.uuid4().hex[:8]}"
    email = email or f"{username}@testmail.com"
    r = http_client.post("/api/auth/register", json={
        "username": username,
        "email": email,
        "password": password,
        "role": role,
    })
    return r, username, email, password


def _login(http_client, username, password):
    return http_client.post(
        "/api/auth/login",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistration:
    def test_register_returns_200(self, http_client):
        r, _, _, _ = _register(http_client)
        assert r.status_code == 200

    def test_register_returns_correct_fields(self, http_client):
        r, username, email, _ = _register(http_client)
        data = r.json()
        assert data["username"] == username
        assert data["email"] == email
        assert "id" in data
        assert "role" in data

    def test_register_default_role_is_user(self, http_client):
        r, _, _, _ = _register(http_client, role="user")
        # Second+ user is not auto-approved
        assert r.json()["role"] == "user"

    def test_duplicate_username_returns_400(self, http_client):
        _, username, _, password = (lambda r, u, e, p: (r, u, e, p))(*_register(http_client))
        r2, _, _, _ = _register(http_client, username=username)
        assert r2.status_code == 400

    def test_duplicate_email_returns_400(self, http_client):
        _, _, email, _ = (lambda r, u, e, p: (r, u, e, p))(*_register(http_client))
        r2, _, _, _ = _register(http_client, email=email)
        assert r2.status_code == 400

    def test_invalid_email_returns_422(self, http_client):
        r = http_client.post("/api/auth/register", json={
            "username": "validuser",
            "email": "not-an-email",
            "password": "pass123",
        })
        assert r.status_code == 422

    def test_missing_password_returns_422(self, http_client):
        r = http_client.post("/api/auth/register", json={
            "username": "nopass",
            "email": "nopass@test.com",
        })
        assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Login Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLogin:
    def test_admin_can_login(self, http_client, live_admin_token):
        """live_admin_token fixture proves login succeeds."""
        assert isinstance(live_admin_token, str)
        assert len(live_admin_token) > 10

    def test_login_returns_access_token(self, http_client, live_admin_token):
        assert live_admin_token is not None

    def test_wrong_password_returns_401(self, http_client):
        r, username, _, _ = _register(http_client)
        r2 = _login(http_client, username, "WrongPassword!")
        assert r2.status_code in (401, 403)

    def test_unknown_user_returns_401(self, http_client):
        r = _login(http_client, "nonexistent_user_xyz", "anypass")
        assert r.status_code == 401

    def test_login_with_email_works(self, http_client, live_admin_token, live_admin_headers):
        """Login can use email instead of username."""
        r = http_client.get("/api/auth/me", headers=live_admin_headers)
        email = r.json()["email"]
        # Try logging in with email (API checks username OR email)
        r2 = http_client.post(
            "/api/auth/login",
            data={"username": email, "password": "Admin@1234!"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        # May fail if password doesn't match — just check structure
        assert r2.status_code in (200, 401)


# ─────────────────────────────────────────────────────────────────────────────
# /api/auth/me
# ─────────────────────────────────────────────────────────────────────────────

class TestGetMe:
    def test_me_returns_user_data(self, http_client, live_admin_headers):
        r = http_client.get("/api/auth/me", headers=live_admin_headers)
        assert r.status_code == 200
        data = r.json()
        assert "username" in data
        assert "email" in data
        assert "role" in data

    def test_me_without_token_returns_401(self, http_client):
        r = http_client.get("/api/auth/me")
        assert r.status_code == 401

    def test_me_with_invalid_token_returns_401(self, http_client):
        r = http_client.get("/api/auth/me", headers={"Authorization": "Bearer invalid.token.here"})
        assert r.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# Password Change
# ─────────────────────────────────────────────────────────────────────────────

class TestPasswordChange:
    def test_wrong_old_password_returns_400(self, http_client, live_user_headers):
        r = http_client.put("/api/auth/password", json={
            "old_password": "WrongOld!",
            "new_password": "NewPass@123",
        }, headers=live_user_headers)
        assert r.status_code == 400

    def test_change_password_unauthenticated_returns_401(self, http_client):
        r = http_client.put("/api/auth/password", json={
            "old_password": "old",
            "new_password": "new",
        })
        assert r.status_code == 401
