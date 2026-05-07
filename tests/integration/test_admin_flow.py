"""
Integration tests for Admin endpoints.

Tests user listing, approval, role change, deletion,
system metrics, analytics, and audit logs.
"""
import uuid
import pytest


def _register_unapproved(http_client, live_admin_headers):
    """Register a new user (will be unapproved) and return (user_id, username, password)."""
    username = f"pending_{uuid.uuid4().hex[:6]}"
    email = f"{username}@testmail.com"
    password = "Pending@123"
    r = http_client.post("/api/auth/register", json={
        "username": username, "email": email,
        "password": password, "role": "user"
    })
    assert r.status_code == 200
    return r.json()["id"], username, password


# ─────────────────────────────────────────────────────────────────────────────
# List Users
# ─────────────────────────────────────────────────────────────────────────────

class TestListUsers:
    def test_admin_can_list_users(self, http_client, live_admin_headers):
        r = http_client.get("/api/admin/users", headers=live_admin_headers)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_regular_user_cannot_list_users(self, http_client, live_user_headers):
        r = http_client.get("/api/admin/users", headers=live_user_headers)
        assert r.status_code == 403

    def test_unauthenticated_cannot_list_users(self, http_client):
        r = http_client.get("/api/admin/users")
        assert r.status_code == 401

    def test_user_list_has_required_fields(self, http_client, live_admin_headers):
        r = http_client.get("/api/admin/users", headers=live_admin_headers)
        users = r.json()
        if users:
            user = users[0]
            for field in ["id", "username", "email", "role", "is_approved"]:
                assert field in user


# ─────────────────────────────────────────────────────────────────────────────
# Approve User
# ─────────────────────────────────────────────────────────────────────────────

class TestApproveUser:
    def test_admin_can_approve_user(self, http_client, live_admin_headers):
        user_id, _, _ = _register_unapproved(http_client, live_admin_headers)
        r = http_client.put(f"/api/admin/users/{user_id}/approve", headers=live_admin_headers)
        assert r.status_code == 200

    def test_approved_user_can_login(self, http_client, live_admin_headers):
        user_id, username, password = _register_unapproved(http_client, live_admin_headers)
        # Approve
        http_client.put(f"/api/admin/users/{user_id}/approve", headers=live_admin_headers)
        # Now login should work
        r = http_client.post(
            "/api/auth/login",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert r.status_code == 200
        assert "access_token" in r.json()

    def test_unapproved_user_cannot_login(self, http_client, live_admin_headers):
        _, username, password = _register_unapproved(http_client, live_admin_headers)
        r = http_client.post(
            "/api/auth/login",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert r.status_code == 403

    def test_non_admin_cannot_approve(self, http_client, live_user_headers, live_admin_headers):
        user_id, _, _ = _register_unapproved(http_client, live_admin_headers)
        r = http_client.put(f"/api/admin/users/{user_id}/approve", headers=live_user_headers)
        assert r.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
# Role Update
# ─────────────────────────────────────────────────────────────────────────────

class TestRoleUpdate:
    def test_admin_can_change_role_to_admin(self, http_client, live_admin_headers):
        user_id, _, _ = _register_unapproved(http_client, live_admin_headers)
        r = http_client.put(f"/api/admin/users/{user_id}/role",
                           json={"role": "admin"}, headers=live_admin_headers)
        assert r.status_code == 200

    def test_invalid_role_returns_400(self, http_client, live_admin_headers):
        user_id, _, _ = _register_unapproved(http_client, live_admin_headers)
        r = http_client.put(f"/api/admin/users/{user_id}/role",
                           json={"role": "superuser"}, headers=live_admin_headers)
        assert r.status_code == 400

    def test_non_admin_cannot_change_role(self, http_client, live_user_headers, live_admin_headers):
        user_id, _, _ = _register_unapproved(http_client, live_admin_headers)
        r = http_client.put(f"/api/admin/users/{user_id}/role",
                           json={"role": "admin"}, headers=live_user_headers)
        assert r.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
# Delete User
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteUser:
    def test_admin_can_delete_user(self, http_client, live_admin_headers):
        user_id, _, _ = _register_unapproved(http_client, live_admin_headers)
        r = http_client.delete(f"/api/admin/users/{user_id}", headers=live_admin_headers)
        assert r.status_code == 200

    def test_admin_cannot_delete_self(self, http_client, live_admin_headers):
        me = http_client.get("/api/auth/me", headers=live_admin_headers).json()
        r = http_client.delete(f"/api/admin/users/{me['id']}", headers=live_admin_headers)
        assert r.status_code == 400

    def test_non_admin_cannot_delete_user(self, http_client, live_user_headers, live_admin_headers):
        user_id, _, _ = _register_unapproved(http_client, live_admin_headers)
        r = http_client.delete(f"/api/admin/users/{user_id}", headers=live_user_headers)
        assert r.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
# System Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemMetrics:
    def test_admin_gets_system_metrics(self, http_client, live_admin_headers):
        r = http_client.get("/api/admin/system", headers=live_admin_headers)
        assert r.status_code == 200
        data = r.json()
        for field in ["users_count", "queries_count", "chunks_count", "pending_count"]:
            assert field in data

    def test_regular_user_cannot_get_metrics(self, http_client, live_user_headers):
        r = http_client.get("/api/admin/system", headers=live_user_headers)
        assert r.status_code == 403

    def test_users_count_is_positive_integer(self, http_client, live_admin_headers):
        r = http_client.get("/api/admin/system", headers=live_admin_headers)
        assert r.json()["users_count"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Analytics
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalytics:
    def test_admin_gets_analytics(self, http_client, live_admin_headers):
        r = http_client.get("/api/admin/analytics", headers=live_admin_headers)
        assert r.status_code == 200
        data = r.json()
        for field in ["top_keywords", "daily_volume", "total_users"]:
            assert field in data

    def test_top_keywords_is_list(self, http_client, live_admin_headers):
        r = http_client.get("/api/admin/analytics", headers=live_admin_headers)
        assert isinstance(r.json()["top_keywords"], list)

    def test_non_admin_cannot_get_analytics(self, http_client, live_user_headers):
        r = http_client.get("/api/admin/analytics", headers=live_user_headers)
        assert r.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
# Audit Logs
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditLogs:
    def test_admin_can_get_audit_logs(self, http_client, live_admin_headers):
        r = http_client.get("/api/admin/audit-logs", headers=live_admin_headers)
        assert r.status_code == 200
        data = r.json()
        assert "logs" in data
        assert "total" in data

    def test_logs_list_is_list(self, http_client, live_admin_headers):
        r = http_client.get("/api/admin/audit-logs", headers=live_admin_headers)
        assert isinstance(r.json()["logs"], list)

    def test_non_admin_cannot_access_audit_logs(self, http_client, live_user_headers):
        r = http_client.get("/api/admin/audit-logs", headers=live_user_headers)
        assert r.status_code == 403

    def test_pagination_limit_param(self, http_client, live_admin_headers):
        r = http_client.get("/api/admin/audit-logs?limit=5&offset=0", headers=live_admin_headers)
        assert r.status_code == 200
        assert len(r.json()["logs"]) <= 5
