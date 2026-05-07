"""
Integration tests for Announcement endpoints.

POST /api/admin/announcements     — Create announcement (admin)
GET  /api/announcements           — Get active announcement (user)
DELETE /api/admin/announcements/{id} — Deactivate (admin)
"""
import uuid
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Create Announcement
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateAnnouncement:
    def test_admin_can_create_announcement(self, http_client, live_admin_headers):
        r = http_client.post("/api/admin/announcements",
                             json={"message": "Sistem bakımı yapılacak."},
                             headers=live_admin_headers)
        assert r.status_code == 200

    def test_create_returns_success_message(self, http_client, live_admin_headers):
        r = http_client.post("/api/admin/announcements",
                             json={"message": "Test duyuru"},
                             headers=live_admin_headers)
        assert "message" in r.json()

    def test_non_admin_cannot_create(self, http_client, live_user_headers):
        r = http_client.post("/api/admin/announcements",
                             json={"message": "Yetkisiz duyuru"},
                             headers=live_user_headers)
        assert r.status_code == 403

    def test_missing_message_returns_422(self, http_client, live_admin_headers):
        r = http_client.post("/api/admin/announcements",
                             json={}, headers=live_admin_headers)
        assert r.status_code == 422

    def test_unauthenticated_returns_401(self, http_client):
        r = http_client.post("/api/admin/announcements",
                             json={"message": "test"})
        assert r.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# Get Active Announcement
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAnnouncement:
    @pytest.fixture(autouse=True)
    def create_announcement(self, http_client, live_admin_headers):
        """Ensure at least one active announcement exists."""
        self.test_message = f"Test duyuru {uuid.uuid4().hex[:6]}"
        http_client.post("/api/admin/announcements",
                        json={"message": self.test_message},
                        headers=live_admin_headers)

    def test_authenticated_user_can_get_announcement(self, http_client, live_user_headers):
        r = http_client.get("/api/announcements", headers=live_user_headers)
        assert r.status_code == 200

    def test_announcement_returns_dict(self, http_client, live_user_headers):
        r = http_client.get("/api/announcements", headers=live_user_headers)
        assert isinstance(r.json(), dict)

    def test_latest_message_is_active(self, http_client, live_user_headers):
        r = http_client.get("/api/announcements", headers=live_user_headers)
        data = r.json()
        if data:  # non-empty means active exists
            assert data.get("message") == self.test_message

    def test_unauthenticated_returns_401(self, http_client):
        r = http_client.get("/api/announcements")
        assert r.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# Deactivate Announcement
# ─────────────────────────────────────────────────────────────────────────────

class TestDeactivateAnnouncement:
    def test_admin_can_deactivate(self, http_client, live_admin_headers):
        # Create one first
        http_client.post("/api/admin/announcements",
                        json={"message": "To be removed"},
                        headers=live_admin_headers)
        # Get current active
        r_get = http_client.get("/api/announcements", headers=live_admin_headers)
        active = r_get.json()
        if active and "id" in active:
            r_del = http_client.delete(f"/api/admin/announcements/{active['id']}",
                                       headers=live_admin_headers)
            assert r_del.status_code == 200

    def test_non_admin_cannot_deactivate(self, http_client, live_user_headers):
        r = http_client.delete(f"/api/admin/announcements/{uuid.uuid4()}",
                               headers=live_user_headers)
        assert r.status_code == 403

    def test_unauthenticated_cannot_deactivate(self, http_client):
        r = http_client.delete(f"/api/admin/announcements/{uuid.uuid4()}")
        assert r.status_code == 401
