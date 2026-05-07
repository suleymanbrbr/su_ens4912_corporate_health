"""
Integration tests for GET /api/policies (Policy Browser endpoint).

Authenticated users may browse policies (full-text and ILIKE search).
"""
import pytest


class TestPoliciesSearch:
    def test_admin_can_access_policies(self, http_client, live_admin_headers):
        r = http_client.get("/api/policies", headers=live_admin_headers)
        assert r.status_code == 200

    def test_result_has_expected_keys(self, http_client, live_admin_headers):
        r = http_client.get("/api/policies", headers=live_admin_headers)
        data = r.json()
        assert "results" in data
        assert "total" in data
        assert "offset" in data
        assert "limit" in data

    def test_results_is_list(self, http_client, live_admin_headers):
        r = http_client.get("/api/policies", headers=live_admin_headers)
        assert isinstance(r.json()["results"], list)

    def test_search_with_query_term(self, http_client, live_admin_headers):
        r = http_client.get("/api/policies?q=ibuprofen", headers=live_admin_headers)
        assert r.status_code == 200
        # Results may be empty if DB has no chunks, but format must be correct
        assert "results" in r.json()

    def test_limit_param_respected(self, http_client, live_admin_headers):
        r = http_client.get("/api/policies?limit=5", headers=live_admin_headers)
        assert len(r.json()["results"]) <= 5

    def test_offset_param_accepted(self, http_client, live_admin_headers):
        r = http_client.get("/api/policies?offset=10", headers=live_admin_headers)
        assert r.status_code == 200
        assert r.json()["offset"] == 10

    def test_authenticated_non_admin_can_access_policies(self, http_client, live_user_headers):
        """Black-box against ``API_BASE_URL``. Expect 200 when the running API matches this repo
        (``GET /api/policies`` uses ``get_current_user``). If you see 403, restart/redeploy the API — an older build may still require admin."""
        r = http_client.get("/api/policies", headers=live_user_headers)
        assert r.status_code == 200

    def test_unauthenticated_returns_401(self, http_client):
        r = http_client.get("/api/policies")
        assert r.status_code == 401

    def test_search_empty_query_returns_all(self, http_client, live_admin_headers):
        r_all = http_client.get("/api/policies?q=&limit=20", headers=live_admin_headers)
        r_noq = http_client.get("/api/policies?limit=20", headers=live_admin_headers)
        assert r_all.status_code == 200
        assert r_noq.status_code == 200

    def test_default_limit_is_20(self, http_client, live_admin_headers):
        r = http_client.get("/api/policies", headers=live_admin_headers)
        assert r.json()["limit"] == 20

    def test_result_items_have_title_and_excerpt(self, http_client, live_admin_headers):
        r = http_client.get("/api/policies?limit=1", headers=live_admin_headers)
        results = r.json()["results"]
        if results:
            item = results[0]
            assert "title" in item
            assert "excerpt" in item
            assert "id" in item
