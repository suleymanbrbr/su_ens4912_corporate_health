"""
Integration tests for Knowledge Graph API endpoints.

Tests: /api/kg/stats, /api/kg/nodes, /api/kg/node/{id},
       /api/kg/subgraph/{id}, /api/kg/path, /api/admin/kg/stats,
       /api/admin/kg/benchmark
"""
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# /api/kg/stats (user-level)
# ─────────────────────────────────────────────────────────────────────────────

class TestKGStats:
    def test_user_can_get_stats(self, http_client, live_user_headers):
        r = http_client.get("/api/kg/stats", headers=live_user_headers)
        assert r.status_code == 200

    def test_stats_has_expected_structure(self, http_client, live_user_headers):
        r = http_client.get("/api/kg/stats", headers=live_user_headers)
        data = r.json()
        # Should return dict (may be empty if no KG built yet)
        assert isinstance(data, dict)

    def test_unauthenticated_returns_401(self, http_client):
        r = http_client.get("/api/kg/stats")
        assert r.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# /api/kg/nodes
# ─────────────────────────────────────────────────────────────────────────────

class TestKGNodes:
    def test_user_can_search_nodes(self, http_client, live_user_headers):
        r = http_client.get("/api/kg/nodes?q=aspirin", headers=live_user_headers)
        assert r.status_code == 200

    def test_search_result_has_nodes_key(self, http_client, live_user_headers):
        r = http_client.get("/api/kg/nodes?q=ibuprofen", headers=live_user_headers)
        assert "nodes" in r.json()

    def test_nodes_is_list(self, http_client, live_user_headers):
        r = http_client.get("/api/kg/nodes", headers=live_user_headers)
        assert isinstance(r.json()["nodes"], list)

    def test_limit_param_respected(self, http_client, live_user_headers):
        r = http_client.get("/api/kg/nodes?limit=5", headers=live_user_headers)
        assert len(r.json()["nodes"]) <= 5

    def test_type_filter_accepted(self, http_client, live_user_headers):
        r = http_client.get("/api/kg/nodes?type=DRUG", headers=live_user_headers)
        assert r.status_code == 200

    def test_unauthenticated_returns_401(self, http_client):
        r = http_client.get("/api/kg/nodes")
        assert r.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# /api/kg/node/{node_id}
# ─────────────────────────────────────────────────────────────────────────────

class TestKGNodeDetail:
    def test_nonexistent_node_returns_404(self, http_client, live_user_headers):
        r = http_client.get("/api/kg/node/nonexistent-id-xyz", headers=live_user_headers)
        assert r.status_code == 404

    def test_unauthenticated_returns_401(self, http_client):
        r = http_client.get("/api/kg/node/some-id")
        assert r.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# /api/kg/path
# ─────────────────────────────────────────────────────────────────────────────

class TestKGPath:
    def test_path_endpoint_accessible(self, http_client, live_user_headers):
        r = http_client.get("/api/kg/path?from_id=a&to_id=b&max_hops=3", headers=live_user_headers)
        assert r.status_code == 200

    def test_path_result_has_path_and_found(self, http_client, live_user_headers):
        r = http_client.get("/api/kg/path?from_id=a&to_id=b", headers=live_user_headers)
        data = r.json()
        assert "path" in data
        assert "found" in data

    def test_unauthenticated_returns_401(self, http_client):
        r = http_client.get("/api/kg/path?from_id=a&to_id=b")
        assert r.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# /api/admin/kg/stats
# ─────────────────────────────────────────────────────────────────────────────

class TestAdminKGStats:
    def test_admin_can_get_kg_stats(self, http_client, live_admin_headers):
        r = http_client.get("/api/admin/kg/stats", headers=live_admin_headers)
        assert r.status_code == 200

    def test_non_admin_returns_403(self, http_client, live_user_headers):
        r = http_client.get("/api/admin/kg/stats", headers=live_user_headers)
        assert r.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
# /api/admin/kg/benchmark
# ─────────────────────────────────────────────────────────────────────────────

class TestKGBenchmark:
    def test_admin_can_run_benchmark(self, http_client, live_admin_headers):
        r = http_client.post("/api/admin/kg/benchmark", headers=live_admin_headers)
        assert r.status_code == 200

    def test_benchmark_returns_hit_rate(self, http_client, live_admin_headers):
        r = http_client.post("/api/admin/kg/benchmark", headers=live_admin_headers)
        data = r.json()
        assert "hit_rate" in data
        assert "total" in data
        assert "hits" in data
        assert "results" in data

    def test_benchmark_hit_rate_is_float(self, http_client, live_admin_headers):
        r = http_client.post("/api/admin/kg/benchmark", headers=live_admin_headers)
        hit_rate = r.json()["hit_rate"]
        assert isinstance(hit_rate, (int, float))
        assert 0.0 <= hit_rate <= 1.0

    def test_benchmark_total_is_10(self, http_client, live_admin_headers):
        r = http_client.post("/api/admin/kg/benchmark", headers=live_admin_headers)
        assert r.json()["total"] == 10

    def test_non_admin_cannot_run_benchmark(self, http_client, live_user_headers):
        r = http_client.post("/api/admin/kg/benchmark", headers=live_user_headers)
        assert r.status_code == 403
