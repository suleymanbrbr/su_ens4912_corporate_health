"""
Locust Stress / Load Test File for SUT Corporate Health API.

Scenarios:
  - LoginUser        : Hammers the auth endpoints (register + login + /me)
  - ChatUser         : Simulates full chat queries
  - KGUser           : Exercises Knowledge Graph endpoints
  - AdminUser        : Exercises admin analytics and user management
  - MixedUser        : Realistic mix of all operations

Usage:
  # Open browser UI at localhost:8089
  locust -f locustfile.py --host=http://localhost:8000

  # Headless 60s run with 50 users
  locust -f locustfile.py --host=http://localhost:8000 \\
    --headless -u 50 -r 10 -t 60s --html=stress_report.html

  # High-load scenario: 100 users
  locust -f locustfile.py --host=http://localhost:8000 \\
    --headless -u 100 -r 20 -t 120s --html=stress_report_100.html
"""
import uuid
import json
import random
from locust import HttpUser, task, between, events

# ── Global tokens cached after first login ────────────────────────────────────
_ADMIN_TOKEN: str = ""
_USER_TOKEN: str = ""

# Preset credentials — must exist in the running DB
ADMIN_CREDENTIALS = {"username": "admin", "password": "Admin@1234!"}
USER_CREDENTIALS  = {"username": "testuser", "password": "User@1234!"}

# ── Sample SUT queries for realistic load ────────────────────────────────────
SUT_QUERIES = [
    "İbuprofen SUT kapsamında ödenir mi?",
    "Kanser tedavisinde hangi raporlar gerekir?",
    "Diyabet ilaçları için endikasyon şartları nelerdir?",
    "Fizik tedavi seans limiti kaçtır?",
    "İşitme cihazı için hangi uzman raporu gerekiyor?",
    "Çocuklarda büyüme hormonu koşulları nelerdir?",
    "MS hastalığı için biyolojik ilaç şartları?",
    "Kronik böbrek hastasına hangi ilaçlar ödeniyor?",
    "Ortopedik protez için ne gerekiyor?",
    "SUT madde 4.2.63 nedir?",
]


# ─────────────────────────────────────────────────────────────────────────────
# LoginUser — pure authentication load
# ─────────────────────────────────────────────────────────────────────────────

class LoginUser(HttpUser):
    """
    Simulates users who repeatedly hit auth endpoints.
    Target: auth endpoint stability under concurrent traffic.
    """
    wait_time = between(0.5, 2)
    weight = 3

    def on_start(self):
        self.username = f"stress_{uuid.uuid4().hex[:8]}"
        self.email = f"{self.username}@stress.test"
        self.password = "Stress@123"
        self._register()
        self._login()

    def _register(self):
        with self.client.post("/api/auth/register", json={
            "username": self.username,
            "email": self.email,
            "password": self.password,
        }, catch_response=True, name="POST /api/auth/register") as r:
            if r.status_code not in (200, 400):
                r.failure(f"Register failed: {r.status_code}")

    def _login(self):
        with self.client.post(
            "/api/auth/login",
            data={"username": self.username, "password": self.password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            catch_response=True,
            name="POST /api/auth/login",
        ) as r:
            if r.status_code == 200:
                self.token = r.json().get("access_token", "")
            elif r.status_code == 403:
                # Unapproved — still a valid state
                self.token = ""
                r.success()
            else:
                r.failure(f"Login failed: {r.status_code}")
                self.token = ""

    @task(3)
    def login(self):
        self._login()

    @task(1)
    def get_me(self):
        if self.token:
            self.client.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {self.token}"},
                name="GET /api/auth/me",
            )


# ─────────────────────────────────────────────────────────────────────────────
# ChatUser — chat query load
# ─────────────────────────────────────────────────────────────────────────────

class ChatUser(HttpUser):
    """
    Simulates approved users sending SUT queries.
    This is the most important load scenario.
    """
    wait_time = between(2, 8)  # chat is inherently slower
    weight = 2

    def on_start(self):
        # Use the shared pre-approved token
        self.token = _get_user_token(self.client)
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.conv_id = str(uuid.uuid4())

    @task(5)
    def chat_query(self):
        query = random.choice(SUT_QUERIES)
        with self.client.post("/api/chat/query", json={
            "query": query,
            "conversation_id": self.conv_id,
            "role": random.choice(["PATIENT", "DOCTOR", "ADMIN"]),
        }, headers=self.headers, stream=True,
           catch_response=True, name="POST /api/chat/query") as r:
            if r.status_code == 200:
                r.success()
            elif r.status_code == 401:
                r.failure("Unauthorized — token expired")
            else:
                r.failure(f"Unexpected status: {r.status_code}")

    @task(2)
    def get_history(self):
        self.client.get("/api/history", headers=self.headers, name="GET /api/history")

    @task(1)
    def save_response(self):
        self.client.post("/api/history/save", json={
            "query": "Test soru",
            "response": "Test yanıt kaydedildi.",
        }, headers=self.headers, name="POST /api/history/save")


# ─────────────────────────────────────────────────────────────────────────────
# KGUser — Knowledge Graph load
# ─────────────────────────────────────────────────────────────────────────────

class KGUser(HttpUser):
    """
    Simulates users browsing the Knowledge Graph.
    """
    wait_time = between(1, 4)
    weight = 2

    def on_start(self):
        self.token = _get_user_token(self.client)
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @task(3)
    def get_kg_nodes(self):
        terms = ["aspirin", "ibuprofen", "diyabet", "kanser", "böbrek"]
        q = random.choice(terms)
        self.client.get(f"/api/kg/nodes?q={q}&limit=10",
                       headers=self.headers, name="GET /api/kg/nodes")

    @task(2)
    def get_kg_stats(self):
        self.client.get("/api/kg/stats", headers=self.headers, name="GET /api/kg/stats")

    @task(1)
    def find_kg_path(self):
        self.client.get("/api/kg/path?from_id=node1&to_id=node2&max_hops=3",
                       headers=self.headers, name="GET /api/kg/path")


# ─────────────────────────────────────────────────────────────────────────────
# AdminUser — admin panel load
# ─────────────────────────────────────────────────────────────────────────────

class AdminUser(HttpUser):
    """
    Simulates admin checking analytics and user lists.
    """
    wait_time = between(3, 10)
    weight = 1

    def on_start(self):
        self.token = _get_admin_token(self.client)
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @task(3)
    def get_analytics(self):
        self.client.get("/api/admin/analytics", headers=self.headers,
                       name="GET /api/admin/analytics")

    @task(2)
    def get_user_list(self):
        self.client.get("/api/admin/users", headers=self.headers,
                       name="GET /api/admin/users")

    @task(2)
    def get_system_metrics(self):
        self.client.get("/api/admin/system", headers=self.headers,
                       name="GET /api/admin/system")

    @task(1)
    def get_audit_logs(self):
        self.client.get("/api/admin/audit-logs?limit=10", headers=self.headers,
                       name="GET /api/admin/audit-logs")

    @task(1)
    def get_active_announcement(self):
        self.client.get("/api/announcements", headers=self.headers,
                       name="GET /api/announcements")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: get/cache shared tokens
# ─────────────────────────────────────────────────────────────────────────────

def _get_admin_token(client) -> str:
    global _ADMIN_TOKEN
    if not _ADMIN_TOKEN:
        r = client.post(
            "/api/auth/login",
            data=ADMIN_CREDENTIALS,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            name="[setup] admin login",
        )
        if r.status_code == 200:
            _ADMIN_TOKEN = r.json().get("access_token", "")
    return _ADMIN_TOKEN


def _get_user_token(client) -> str:
    global _USER_TOKEN
    if not _USER_TOKEN:
        r = client.post(
            "/api/auth/login",
            data=USER_CREDENTIALS,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            name="[setup] user login",
        )
        if r.status_code == 200:
            _USER_TOKEN = r.json().get("access_token", "")
    return _USER_TOKEN
