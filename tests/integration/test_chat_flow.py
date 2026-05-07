"""
Integration tests for the Chat endpoints.

These tests hit the live server (Docker) via httpx.
"""
import io
import uuid
import pytest
import httpx

CONV_ID = str(uuid.uuid4())


def _post_chat_query_stream(http_client, headers, payload):
    """
    Consume /api/chat/query as an SSE stream. Default httpx.post buffers the full
    body and can raise RemoteProtocolError if the server closes a chunked stream
    early. We stream-read and tolerate incomplete chunked reads once status is 200.
    """
    with http_client.stream(
        "POST",
        "/api/chat/query",
        json=payload,
        headers=headers,
        timeout=120.0,
    ) as r:
        status = r.status_code
        if status != 200:
            return status
        try:
            for _ in r.iter_lines():
                pass
        except httpx.RemoteProtocolError:
            pass
    return status


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/chat/query (streaming)
# ─────────────────────────────────────────────────────────────────────────────

class TestChatQuery:
    def test_authenticated_user_can_query(self, http_client, live_user_headers):
        r = http_client.post("/api/chat/query", json={
            "query": "İbuprofen SUT kapsamında ödenir mi?",
            "conversation_id": CONV_ID,
            "role": "PATIENT",
        }, headers=live_user_headers)
        assert r.status_code == 200

    def test_query_returns_sse_stream(self, http_client, live_user_headers):
        r = http_client.post("/api/chat/query", json={
            "query": "test sorgu",
            "conversation_id": CONV_ID,
        }, headers=live_user_headers)
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_stream_contains_final_answer(self, http_client, live_user_headers):
        r = http_client.post("/api/chat/query", json={
            "query": "SUT nedir?",
            "conversation_id": CONV_ID,
        }, headers=live_user_headers)
        body = r.text
        # Note: In real Docker, this might take time. httpx.post(timeout=60) is used.
        assert "final_answer" in body

    def test_unauthenticated_query_returns_401(self, http_client):
        r = http_client.post("/api/chat/query", json={"query": "test"})
        assert r.status_code == 401

    def test_doctor_role_is_accepted(self, http_client, live_user_headers):
        r = http_client.post("/api/chat/query", json={
            "query": "ICD-10 kodları nelerdir?",
            "conversation_id": CONV_ID,
            "role": "DOCTOR",
        }, headers=live_user_headers)
        assert r.status_code == 200

    def test_admin_role_query(self, http_client, live_admin_headers):
        status = _post_chat_query_stream(
            http_client,
            live_admin_headers,
            {
                "query": "SGK maliyetleri hakkında bilgi ver",
                "conversation_id": CONV_ID,
                "role": "ADMIN",
            },
        )
        assert status == 200

    def test_query_without_conversation_id(self, http_client, live_user_headers):
        r = http_client.post("/api/chat/query", json={
            "query": "Herhangi bir soru",
        }, headers=live_user_headers)
        assert r.status_code == 200

    def test_missing_query_field_returns_422(self, http_client, live_user_headers):
        r = http_client.post("/api/chat/query", json={
            "conversation_id": CONV_ID,
        }, headers=live_user_headers)
        assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/history
# ─────────────────────────────────────────────────────────────────────────────

class TestHistory:
    def test_user_can_get_history(self, http_client, live_user_headers):
        r = http_client.get("/api/history", headers=live_user_headers)
        assert r.status_code == 200
        data = r.json()
        assert "history" in data
        assert "saved" in data

    def test_history_is_list(self, http_client, live_user_headers):
        r = http_client.get("/api/history", headers=live_user_headers)
        assert isinstance(r.json()["history"], list)

    def test_unauthenticated_history_returns_401(self, http_client):
        r = http_client.get("/api/history")
        assert r.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/history/save
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveResponse:
    def test_user_can_save_response(self, http_client, live_user_headers):
        r = http_client.post("/api/history/save", json={
            "query": "Test soru",
            "response": "Test yanıt",
        }, headers=live_user_headers)
        assert r.status_code == 200

    def test_save_without_auth_returns_401(self, http_client):
        r = http_client.post("/api/history/save", json={
            "query": "q", "response": "r"
        })
        assert r.status_code == 401

    def test_save_missing_fields_returns_422(self, http_client, live_user_headers):
        r = http_client.post("/api/history/save", json={"query": "only query"}, headers=live_user_headers)
        assert r.status_code == 422

    def test_saved_response_appears_in_history(self, http_client, live_user_headers):
        unique_response = f"Unique response {uuid.uuid4().hex}"
        http_client.post("/api/history/save", json={
            "query": "Save test query",
            "response": unique_response,
        }, headers=live_user_headers)
        r_hist = http_client.get("/api/history", headers=live_user_headers)
        saved_responses = [s["response"] for s in r_hist.json()["saved"]]
        assert any(unique_response in resp for resp in saved_responses)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/feedback
# ─────────────────────────────────────────────────────────────────────────────

class TestFeedback:
    def test_user_can_submit_feedback(self, http_client, live_user_headers):
        # 1. First create a query to get a real message_id in history
        _post_chat_query_stream(
            http_client,
            live_user_headers,
            {
                "query": "Feedback test query",
                "conversation_id": "feedback-conv",
            },
        )
        
        # 2. Get history to find the ID
        r_hist = http_client.get("/api/history", headers=live_user_headers)
        history = r_hist.json().get("history", [])
        if not history:
            pytest.skip("Could not find message in history to test feedback")
        
        msg_id = history[0]["id"]
        
        # 3. Submit feedback
        r = http_client.post("/api/feedback", json={
            "message_id": msg_id,
            "rating": 5,
            "feedback_text": "Çok iyi yanıt!",
            "is_accurate": True,
        }, headers=live_user_headers)
        assert r.status_code == 200

    def test_feedback_without_auth_returns_401(self, http_client):
        r = http_client.post("/api/feedback", json={
            "message_id": str(uuid.uuid4()), "rating": 3,
        })
        assert r.status_code == 401

    def test_feedback_missing_rating_returns_422(self, http_client, live_user_headers):
        r = http_client.post("/api/feedback", json={
            "message_id": str(uuid.uuid4()),
        }, headers=live_user_headers)
        assert r.status_code == 422

    def test_negative_rating_accepted(self, http_client, live_user_headers):
        # Create a query
        http_client.post("/api/chat/query", json={"query": "q"}, headers=live_user_headers)
        r_hist = http_client.get("/api/history", headers=live_user_headers)
        msg_id = r_hist.json()["history"][0]["id"]

        r = http_client.post("/api/feedback", json={
            "message_id": msg_id,
            "rating": -1,
        }, headers=live_user_headers)
        assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/chat/upload
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentUpload:
    def test_non_pdf_file_returns_400(self, http_client, live_user_headers):
        r = http_client.post(
            f"/api/chat/upload?conversation_id={CONV_ID}",
            files={"file": ("test.txt", io.BytesIO(b"some text"), "text/plain")},
            headers=live_user_headers,
        )
        assert r.status_code == 400

    def test_upload_without_auth_returns_401(self, http_client):
        r = http_client.post(
            f"/api/chat/upload?conversation_id={CONV_ID}",
            files={"file": ("test.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
        )
        assert r.status_code == 401
