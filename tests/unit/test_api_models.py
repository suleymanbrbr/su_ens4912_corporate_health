"""
Unit tests for Pydantic models used in api_server.py

Since api_server.py requires FastAPI (only available inside Docker),
we redeclare the models here identically — this tests the same validation
logic that the production models enforce.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import pytest
from typing import Optional
from pydantic import BaseModel, EmailStr, ValidationError


# ── Redeclare models exactly as in api_server.py ──────────────────────────────

class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: Optional[str] = "user"


class ChatQuery(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    role: Optional[str] = "PATIENT"


class PasswordChange(BaseModel):
    old_password: str
    new_password: str


class SaveResponse(BaseModel):
    query: str
    response: str


class RoleUpdate(BaseModel):
    role: str


class AnnouncementCreate(BaseModel):
    message: str


class FeedbackCreate(BaseModel):
    message_id: str
    rating: int
    feedback_text: str = ""
    is_accurate: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# UserRegister
# ─────────────────────────────────────────────────────────────────────────────

class TestUserRegisterModel:
    def test_valid_registration(self):
        user = UserRegister(username="alice", email="alice@example.com", password="pass123")
        assert user.username == "alice"

    def test_default_role_is_user(self):
        user = UserRegister(username="bob", email="bob@example.com", password="pass")
        assert user.role == "user"

    def test_invalid_email_raises_validation_error(self):
        with pytest.raises(ValidationError):
            UserRegister(username="bad", email="not-an-email", password="pass")

    def test_missing_username_raises_error(self):
        with pytest.raises(ValidationError):
            UserRegister(email="e@test.com", password="pass")

    def test_missing_password_raises_error(self):
        with pytest.raises(ValidationError):
            UserRegister(username="test", email="e@test.com")

    def test_custom_role_accepted(self):
        user = UserRegister(username="admin", email="a@test.com", password="pass", role="admin")
        assert user.role == "admin"

    def test_username_stored_correctly(self):
        user = UserRegister(username="suleymanberber", email="s@example.com", password="p")
        assert user.username == "suleymanberber"


# ─────────────────────────────────────────────────────────────────────────────
# ChatQuery
# ─────────────────────────────────────────────────────────────────────────────

class TestChatQueryModel:
    def test_valid_query(self):
        q = ChatQuery(query="İbuprofen nedir?")
        assert q.query == "İbuprofen nedir?"

    def test_default_role_is_patient(self):
        q = ChatQuery(query="test")
        assert q.role == "PATIENT"

    def test_optional_conversation_id_defaults_to_none(self):
        q = ChatQuery(query="test")
        assert q.conversation_id is None

    def test_custom_conversation_id(self):
        q = ChatQuery(query="test", conversation_id="abc-123")
        assert q.conversation_id == "abc-123"

    def test_doctor_role(self):
        q = ChatQuery(query="test", role="DOCTOR")
        assert q.role == "DOCTOR"

    def test_admin_role(self):
        q = ChatQuery(query="test", role="ADMIN")
        assert q.role == "ADMIN"

    def test_empty_query_accepted(self):
        q = ChatQuery(query="")
        assert q.query == ""

    def test_missing_query_raises(self):
        with pytest.raises(ValidationError):
            ChatQuery()

    def test_unicode_query(self):
        q = ChatQuery(query="Türkçe karakter testi: çğşüöı")
        assert "Türkçe" in q.query


# ─────────────────────────────────────────────────────────────────────────────
# FeedbackCreate
# ─────────────────────────────────────────────────────────────────────────────

class TestFeedbackCreateModel:
    def test_valid_feedback(self):
        fb = FeedbackCreate(message_id="msg-1", rating=5, feedback_text="Çok iyi!", is_accurate=True)
        assert fb.rating == 5

    def test_default_is_accurate_true(self):
        fb = FeedbackCreate(message_id="msg-1", rating=3)
        assert fb.is_accurate is True

    def test_default_feedback_text_empty(self):
        fb = FeedbackCreate(message_id="msg-1", rating=1)
        assert fb.feedback_text == ""

    def test_negative_rating_accepted(self):
        fb = FeedbackCreate(message_id="x", rating=-1)
        assert fb.rating == -1

    def test_missing_rating_raises_error(self):
        with pytest.raises(ValidationError):
            FeedbackCreate(message_id="x")

    def test_missing_message_id_raises(self):
        with pytest.raises(ValidationError):
            FeedbackCreate(rating=3)

    def test_is_accurate_false(self):
        fb = FeedbackCreate(message_id="x", rating=1, is_accurate=False)
        assert fb.is_accurate is False


# ─────────────────────────────────────────────────────────────────────────────
# PasswordChange
# ─────────────────────────────────────────────────────────────────────────────

class TestPasswordChangeModel:
    def test_valid_password_change(self):
        pc = PasswordChange(old_password="old", new_password="new")
        assert pc.new_password == "new"

    def test_missing_old_password_raises(self):
        with pytest.raises(ValidationError):
            PasswordChange(new_password="new")

    def test_missing_new_password_raises(self):
        with pytest.raises(ValidationError):
            PasswordChange(old_password="old")

    def test_passwords_can_be_same(self):
        pc = PasswordChange(old_password="same", new_password="same")
        assert pc.old_password == pc.new_password


# ─────────────────────────────────────────────────────────────────────────────
# RoleUpdate
# ─────────────────────────────────────────────────────────────────────────────

class TestRoleUpdateModel:
    def test_admin_role_valid(self):
        ru = RoleUpdate(role="admin")
        assert ru.role == "admin"

    def test_user_role_valid(self):
        ru = RoleUpdate(role="user")
        assert ru.role == "user"

    def test_missing_role_raises(self):
        with pytest.raises(ValidationError):
            RoleUpdate()

    def test_role_stored_as_string(self):
        ru = RoleUpdate(role="admin")
        assert isinstance(ru.role, str)


# ─────────────────────────────────────────────────────────────────────────────
# AnnouncementCreate
# ─────────────────────────────────────────────────────────────────────────────

class TestAnnouncementCreateModel:
    def test_valid_announcement(self):
        ann = AnnouncementCreate(message="Bakım yapılacak.")
        assert ann.message == "Bakım yapılacak."

    def test_missing_message_raises(self):
        with pytest.raises(ValidationError):
            AnnouncementCreate()

    def test_turkish_message(self):
        ann = AnnouncementCreate(message="Sistem şu anda bakımda. Lütfen bekleyiniz.")
        assert "bakımda" in ann.message


# ─────────────────────────────────────────────────────────────────────────────
# SaveResponse
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveResponseModel:
    def test_valid_save_response(self):
        sr = SaveResponse(query="Sorum bu", response="Yanıt bu")
        assert sr.query == "Sorum bu"

    def test_missing_query_raises(self):
        with pytest.raises(ValidationError):
            SaveResponse(response="r")

    def test_missing_response_raises(self):
        with pytest.raises(ValidationError):
            SaveResponse(query="q")
