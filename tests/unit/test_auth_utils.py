"""
Unit tests for auth_utils.py

Tests password hashing, verification, JWT creation, and decoding.
All tests are fully isolated — no database or network calls.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import time
import pytest
from datetime import timedelta
from unittest.mock import patch

import auth_utils


# ─────────────────────────────────────────────────────────────────────────────
# Password Utilities
# ─────────────────────────────────────────────────────────────────────────────

class TestPasswordHashing:
    """Tests for get_password_hash and verify_password."""

    def test_hash_is_not_plaintext(self):
        hashed = auth_utils.get_password_hash("mypassword123")
        assert hashed != "mypassword123"

    def test_hash_starts_with_bcrypt_prefix(self):
        hashed = auth_utils.get_password_hash("securepass")
        assert hashed.startswith("$2b$") or hashed.startswith("$2a$")

    def test_correct_password_verifies(self):
        password = "CorrectHorse42!"
        hashed = auth_utils.get_password_hash(password)
        assert auth_utils.verify_password(password, hashed) is True

    def test_wrong_password_fails(self):
        hashed = auth_utils.get_password_hash("original")
        assert auth_utils.verify_password("wrong", hashed) is False

    def test_empty_string_password(self):
        """Edge case: empty string can be hashed and verified."""
        hashed = auth_utils.get_password_hash("")
        assert auth_utils.verify_password("", hashed) is True
        assert auth_utils.verify_password("notempty", hashed) is False

    def test_unicode_password(self):
        password = "şifremTürkçe!42"
        hashed = auth_utils.get_password_hash(password)
        assert auth_utils.verify_password(password, hashed) is True

    def test_two_hashes_of_same_password_differ(self):
        """bcrypt uses random salt → two hashes of the same password are different."""
        h1 = auth_utils.get_password_hash("samepassword")
        h2 = auth_utils.get_password_hash("samepassword")
        assert h1 != h2
        # But both must verify correctly
        assert auth_utils.verify_password("samepassword", h1)
        assert auth_utils.verify_password("samepassword", h2)


# ─────────────────────────────────────────────────────────────────────────────
# JWT Token Creation
# ─────────────────────────────────────────────────────────────────────────────

class TestJWTCreation:
    """Tests for create_access_token."""

    def test_creates_non_empty_token(self):
        token = auth_utils.create_access_token({"sub": "user1", "role": "user"})
        assert isinstance(token, str)
        assert len(token) > 20

    def test_token_has_three_parts(self):
        """JWT format: header.payload.signature"""
        token = auth_utils.create_access_token({"sub": "user1"})
        parts = token.split(".")
        assert len(parts) == 3

    def test_custom_expiry_is_respected(self):
        """Token with 1-hour expiry must be decodable immediately."""
        token = auth_utils.create_access_token(
            {"sub": "user1"}, expires_delta=timedelta(hours=1)
        )
        payload = auth_utils.decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == "user1"

    def test_token_contains_correct_subject(self):
        token = auth_utils.create_access_token({"sub": "testuser", "role": "admin"})
        payload = auth_utils.decode_access_token(token)
        assert payload["sub"] == "testuser"
        assert payload["role"] == "admin"

    def test_expired_token_returns_none(self):
        """Token expired in the past must not decode."""
        token = auth_utils.create_access_token(
            {"sub": "expireduser"}, expires_delta=timedelta(seconds=-1)
        )
        payload = auth_utils.decode_access_token(token)
        assert payload is None


# ─────────────────────────────────────────────────────────────────────────────
# JWT Token Decoding
# ─────────────────────────────────────────────────────────────────────────────

class TestJWTDecoding:
    """Tests for decode_access_token."""

    def test_valid_token_returns_payload(self):
        token = auth_utils.create_access_token({"sub": "alice", "role": "user"})
        payload = auth_utils.decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == "alice"

    def test_tampered_token_returns_none(self):
        token = auth_utils.create_access_token({"sub": "alice"})
        tampered = token[:-5] + "XXXXX"
        assert auth_utils.decode_access_token(tampered) is None

    def test_garbage_string_returns_none(self):
        assert auth_utils.decode_access_token("not.a.token") is None

    def test_empty_string_returns_none(self):
        assert auth_utils.decode_access_token("") is None

    def test_token_with_extra_claims(self):
        """Custom claims survive the encode/decode cycle."""
        token = auth_utils.create_access_token({
            "sub": "bob",
            "role": "admin",
            "department": "IT",
        })
        payload = auth_utils.decode_access_token(token)
        assert payload["department"] == "IT"

    def test_wrong_secret_key_returns_none(self):
        """Decoding with a different secret must fail."""
        token = auth_utils.create_access_token({"sub": "charlie"})
        original_key = auth_utils.SECRET_KEY
        auth_utils.SECRET_KEY = "completely-different-secret"
        payload = auth_utils.decode_access_token(token)
        auth_utils.SECRET_KEY = original_key  # restore
        assert payload is None


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthConstants:
    def test_algorithm_is_hs256(self):
        assert auth_utils.ALGORITHM == "HS256"

    def test_expiry_is_24h(self):
        assert auth_utils.ACCESS_TOKEN_EXPIRE_MINUTES == 60 * 24
