"""
secrets_utils.py — Symmetric encryption helpers for per-user API keys.

Uses Fernet (AES-128-CBC + HMAC) from the `cryptography` library.
The encryption key is read once from the API_KEY_ENCRYPTION_KEY env var
and cached at module level.

Operational notes:
- Generate a key with `python -c "from secrets_utils import generate_fernet_key; print(generate_fernet_key())"`
- Rotating the Fernet key invalidates all stored ciphertexts — re-encrypt them
  with MultiFernet if you ever rotate in production.
"""

import os
from cryptography.fernet import Fernet, InvalidToken  # noqa: F401  (re-exported for callers)

_FERNET: Fernet | None = None


def _get_fernet() -> Fernet:
    global _FERNET
    if _FERNET is None:
        key = os.getenv("API_KEY_ENCRYPTION_KEY")
        if not key:
            raise RuntimeError(
                "API_KEY_ENCRYPTION_KEY env var is required. "
                "Generate one with `python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"`"
            )
        _FERNET = Fernet(key.encode() if isinstance(key, str) else key)
    return _FERNET


def encrypt_api_key(plaintext: str) -> bytes:
    """Encrypt an API key. Returns Fernet token bytes — store as BYTEA in Postgres."""
    return _get_fernet().encrypt(plaintext.encode("utf-8"))


def decrypt_api_key(ciphertext: bytes) -> str:
    """Decrypt a Fernet token back to the plaintext API key string."""
    return _get_fernet().decrypt(bytes(ciphertext)).decode("utf-8")


def make_key_hint(plaintext: str) -> str:
    """Return last 4 chars (or full if shorter) for UI display."""
    if not plaintext:
        return ""
    return plaintext[-4:] if len(plaintext) > 4 else plaintext


def generate_fernet_key() -> str:
    """For ops: generate a new key to put in env."""
    return Fernet.generate_key().decode()
