import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import JWTError, jwt
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
# JWT_SECRET_KEY is REQUIRED — refuse to start without it rather than silently
# falling back to a well-known default that would break security across deploys.
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET_KEY env var is required. "
        "Generate one with `python -c \"import secrets; print(secrets.token_urlsafe(64))\"`"
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# bcrypt has a hard 72-byte input limit; recent bcrypt versions (>=4.1) raise
# ValueError instead of silently truncating. Pre-truncate so long passwords
# still register rather than crashing with a 500. We use the `bcrypt` package
# directly because passlib 1.7.4 + bcrypt 4.1+ are incompatible (passlib's
# internal `detect_wrap_bug` probe overflows the 72-byte limit at init time).
_BCRYPT_MAX_BYTES = 72
_BCRYPT_ROUNDS = 12


def _to_bcrypt_bytes(password: str) -> bytes:
    """Encode password to bytes and truncate to <=72 utf-8 bytes."""
    if not isinstance(password, str):
        password = str(password or "")
    encoded = password.encode("utf-8")
    if len(encoded) > _BCRYPT_MAX_BYTES:
        encoded = encoded[:_BCRYPT_MAX_BYTES]
    return encoded


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a bcrypt hash. Returns False on any failure."""
    if not hashed_password:
        return False
    try:
        return bcrypt.checkpw(
            _to_bcrypt_bytes(plain_password),
            hashed_password.encode("utf-8") if isinstance(hashed_password, str) else hashed_password,
        )
    except (ValueError, TypeError) as e:
        logger.warning(f"verify_password failed [{type(e).__name__}]: {e}")
        return False


def get_password_hash(password: str) -> str:
    """Generate a bcrypt hash for a password. Returns the hash as a string."""
    salt = bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)
    hashed = bcrypt.hashpw(_to_bcrypt_bytes(password), salt)
    return hashed.decode("utf-8")


def _utcnow() -> datetime:
    """Timezone-aware UTC `now` — replaces deprecated datetime.utcnow()."""
    return datetime.now(timezone.utc)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = _utcnow() + expires_delta
    else:
        # Always use the configured default (24h) when no explicit expiry is given.
        # This replaces the prior 15-minute hardcoded fallback that was inconsistent
        # with ACCESS_TOKEN_EXPIRE_MINUTES.
        expire = _utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
