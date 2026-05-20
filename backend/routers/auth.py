# routers/auth.py — authentication and account endpoints.
#
# Routes:
#   POST /api/auth/register
#   POST /api/auth/login
#   GET  /api/auth/me
#   PUT  /api/auth/password
#
# Path-prefix is kept identical to the legacy api_server.py wiring so the
# frontend continues to function without changes.

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr

from auth_utils import (
    create_access_token,
    get_password_hash,
    verify_password,
)
from deps import (
    MIN_PASSWORD_LENGTH,
    VALID_USER_ROLES,
    db_execute,
    db_session,
    get_current_user,
    log_audit,
)


router = APIRouter(prefix="/api/auth", tags=["auth"])


# --- Pydantic models ---
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: Optional[str] = "user"


class Token(BaseModel):
    access_token: str
    token_type: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str
    is_approved: int


class PasswordChange(BaseModel):
    old_password: str
    new_password: str


# --- Endpoints ---
@router.post("/register", response_model=UserResponse)
async def register(user: UserRegister):
    # Server-side password policy. The frontend also enforces this, but never
    # trust the client.  We accept any 8+ char password (bcrypt then caps at
    # 72 utf-8 bytes); existing accounts are not affected.
    if len(user.password or "") < MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Şifre en az {MIN_PASSWORD_LENGTH} karakter olmalıdır.",
        )

    # Basic username hygiene — keep API response shape but avoid empty / whitespace.
    if not user.username or not user.username.strip():
        raise HTTPException(status_code=400, detail="Kullanıcı adı zorunludur.")

    with db_session() as conn:
        cur = db_execute(conn, "SELECT id FROM users WHERE username = %s OR email = %s", (user.username, user.email))
        existing = cur.fetchone()
        cur.close()
        if existing:
            raise HTTPException(status_code=400, detail="User already exists")

        user_id = str(uuid.uuid4())
        hashed_pwd = get_password_hash(user.password)
        role = user.role if user.role in VALID_USER_ROLES else "user"

        cur = db_execute(conn, "SELECT COUNT(*) FROM users")
        user_count = cur.fetchone()[0]
        cur.close()
        is_approved = 1 if user_count == 0 else 0

        db_execute(
            conn,
            "INSERT INTO users (id, username, email, hashed_password, role, is_approved) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (user_id, user.username, user.email, hashed_pwd, role, is_approved),
        )
        log_audit(
            conn,
            "register",
            user_id=user_id,
            entity_type="user",
            entity_id=user_id,
            details={"username": user.username, "roles": role},
        )
        conn.commit()
        return {
            "id": user_id,
            "username": user.username,
            "email": user.email,
            "role": role,
            "is_approved": is_approved,
        }


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    with db_session() as conn:
        cur = db_execute(
            conn,
            "SELECT * FROM users WHERE username = %s OR email = %s",
            (form_data.username, form_data.username),
        )
        user = cur.fetchone()
        cur.close()

        if not user or not verify_password(form_data.password, user["hashed_password"]):
            raise HTTPException(status_code=401, detail="Incorrect credentials")

        if user["is_approved"] == 0:
            raise HTTPException(status_code=403, detail="Hesabınız henüz onaylanmamıştır.")

        log_audit(conn, "login", user_id=user["id"])
        conn.commit()
        username = user["username"]
        role = user["role"]

    access_token = create_access_token(data={"sub": username, "role": role})
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
async def me(current_user: dict = Depends(get_current_user)):
    return current_user


@router.put("/password")
async def change_password(data: PasswordChange, current_user: dict = Depends(get_current_user)):
    # Enforce the same min-length policy on password change.
    if len(data.new_password or "") < MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Yeni şifre en az {MIN_PASSWORD_LENGTH} karakter olmalıdır.",
        )

    with db_session() as conn:
        cur = db_execute(conn, "SELECT hashed_password FROM users WHERE id = %s", (current_user["id"],))
        user = cur.fetchone()
        cur.close()
        if not user or not verify_password(data.old_password, user["hashed_password"]):
            raise HTTPException(status_code=400, detail="Mevcut şifre yanlış")

        new_hashed = get_password_hash(data.new_password)
        db_execute(conn, "UPDATE users SET hashed_password = %s WHERE id = %s", (new_hashed, current_user["id"]))
        log_audit(conn, "password_change", user_id=current_user["id"])
        conn.commit()
    return {"message": "Şifre başarıyla güncellendi."}
