# Authentication: JWT tokens + bcrypt password hashing.

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import jwt
import bcrypt
from fastapi import HTTPException, Header
from sqlalchemy import text

import config
import memory as memory_module

logger = logging.getLogger(__name__)

SECRET_KEY = getattr(config, 'JWT_SECRET_KEY', None)
ALGORITHM = getattr(config, 'JWT_ALGORITHM', 'HS256')
ACCESS_TOKEN_EXPIRE_DAYS = getattr(config, 'JWT_EXPIRE_DAYS', 7)

if not SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY must be set in config.py or environment")

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


#JWT Token Lifecycle

def create_access_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    
    payload = {
        "sub": user_id,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp())
    }
    
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    logger.info(f"[AUTH] Token created for user '{user_id}', expires at {expire.isoformat()}")
    return token


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        logger.debug(f"[AUTH] Token decoded successfully: sub={payload.get('sub')}")
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("[AUTH] Token rejected: EXPIRED")
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidSignatureError:
        logger.error("[AUTH] Token rejected: INVALID SIGNATURE (wrong SECRET_KEY?)")
        raise HTTPException(status_code=401, detail="Invalid token signature")
    except jwt.DecodeError:
        logger.error("[AUTH] Token rejected: MALFORMED (not a valid JWT)")
        raise HTTPException(status_code=401, detail="Malformed token")
    except jwt.InvalidTokenError as e:
        logger.error(f"[AUTH] Token rejected: {type(e).__name__}: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")


def verify_token(token: str) -> str:
    """Decode token and return the user_id (sub claim)."""
    payload = decode_token(token)
    user_id = payload.get("sub")
    if not user_id:
        logger.error("[AUTH] Token has no 'sub' claim")
        raise HTTPException(status_code=401, detail="Token missing user identity")
    return user_id


#FastAPI Dependency

def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    logger.debug(f"[AUTH] Received Authorization header: {authorization[:30] + '...' if authorization else 'None'}")
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    if not authorization.startswith("Bearer "):
        logger.warning(f"[AUTH] Bad header format: {authorization[:30]}")
        raise HTTPException(status_code=401, detail="Authorization header must start with 'Bearer '")
    
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty token")
    
    return verify_token(token)


#SQL Server User Management

def ensure_password_column():
    with memory_module.engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'users' AND COLUMN_NAME = 'password_hash'
        """))
        has_column = result.scalar() > 0
    
    if not has_column:
        raise RuntimeError(
            "The 'users' table is missing the 'password_hash' column.\n"
            "Run this in SSMS:\n"
            "ALTER TABLE users ADD password_hash NVARCHAR(255) NULL;"
        )


def register_user(user_id: str, password: str) -> bool:

    ensure_password_column()
    
    # Check if user already exists
    with memory_module.engine.connect() as conn:
        result = conn.execute(
            text("SELECT 1 FROM users WHERE user_id = :user_id"),
            {"user_id": user_id}
        )
        if result.fetchone():
            logger.info(f"[AUTH] Registration failed: user '{user_id}' already exists")
            return False
    
    # Hash password and insert user
    hashed = hash_password(password)
    with memory_module.engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO users (user_id, name, password_hash)
                VALUES (:user_id, :name, :password_hash)
            """),
            {"user_id": user_id, "name": user_id, "password_hash": hashed}
        )
    
    logger.info(f"[AUTH] User '{user_id}' registered successfully")
    return True


def authenticate_user(user_id: str, password: str) -> Optional[str]:
    ensure_password_column()
    
    with memory_module.engine.connect() as conn:
        result = conn.execute(
            text("SELECT password_hash FROM users WHERE user_id = :user_id"),
            {"user_id": user_id}
        )
        row = result.fetchone()
    
    if not row or not row[0]:
        logger.warning(f"[AUTH] Login failed for '{user_id}': user not found or no password")
        return None
    
    stored_hash = row[0]
    if not verify_password(password, stored_hash):
        logger.warning(f"[AUTH] Login failed for '{user_id}': wrong password")
        return None
    
    token = create_access_token(user_id)
    logger.info(f"[AUTH] User '{user_id}' logged in, token issued")
    return token