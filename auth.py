"""
auth.py — Simple JWT authentication.
Uses PyJWT (pure Python) and hashlib for password hashing.
User store: users.json (flat file, easy to swap for a DB later).
"""
import json
import hashlib
import hmac
import os
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
USERS_FILE = Path("users.json")
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "sleep-ai-dev-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # 8 hours


# ── Lazy-import PyJWT so the app still starts if it's not installed ──────────
def _get_jwt():
    try:
        import jwt
        return jwt
    except ImportError:
        logger.warning("PyJWT not installed — auth will be disabled")
        return None


# ── User store ────────────────────────────────────────────────────────────────
def _load_users() -> dict:
    if USERS_FILE.exists():
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_users(users: dict) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


# ── Password hashing ──────────────────────────────────────────────────────────
def _hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Return (hash_hex, salt). Generate a new salt if not provided."""
    if salt is None:
        salt = os.urandom(16).hex()
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return key.hex(), salt


def verify_password(plain: str, stored_hash: str, salt: str) -> bool:
    computed, _ = _hash_password(plain, salt)
    return hmac.compare_digest(computed, stored_hash)


# ── User CRUD ─────────────────────────────────────────────────────────────────
def register_user(username: str, password: str, email: str) -> bool:
    """Register a new user. Returns False if username or email already taken."""
    users = _load_users()
    if username in users:
        return False
    # Check email uniqueness
    for u, data in users.items():
        if data.get("email") == email:
            return False
            
    pw_hash, salt = _hash_password(password)
    users[username] = {"hash": pw_hash, "salt": salt, "email": email}
    _save_users(users)
    logger.info("New user registered: %s (%s)", username, email)
    return True


def authenticate_user(username: str, password: str) -> bool:
    """Return True if credentials are valid."""
    users = _load_users()
    if username not in users:
        return False
    record = users[username]
    return verify_password(password, record["hash"], record["salt"])


# ── JWT ───────────────────────────────────────────────────────────────────────
def create_access_token(username: str) -> Optional[str]:
    jwt = _get_jwt()
    if jwt is None:
        return None
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[str]:
    """Return username if token is valid, else None."""
    jwt = _get_jwt()
    if jwt is None:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except Exception as exc:
        logger.debug("Token verification failed: %s", exc)
        return None


# ── FastAPI dependency helper ─────────────────────────────────────────────────
def get_current_user(authorization: Optional[str] = None) -> Optional[str]:
    """
    Extract and validate the Bearer token from an Authorization header value.
    Returns the username or None if auth fails / not provided.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    return verify_token(token)
