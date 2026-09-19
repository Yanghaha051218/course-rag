import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request, status

from course_rag_api.models import User
from course_rag_api.storage import SQLiteStore


_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1


def hash_password(password: str) -> str:
    if not password or len(password) > 256:
        raise ValueError("password must contain 1-256 characters")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
    )
    encode = base64.urlsafe_b64encode
    return "$".join(
        (
            "scrypt",
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            encode(salt).decode("ascii"),
            encode(digest).decode("ascii"),
        )
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, n, r, p, salt_text, digest_text = encoded.split("$", 5)
        if scheme != "scrypt":
            return False
        cost = int(n)
        if cost != _SCRYPT_N or int(r) != _SCRYPT_R or int(p) != _SCRYPT_P:
            return False
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=cost,
            r=int(r),
            p=int(p),
            maxmem=32 * 1024 * 1024,
        )
    except (ValueError, TypeError, UnicodeDecodeError):
        return False
    return hmac.compare_digest(actual, expected)


def session_digest(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def issue_session(store: SQLiteStore, user_id: str, ttl_seconds: int) -> str:
    now = datetime.now(timezone.utc)
    token = new_session_token()
    store.create_session(
        user_id,
        session_digest(token),
        now.isoformat(),
        (now + timedelta(seconds=ttl_seconds)).isoformat(),
    )
    return token


def authenticate_request(
    request: Request, store: SQLiteStore, cookie_name: str
) -> User:
    token = request_session_token(request, cookie_name)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = store.get_user_by_session(
        session_digest(token), datetime.now(timezone.utc).isoformat()
    )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def request_session_token(request: Request, cookie_name: str) -> str:
    authorization = request.headers.get("authorization", "")
    token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    return token or request.cookies.get(cookie_name, "")
