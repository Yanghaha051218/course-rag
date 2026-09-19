import base64
import hashlib
import hmac
import secrets


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
