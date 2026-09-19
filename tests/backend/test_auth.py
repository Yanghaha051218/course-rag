from datetime import datetime, timedelta, timezone

from course_rag_api.auth import hash_password, session_digest, verify_password
from course_rag_api.storage import SQLiteStore


def test_password_hash_round_trip_does_not_store_plaintext() -> None:
    password = "correct horse battery staple"

    encoded = hash_password(password)

    assert password not in encoded
    assert verify_password(password, encoded)
    assert not verify_password("wrong password", encoded)


def test_store_creates_user_and_expires_sessions(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    user = store.create_user("alice@example.com", hash_password("password-123"))
    token = "session-token"
    now = datetime.now(timezone.utc)

    store.create_session(
        user.id,
        session_digest(token),
        now.isoformat(),
        (now + timedelta(hours=1)).isoformat(),
    )

    assert store.get_user_by_session(session_digest(token), now.isoformat()) == user
    assert store.get_user_by_session(
        session_digest(token), (now + timedelta(hours=2)).isoformat()
    ) is None


def test_courses_are_scoped_to_owner(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    alice = store.create_user("alice@example.com", hash_password("password-123"))
    bob = store.create_user("bob@example.com", hash_password("password-123"))
    course = store.create_course("Private ODE", owner_id=alice.id)

    assert store.list_courses_for_user(alice.id) == (course,)
    assert store.list_courses_for_user(bob.id) == ()
    assert store.get_course_for_user(course.id, alice.id) == course
    assert store.get_course_for_user(course.id, bob.id) is None
