from pathlib import Path

from fastapi.testclient import TestClient

from course_rag_api.main import app, get_store
from course_rag_api.storage import SQLiteStore


def test_api_requires_authentication_for_courses(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    app.dependency_overrides[get_store] = lambda: store
    try:
        response = TestClient(app).get("/courses")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}


def test_registered_users_only_see_their_own_courses(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    app.dependency_overrides[get_store] = lambda: store
    client = TestClient(app)
    try:
        alice = client.post(
            "/auth/register",
            json={"email": "alice@example.com", "password": "password-123"},
        )
        created = client.post("/courses", json={"name": "Alice ODE"})
        alice_course_id = created.json()["id"]
        logged_out = client.post("/auth/logout")
        bob = client.post(
            "/auth/register",
            json={"email": "bob@example.com", "password": "password-123"},
        )
        bob_courses = client.get("/courses")
        foreign_documents = client.get(f"/courses/{alice_course_id}/documents")
    finally:
        app.dependency_overrides.clear()

    assert alice.status_code == 201
    assert created.status_code == 201
    assert logged_out.status_code == 204
    assert bob.status_code == 201
    assert bob_courses.status_code == 200
    assert bob_courses.json() == {"items": []}
    assert foreign_documents.status_code == 404


def test_login_rejects_wrong_password(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    app.dependency_overrides[get_store] = lambda: store
    client = TestClient(app)
    try:
        client.post(
            "/auth/register",
            json={"email": "alice@example.com", "password": "password-123"},
        )
        response = client.post(
            "/auth/login",
            json={"email": "alice@example.com", "password": "wrong-password"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid email or password"}
