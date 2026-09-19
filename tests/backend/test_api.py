from pathlib import Path

from fastapi.testclient import TestClient

from course_rag_api.generation import Citation, FinalResponse, FinalStatus
from course_rag_api.auth import hash_password
from course_rag_api.main import (
    app,
    get_answer_service,
    get_current_user,
    get_indexer,
    get_retriever,
    get_store,
)
from course_rag_api.models import IndexingSummary, RetrievedChunk
from course_rag_api.storage import SQLiteStore


def _authenticate_test_user(store: SQLiteStore):
    user = store.create_user("test@example.com", hash_password("password-123"))
    app.dependency_overrides[get_current_user] = lambda: user
    return user


def test_create_and_list_courses(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    _authenticate_test_user(store)
    app.dependency_overrides[get_store] = lambda: store
    client = TestClient(app)
    try:
        created = client.post("/courses", json={"name": "  Differential Equations  "})
        listed = client.get("/courses")
    finally:
        app.dependency_overrides.clear()

    assert created.status_code == 201
    assert created.json()["name"] == "Differential Equations"
    assert listed.status_code == 200
    assert listed.json() == {"items": [created.json()]}


def test_create_course_rejects_blank_name(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    _authenticate_test_user(store)
    app.dependency_overrides[get_store] = lambda: store
    try:
        response = TestClient(app).post("/courses", json={"name": "   "})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_uploads_and_lists_document(tmp_path: Path, monkeypatch) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    user = _authenticate_test_user(store)
    course = store.create_course("ODE", owner_id=user.id)

    class Indexer:
        def index_document(self, document_id: str) -> IndexingSummary:
            return IndexingSummary(course.id, document_id, 1, 1, "test")

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_indexer] = lambda: Indexer()
    monkeypatch.setattr("course_rag_api.main.settings.upload_path", tmp_path / "uploads")
    client = TestClient(app)
    try:
        uploaded = client.post(
            f"/courses/{course.id}/documents?filename=lesson.txt",
            content=b"The coefficient is 7.25.",
            headers={"content-type": "application/octet-stream"},
        )
        listed = client.get(f"/courses/{course.id}/documents")
    finally:
        app.dependency_overrides.clear()

    assert uploaded.status_code == 201
    assert uploaded.json()["filename"] == "lesson.txt"
    assert listed.json()["items"][0]["id"] == uploaded.json()["id"]
    assert listed.json()["items"][0]["size_bytes"] == len(b"The coefficient is 7.25.")


def test_deletes_document_and_removes_uploaded_file(tmp_path: Path, monkeypatch) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    user = _authenticate_test_user(store)
    course = store.create_course("ODE", owner_id=user.id)

    class Indexer:
        def index_document(self, document_id: str) -> IndexingSummary:
            return IndexingSummary(course.id, document_id, 1, 1, "test")

        def delete_document(self, course_id: str, document_id: str):
            return store.delete_document(course_id, document_id)

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_indexer] = lambda: Indexer()
    monkeypatch.setattr("course_rag_api.main.settings.upload_path", tmp_path / "uploads")
    client = TestClient(app)
    try:
        uploaded = client.post(
            f"/courses/{course.id}/documents?filename=lesson.txt",
            content=b"The coefficient is 7.25.",
            headers={"content-type": "application/octet-stream"},
        )
        uploaded_path = next((tmp_path / "uploads").rglob("lesson.txt"))
        deleted = client.delete(
            f"/courses/{course.id}/documents/{uploaded.json()['id']}"
        )
        listed = client.get(f"/courses/{course.id}/documents")
    finally:
        app.dependency_overrides.clear()

    assert uploaded.status_code == 201
    assert deleted.status_code == 204
    assert not uploaded_path.exists()
    assert listed.json() == {"items": []}


def test_upload_rejects_course_quota(tmp_path: Path, monkeypatch) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    user = _authenticate_test_user(store)
    course = store.create_course("ODE", owner_id=user.id)

    class Indexer:
        def index_document(self, document_id: str) -> IndexingSummary:
            return IndexingSummary(course.id, document_id, 1, 1, "test")

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_indexer] = lambda: Indexer()
    monkeypatch.setattr("course_rag_api.main.settings.upload_path", tmp_path / "uploads")
    monkeypatch.setattr("course_rag_api.main.settings.max_course_bytes", 5)
    try:
        response = TestClient(app).post(
            f"/courses/{course.id}/documents?filename=lesson.txt",
            content=b"123456",
            headers={"content-type": "application/octet-stream"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 413


def test_returns_answered_or_abstained_shape(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    user = _authenticate_test_user(store)
    course = store.create_course("ODE", owner_id=user.id)

    class Answers:
        def answer_question(self, **_: object) -> FinalResponse:
            return FinalResponse(
                FinalStatus.ANSWERED,
                "The coefficient is 7.25.",
                (Citation("chunk-1", "lesson.txt", "section", 1, 1),),
                None,
            )

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_answer_service] = lambda: Answers()
    try:
        response = TestClient(app).post(
            f"/courses/{course.id}/questions", json={"question": "What is it?"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "status": "ANSWERED",
        "answer": "The coefficient is 7.25.",
        "citations": [{"chunk_id": "chunk-1", "filename": "lesson.txt", "source_type": "section", "source_start": 1, "source_end": 1}],
        "reason": None,
    }


def test_returns_ranked_evidence_without_generation(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    user = _authenticate_test_user(store)
    course = store.create_course("ODE", owner_id=user.id)

    class EvidenceRetriever:
        def retrieve(self, **_: object) -> tuple[RetrievedChunk, ...]:
            return (
                RetrievedChunk(
                    "chunk-1",
                    course.id,
                    "document-1",
                    "lesson.md",
                    "The coefficient is 7.25.",
                    0.91,
                    0,
                    "section",
                    2,
                    2,
                ),
            )

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_retriever] = lambda: EvidenceRetriever()
    try:
        response = TestClient(app).post(
            f"/courses/{course.id}/evidence", json={"question": "What is the coefficient?"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "question": "What is the coefficient?",
        "items": [
            {
                "rank": 1,
                "chunk_id": "chunk-1",
                "filename": "lesson.md",
                "text": "The coefficient is 7.25.",
                "score": 0.91,
                "source_type": "section",
                "source_start": 2,
                "source_end": 2,
            }
        ],
    }
