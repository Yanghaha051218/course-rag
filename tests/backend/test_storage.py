import sqlite3
from pathlib import Path

import pytest

from course_rag_api.models import Chunk, Document
from course_rag_api.storage import SQLiteStore


CREATED_AT = "2026-01-01T00:00:00+00:00"


def _document(document_id: str, course_id: str, checksum: str) -> Document:
    return Document(
        id=document_id,
        course_id=course_id,
        filename=f"{document_id}.txt",
        file_type="txt",
        source_path=f"/tmp/{document_id}.txt",
        checksum=checksum,
        page_or_unit_count=1,
        created_at=CREATED_AT,
    )


def _chunk(chunk_id: str, document_id: str, course_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        course_id=course_id,
        document_id=document_id,
        text=text,
        chunk_index=0,
        source_type="section",
        source_start=1,
        source_end=1,
        created_at=CREATED_AT,
    )


def test_store_persists_courses_documents_chunks_and_isolates_courses(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    course_a = store.create_course("Course A")
    course_b = store.create_course("Course B")
    document_a = _document("document-a", course_a.id, "a" * 64)
    document_b = _document("document-b", course_b.id, "b" * 64)
    store.persist_document(document_a, (_chunk("chunk-a", "document-a", course_a.id, "A"),))
    store.persist_document(document_b, (_chunk("chunk-b", "document-b", course_b.id, "B"),))

    assert store.get_course(course_a.id) == course_a
    assert store.list_documents(course_a.id) == (document_a,)
    assert [chunk.text for chunk in store.list_chunks(course_a.id)] == ["A"]
    assert [chunk.text for chunk in store.list_chunks(course_b.id)] == ["B"]


def test_store_foreign_keys_reject_cross_course_chunk_and_roll_back(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    course_a = store.create_course("Course A")
    course_b = store.create_course("Course B")
    document = _document("document-a", course_a.id, "a" * 64)
    mismatched = _chunk("chunk-a", document.id, course_b.id, "wrong course")

    with pytest.raises(sqlite3.IntegrityError):
        store.persist_document(document, (mismatched,))

    assert store.list_documents(course_a.id) == ()


def test_store_rejects_document_for_missing_course(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")

    with pytest.raises(sqlite3.IntegrityError):
        store.persist_document(
            _document("document-a", "missing-course", "a" * 64), ()
        )


def test_store_deletes_document_and_its_chunks(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    course = store.create_course("Course A")
    document = _document("document-a", course.id, "a" * 64)
    store.persist_document(document, (_chunk("chunk-a", document.id, course.id, "A"),))

    deleted = store.delete_document(course.id, document.id)

    assert deleted == document
    assert store.list_documents(course.id) == ()
    assert store.list_chunks(course.id) == ()
