from pathlib import Path

import pymupdf
import pytest

from course_rag_api.errors import (
    DocumentTooLargeError,
    EmptyDocumentError,
    MissingDocumentError,
    UnsupportedFileTypeError,
)
from course_rag_api.ingestion import ingest_document
from course_rag_api.storage import SQLiteStore


def test_ingestion_persists_document_and_chunks(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    course = store.create_course("Orbital Gardening")
    path = tmp_path / "lesson.txt"
    path.write_text(
        "The Blueleaf coefficient is 7.25.\n\n"
        "Harvest begins after eighteen measured minutes."
    )

    summary = ingest_document(
        store=store,
        course_id=course.id,
        file_path=path,
        chunk_target_size=7,
        chunk_overlap=2,
        max_document_bytes=10_000,
    )

    documents = store.list_documents(course.id)
    chunks = store.list_chunks(course.id)
    assert summary.status == "ingested"
    assert summary.document_id == documents[0].id
    assert summary.source_unit_count == 2
    assert summary.chunk_count == len(chunks)
    assert documents[0].filename == "lesson.txt"
    assert all(chunk.course_id == course.id for chunk in chunks)


def test_duplicate_is_returned_but_same_content_in_two_courses_is_allowed(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    course_a = store.create_course("Course A")
    course_b = store.create_course("Course B")
    path = tmp_path / "shared.md"
    path.write_text("The synthetic sunlamp cycle is exactly four hours.")

    first = ingest_document(store=store, course_id=course_a.id, file_path=path)
    duplicate = ingest_document(store=store, course_id=course_a.id, file_path=path)
    other_course = ingest_document(store=store, course_id=course_b.id, file_path=path)

    assert first.status == "ingested"
    assert duplicate.status == "duplicate"
    assert duplicate.document_id == first.document_id
    assert len(store.list_documents(course_a.id)) == 1
    assert other_course.status == "ingested"
    assert other_course.document_id != first.document_id
    assert {chunk.course_id for chunk in store.list_chunks(course_a.id)} == {
        course_a.id
    }
    assert {chunk.course_id for chunk in store.list_chunks(course_b.id)} == {
        course_b.id
    }


def test_ingestion_summary_counts_physical_pdf_pages_consistently(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    course = store.create_course("Page Count Course")
    path = tmp_path / "pages.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Page one has text.")
    document.new_page()
    document.save(path)
    document.close()

    first = ingest_document(store=store, course_id=course.id, file_path=path)
    duplicate = ingest_document(store=store, course_id=course.id, file_path=path)

    assert first.source_unit_count == duplicate.source_unit_count == 2


@pytest.mark.parametrize(
    ("filename", "content", "error"),
    [
        ("empty.txt", "   \n", EmptyDocumentError),
        ("unsupported.csv", "data", UnsupportedFileTypeError),
    ],
)
def test_ingestion_failures_leave_no_document(
    tmp_path: Path, filename: str, content: str, error: type[Exception]
) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    course = store.create_course("Failure Course")
    path = tmp_path / filename
    path.write_text(content)

    with pytest.raises(error):
        ingest_document(store=store, course_id=course.id, file_path=path)

    assert store.list_documents(course.id) == ()


def test_ingestion_rejects_missing_and_oversized_files(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    course = store.create_course("Boundary Course")

    with pytest.raises(MissingDocumentError):
        ingest_document(
            store=store, course_id=course.id, file_path=tmp_path / "missing.txt"
        )

    path = tmp_path / "large.txt"
    path.write_text("more than five bytes")
    with pytest.raises(DocumentTooLargeError):
        ingest_document(
            store=store,
            course_id=course.id,
            file_path=path,
            max_document_bytes=5,
        )
