import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from course_rag_api.errors import CourseNotFoundError, DocumentNotFoundError
from course_rag_api.models import Chunk, Course, Document


_SCHEMA = """
CREATE TABLE IF NOT EXISTS courses (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    source_path TEXT NOT NULL,
    checksum TEXT NOT NULL,
    page_or_unit_count INTEGER NOT NULL CHECK (page_or_unit_count > 0),
    created_at TEXT NOT NULL,
    UNIQUE (course_id, checksum),
    UNIQUE (id, course_id),
    FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    text TEXT NOT NULL CHECK (length(trim(text)) > 0),
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    source_type TEXT NOT NULL CHECK (source_type IN ('page', 'slide', 'section')),
    source_start INTEGER NOT NULL CHECK (source_start > 0),
    source_end INTEGER NOT NULL CHECK (source_end >= source_start),
    created_at TEXT NOT NULL,
    UNIQUE (document_id, chunk_index),
    FOREIGN KEY (document_id, course_id)
        REFERENCES documents(id, course_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS documents_course_id_idx ON documents(course_id);
CREATE INDEX IF NOT EXISTS chunks_course_id_idx ON chunks(course_id);
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);
"""


class SQLiteStore:
    """Minimal course-scoped metadata store backed by SQLite."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def create_course(self, name: str) -> Course:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("course name must not be empty")
        course = Course(
            id=str(uuid4()),
            name=clean_name,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO courses (id, name, created_at) VALUES (?, ?, ?)",
                (course.id, course.name, course.created_at),
            )
        return course

    def get_course(self, course_id: str) -> Course:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, name, created_at FROM courses WHERE id = ?",
                (course_id,),
            ).fetchone()
        if row is None:
            raise CourseNotFoundError(f"Course does not exist: {course_id}")
        return Course(**dict(row))

    def find_document_by_checksum(
        self, course_id: str, checksum: str
    ) -> Document | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, course_id, filename, file_type, source_path, checksum,
                       page_or_unit_count, created_at
                FROM documents
                WHERE course_id = ? AND checksum = ?
                """,
                (course_id, checksum),
            ).fetchone()
        return None if row is None else Document(**dict(row))

    def get_document(self, document_id: str) -> Document:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, course_id, filename, file_type, source_path, checksum,
                       page_or_unit_count, created_at
                FROM documents WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
        if row is None:
            raise DocumentNotFoundError(f"Document does not exist: {document_id}")
        return Document(**dict(row))

    def persist_document(
        self, document: Document, chunks: Sequence[Chunk]
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id, course_id, filename, file_type, source_path, checksum,
                    page_or_unit_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.course_id,
                    document.filename,
                    document.file_type,
                    document.source_path,
                    document.checksum,
                    document.page_or_unit_count,
                    document.created_at,
                ),
            )
            connection.executemany(
                """
                INSERT INTO chunks (
                    id, course_id, document_id, text, chunk_index, source_type,
                    source_start, source_end, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        chunk.id,
                        chunk.course_id,
                        chunk.document_id,
                        chunk.text,
                        chunk.chunk_index,
                        chunk.source_type,
                        chunk.source_start,
                        chunk.source_end,
                        chunk.created_at,
                    )
                    for chunk in chunks
                ),
            )

    def list_documents(self, course_id: str) -> tuple[Document, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, course_id, filename, file_type, source_path, checksum,
                       page_or_unit_count, created_at
                FROM documents WHERE course_id = ? ORDER BY created_at, id
                """,
                (course_id,),
            ).fetchall()
        return tuple(Document(**dict(row)) for row in rows)

    def list_chunks(self, course_id: str) -> tuple[Chunk, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, course_id, document_id, text, chunk_index, source_type,
                       source_start, source_end, created_at
                FROM chunks
                WHERE course_id = ?
                ORDER BY document_id, chunk_index
                """,
                (course_id,),
            ).fetchall()
        return tuple(Chunk(**dict(row)) for row in rows)

    def list_document_chunks(
        self, course_id: str, document_id: str
    ) -> tuple[Chunk, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, course_id, document_id, text, chunk_index, source_type,
                       source_start, source_end, created_at
                FROM chunks
                WHERE course_id = ? AND document_id = ?
                ORDER BY chunk_index
                """,
                (course_id, document_id),
            ).fetchall()
        return tuple(Chunk(**dict(row)) for row in rows)

    def get_chunks_with_filenames(
        self, course_id: str, chunk_ids: Sequence[str]
    ) -> dict[str, tuple[Chunk, str]]:
        if not chunk_ids:
            return {}
        placeholders = ", ".join("?" for _ in chunk_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT c.id, c.course_id, c.document_id, c.text, c.chunk_index,
                       c.source_type, c.source_start, c.source_end, c.created_at,
                       d.filename
                FROM chunks AS c
                JOIN documents AS d
                  ON d.id = c.document_id AND d.course_id = c.course_id
                WHERE c.course_id = ? AND c.id IN ({placeholders})
                """,
                (course_id, *chunk_ids),
            ).fetchall()
        records: dict[str, tuple[Chunk, str]] = {}
        for row in rows:
            values = dict(row)
            filename = values.pop("filename")
            chunk = Chunk(**values)
            records[chunk.id] = (chunk, filename)
        return records

    def count_document_chunks(self, course_id: str, document_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT count(*) FROM chunks WHERE course_id = ? AND document_id = ?",
                (course_id, document_id),
            ).fetchone()
        return int(row[0])
