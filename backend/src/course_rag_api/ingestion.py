from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from course_rag_api.chunking import chunk_source_units
from course_rag_api.errors import (
    DocumentTooLargeError,
    EmptyDocumentError,
    MissingDocumentError,
    UnsupportedFileTypeError,
)
from course_rag_api.models import Document, IngestionSummary
from course_rag_api.parsers import SUPPORTED_EXTENSIONS, parse_document
from course_rag_api.storage import SQLiteStore


def _checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_document(
    *,
    store: SQLiteStore,
    course_id: str,
    file_path: Path,
    chunk_target_size: int = 600,
    chunk_overlap: int = 100,
    max_document_bytes: int = 50 * 1024 * 1024,
) -> IngestionSummary:
    """Parse, chunk, and atomically persist one course-owned document."""
    store.get_course(course_id)
    path = Path(file_path)
    if not path.is_file():
        raise MissingDocumentError(f"Document does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(f"Unsupported file type: {suffix or '<none>'}")
    size = path.stat().st_size
    if size == 0:
        raise EmptyDocumentError(f"{path.name} is empty")
    if size > max_document_bytes:
        raise DocumentTooLargeError(
            f"{path.name} is {size} bytes; limit is {max_document_bytes}"
        )

    checksum = _checksum(path)
    existing = store.find_document_by_checksum(course_id, checksum)
    if existing is not None:
        return IngestionSummary(
            document_id=existing.id,
            filename=existing.filename,
            file_type=existing.file_type,
            course_id=course_id,
            source_unit_count=existing.page_or_unit_count,
            chunk_count=store.count_document_chunks(course_id, existing.id),
            status="duplicate",
        )

    parsed = parse_document(path)
    document_id = str(uuid5(NAMESPACE_URL, f"course-rag:{course_id}:{checksum}"))
    created_at = datetime.now(timezone.utc).isoformat()
    document = Document(
        id=document_id,
        course_id=course_id,
        filename=path.name,
        file_type=suffix.removeprefix("."),
        source_path=str(path.resolve()),
        checksum=checksum,
        page_or_unit_count=parsed.unit_count,
        created_at=created_at,
    )
    chunks = chunk_source_units(
        course_id=course_id,
        document_id=document_id,
        units=parsed.units,
        target_size=chunk_target_size,
        overlap=chunk_overlap,
        created_at=created_at,
    )
    store.persist_document(document, chunks)
    return IngestionSummary(
        document_id=document.id,
        filename=document.filename,
        file_type=document.file_type,
        course_id=course_id,
        source_unit_count=parsed.unit_count,
        chunk_count=len(chunks),
        status="ingested",
    )
