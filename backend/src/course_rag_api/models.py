from dataclasses import dataclass
from typing import Literal


SourceType = Literal["page", "slide", "section"]


@dataclass(frozen=True, slots=True)
class SourceUnit:
    text: str
    source_type: SourceType
    source_number: int


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    units: tuple[SourceUnit, ...]
    unit_count: int


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    course_id: str
    document_id: str
    text: str
    chunk_index: int
    source_type: SourceType
    source_start: int
    source_end: int
    created_at: str


@dataclass(frozen=True, slots=True)
class Course:
    id: str
    name: str
    created_at: str
    owner_id: str | None = None


@dataclass(frozen=True, slots=True)
class User:
    id: str
    email: str
    created_at: str


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    course_id: str
    filename: str
    file_type: str
    source_path: str
    checksum: str
    page_or_unit_count: int
    created_at: str


@dataclass(frozen=True, slots=True)
class IngestionSummary:
    document_id: str
    filename: str
    file_type: str
    course_id: str
    source_unit_count: int
    chunk_count: int
    status: Literal["ingested", "duplicate"]


@dataclass(frozen=True, slots=True)
class IndexingSummary:
    course_id: str
    document_id: str | None
    total_count: int
    indexed_count: int
    collection_name: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    course_id: str
    document_id: str
    filename: str
    text: str
    score: float
    chunk_index: int
    source_type: SourceType
    source_start: int
    source_end: int
