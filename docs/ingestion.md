# Document ingestion

Milestone 1-A provides a local developer workflow that converts one document
owned by one course into citation-preserving chunks stored in SQLite.

```text
Document -> parser -> SourceUnit -> chunker -> SQLite
                                      |
                                      +-> future embedding/indexing
```

## Supported formats and source locations

| Format | Extraction unit | Stored source type | Locator behavior |
| --- | --- | --- | --- |
| PDF | page | `page` | Physical 1-based page number |
| PPTX | slide | `slide` | Physical 1-based slide number |
| DOCX | non-empty paragraph | `section` | 1-based paragraph position |
| Markdown/TXT | non-empty text block | `section` | 1-based block position |

PDF pages and PPTX slides without text are skipped while their physical numbers
remain intact. DOCX parsing cannot recover reliable rendered page boundaries,
so it deliberately records paragraph positions instead of invented page numbers.
PPTX extraction covers normal text-bearing shapes in top-to-bottom,
left-to-right order; diagrams and images are not interpreted.

## Chunking

The v0.1 chunker uses whitespace-delimited words as a deterministic size
approximation. A word count is not claimed to equal model tokens. It prefers
sentence or source-unit endings before the configured target of 600 words and
starts the next chunk with up to 100 overlapping words.

Chunks may span adjacent pages or slides. Each chunk stores the first and last
source number represented by its words, so a page 10 to page 11 chunk retains
`source_start = 10` and `source_end = 11`. Empty chunks are never persisted.

Configure the defaults with:

```text
COURSE_RAG_CHUNK_TARGET_SIZE
COURSE_RAG_CHUNK_OVERLAP
```

## Persistence and duplicate policy

SQLite stores `courses`, `documents`, and `chunks`. Foreign keys and a composite
document/course reference prevent cross-course chunk ownership. Read methods for
documents and chunks require a course ID.

Each file receives a SHA-256 checksum. When the same bytes are ingested into the
same course, ingestion returns the existing document with status `duplicate`.
The same bytes may be ingested into a different course. Parsing and chunking
happen before the transaction that writes the document and its chunks, so a
failed parse does not leave a partial document.

The default database path is `runtime/db/course-rag.sqlite3`; runtime databases
are ignored by Git.

## Developer CLI

```bash
course-rag create-course "Orbital Gardening Demo"
course-rag ingest --course <course-id> path/to/material.pdf
```

Use `--database path/to/file.sqlite3` before the subcommand to select another
database, including a temporary database for testing.

## Current limitations

- No OCR; image-only PDFs fail with an explicit no-extractable-text error.
- No diagram, image, speaker-note, or embedded-media understanding.
- DOCX tables are not extracted and DOCX page numbers are unavailable.
- Markdown is treated as text blocks; its syntax is not converted into a full
  semantic document tree.
- Embedding and retrieval are separate commands; ingestion never calls an
  external embedding provider automatically.
- No OCR, generation, or evidence gate is performed by the ingestion path; the
  local product prototype provides upload and document-management endpoints.
