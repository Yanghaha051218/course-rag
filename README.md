# CourseRAG

CourseRAG is a citation-first, closed-corpus retrieval-augmented generation
system for course materials. Its central promise is that course-content answers
will be grounded in evidence retrieved from materials supplied by the user.
When the selected course does not contain enough evidence, the system will
abstain instead of filling gaps with a model's pretrained knowledge.

## Status

**Early development — Milestone 1-A.**

The repository now implements local document ingestion, source-aware parsing,
deterministic chunking, SQLite metadata persistence, a developer CLI, tests,
synthetic fixtures, and CI. Embeddings, retrieval, evidence gating, model calls,
citations, and question answering are not implemented yet.

## Grounding principle

The planned answer path is:

```text
Question + selected course
  -> retrieve course-scoped evidence
  -> validate evidence sufficiency
  -> if insufficient: abstain without calling a generation model
  -> if sufficient: generate only from the evidence
  -> validate and return citations
```

Grounding will be enforced in backend control flow as well as in model
instructions. Prompting alone is not considered an enforcement boundary. See
[the grounding policy](docs/grounding-policy.md) for the normative rules.

## Architecture

- **Frontend:** Next.js and TypeScript.
- **Backend:** FastAPI and Python.
- **Application metadata:** SQLite for courses, documents, and chunks.
- **Vector storage:** Qdrant, partitioned and filtered by course.
- **Document formats:** PDF, PPTX, DOCX, Markdown, and text are ingestible.
- **Models:** provider interfaces for embeddings and generation, with OpenAI as
  an initial provider.
- **Quality gates:** retrieval sufficiency, course-isolation checks, citation
  validation, and explicit abstention.

The fuller component and data-flow design is in
[docs/architecture.md](docs/architecture.md).

## Implemented in Milestone 1-A

- Course creation and course-owned document metadata.
- PDF page, PPTX slide, DOCX paragraph, Markdown, and text parsing.
- Deterministic word-budget chunking with overlap and source ranges.
- Transactional SQLite persistence and same-course duplicate detection.
- Developer-facing ingestion CLI.

## Planned features

- Embeddings and course-filtered vector retrieval.
- Evidence sufficiency checks before any generation request.
- Answers with locatable citations into the source materials.
- Explicit unsupported or insufficient-evidence responses.
- Pluggable embedding and generation providers.
- Question-answering and chat interfaces.

These are roadmap items, not claims about the current implementation.

## Repository layout

```text
course-rag/
├── backend/                 FastAPI application and Python package metadata
├── frontend/                Next.js TypeScript application
├── docs/                    Architecture, grounding, and ingestion behavior
├── examples/example-course Tiny original test-only course materials
├── runtime/                 Local uploads, databases, and vector data (ignored)
└── tests/backend/           Backend API tests
```

## Prerequisites

- Python 3.11 or newer (CI uses Python 3.12).
- Node.js 22.13 or newer (CI uses Node.js 24).
- pnpm 11.19.0.

## Start from a clean clone

Run all commands from the repository root unless a step says otherwise.

1. Create a local environment file. It contains no real credentials by
   default:

   ```bash
   cp .env.example .env
   ```

2. Create the Python environment and install the pinned backend dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e "backend[test]"
   ```

3. Run the backend tests:

   ```bash
   cd backend
   python -m pytest -q
   cd ..
   ```

4. Start the backend:

   ```bash
   python -m uvicorn course_rag_api.main:app \
     --app-dir backend/src \
     --host 127.0.0.1 \
     --port 8000 \
     --reload
   ```

   In another terminal, verify `http://127.0.0.1:8000/health`:

   ```bash
   curl --fail http://127.0.0.1:8000/health
   ```

5. Install and start the frontend:

   ```bash
   cd frontend
   pnpm install --frozen-lockfile
   pnpm dev
   ```

   Open `http://127.0.0.1:3000`.

No API key is required for Milestone 1-A. The OpenAI setting in `.env.example` is
reserved for a future provider and should remain empty for now.

## Ingest documents locally

Generate the optional synthetic binary examples under ignored runtime storage:

```bash
python examples/example-course/generate_documents.py runtime/example-course
```

Create a course, copy the printed course ID, then ingest a document:

```bash
course-rag create-course "Orbital Gardening Demo"
course-rag ingest --course <course-id> runtime/example-course/orbital-gardening.pdf
```

The default database is `runtime/db/course-rag.sqlite3`. Configure its path,
chunk sizes, overlap, and input-size limit through the `COURSE_RAG_*` settings
shown in `.env.example`. See [docs/ingestion.md](docs/ingestion.md) for format
and provenance details.

## Verification commands

```bash
cd backend && python -m pytest -q
cd ../frontend && pnpm peers check
pnpm run lint
pnpm run typecheck
pnpm run build
```

## Privacy and repository hygiene

Course materials may contain copyrighted, private, or personally identifiable
information. Runtime uploads, SQLite files, Qdrant data, local environment
files, and build outputs are ignored by Git.

**Never commit real course materials to a public repository.** Only synthetic,
original fixtures created specifically for testing belong in `examples/`.
Never commit API keys or a populated `.env` file.

The project should remain private until the planned public-release audit checks
licensing, secrets, test fixtures, dependency risk, documentation, and tracked
files.
