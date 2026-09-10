# CourseRAG

CourseRAG is a citation-first, closed-corpus retrieval-augmented generation
system for course materials. Its central promise is that course-content answers
will be grounded in evidence retrieved from materials supplied by the user.
When the selected course does not contain enough evidence, the system will
abstain instead of filling gaps with a model's pretrained knowledge.

## Status

**Early development — Milestone 2-B.**

The repository now implements local document ingestion, source-aware parsing,
deterministic chunking, SQLite metadata persistence, embedding providers,
Qdrant indexing, course-scoped vector retrieval, deterministic retrieval
evaluation, and a calibrated Evidence Gate that returns allow or abstain. Answer
generation, final citations, and question answering are not implemented yet.

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
- **Vector storage:** local Qdrant collections with mandatory course filters.
- **Document formats:** PDF, PPTX, DOCX, Markdown, and text are ingestible.
- **Embeddings:** deterministic offline and OpenAI providers behind one small
  interface.
- **Generation:** planned; no generation provider or model call exists yet.
- **Evidence gate (implemented):** provider-bound retrieval sufficiency and
  explicit abstention; citation validation is planned.

The fuller component and data-flow design is in
[docs/architecture.md](docs/architecture.md).

## Implemented through Milestone 2-B

- Course creation and course-owned document metadata.
- PDF page, PPTX slide, DOCX paragraph, Markdown, and text parsing.
- Deterministic word-budget chunking with overlap and source ranges.
- Transactional SQLite persistence and same-course duplicate detection.
- Deterministic offline embeddings and an OpenAI embedding provider.
- Idempotent Qdrant vector indexing with stable chunk UUIDs.
- Required-course retrieval returning ranked, provenance-preserving chunks.
- Offline retrieval evaluation with Hit@k, Recall@k, raw-score collection, and
  explicit cross-course isolation checks.
- Provider-bound calibration artifacts and a top-1 retrieval abstention gate.
- Developer-facing ingestion CLI.

## Planned features

- Evidence sufficiency checks before any generation request.
- Answers with locatable citations into the source materials.
- Explicit unsupported or insufficient-evidence responses.
- Additional embedding providers and a future generation provider.
- Question-answering and chat interfaces.

These are roadmap items, not claims about the current implementation.

## Repository layout

```text
course-rag/
├── backend/                 FastAPI application and Python package metadata
├── frontend/                Next.js TypeScript application
├── docs/                    Architecture, grounding, ingestion, and retrieval
├── examples/                Two original contradictory synthetic courses
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

No API key is required for the default deterministic provider. To use OpenAI
embeddings, set `COURSE_RAG_EMBEDDING_PROVIDER=openai` and supply
`OPENAI_API_KEY`; never commit the populated file.

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

## Index and retrieve evidence

Index all currently unindexed chunks in a course, then retrieve raw evidence:

```bash
course-rag index-course --course <course-id>
course-rag retrieve \
  --course <course-id> \
  --query "What is the Blueleaf coefficient?" \
  --limit 5
```

You can also run `course-rag index-document --document <document-id>`. Qdrant
data defaults to ignored `runtime/qdrant/`. Retrieval prints similarity scores
and source chunks; it does not generate an answer or apply an evidence threshold.
See [docs/retrieval.md](docs/retrieval.md).

## Evaluate retrieval

Run the checked-in fictional benchmark without network access:

```bash
course-rag evaluate-retrieval \
  examples/retrieval-evaluation/dataset.json \
  --output runtime/evaluations/m2a.json
```

The report measures retrieval ranking and course isolation; it does not choose
an evidence threshold or make a sufficiency decision. See
[docs/retrieval-evaluation.md](docs/retrieval-evaluation.md).

## Calibrate and inspect evidence

```bash
course-rag calibrate-retrieval \
  examples/retrieval-evaluation/dataset.json \
  --output runtime/evaluations/m2b-calibration.json

course-rag inspect-evidence \
  --course <course-id> \
  --query "What is the Blueleaf coefficient?" \
  --calibration runtime/evaluations/m2b-calibration.json
```

Calibration measures a synthetic corpus and binds a threshold to one embedding
provider/model/dimension. The gate preserves retrieved provenance on `ALLOW` and
returns `ABSTAIN` otherwise; it never generates a natural-language answer. See
[docs/evidence-gating.md](docs/evidence-gating.md).

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

When the OpenAI embedding provider is selected, document chunk text and query
text are sent to OpenAI for embedding. The deterministic provider stays local
and exists only for reproducible development and testing.

**Never commit real course materials to a public repository.** Only synthetic,
original fixtures created specifically for testing belong in `examples/`.
Never commit API keys or a populated `.env` file.

The project should remain private until the planned public-release audit checks
licensing, secrets, test fixtures, dependency risk, documentation, and tracked
files.
