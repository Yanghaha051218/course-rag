# CourseRAG

[中文说明](README.zh-CN.md)

CourseRAG is a citation-first, closed-corpus retrieval-augmented generation
system for course materials. Its central promise is that course-content answers
will be grounded in evidence retrieved from materials supplied by the user.
When the selected course does not contain enough evidence, the system will
abstain instead of filling gaps with a model's pretrained knowledge.

## Status

**Early development — Milestone 5-C local product prototype.**

The repository now implements local document ingestion, source-aware parsing,
deterministic chunking, SQLite metadata persistence, embedding providers,
Qdrant indexing, course-scoped vector retrieval, deterministic retrieval
evaluation, the M2-B similarity-gate baseline, evidence support verification,
and a backend grounded-generation service. A local real-course benchmark runner
measures those stages without committing course materials. The repository also
has a minimal local workflow for creating a course, uploading documents, asking
questions, and viewing ranked source evidence without a paid generation API.
Uploaded materials can be listed and deleted; uploads are limited to 50 MiB per
file and 500 MiB per course by default.
The local prototype now has cookie-based account sessions, scopes courses to
their owner, and includes basic API security hardening; it is still not
production-ready.

The SupportVerifier has human-reviewed gold datasets, but its live semantic
evaluation is still pending. Do not deploy this prototype for real users yet:
the current authentication limiter is local and per-process; account recovery
and deployment hardening are still planned.

## Grounding principle

The planned answer path is:

```text
Question + selected course
  -> retrieve course-scoped evidence
  -> verify support from the retrieved evidence
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
- **Embeddings:** deterministic test, local FastEmbed semantic, and OpenAI
  providers behind one small interface.
- **Generation (implemented service):** an OpenAI Responses API provider accepts
  only `SUPPORTED` evidence chunks and returns strict structured claims.
- **Similarity-gate baseline (implemented):** a provider-bound M2-B threshold
  retained for comparison, not a prerequisite for support verification.
- **Support verification (implemented):** a fail-closed verifier binds
  `SUPPORTED` decisions to retrieved, provenance-checked chunk IDs.

The fuller component and data-flow design is in
[docs/architecture.md](docs/architecture.md).

## Implemented

- Course creation and course-owned document metadata.
- PDF page, PPTX slide, DOCX paragraph, Markdown, and text parsing.
- Deterministic word-budget chunking with overlap and source ranges.
- Transactional SQLite persistence and same-course duplicate detection.
- Deterministic offline embeddings, a local FastEmbed semantic provider, and
  an OpenAI embedding provider.
- Idempotent Qdrant vector indexing with stable chunk UUIDs.
- Required-course retrieval returning ranked, provenance-preserving chunks.
- Offline retrieval evaluation with Hit@k, Recall@k, raw-score collection, and
  explicit cross-course isolation checks.
- Provider-bound calibration artifacts and a top-1 retrieval abstention gate.
- Support verification with `SUPPORTED`, `INSUFFICIENT`, and `CONFLICTING`
  outcomes; only verified chunk IDs may support `SUPPORTED`.
- Grounded-generation service that emits only `ANSWERED` responses with
  validated source citations, or explicit `ABSTAINED` responses.
- Citation validation that rejects invented, foreign-course, and unapproved
  chunk bindings before an answer is returned.
- Local benchmark runner with separate retrieval, support, citation, and
  failure-analysis metrics for human-curated course questions.
- Developer-facing ingestion CLI.
- Local course, document-upload, evidence-retrieval, and grounded-generation
  HTTP endpoints.
- Document size metadata, course-level upload quotas, and course-scoped document
  deletion.
- Local registration/login with httpOnly sessions and owner-scoped course APIs.
- Minimal responsive Next.js interface for the free retrieval-only workflow.

## Planned features

- Additional embedding and generation providers.
- Account recovery, shared production rate limiting, deployment hardening, and
  conversation history.

These are roadmap items, not claims about the current implementation.

## Repository layout

```text
course-rag/
├── backend/                 FastAPI application and Python package metadata
├── frontend/                Next.js TypeScript application
├── docs/                    Architecture, grounding, ingestion, and retrieval
├── benchmarks/               Benchmark schema, synthetic demo, local ignore rules
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

No API key is required for the frontend's retrieval-only workflow. For useful
local semantic retrieval, set `COURSE_RAG_EMBEDDING_PROVIDER=fastembed`; its first
use downloads `BAAI/bge-small-en-v1.5` into ignored `runtime/models/fastembed/`
and processes course text locally. Documents must be indexed with the same
provider used for retrieval. The separate grounded-generation endpoint still
requires an OpenAI key with available API credit and fails closed on provider
errors. To use OpenAI embeddings instead, set
`COURSE_RAG_EMBEDDING_PROVIDER=openai` and supply `OPENAI_API_KEY`; never
commit the populated file.

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

## Verify evidence support

The M3-A verifier is separate from the M2-B similarity threshold. It receives
only the selected question and course-scoped retrieved chunks, then returns a
structured status and chunk bindings—never an answer:

```bash
course-rag verify-support \
  --course <course-id> \
  --query "Who discovered the Blueleaf coefficient?"
```

The only current verifier provider is OpenAI. Configure it with
`COURSE_RAG_VERIFIER_PROVIDER=openai`,
`COURSE_RAG_VERIFIER_MODEL=<model>`, and `OPENAI_API_KEY`. See
[docs/support-verification.md](docs/support-verification.md). Automated tests
mock this boundary; no live verifier result is claimed.

## Benchmark an authorized local course

Create a human-curated benchmark outside Git using the documented
[benchmark schema](benchmarks/schema.md), then run it only after local ingestion
and indexing:

```bash
course-rag benchmark \
  --dataset benchmarks/local/physics2.json \
  --course <course-id> \
  --output runtime/evaluations/physics2-retrieval.json
```

The default run measures retrieval only. Add `--verify-support` to evaluate the
configured support verifier, or `--verify-support --generate` to include the
grounded generator and citation validation. The JSON report keeps these metrics
separate and contains failure analysis. See [docs/benchmarking.md](docs/benchmarking.md).

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

Parsing, SQLite, and local Qdrant remain local. When the OpenAI embedding
provider is selected, document chunk text and query text are sent to OpenAI for
embedding. When the OpenAI support verifier is selected, the question and only
the retrieved evidence text/provenance for the selected course are sent to
OpenAI. When the OpenAI generator is selected, it receives the question and
only the specific chunks bound by a `SUPPORTED` verifier decision. FastEmbed
downloads its model on first use, then processes embedding inputs locally. The
deterministic embedding provider stays local and exists only for reproducible
development and testing.

**Never commit real course materials to a public repository.** Only synthetic,
original fixtures created specifically for testing belong in `examples/`.
Never commit API keys or a populated `.env` file.

Real-course benchmark files, source documents, and generated reports stay local:
`benchmarks/local/` and `runtime/` are ignored. Run benchmarks only with course
materials you are authorized to use.

The project should remain private until the planned public-release audit checks
licensing, secrets, test fixtures, dependency risk, documentation, and tracked
files.
