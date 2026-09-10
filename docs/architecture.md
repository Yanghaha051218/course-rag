# CourseRAG architecture

## Scope and status

Milestone 2-B implements local document parsing and persistence, embedding,
Qdrant indexing, course-scoped retrieval, offline retrieval evaluation, and a
provider-bound Evidence Gate behind a developer CLI. The FastAPI surface
remains the Milestone 0 `/health` endpoint, and the Next.js page remains
static. Components labelled **planned** do not exist yet.

## Primary invariant

For course-content questions, generation is downstream of successful retrieval
and evidence validation. The application must not call a generation provider
when the selected course does not yield sufficient evidence.

```mermaid
flowchart TD
    Q[Question + selected course ID] --> S[Validate request and course scope]
    S --> R[Retrieve only course-scoped chunks]
    R --> G{Evidence sufficient?}
    G -- No --> A[Return explicit abstention]
    G -- Yes --> L[Call generation provider with retrieved evidence]
    L --> C[Validate claims and citations]
    C --> O[Return grounded answer with citations]
```

The `No` branch ends before provider invocation. A prompt asking a model to
abstain is a useful secondary control, but it cannot replace this backend branch.

The implemented path stops at structured retrieval results:

```mermaid
flowchart TD
    D[Course-owned document] --> P[Format parser]
    P --> U[SourceUnit with page, slide, or section]
    U --> K[Deterministic chunker]
    K --> S[(SQLite courses, documents, chunks)]
    S --> E[EmbeddingProvider]
    E --> V[(Qdrant)]
    V --> R[Course-filtered Retriever]
    R --> O[RetrievedChunk array]
    O --> M[Offline retrieval measurements]
    O --> G[Calibrated Evidence Gate]
    G --> F[Future generator on ALLOW only]
    G --> A[Explicit abstain]
```

## Major components

### Web application

The Next.js frontend will present course selection, document-management status,
questions, answers, citations, and abstention states. It will communicate with
the backend through typed HTTP contracts. It remains static in Milestone 2-B.

### API and orchestration

The FastAPI backend owns request validation and the answer workflow. Planned
orchestration responsibilities are course authorization, retrieval, evidence
gating, provider invocation, citation validation, and stable response shapes.
The only current endpoint is `GET /health`.

### Ingestion pipeline

The implemented ingestion pipeline:

1. requires one existing course ID and a supported local file;
2. validates existence, extension, non-empty size, and configured size limit;
3. extracts page, slide, or honest section units from PDF, PPTX, DOCX,
   Markdown, or text;
4. creates overlapping chunks carrying course, document, and source ranges;
5. writes document metadata and chunks in one SQLite transaction; and
6. returns the existing document when the same checksum already belongs to the
   same course.

Ingestion remains separate from indexing so embedding-provider failures do not
affect document persistence. Uploaded documents remain untrusted input; the CLI
applies file-size and type limits and returns explicit errors.

### Metadata store

SQLite stores courses, documents, chunk text, checksums, and source locators.
Foreign keys link documents to courses and use `(document_id, course_id)` for
chunks, preventing a chunk from claiming a different course than its document.
Course-scoped indexes support later retrieval. Database files live under
`runtime/` and are ignored by Git.

### Embedding and vector index

Qdrant stores vectors and compact provenance payloads; authoritative chunk text
remains in SQLite. Stable chunk UUIDs are Qdrant point IDs. One collection is
derived from provider, model, dimension, and a configured prefix. Collection
metadata is checked before use, so incompatible embeddings cannot mix.

Every Qdrant query requires a course ID and applies it inside `query_points` as
an exact payload filter. Returned payloads are then checked against SQLite;
missing or contradictory provenance is an explicit consistency error.

### Provider boundary

The implemented `EmbeddingProvider` interface exposes provider/model identity,
dimension, batch document embedding, and query embedding. The deterministic
provider supports offline tests; OpenAI is the external embedding option.

Generation providers remain planned. Credentials come from environment
variables and are never stored in vectors, logs, fixtures, or Git.

### Retrieval evaluation

The evaluation runner creates temporary SQLite and in-memory Qdrant stores,
reuses the production ingestion, indexing, and retrieval path, and reports
Hit@k, Recall@k, raw scores, and course-isolation status. It makes no evidence
sufficiency decision. The checked-in corpus and labels are synthetic and
human-readable; optional JSON output belongs under ignored runtime storage.

### Evidence gate (implemented) and citation validator (planned)

The Evidence Gate verifies provider identity against an empirical calibration
artifact, rejects empty or foreign-course results, and applies a calibrated
top-1 similarity policy. It returns a structured allow/abstain decision and
preserves `RetrievedChunk` provenance only for allow. It neither generates an
answer nor decides factual truth; calibration labels remain evaluation-only.

After future generation, a citation validator will reject or downgrade unsupported
claims, verify that cited chunk IDs were in the approved evidence set, and
return locators suitable for the source format.

## Course isolation

Course isolation is a correctness and privacy boundary, not a ranking hint.
Current and future defenses are layered:

1. every persisted document and chunk is assigned one course ID at ingestion;
2. SQLite composite foreign keys prevent document/chunk course mismatches;
3. the storage interface exposes chunk listing only with an explicit course ID;
4. automated tests prove two courses remain distinct;
5. Qdrant retrieval applies an exact `course_id` filter before returning hits;
6. backend retrieval verifies Qdrant provenance against SQLite; and
7. future answer orchestration must reject mismatched evidence before prompting.

Documents or chunks from Course A must never appear in retrieval, prompts,
answers, or citations for Course B.

## Runtime and trust boundaries

- Browser requests, uploaded files, extracted text, retrieved chunks, provider
  responses, and model output are untrusted at their respective boundaries.
- The backend is the policy enforcement point for course scope and generation
  eligibility.
- Runtime data belongs under `runtime/` and remains local by default.
- The frontend must not receive provider credentials.
- Errors must be explicit and observable; failed parsing, retrieval, or provider
  calls must not be converted into fabricated answers.

## Intended request outcome types

The future answer API should return a stable discriminated outcome:

- `answered`: grounded answer plus validated citations;
- `insufficient_evidence`: explicit abstention without a generation call; or
- `error`: an operational failure, distinct from lack of evidence.

The precise endpoint and schema will be designed in the milestone that
implements the answer path.
