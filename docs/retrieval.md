# Vector indexing and retrieval

Milestone 1-B converts SQLite chunks into vectors and returns ranked source
chunks for one required course. Retrieval itself stops before support
verification or generation; M3-A consumes its `RetrievedChunk[]` output.

```text
SQLite chunks -> EmbeddingProvider -> Qdrant
query + course_id -> query embedding -> filtered Qdrant search -> SQLite join
                  -> RetrievedChunk[]
```

## Embedding providers

`EmbeddingProvider` exposes provider name, model name, dimension, batch document
embedding, and query embedding. Retrieval and indexing depend only on this
contract.

The default `deterministic-hash-v1` provider creates normalized lexical hashing
vectors without network access. It exists for deterministic offline testing and
development and is not intended as a high-quality production semantic embedding
model. Its behavior does not represent OpenAI embedding quality.

The OpenAI provider uses the current Python SDK embeddings resource and supports
batched inputs. Select it with:

```text
COURSE_RAG_EMBEDDING_PROVIDER=openai
COURSE_RAG_EMBEDDING_MODEL=text-embedding-3-small  # optional default
COURSE_RAG_EMBEDDING_DIMENSION=1536                # optional default
OPENAI_API_KEY=...                                 # never commit
```

No provider performs an API request at import time. When OpenAI is selected,
document chunk text and query text are sent to OpenAI for embedding. Automated
tests use a fake client and never send document text over the network.

## Qdrant design

Local development uses persistent Qdrant storage at `runtime/qdrant/`; tests use
isolated in-memory clients and require no server or Docker.

The collection name is derived from the configured prefix plus a stable hash of
the provider name, model name, and vector dimension. The same identity reuses a
collection; a different model or dimension gets a different collection.
Collection metadata is validated before use, preventing silent mixing if a
collection is altered or collides.

Each point ID is the stable SQLite chunk UUID. Its compact payload contains:

```text
chunk_id, course_id, document_id, chunk_index,
source_type, source_start, source_end, filename
```

Authoritative chunk text remains in SQLite and is not duplicated in Qdrant.

## Indexing lifecycle and idempotency

`index-document` indexes one document. `index-course` checks every course chunk
and embeds only point IDs absent from the compatible Qdrant collection. Upserted
point IDs are stable, so retries do not duplicate vectors. Batch failures remain
explicit; rerunning safely continues with still-missing IDs.

## Course-scoped retrieval

The public retrieval contract requires `course_id`, query text, and a positive
limit from 1 through 100. Qdrant receives an exact `course_id` payload filter inside
`query_points`; results are never globally searched and then filtered afterward.
The backend also verifies returned course and provenance metadata against
SQLite. A missing SQLite chunk or mismatch raises an index consistency error.
Query text is embedded in memory and is not written to SQLite or Qdrant.

Results are ordered by Qdrant similarity score descending, then chunk ID for
stable tie ordering. The score is returned unchanged as a vector similarity
score. It is not a factual-confidence percentage. The M2-B Evidence Gate uses
a provider-bound calibration artifact; raw scores are never globally portable.
M3-A does not require that strict threshold before Support Verification.

## Current limitations

- The small synthetic benchmark measures regression behavior, not a semantic
  quality guarantee for the offline hashing provider.
- No hybrid keyword search, reranking, query rewriting, or conversation history.
- Course indexing currently loads one course's chunk metadata at once; add
  pagination only when measured course sizes require it.
- This milestone configures embedded local Qdrant only, not a remote cluster.
- The M2-B gate is calibrated only on the synthetic corpus and deterministic
  provider; production calibration requires representative data and the chosen
  embedding provider.
- Support Verification is an architecture and safety boundary, not a demonstrated
  semantic-quality result until a live verifier is evaluated separately.
- No answer generation, final citation formatting, or chat UI.

## Official API references

- OpenAI Python embeddings resource:
  https://github.com/openai/openai-python/blob/main/src/openai/resources/embeddings.py
- Qdrant local and filtered-query quickstart:
  https://qdrant.tech/documentation/quickstart/
- Qdrant collection configuration:
  https://qdrant.tech/documentation/manage-data/collections/
- Qdrant UUID point IDs and upserts:
  https://qdrant.tech/documentation/manage-data/points/
