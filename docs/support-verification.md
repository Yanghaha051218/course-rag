# Evidence Support Verification

## Purpose

Retrieval similarity and evidence sufficiency are different problems.
Similarity asks how related a chunk is to a query; Support Verification asks
whether the supplied course-scoped chunks explicitly support answering the
question actually asked. M3-A implements the latter classification step but
does not generate an answer.

```text
Question -> course-scoped retrieval -> RetrievedChunk[] -> Support Verifier
                                                        -> SUPPORTED
                                                        -> INSUFFICIENT
                                                        -> CONFLICTING
```

The M2-B calibrated top-1 threshold remains an evaluated comparison baseline.
It is not required before invoking this verifier because its strict threshold
allowed 0/6 answerable holdout cases in the current synthetic experiment.

## Contract and invariants

`SupportVerifier` receives a question, selected course ID, and retrieved chunks.
It returns `SupportDecision` with a status, machine-readable reason, verifier
provider/model identity, and `supporting_chunk_ids`.

- `SUPPORTED` requires at least one unique supporting chunk ID.
- `INSUFFICIENT` has no supporting chunk IDs.
- `CONFLICTING` fails closed; any returned IDs must be unique retrieved chunks.
- Every returned ID is checked to exist in the retrieved set, belong to the
  selected course, and match SQLite document/chunk provenance.
- Empty evidence, foreign evidence, malformed output, invented IDs, and verifier
  exceptions become non-supported decisions. The grounded generator is never
  called for these decisions.

The verifier receives no expected answers, unrelated course materials, web
search, File Search, or external tools. Evidence text is untrusted data and is
instructed not to control verifier behavior.

## OpenAI provider

The only production verifier provider in M3-A is OpenAI via the Responses API.
It uses a strict JSON Schema response format, does not pass a `tools` argument,
and sets `store=false`. The request contains only the question plus retrieved
chunk IDs, locators, filenames, and text from the selected course. It never
contains all courses or the evaluation's expected answers.

Configure it locally:

```text
COURSE_RAG_VERIFIER_PROVIDER=openai
COURSE_RAG_VERIFIER_MODEL=gpt-4o-mini
OPENAI_API_KEY=...  # never commit
```

No API request occurs at import. Selecting this provider without a key returns a
clear configuration error. Unit tests inject a mock SDK client and make no
network requests. A live semantic evaluation was not performed because no API
credential was supplied.

The Responses API supports `text.format` JSON Schema with `strict: true`; its
`store` option controls response storage. References:

- https://platform.openai.com/docs/api-reference/responses
- https://platform.openai.com/docs/guides/structured-outputs
- https://platform.openai.com/docs/models/default-usage-policies-by-endpoint

## Developer CLI

```bash
course-rag verify-support \
  --course <course-id> \
  --query "Who discovered the Blueleaf coefficient?"
```

The command prints the structured decision, verifier identity, retrieved chunk
locators, and validated supporting chunk IDs. It never prints a generated
factual answer.

## Limitations

Structured output and backend validation make the boundary safer, but a mocked
test double does not establish semantic quality. Before production use, evaluate
a live provider on a preregistered synthetic calibration/holdout protocol and
report support, false-support, and conflict-detection metrics separately.
