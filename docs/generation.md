# Grounded answer generation

## Purpose

Milestone 3-B implements the backend service that turns a validated
`SUPPORTED` support decision into either an `ANSWERED` response with source
citations or an `ABSTAINED` response. It is a service boundary only: no question
answering HTTP endpoint, frontend flow, or chat interface exists yet.

```text
question + course
  -> course-scoped retrieval
  -> Support Verifier
  -> SUPPORTED with bound chunk IDs only
  -> Generator receives exactly those chunks
  -> validate claim bindings and source locators
  -> ANSWERED | ABSTAINED
```

`INSUFFICIENT`, `CONFLICTING`, empty evidence, invalid provenance, and verifier
failures end before generation. A generator exception or invalid citation
binding also fails closed as `ABSTAINED`; raw model output is never returned.

## Structured contract

The generator returns:

```json
{
  "answer": "Claim one.\nClaim two.",
  "claims": [
    {"text": "Claim one.", "supporting_chunk_ids": ["chunk-id"]},
    {"text": "Claim two.", "supporting_chunk_ids": ["chunk-id"]}
  ]
}
```

The backend requires non-empty unique chunk IDs for every claim. To avoid
unbound prose, `answer` must be exactly the newline-joined claim texts. Every
claim ID must refer to one of the verifier-bound chunks from the selected
course. The returned citation contains that chunk ID plus its original filename
and source range; the provider never supplies a citation locator.

## OpenAI boundary

The OpenAI generator uses the Responses API with strict JSON Schema,
`store=false`, and no tools, web search, or file search. Its request contains
only the question and the verified chunk IDs, filenames, source locators, and
text. It does not request evaluation answers or material from another course.
The API key comes only from `OPENAI_API_KEY`; no request happens at import time.

Model instructions direct the provider to use only the supplied evidence and to
treat evidence text as untrusted data. This is defense in depth, not the
security boundary: backend support gating and citation validation enforce the
policy procedurally.

The integration follows the official [Responses API](https://platform.openai.com/docs/api-reference/responses), [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs), and [data controls](https://platform.openai.com/docs/models/default-usage-policies-by-endpoint).

## Evaluation status

The evaluation helper keeps answered-supported, missed-supported,
unsupported-generated, correctly abstained unsupported, abstained-conflict, and
citation-validation-failure counts separate. Unit tests use mocked OpenAI
responses and synthetic documents; no live generator semantic evaluation was
run because no API credential was supplied. These tests validate the boundary,
not answer quality or factual accuracy.
