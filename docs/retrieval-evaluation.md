# Retrieval evaluation

Milestone 2-A measures the existing course-scoped retriever before any evidence
gate is designed. The checked-in dataset is
[`examples/retrieval-evaluation/dataset.json`](../examples/retrieval-evaluation/dataset.json).
It uses only original fictional materials and fixed chunk settings so every
expected evidence reference is reviewable as a filename, chunk index, and source
range.

## What it measures

The dataset measures whether labeled relevant chunks appear in the ranked
results and whether every returned chunk belongs to the selected course. It
contains exact, paraphrase, cross-document, multi-chunk, unsupported, near-miss,
contradictory-course, and vocabulary-overlap cases.

For answerable cases:

- **Hit@k** is `1` when at least one expected evidence chunk occurs in the first
  `k` results, otherwise `0`. The report averages this over answerable cases.
- **Recall@k** is the fraction of that case's expected evidence chunks found in
  the first `k` results. The report averages case recall over answerable cases.

Unsupported cases never count as hits merely because retrieval returned a
chunk. Their top-1 score and complete returned score list up to the requested
limit are retained separately. All values are the vector store's raw similarity
scores.

The runner creates temporary SQLite and in-memory Qdrant stores, ingests and
indexes the declared corpus, and deletes those stores after the run. Queries are
not written to the production database. Retrieval still executes the mandatory
Qdrant `course_id` filter and the existing authoritative SQLite hydration and
provenance checks. Source provenance provides a deterministic tie order for
equal scores in the small evaluation corpus.

## What it does not measure

This is not an answer-quality, factuality, citation, generation, or user-facing
evaluation. It does not decide whether retrieved evidence is sufficient and it
does not measure the quality of an OpenAI embedding model when run with the
default deterministic lexical provider.

A raw cosine similarity score is not a calibrated probability and must not be
displayed as confidence. Its scale depends on the embedding provider, corpus,
query distribution, and competing chunks. Retrieval quality asks whether the
right evidence was ranked; evidence sufficiency asks whether that evidence can
support the requested factual claim. Those are separate decisions.

Evidence-threshold selection is deliberately deferred to M2-B. M2-B should use
the observed answerable and unsupported score distributions, expand the dataset
where needed, and choose calibration and abstention policy explicitly. M2-A
does not provide a threshold or sufficiency decision.

## Run it

From the repository root, with the backend installed:

```bash
course-rag evaluate-retrieval \
  examples/retrieval-evaluation/dataset.json \
  --output runtime/evaluations/m2a.json
```

The default deterministic provider requires no API key or network access. The
same command uses the configured embedding provider, so a manual OpenAI run is
possible by setting the existing embedding environment variables. Automated
tests never select OpenAI.

Representative deterministic-provider output:

```text
Retrieval evaluation
Cases: 8 (answerable=6, unsupported=2)
Hit@1: 6/6 (1.000)
Recall@1: 0.833
Hit@3: 6/6 (1.000)
Recall@3: 1.000
Hit@5: 6/6 (1.000)
Recall@5: 1.000
Isolation: PASS
Unsupported top-1 scores:
- unsupported-orbital-temperature: 0.205196
- blueleaf-fertilizer-near-miss: 0.173422
JSON: runtime/evaluations/m2a.json
```

The JSON artifact contains per-case raw scores, retrieved provenance, hits,
recall, and isolation status. `runtime/evaluations/` is gitignored.
