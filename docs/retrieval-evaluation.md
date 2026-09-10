# Retrieval evaluation

Milestone 2-A established the retrieval measurement harness. M2-B expands the
checked-in dataset and uses an explicit calibration/holdout split before adding
the Evidence Gate. The dataset is
[`examples/retrieval-evaluation/dataset.json`](../examples/retrieval-evaluation/dataset.json).
It uses only original fictional materials and fixed chunk settings so every
expected evidence reference is reviewable as a filename, chunk index, and source
range.

## What it measures

The dataset measures whether labeled relevant chunks appear in the ranked
results and whether every returned chunk belongs to the selected course. It
contains exact, paraphrase, cross-document, multi-chunk, unsupported,
plausible-domain, near-miss, wrong-attribute, adversarial lexical-overlap,
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

The M2-B calibration command chooses a provider-bound baseline threshold from
the calibration split and reports holdout results separately. Its current
deterministic baseline reported holdout answerable acceptance of **0/6** and
unsupported false accepts of **0/6**. This result is preserved rather than tuned
away: similarity alone is not the final evidence-sufficiency decision. See
[evidence gating](evidence-gating.md) for the objective and limitations.

M3-A adds separate Support Verification metrics: supported answerable, missed
answerable, incorrectly supported unsupported, correctly rejected unsupported,
and correctly detected conflicts. These are not combined with Hit@k or Recall@k.
The same 40 labels map answerable to `SUPPORTED` and unsupported to
`INSUFFICIENT`; `support_conflicts` supplies a separate same-course conflict
fixture. Live semantic metrics require a real verifier run and are not claimed
by mocked tests.

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
Cases: 40 (answerable=20, unsupported=20)
Hit@1: 20/20 (1.000)
Recall@1: 0.875
Hit@3: 20/20 (1.000)
Recall@3: 1.000
Hit@5: 20/20 (1.000)
Recall@5: 1.000
Isolation: PASS
Unsupported top-1 scores:
- comet-moss-tray-color: 0.780190
- solar-vine-wavelength: 0.484200
JSON: runtime/evaluations/m3a-retrieval.json
```

The JSON artifact contains per-case raw scores, retrieved provenance, hits,
recall, and isolation status. `runtime/evaluations/` is gitignored.
