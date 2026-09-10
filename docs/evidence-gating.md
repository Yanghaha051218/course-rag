# Calibrated evidence gate

Milestone 2-B adds a decision layer between retrieval and any future answer
generation:

```text
selected course + query -> course-scoped retrieval -> Evidence Gate -> allow | abstain
```

The gate does not generate text, inspect ground-truth labels, use keywords, or
know answers. It receives only the selected course ID, ranked `RetrievedChunk`
objects, and a calibration artifact. On `allow`, it preserves the retrieved
chunks and their full provenance for a future generator. On `abstain`, it
returns no evidence payload.

## Calibration

`examples/retrieval-evaluation/dataset.json` is versioned and explicitly split
into 28 calibration and 12 holdout cases. Labels and expected provenance are
part of the dataset, not inferred from similarity scores.

The baseline policy is `top1_threshold`: allow only when the top retrieved raw
similarity score is at least the empirically selected threshold. Candidates are
the observed calibration top-1 scores plus an always-abstain boundary. They are
ranked deterministically by:

1. fewest unsupported cases allowed;
2. most answerable cases allowed; then
3. highest threshold when the first two outcomes tie.

This makes false acceptance more costly than false abstention without silently
hard-coding a threshold. It can still reveal poor separability; it is not an
accuracy optimizer.

The runner also reports top-1/top-2/top-3 scores, top-1-minus-top-2 margins,
counts, and min/max/mean/median/p10/p25/p75/p90 distributions for answerable
and unsupported cases. A margin candidate is compared but is not automatically
selected. The checked-in baseline remains top-1 only unless a future, documented
evaluation justifies extra complexity.

## Provider binding

Calibration artifacts contain provider name, model name, embedding dimension,
dataset version, retrieval limit, and chunk configuration. The Evidence Gate
fails explicitly if the runtime provider, model, or dimension differs. A
deterministic-provider threshold must never be reused for OpenAI embeddings.

## Run locally

```bash
course-rag calibrate-retrieval \
  examples/retrieval-evaluation/dataset.json \
  --output runtime/evaluations/m2b-calibration.json

course-rag inspect-evidence \
  --course <course-id> \
  --query "What is the Blueleaf coefficient?" \
  --calibration runtime/evaluations/m2b-calibration.json
```

The default deterministic provider is offline. A live OpenAI calibration is
optional and requires an explicitly configured key; it is never run in tests.

## Limitations

Raw vector similarity is a retrieval signal, not factual confidence. The corpus
is fictional and the default embedding provider is lexical, so the resulting
threshold is only a reproducible synthetic baseline. It does not prove reliable
grounded answering or a hallucination rate. The current holdout result must be
read as an evaluation finding, not tuned away by editing holdout labels.

Generation and generated-answer citation validation remain disabled.
