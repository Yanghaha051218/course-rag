# Real-course pilot: shareable aggregate report

Copy this template to an ignored path such as
`runtime/evaluations/real-course-pilot.md`. Do not commit a completed report,
course names, filenames, source excerpts, benchmark questions, model outputs,
instructor information, or credentials.

## Scope and configuration

- Pilot status: `<completed | structural only | not run>`
- Corpus authorization confirmed: `<yes | no>`
- Embedding provider / model / dimension: `<provider> / <model> / <dimension>`
- Support verifier provider / model: `<provider> / <model | not run>`
- Generator provider / model: `<provider> / <model | not run>`
- Retrieval `k`: `<integer>`
- Configuration fixed before the run: `<yes | no>`

## Anonymized corpus and benchmark

| Measure | Count |
| --- | ---: |
| Documents | `<n>` |
| Source units (pages/slides/sections) | `<n>` |
| Chunks | `<n>` |
| Benchmark questions | `<n>` |
| Answerable | `<n>` |
| Unsupported | `<n>` |
| Near-miss / wrong-attribute | `<n>` |
| Multi-source | `<n>` |

## Retrieval

| Metric | Value |
| --- | ---: |
| Hit@1 / @3 / @5 | `<x> / <x> / <x>` |
| Recall@1 / @3 / @5 | `<x> / <x> / <x>` |
| MRR | `<x>` |

## Support verification

| Outcome | Count |
| --- | ---: |
| Answerable → SUPPORTED | `<n>` |
| Answerable → not SUPPORTED | `<n>` |
| Unsupported or near-miss → SUPPORTED | `<n>` |
| Unsupported or near-miss → not SUPPORTED | `<n>` |
| Support acceptance rate | `<x>` |
| False support rate | `<x>` |

## Grounded generation and citations

| Outcome | Count / rate |
| --- | ---: |
| ANSWERED | `<n>` |
| ABSTAINED | `<n>` |
| Answerable end-to-end coverage | `<n> / <n> = x>` |
| Unsupported generation rate | `<n> / <n> = x>` |
| Structural citation validity | `<n> / <n> = x>` |
| Citation binding failures | `<n>` |
| Unsupported claim rate | `<n> / <n> = x>` |
| Human semantic citation review | `<n> supporting / <n> reviewed = x>` |

## Failure analysis and manual audit

| Primary failure category | Count |
| --- | ---: |
| Retrieval failure | `<n>` |
| Support false negative | `<n>` |
| Generation failure | `<n>` |
| Citation validation failure | `<n>` |
| Correct unsupported abstention | `<n>` |
| False support | `<n>` |
| False generation | `<n>` |

Manually reviewed generated answers: `<n>`. Aggregate substantive claims with
support: `<n> / <n>`. Aggregate citations supporting their claim wording:
`<n> / <n>`. Record only anonymized failure types, never excerpts.

## M2-B threshold baseline

Only complete this comparison when the embedding provider, model, dimension,
and indexing configuration match the calibration artifact. Otherwise state that
the synthetic `0.782508` threshold is provider-specific and was not applied as
a production decision rule.

- Threshold-only answerable acceptance: `<x | not applicable>`
- Threshold-only unsupported false acceptance: `<x | not applicable>`
- SupportVerifier production result: `<aggregate comparison>`
