# M4-A benchmark schema

Benchmark files are UTF-8 JSON. They record human annotation about whether a
question is supported and where its evidence should be found; they never store a
reference answer or generated answer.

```json
{
  "version": "course-benchmark-v1",
  "cases": [
    {
      "id": "physics-gauss-law-001",
      "course_id": "physics2",
      "question": "State Gauss's law.",
      "type": "answerable",
      "category": "direct_factual",
      "expected": {
        "supporting_sources": [
          {"document": "week-03.pdf", "source_type": "page", "start": 12, "end": 12}
        ]
      }
    }
  ]
}
```

All top-level and case fields are required. IDs are unique. `course_id` is a
human-maintained benchmark label, not a SQLite UUID: the runner receives the
local course UUID separately through `--course`. One dataset represents one
course, so every case must use the same `course_id` label.

## Case types

- `answerable` requires one or more `supporting_sources` and should lead to
  `SUPPORTED`.
- `unsupported` requires an empty source list and should lead to `INSUFFICIENT`.
- `conflict` requires at least two sources and should lead to `CONFLICTING`.

Categories are `direct_factual`, `concept_explanation`, `multi_document`,
`near_miss`, and `citation_sensitive`. They describe the question design, not
the expected wording of an answer.

Each source is `{document, source_type, start, end}`. `document` is the original
filename only, while `source_type` is `page`, `slide`, or `section`. Ranges are
inclusive positive integers and must match the locally ingested source locator.
