# Synthetic retrieval evaluation corpus

This directory contains only small, original fixtures written for CourseRAG.
`dataset.json` defines two contradictory fictional courses, their documents,
and labeled retrieval cases. The fixed chunk settings make chunk-level evidence
references reproducible.

The cases cover exact lookup, paraphrase, same-course cross-document retrieval,
multi-chunk evidence, unsupported and near-miss questions, contradictory facts
between courses, and vocabulary overlap between courses.
