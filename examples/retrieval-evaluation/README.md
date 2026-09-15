# Synthetic retrieval evaluation corpus

This directory contains only small, original fixtures written for CourseRAG.
`dataset.json` version `m2b-v1` defines two contradictory fictional courses,
their documents, and 40 explicitly split labeled cases (28 calibration, 12
holdout). Fixed chunk settings make chunk-level evidence references reproducible.

The cases cover exact lookup, paraphrase, same-course cross-document retrieval,
multi-chunk evidence, unsupported and near-miss questions, wrong attributes,
adversarial lexical overlap, contradictory facts between courses, and vocabulary
overlap between courses.
