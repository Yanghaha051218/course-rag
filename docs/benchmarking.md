# Real-course benchmarking

## Purpose

M4-A evaluates CourseRAG on authorized local course material rather than
claiming that synthetic tests establish real-world quality. It answers separate
questions about retrieval, support verification, citation validation, and
failure modes. It does not add a UI, deployment, course scraping, or automatic
question generation.

## Privacy

Real lecture notes, slides, assignments, and benchmark questions may be
copyrighted or private. They are intentionally excluded from Git history:

- put local benchmark JSON under `benchmarks/local/`;
- keep source files outside the repository or under ignored `runtime/` storage;
- write reports under `runtime/evaluations/`; and
- publish only authorized, aggregated results after review.

The repository includes only the schema and a tiny fictional
[`synthetic-demo.json`](../benchmarks/examples/synthetic-demo.json), which CI
loads and exercises without external services. Never commit real PDFs, PPTX,
DOCX, course names, professor names, exam material, API keys, or a populated
local benchmark.

## Human annotation workflow

1. An instructor or student with permission chooses a source passage.
2. They write one question and classify it as `answerable`, `unsupported`, or
   `conflict`.
3. For answerable or conflict questions, they record the original filename,
   source type, and inclusive page, slide, or section range.
4. They choose a category: direct factual, concept explanation, multi-document,
   near miss, or citation sensitive.
5. A second authorized reviewer checks the locator and label before a real run.

Do not generate benchmark questions automatically. That would risk measuring
model-generated assumptions rather than course evidence. The full format is in
[`benchmarks/schema.md`](../benchmarks/schema.md); it deliberately has no answer
field.

## Local workflow

First create, ingest, and index an authorized course manually:

```bash
course-rag create-course "Authorized Physics 2"
course-rag ingest --course <course-id> /authorized/path/week-01.pdf
course-rag ingest --course <course-id> /authorized/path/week-02.pdf
course-rag index-course --course <course-id>
```

Then run retrieval-only benchmarking:

```bash
course-rag benchmark \
  --dataset benchmarks/local/physics2.json \
  --course <course-id> \
  --output runtime/evaluations/physics2-retrieval.json
```

Use `--verify-support` to call the configured verifier, or add `--generate` to
also exercise the grounded generator and citation validator. Generation requires
support verification, and both OpenAI stages require `OPENAI_API_KEY`. No model
call happens in the default retrieval-only run. The runner requires a retrieval
limit of at least five so Hit@1, Hit@3, and Hit@5 are meaningful.

The report records the dataset version; embedding provider, model, and dimension;
retrieval limit; and enabled verifier/generator identities. It contains source
locators and local questions but never course text. Treat the JSON as local data.

## Metrics

Retrieval is evaluated only on cases with expected evidence:

- **Hit@1/3/5:** at least one expected locator appears by that rank.
- **Recall@1/3/5:** fraction of expected locators found by that rank.
- **FullEvidence@1/3/5:** all expected locators appear by that rank.
- **MRR:** reciprocal rank of the first expected locator, averaged over cases.

Hit measures whether retrieval found any relevant evidence. Recall measures how
much of the annotated evidence set it found. FullEvidence is especially useful
for multi-region questions: it is zero until every required locator is present,
even if Hit is already one.

When enabled, support metrics are separate:

- **support_accept_rate:** answerable cases classified `SUPPORTED`;
- **false_support_rate:** unsupported cases incorrectly classified `SUPPORTED`;
- **conflict_detection_rate:** conflict cases classified `CONFLICTING`.

When generation is enabled, an `ANSWERED` result has already passed the M3-B
claim-to-approved-chunk and locator validation. `citation_validity_rate` is the
fraction of evaluated citation outcomes accepted by that validator.
`unsupported_claim_rate` is the fraction of supported generation attempts
rejected specifically for invalid claim bindings. It is not a semantic entailment
metric: assess claim wording separately during human failure review.

## Failure analysis

Every report includes per-case failures, separate from aggregate scores:

- `retrieval_failure` — an expected locator was absent through rank five;
- `support_verification_failure` — the observed support status differs from the
  annotated type;
- `citation_failure` — claim bindings or locators failed M3-B validation;
- `generation_failure` — a supported attempt abstained for another generator
  error; and
- `dataset_error` — reserved for invalid annotations, which fail dataset loading
  before a report is produced.

A real pilot has not been run because no authorized course corpus was provided.
Do not infer real-course performance from the synthetic demo.
