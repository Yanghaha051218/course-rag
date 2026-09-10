# Grounding policy

## Purpose

This policy defines what CourseRAG may treat as evidence and how it must behave
when answering questions about a selected course. It is normative for future
retrieval and answer-generation work. Milestone 3-A adds an evidence Support
Verifier after retrieval, while preserving the M2-B calibrated similarity gate
as an experimental baseline. Neither component generates answers.

## Three distinct concepts

### Model pretrained knowledge

Pretrained knowledge is information encoded in a model before the current
CourseRAG request. It may help the model understand language, follow the response
format, and reason over supplied passages. It is **not evidence** for a factual
claim about course content and has no acceptable source locator in this system.

### Retrieved course evidence

Retrieved course evidence is content that:

- came from a user-provided document successfully ingested for the selected
  course;
- was returned by retrieval constrained to that exact course;
- retains a validated document ID, chunk ID, course ID, and source locator; and
- is included in the approved evidence set passed to generation.

Only this evidence may support factual course-content claims.

### Closed corpus

"Closed corpus" means the answerable factual universe is limited to the
user-provided materials belonging to the currently selected course. The model's
pretrained knowledge, unrelated courses, web search, external databases, and
uncited inference are outside that corpus.

Closed corpus does not mean the model forgets its training. It means backend
control flow and answer validation prevent that training from being used as an
unsupported factual source.

## Required behavior

1. Every material factual claim in a course-content answer must be supported by
   one or more retrieved passages from the selected course.
2. Citations must identify evidence that was actually retrieved and approved for
   that request; the model may not invent document names, chunk IDs, or locators.
3. If evidence is missing, too weak, contradictory without resolution, or does
   not cover the asked claim, the system must return an explicit
   `insufficient_evidence` response.
4. When evidence is insufficient, backend code must stop the workflow before any
   generation-provider call.
5. Operational failures are errors, not evidence insufficiency, and must not be
   silently converted into an answer or abstention.
6. Evidence from any course other than the selected course is forbidden, even if
   it would answer the question correctly.

## Enforcement layers

Strict grounding will be enforced in both of these ways:

1. **Course-scoped vector retrieval (implemented):** Qdrant filters the selected
   course and SQLite validates provenance.
2. **Similarity-gate baseline (implemented):** a provider-bound M2-B threshold
   remains available for evaluation and comparison; it is not a prerequisite for
   M3-A support verification.
3. **Support Verifier (implemented):** it receives only course-scoped retrieved
   evidence and returns `SUPPORTED`, `INSUFFICIENT`, or `CONFLICTING` with
   validated chunk bindings.
4. **Generator instructions (planned):** a future generator will receive only
   approved evidence.
5. **Citation/support validation (planned):** generated claims will be checked
   against approved evidence.

### A. Procedurally in backend code

Backend orchestration requires a selected course, retrieves with a storage-level
course filter, validates every result's course ID and SQLite provenance, runs an
explicit support verifier, and prevents future generation-provider invocation
unless the result is `SUPPORTED`. It will then validate returned citations
against the exact approved evidence set.

This is the primary enforcement mechanism and must be covered by tests,
including tests that assert the generation provider was not called.

### B. Through model instructions

When generation is permitted, the model will be instructed to use only the
provided evidence, attach citations, avoid unsupported claims, and state when a
requested detail is not present.

Model instructions are defense in depth. **Prompting alone is not sufficient
enforcement.** Models can misunderstand instructions, be influenced by prompt
injection inside documents, or produce plausible unsupported text. Permissions,
course isolation, evidence gating, and citation validation therefore remain in
deterministic backend code.

## Evidence-sufficiency policy

Embedding similarity answers "How related is this retrieved text to the
query?" It does not reliably answer "Does this text contain enough evidence to
answer the query?" The M2-B top-1 experiment found overlap between answerable
and unsupported scores, including zero answerable holdout allows at its strict
threshold. The threshold remains a provider-specific baseline, not factual
confidence or a mandatory M3-A gate.

The Support Verifier evaluates the relationship between the actual question and
the retrieved evidence. Its structured result is still untrusted until backend
code validates its status and evidence bindings. Whatever future method is
chosen must be testable without a live model and conservative under ambiguity.

At minimum, evidence is insufficient when:

- retrieval returns no eligible chunks;
- all matches are outside the selected course;
- the retrieved passages do not address the key entities or relation in the
  question;
- a requested claim lacks a supporting passage and locator; or
- provenance validation fails.

## Permitted model contribution

After the evidence gate passes, a model may summarize, compare, organize,
calculate from, or explain the approved evidence. It may use general language
ability to produce a clear response. It may not add facts merely because they
are likely, widely known, or present in its training.

If an inference is allowed in a later product design, it must be clearly marked,
traceable to cited premises in the selected course, and validated by explicit
backend policy rather than accepted solely because the model labels it an
inference.
