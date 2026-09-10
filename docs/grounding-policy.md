# Grounding policy

## Purpose

This policy defines what CourseRAG may treat as evidence and how it must behave
when answering questions about a selected course. It is normative for future
retrieval and answer-generation work. Milestone 2-B adds a calibrated,
provider-bound Evidence Gate for raw course-scoped evidence, but does not
generate answers.

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
2. **Calibrated Evidence Gate (implemented):** a provider-bound retrieval policy
   permits or abstains without generating an answer.
3. **Generator instructions (planned):** a future generator will receive only
   approved evidence.
4. **Citation/support validation (planned):** generated claims will be checked
   against approved evidence.

### A. Procedurally in backend code

Backend orchestration will require a selected course, retrieve with a
storage-level course filter, verify every result's course ID, run an explicit
evidence-sufficiency gate, and prevent provider invocation on the insufficient
branch. It will then validate returned citations against the exact approved
evidence set.

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

The first top-1 retrieval policy is calibrated from an explicit synthetic split.
Its threshold is provider-specific and is not factual confidence. Whatever
future method is chosen must be deterministic at the
orchestration boundary, testable without a live model, and conservative under
ambiguity.

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
