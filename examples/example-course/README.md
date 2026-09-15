# Synthetic example course

This fixture is tiny, original, and created specifically for CourseRAG tests. It
does not reproduce a textbook, paper, lecture, assignment, or real university
course. Ingestion tests may use it without requiring private or copyrighted
material.

The Markdown and text files are committed directly. Generate the original PDF,
PPTX, and DOCX examples under ignored runtime storage with:

```bash
python examples/example-course/generate_documents.py runtime/example-course
```

The generated documents contain identifiable fictional facts, multiple source
units, and no copied course content.
