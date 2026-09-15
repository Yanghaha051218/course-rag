"use client";

import { FormEvent, useEffect, useState } from "react";

type Course = { id: string; name: string; created_at: string };
type Citation = {
  chunk_id: string;
  filename: string;
  source_type: string;
  source_start: number;
  source_end: number;
};
type Answer = {
  status: "ANSWERED" | "ABSTAINED";
  answer: string | null;
  citations: Citation[];
  reason: string | null;
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api-backend${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "Request failed");
  }
  return response.json();
}

export default function Home() {
  const [courses, setCourses] = useState<Course[]>([]);
  const [courseId, setCourseId] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [message, setMessage] = useState("Loading courses…");
  const [busy, setBusy] = useState(false);

  async function refresh() {
    try {
      const data = await api<{ items: Course[] }>("/courses");
      setCourses(data.items);
      setCourseId((value) => value || data.items[0]?.id || "");
      setMessage(data.items.length ? "Ready" : "Create your first course to begin.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not reach the API");
    }
  }

  useEffect(() => {
    void api<{ items: Course[] }>("/courses")
      .then((data) => {
        setCourses(data.items);
        setCourseId(data.items[0]?.id || "");
        setMessage(data.items.length ? "Ready" : "Create your first course to begin.");
      })
      .catch((error) =>
        setMessage(error instanceof Error ? error.message : "Could not reach the API"),
      );
  }, []);

  async function createCourse(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setBusy(true);
    try {
      const course = await api<Course>("/courses", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name: form.get("name") }),
      });
      event.currentTarget.reset();
      await refresh();
      setCourseId(course.id);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not create course");
    } finally {
      setBusy(false);
    }
  }

  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const file = new FormData(event.currentTarget).get("document");
    if (!(file instanceof File) || !courseId) return;
    setBusy(true);
    setMessage(`Uploading ${file.name}…`);
    try {
      await api(`/courses/${courseId}/documents?filename=${encodeURIComponent(file.name)}`, {
        method: "POST",
        headers: { "content-type": "application/octet-stream" },
        body: file,
      });
      event.currentTarget.reset();
      setMessage(`${file.name} is indexed and ready.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  async function ask(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const question = new FormData(event.currentTarget).get("question");
    if (!courseId) return;
    setBusy(true);
    setAnswer(null);
    setMessage("Checking course evidence…");
    try {
      setAnswer(
        await api<Answer>(`/courses/${courseId}/questions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ question }),
        }),
      );
      setMessage("Ready");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Question failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">Citation-first course research</p>
          <h1>CourseRAG</h1>
        </div>
        <p className="status" role="status">{message}</p>
      </header>
      <div className="workspace">
        <aside aria-label="Course setup">
          <section>
            <h2>Your course</h2>
            <label htmlFor="course">Selected course</label>
            <select id="course" value={courseId} onChange={(event) => setCourseId(event.target.value)} disabled={busy}>
              <option value="">Choose a course</option>
              {courses.map((course) => <option key={course.id} value={course.id}>{course.name}</option>)}
            </select>
          </section>
          <form onSubmit={createCourse}>
            <label htmlFor="name">New course</label>
            <div className="row">
              <input id="name" name="name" required maxLength={120} placeholder="Differential Equations" />
              <button disabled={busy}>Add</button>
            </div>
          </form>
          <form onSubmit={upload}>
            <label htmlFor="document">Add course material</label>
            <input id="document" name="document" type="file" required accept=".pdf,.pptx,.docx,.md,.txt" />
            <button disabled={busy || !courseId}>Upload and index</button>
            <small>PDF, PPTX, DOCX, Markdown or text · 50 MB max</small>
          </form>
        </aside>
        <section className="conversation">
          <div>
            <p className="eyebrow">Evidence-bound answers</p>
            <h2>Ask your materials</h2>
            <p className="intro">CourseRAG searches only the selected course. If evidence is insufficient or conflicting, it abstains.</p>
          </div>
          <form className="ask" onSubmit={ask}>
            <label htmlFor="question">Question</label>
            <textarea id="question" name="question" required maxLength={4000} placeholder="What does the course say about…?" />
            <button disabled={busy || !courseId}>Check evidence and answer</button>
          </form>
          {answer && (
            <article className={`result ${answer.status.toLowerCase()}`} aria-live="polite">
              <p className="result-label">{answer.status}</p>
              <p className="answer">{answer.answer || "I can’t answer this from the available course evidence."}</p>
              {answer.reason && <p className="reason">Reason: {answer.reason}</p>}
              {answer.citations.length > 0 && (
                <>
                  <h3>Sources</h3>
                  <ul>
                    {answer.citations.map((citation) => (
                      <li key={citation.chunk_id}>
                        <strong>{citation.filename}</strong>
                        <span>
                          {citation.source_type} {citation.source_start}
                          {citation.source_end === citation.source_start ? "" : `–${citation.source_end}`}
                        </span>
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </article>
          )}
        </section>
      </div>
    </main>
  );
}
