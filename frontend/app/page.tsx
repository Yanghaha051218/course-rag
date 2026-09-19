"use client";

import { FormEvent, useEffect, useState } from "react";

type Course = { id: string; name: string; created_at: string };
type User = { id: string; email: string; created_at: string };
type Document = {
  id: string;
  filename: string;
  file_type: string;
  source_units: number;
  size_bytes: number;
  created_at: string;
};
type Evidence = {
  rank: number;
  chunk_id: string;
  filename: string;
  text: string;
  score: number;
  source_type: string;
  source_start: number;
  source_end: number;
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api-backend${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "Request failed");
  }
  if (response.status === 204) return undefined as T;
  return response.json();
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function Home() {
  const [courses, setCourses] = useState<Course[]>([]);
  const [user, setUser] = useState<User | null>(null);
  const [courseId, setCourseId] = useState("");
  const [documents, setDocuments] = useState<Document[]>([]);
  const [evidence, setEvidence] = useState<Evidence[] | null>(null);
  const [message, setMessage] = useState("Loading courses…");
  const [busy, setBusy] = useState(false);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");

  async function authenticate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setBusy(true);
    try {
      const nextUser = await api<User>(`/auth/${authMode}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ email: form.get("email"), password: form.get("password") }),
      });
      setUser(nextUser);
      setMessage("Loading your courses…");
      await refresh();
      event.currentTarget.reset();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Authentication failed");
    } finally {
      setBusy(false);
    }
  }

  async function logout() {
    setBusy(true);
    try {
      await api<void>("/auth/logout", { method: "POST" });
    } finally {
      setUser(null);
      setCourses([]);
      setCourseId("");
      setDocuments([]);
      setEvidence(null);
      setMessage("Signed out");
      setBusy(false);
    }
  }

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
    void api<User>("/auth/me")
      .then((currentUser) => {
        setUser(currentUser);
        return api<{ items: Course[] }>("/courses");
      })
      .then((data) => {
        setCourses(data.items);
        setCourseId(data.items[0]?.id || "");
        setMessage(data.items.length ? "Ready" : "Create your first course to begin.");
      })
      .catch((error) => {
        setUser(null);
        setMessage(error instanceof Error && error.message === "Authentication required" ? "Sign in to manage your courses." : error instanceof Error ? error.message : "Could not reach the API");
      });
  }, []);

  useEffect(() => {
    if (!courseId) {
      return;
    }
    void api<{ items: Document[] }>(`/courses/${courseId}/documents`)
      .then((data) => setDocuments(data.items))
      .catch((error) => setMessage(error instanceof Error ? error.message : "Could not load documents"));
  }, [courseId]);

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
      const documents = await api<{ items: Document[] }>(`/courses/${courseId}/documents`);
      setDocuments(documents.items);
      setMessage(`${file.name} is indexed and ready.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  async function removeDocument(document: Document) {
    if (!courseId || !window.confirm(`Delete ${document.filename}?`)) return;
    setBusy(true);
    try {
      await api(`/courses/${courseId}/documents/${document.id}`, { method: "DELETE" });
      setDocuments((items) => items.filter((item) => item.id !== document.id));
      setMessage(`${document.filename} was deleted.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not delete document");
    } finally {
      setBusy(false);
    }
  }

  async function ask(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const question = new FormData(event.currentTarget).get("question");
    if (!courseId) return;
    setBusy(true);
    setEvidence(null);
    setMessage("Searching course evidence…");
    try {
      const response = await api<{ items: Evidence[] }>(
        `/courses/${courseId}/evidence`,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ question }),
        },
      );
      setEvidence(response.items);
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
        <div className="header-actions">
          {user && <span className="status">{user.email}</span>}
          {user && <button type="button" className="sign-out" onClick={() => void logout()} disabled={busy}>Sign out</button>}
          <p className="status" role="status">{message}</p>
        </div>
      </header>
      {!user ? (
        <section className="auth-card" aria-labelledby="auth-heading">
          <p className="eyebrow">Private course workspace</p>
          <h2 id="auth-heading">{authMode === "login" ? "Sign in to CourseRAG" : "Create your CourseRAG account"}</h2>
          <p className="intro">Your courses and uploaded materials are isolated to your account.</p>
          <form onSubmit={authenticate}>
            <label htmlFor="email">Email</label>
            <input id="email" name="email" type="email" required maxLength={254} autoComplete="email" />
            <label htmlFor="password">Password</label>
            <input id="password" name="password" type="password" required minLength={8} maxLength={256} autoComplete={authMode === "login" ? "current-password" : "new-password"} />
            <button disabled={busy}>{authMode === "login" ? "Sign in" : "Register"}</button>
          </form>
          <button type="button" className="auth-toggle" onClick={() => setAuthMode(authMode === "login" ? "register" : "login")} disabled={busy}>
            {authMode === "login" ? "Need an account? Register" : "Already have an account? Sign in"}
          </button>
        </section>
      ) : (
      <div className="workspace">
        <aside aria-label="Course setup">
          <section>
            <h2>Your course</h2>
            <label htmlFor="course">Selected course</label>
            <select id="course" value={courseId} onChange={(event) => { setCourseId(event.target.value); if (!event.target.value) setDocuments([]); }} disabled={busy}>
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
            <small>PDF, PPTX, DOCX, Markdown or text · 50 MB per file · course quota enforced by the server</small>
          </form>
          <section className="documents" aria-labelledby="documents-heading">
            <h2 id="documents-heading">Course materials</h2>
            {!courseId || documents.length === 0 ? (
              <p className="reason">No materials uploaded yet.</p>
            ) : (
              <ul>
                {documents.map((document) => (
                  <li key={document.id} className="document-item">
                    <div>
                      <strong>{document.filename}</strong>
                      <span>{formatBytes(document.size_bytes)} · {document.source_units} source units</span>
                    </div>
                    <button
                      type="button"
                      className="document-delete"
                      aria-label={`Delete ${document.filename}`}
                      disabled={busy}
                      onClick={() => void removeDocument(document)}
                    >
                      Delete
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </aside>
        <section className="conversation">
          <div>
            <p className="eyebrow">Free local retrieval</p>
            <h2>Search your materials</h2>
            <p className="intro">CourseRAG searches only the selected course and shows the original evidence. No generation model or paid API is used.</p>
          </div>
          <form className="ask" onSubmit={ask}>
            <label htmlFor="question">Question</label>
            <textarea id="question" name="question" required maxLength={4000} placeholder="What does the course say about…?" />
            <button disabled={busy || !courseId}>Find course evidence</button>
          </form>
          {evidence && (
            <section className="evidence" aria-live="polite">
              <p className="result-label">Retrieved evidence</p>
              {evidence.length === 0 ? (
                <p className="reason">No course evidence was retrieved.</p>
              ) : evidence.map((item) => (
                <article key={item.chunk_id} className="evidence-item">
                  <div>
                    <strong>#{item.rank} · {item.filename}</strong>
                    <span>{item.source_type} {item.source_start}{item.source_end === item.source_start ? "" : `–${item.source_end}`} · score {item.score.toFixed(3)}</span>
                  </div>
                  <p>{item.text}</p>
                </article>
              ))}
            </section>
          )}
        </section>
      </div>
      )}
    </main>
  );
}
