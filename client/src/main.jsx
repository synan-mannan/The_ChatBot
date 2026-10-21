import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./styles.css";

function getSessionId() {
  const saved = localStorage.getItem("aijob-session-id");
  if (saved) return saved;
  const created = crypto.randomUUID();
  localStorage.setItem("aijob-session-id", created);
  return created;
}

const SESSION_ID = getSessionId();
const TOOL_LABELS = {
  parse_resume: "Resume profile",
  recommend_jobs: "Job matching",
  get_job_details: "Job details",
  apply_to_job: "Application recorded",
  list_applications: "Application history"
};

async function responseJson(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "Something went wrong. Please try again.");
  }
  return payload;
}

function ProfileCard({ profile }) {
  const contact = [profile.email, profile.phone].filter(Boolean).join(" · ");
  return (
    <section className="profile-card" aria-label="Extracted resume profile">
      <div className="profile-heading">
        <div className="avatar">{(profile.name || "Y").slice(0, 1).toUpperCase()}</div>
        <div>
          <h3>{profile.name || "Resume profile"}</h3>
          <p>{profile.designation || "Role not specified"}</p>
        </div>
      </div>
      {contact && <p className="profile-contact">{contact}</p>}
      {profile.experience_years != null && (
        <p className="profile-meta">{profile.experience_years} years of experience</p>
      )}
      {!!profile.skills?.length && (
        <div className="skill-list">
          {profile.skills.map((skill) => <span className="skill-pill" key={skill}>{skill}</span>)}
        </div>
      )}
      {!!profile.education?.length && (
        <div className="profile-section">
          <span className="eyebrow">Education</span>
          {profile.education.map((item, index) => <p key={`${item}-${index}`}>{item}</p>)}
        </div>
      )}
      {!!profile.projects?.length && (
        <div className="profile-section">
          <span className="eyebrow">Projects</span>
          {profile.projects.map((item, index) => <p key={`${item}-${index}`}>{item}</p>)}
        </div>
      )}
    </section>
  );
}

function JobCards({ jobs, onApply }) {
  if (!jobs?.length) return null;
  return (
    <div className="job-list">
      {jobs.map((recommendation, index) => {
        const job = recommendation.job;
        return (
          <article className="job-card" key={job.id}>
            <div className="job-topline">
              <span className="job-number">MATCH {String(index + 1).padStart(2, "0")}</span>
              <span className="match-score">{Math.round(recommendation.score * 100)}% match</span>
            </div>
            <h3>{job.title}</h3>
            <p className="job-company">{job.company} <span>·</span> {job.location}</p>
            <div className="job-tags">
              {job.is_remote && <span className="remote-tag">Remote</span>}
              {job.experience_required !== "Not specified" && (
                <span>{job.experience_required}</span>
              )}
              {job.skills.slice(0, 4).map((skill) => <span key={skill}>{skill}</span>)}
            </div>
            <ul className="match-reasons">
              {recommendation.match_reasons.map((reason) => <li key={reason}>{reason}</li>)}
            </ul>
            <div className="job-actions">
              <button className="button button-small" onClick={() => onApply(job.id)}>
                Record application
              </button>
              <a href={job.source_link} target="_blank" rel="noreferrer">
                View original listing ↗
              </a>
            </div>
            <p className="job-source">Source: {job.source}</p>
          </article>
        );
      })}
    </div>
  );
}

function ApplicationList({ applications }) {
  if (!applications?.length) return <p className="empty-state">No applications recorded yet.</p>;
  return (
    <div className="applications-list">
      {applications.map((application) => (
        <div className="application-row" key={`${application.job_id}-${application.applied_date}`}>
          <div>
            <strong>{application.title}</strong>
            <span>{application.company} · {new Date(application.applied_date).toLocaleDateString()}</span>
          </div>
          <span className="status-pill">{application.status}</span>
        </div>
      ))}
    </div>
  );
}

function ToolChips({ tools }) {
  if (!tools?.length) return null;
  return (
    <div className="tool-chips" aria-label="Tools used">
      {tools.map((tool, index) => (
        <span className="tool-chip" key={`${tool.name}-${index}`}>
          <span className="tool-sparkle">✦</span> Used: {TOOL_LABELS[tool.name] || tool.name}
        </span>
      ))}
    </div>
  );
}

function Message({ message, onApply }) {
  const isUser = message.role === "user";
  return (
    <div className={`message-row ${isUser ? "message-user" : "message-assistant"}`}>
      {!isUser && <div className="assistant-mark" aria-hidden="true">✳</div>}
      <div className="message-content">
        <div className={`bubble ${isUser ? "bubble-user" : "bubble-assistant"}`}>
          {isUser
            ? <p>{message.content}</p>
            : <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>}
        </div>
        {!isUser && <ToolChips tools={message.toolCalls} />}
        {!isUser && message.profile && <ProfileCard profile={message.profile} />}
        {!isUser && <JobCards jobs={message.jobs} onApply={onApply} />}
        {!isUser && message.applications && <ApplicationList applications={message.applications} />}
      </div>
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState([{
    id: "welcome",
    role: "assistant",
    content: "Hi, I’m **AIJob**. Upload your resume and I’ll help you find roles that fit, refine your search, and keep track of applications."
  }]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const fileInput = useRef(null);
  const endOfMessages = useRef(null);

  useEffect(() => {
    endOfMessages.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  async function sendMessage(text = input) {
    const messageText = text.trim();
    if (!messageText || busy) return;
    setError("");
    setInput("");
    setMessages((current) => [...current, {
      id: crypto.randomUUID(),
      role: "user",
      content: messageText
    }]);
    setBusy(true);
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: SESSION_ID, message: messageText })
      });
      const payload = await responseJson(response);
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: "assistant",
        content: payload.reply,
        toolCalls: payload.tool_calls || [],
        jobs: payload.jobs,
        profile: payload.profile,
        applications: payload.applications
      }]);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  async function uploadResume(file) {
    if (!file || busy) return;
    setError("");
    if (file.size > 5 * 1024 * 1024) {
      setError("That resume is over 5 MB. Please choose a smaller file.");
      return;
    }
    setBusy(true);
    const form = new FormData();
    form.append("session_id", SESSION_ID);
    form.append("file", file);
    setMessages((current) => [...current, {
      id: crypto.randomUUID(),
      role: "user",
      content: `Uploaded ${file.name}`
    }]);
    try {
      const response = await fetch("/api/upload", { method: "POST", body: form });
      const payload = await responseJson(response);
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: "assistant",
        content: payload.message || "Your resume is ready. I’ve extracted the profile below.",
        profile: payload.profile,
        toolCalls: [{ name: "parse_resume" }]
      }]);
    } catch (requestError) {
      setError(requestError.message);
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: "assistant",
        content: requestError.message
      }]);
    } finally {
      setBusy(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  function applyToJob(jobId) {
    sendMessage(`Record my application for dataset job ID id:${jobId}`);
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="AIJob home">
          <span className="brand-icon">✳</span>
          <span>AIJob</span>
        </a>
        <div className="sidebar-copy">
          <span className="eyebrow">YOUR CAREER, IN MOTION</span>
          <h1>Find work that <em>fits you.</em></h1>
          <p>One conversation from your resume to your next opportunity.</p>
        </div>
        <div className="sidebar-steps">
          <div><span>01</span><p>Share your resume</p></div>
          <div><span>02</span><p>Explore matched roles</p></div>
          <div><span>03</span><p>Track applications</p></div>
        </div>
        <div className="sidebar-foot">
          <span className="online-dot" /> Private by design
          <p>No account needed. Your chat stays in this browser session.</p>
        </div>
      </aside>

      <section className="chat-panel">
        <header className="chat-header">
          <div className="header-assistant">
            <div className="assistant-mark">✳</div>
            <div><strong>AIJob Assistant</strong><span><i /> Here to help with your search</span></div>
          </div>
          <button className="header-action" onClick={() => sendMessage("Show my applied jobs")}>
            <span>▤</span> My applications
          </button>
        </header>

        <div className="conversation">
          <div className="date-divider"><span>YOUR JOB SEARCH STARTS HERE</span></div>
          {messages.map((message) => (
            <Message key={message.id} message={message} onApply={applyToJob} />
          ))}
          {busy && (
            <div className="message-row message-assistant typing-row">
              <div className="assistant-mark">✳</div>
              <div className="typing-indicator" aria-label="AIJob is thinking">
                <span /><span /><span />
              </div>
            </div>
          )}
          <div ref={endOfMessages} />
        </div>

        <div className="composer-wrap">
          {error && <div className="error-banner" role="alert">{error}</div>}
          <form className="composer" onSubmit={(event) => { event.preventDefault(); sendMessage(); }}>
            <input
              ref={fileInput}
              className="visually-hidden"
              type="file"
              accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              onChange={(event) => uploadResume(event.target.files?.[0])}
              aria-label="Upload a PDF or DOCX resume"
            />
            <button
              className="upload-button"
              type="button"
              onClick={() => fileInput.current?.click()}
              disabled={busy}
              aria-label="Upload resume"
              title="Upload resume (PDF or DOCX)"
            >＋</button>
            <input
              className="message-input"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask about your next opportunity..."
              aria-label="Message AIJob"
              disabled={busy}
            />
            <button className="send-button" type="submit" disabled={busy || !input.trim()} aria-label="Send message">
              ↑
            </button>
          </form>
          <p className="composer-note">AIJob can make mistakes. Always verify details with the original listing.</p>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode><App /></React.StrictMode>
);
