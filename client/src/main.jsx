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
  apply_to_job: "Application status",
  generate_cover_letter: "Cover letter draft",
  approve_cover_letter: "Cover letter approved",
  save_application_answers: "Application answers saved",
  list_applications: "Applications list"
};
const APPLICATION_STATUS_LABELS = {
  submitted: "Submitted",
  dry_run: "Dry run",
  review_ready: "Ready for review — not submitted",
  needs_input: "Answers needed",
  needs_manual: "Needs manual action",
  failed: "Failed",
  retry_later: "Retry later",
  in_progress: "In progress",
  duplicate: "Already recorded"
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
                Apply
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
            {application.method && <span>Method: {application.method}</span>}
            {application.detail && <p className="application-detail">{application.detail}</p>}
            {application.apply_url && (
              <a className="application-link" href={application.apply_url} target="_blank" rel="noreferrer">
                Open application page ↗
              </a>
            )}
            {application.detected_ats && (
              <span>ATS: {application.detected_ats}</span>
            )}
            {!!application.filled?.length && (
              <div className="application-report-section">
                <strong>Fields filled</strong>
                <ul>
                  {application.filled.map((field, index) => (
                    <li key={`${field.label}-${index}`}>{field.label}</li>
                  ))}
                </ul>
              </div>
            )}
            {!!application.needs_input?.length && (
              <div className="application-report-section">
                <strong>Reply in chat with these answers</strong>
                <ul>
                  {application.needs_input.map((field, index) => (
                    <li key={`${field.label}-${index}`}>
                      {field.label}{field.required ? " (required)" : ""}
                      {!!field.options?.length && <span> — {field.options.join(", ")}</span>}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {application.screenshot_url && (
              <img
                className="application-preview"
                src={application.screenshot_url}
                alt={`Application preview for ${application.title}`}
                loading="lazy"
              />
            )}
            {application.debug_screenshot_url && (
              <a className="application-link" href={application.debug_screenshot_url} target="_blank" rel="noreferrer">
                Open debug screenshot ↗
              </a>
            )}
          </div>
          <ApplicationStatus
            status={
              application.status === "review_ready" && (application.filled?.length || 0) < 3
                ? "needs_manual"
                : application.status
            }
          />
        </div>
      ))}
    </div>
  );
}

function ApplicationStatus({ status }) {
  return (
    <span className={`status-pill status-${status}`}>
      {APPLICATION_STATUS_LABELS[status] || status}
    </span>
  );
}

function ApplicationOutcome({ result, onSubmit, onCancel }) {
  const status = result.status === "review_ready" && (result.filled?.length || 0) < 3
    ? "needs_manual"
    : result.status;
  const canSubmit = ["review_ready", "dry_run"].includes(status);
  const canCancel = ["review_ready", "dry_run", "needs_input"].includes(status);
  return (
    <section className="application-outcome" aria-label="Application result">
      <div className="application-outcome-heading">
        <strong>{result.title} {result.company ? `· ${result.company}` : ""}</strong>
        <ApplicationStatus status={status} />
      </div>
      {result.detail && <p>{result.detail}</p>}
      {result.apply_url && (
        <a className="application-link" href={result.apply_url} target="_blank" rel="noreferrer">
          Open application page ↗
        </a>
      )}
      {result.detected_ats && <p>Detected ATS: {result.detected_ats}</p>}
      {result.screenshot_url && (
        <img
          className="application-preview"
          src={result.screenshot_url}
          alt={`Filled application preview for ${result.title}`}
          loading="lazy"
        />
      )}
      {!!result.filled?.length && (
        <div className="application-report-section">
          <strong>Fields filled</strong>
          <ul>
            {result.filled.map((field, index) => (
              <li key={`${field.label}-${index}`}>{field.label} <span>({field.key})</span></li>
            ))}
          </ul>
        </div>
      )}
      {!!result.needs_input?.length && (
        <div className="application-report-section">
          <strong>Reply in chat with these answers</strong>
          <ul>
            {result.needs_input.map((field, index) => (
              <li key={`${field.label}-${index}`}>
                {field.label}{field.required ? " (required)" : ""}
                {!!field.options?.length && <span> — {field.options.join(", ")}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
      {result.debug_screenshot_url && (
        <a className="application-link" href={result.debug_screenshot_url} target="_blank" rel="noreferrer">
          Open debug screenshot ↗
        </a>
      )}
      {!!result.discovered_form_labels?.length && (
        <div className="application-report-section">
          <strong>Form labels found</strong>
          <ul>
            {result.discovered_form_labels.map((label, index) => (
              <li key={`${label}-${index}`}>{label}</li>
            ))}
          </ul>
        </div>
      )}
      {(canSubmit || canCancel) && (
        <div className="job-actions">
          {canSubmit && (
            <button
              className="button button-small"
              onClick={() => onSubmit(result.job_id)}
              disabled={result.status === "dry_run"}
              title={result.status === "dry_run" ? "Submission is disabled by server configuration." : ""}
            >
              Submit
            </button>
          )}
          <button className="button button-small button-secondary" onClick={onCancel}>
            Cancel
          </button>
        </div>
      )}
    </section>
  );
}

function ApplicationConfirmation({ confirmation, onConfirm, onCancel }) {
  return (
    <section className="application-confirmation" aria-label="Confirm application">
      <p><strong>{confirmation.title}</strong> · {confirmation.company}</p>
      <p>Plan: {confirmation.plan}.</p>
      <p>Nothing will be sent until you confirm.</p>
      <div className="job-actions">
        <button className="button button-small" onClick={() => onConfirm(confirmation.job_id)}>
          {confirmation.plan?.includes("prepare and review") ? "Prepare for review" : "Yes, submit"}
        </button>
        <button className="button button-small button-secondary" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </section>
  );
}

function ToolChips({ tools, result, confirmation }) {
  if (!tools?.length) return null;
  return (
    <div className="tool-chips" aria-label="Tools used">
      {tools.map((tool, index) => (
        <span className="tool-chip" key={`${tool.name}-${index}`}>
          <span className="tool-sparkle">✦</span> Used: {
            tool.name === "apply_to_job"
              ? confirmation
                ? "Application plan"
                : ["review_ready", "dry_run"].includes(result?.status)
                  ? "Application form prepared"
                  : "Application status"
              : TOOL_LABELS[tool.name] || tool.name
          }
        </span>
      ))}
    </div>
  );
}

function Message({ message, onApply, onConfirm, onSubmit, onCancel }) {
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
        {!isUser && (
          <ToolChips
            tools={message.toolCalls}
            result={message.applicationResult}
            confirmation={message.applicationConfirmation}
          />
        )}
        {!isUser && message.profile && <ProfileCard profile={message.profile} />}
        {!isUser && <JobCards jobs={message.jobs} onApply={onApply} />}
        {!isUser && message.applications && <ApplicationList applications={message.applications} />}
        {!isUser && message.applicationConfirmation && (
          <ApplicationConfirmation
            confirmation={message.applicationConfirmation}
            onConfirm={onConfirm}
            onCancel={onCancel}
          />
        )}
        {!isUser && message.applicationResult && (
          <ApplicationOutcome
            result={message.applicationResult}
            onSubmit={onSubmit}
            onCancel={onCancel}
          />
        )}
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
      setMessages((current) => {
        const profileAlreadyShown = current.some(
          (message) => message.role === "assistant" && message.profile
        );
        return [...current, {
          id: crypto.randomUUID(),
          role: "assistant",
          content: payload.reply,
          toolCalls: payload.tool_calls || [],
          jobs: payload.jobs,
          profile: profileAlreadyShown ? undefined : payload.profile,
          applications: payload.applications,
          applicationConfirmation: payload.application_confirmation,
          applicationResult: payload.application_result
        }];
      });
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

  async function applyToJob(jobId, confirm = false, submit = false) {
    if (busy) return;
    setError("");
    setBusy(true);
    try {
      const response = await fetch("/api/applications/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: SESSION_ID, job_id: jobId, confirm, submit })
      });
      const payload = await responseJson(response);
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: "assistant",
        content: payload.message,
        toolCalls: [{ name: "apply_to_job" }],
        applicationConfirmation: payload.status === "confirmation_required" ? payload : undefined,
        applicationResult: [
          "submitted", "dry_run", "review_ready", "needs_input",
          "needs_manual", "failed", "retry_later", "duplicate"
        ]
          .includes(payload.status) ? payload : undefined
      }]);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  async function clearSession() {
    if (busy || !window.confirm("Clear this session, including your saved profile and previews?")) return;
    setError("");
    setBusy(true);
    try {
      const response = await fetch(
        `/api/session?session_id=${encodeURIComponent(SESSION_ID)}`,
        { method: "DELETE" }
      );
      if (!response.ok) await responseJson(response);
      setMessages([{
        id: "welcome",
        role: "assistant",
        content: "Hi, I’m **AIJob**. Upload your resume and I’ll help you find roles that fit, refine your search, and keep track of applications."
      }]);
      setInput("");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
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
          <div className="header-actions">
            <button className="header-action" onClick={() => sendMessage("Show my applied jobs")}>
              <span>▤</span> My applications
            </button>
            <button className="header-action" onClick={clearSession} disabled={busy}>
              Clear session
            </button>
          </div>
        </header>

        <div className="conversation">
          <div className="date-divider"><span>YOUR JOB SEARCH STARTS HERE</span></div>
          {messages.map((message) => (
            <Message
              key={message.id}
              message={message}
              onApply={applyToJob}
              onConfirm={(jobId) => applyToJob(jobId, true)}
              onSubmit={(jobId) => applyToJob(jobId, false, true)}
              onCancel={() => sendMessage("cancel")}
            />
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
