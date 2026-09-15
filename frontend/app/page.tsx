"use client";

/**
 * The Phase 4 interface: upload a document, ask about it, read a cited answer.
 *
 * Deliberately one screen and no more. The comparison view, the KFS diff table
 * and the risk panel belong to later phases; what this has to prove is that the
 * whole chain works in a browser -- a real document in, a grounded answer out,
 * with every claim traceable to the clause it came from.
 *
 * Citations render beside the answer rather than under it, because they are not
 * a footnote. They are the reason the answer is allowed to exist.
 */

import { useCallback, useRef, useState } from "react";
import {
  askQuestion,
  uploadDocument,
  type ChatMeta,
  type Citation,
  type CoverageGap,
  type Fallback,
  type UploadResult,
} from "@/lib/api";

const SUGGESTIONS = [
  "Is there a lock-in period before I can prepay my loan, and what would it cost?",
  "What penal charges apply if I miss an EMI payment?",
  "How much am I paying for insurance and was it optional?",
  "If interest rates go up, can the lender extend my tenure without asking me?",
];

type Turn = {
  question: string;
  answer: string;
  citations: Citation[];
  meta: ChatMeta | null;
  stripped: string[];
  degraded: string | null;
  fallback: Fallback | null;
  coverage: CoverageGap[];
  streaming: boolean;
  error: string | null;
};

export default function Home() {
  const [document, setDocument] = useState<UploadResult | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const handleUpload = useCallback(async (file: File) => {
    setUploading(true);
    setUploadError(null);
    try {
      setDocument(await uploadDocument(file));
      setTurns([]);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : String(error));
    } finally {
      setUploading(false);
    }
  }, []);

  const ask = useCallback(
    async (text: string) => {
      if (!document || !text.trim() || busy) return;
      setBusy(true);
      setQuestion("");

      const index = turns.length;
      setTurns((previous) => [
        ...previous,
        {
          question: text,
          answer: "",
          citations: [],
          meta: null,
          stripped: [],
          degraded: null,
          fallback: null,
          coverage: [],
          streaming: true,
          error: null,
        },
      ]);

      const update = (patch: Partial<Turn>) =>
        setTurns((previous) =>
          previous.map((turn, i) => (i === index ? { ...turn, ...patch } : turn)),
        );

      try {
        await askQuestion(text, document.document_id, document.version, {
          onMeta: (meta) => update({ meta }),
          onCitations: (citations) => update({ citations }),
          onDelta: (delta) =>
            setTurns((previous) =>
              previous.map((turn, i) =>
                i === index ? { ...turn, answer: turn.answer + delta } : turn,
              ),
            ),
          // The hosted model failed: its partial draft is void, and the
          // fallback's answer streams into a cleared box under a notice.
          onFallback: (fallback) => update({ fallback, streaming: false }),
          onCoverage: (coverage) => update({ coverage }),
          onDegraded: ({ message }) => update({ degraded: message, answer: "" }),
          // Verification can only remove sentences, so the checked answer always
          // replaces the draft rather than adding to it.
          onVerified: ({ text: verified, stripped }) =>
            update({ answer: verified, stripped }),
          onError: (message) => update({ error: message, streaming: false }),
          onDone: () => update({ streaming: false }),
        });
      } catch (error) {
        // A thrown request (API down, network drop) must not leave the button stuck.
        update({ error: error instanceof Error ? error.message : String(error), streaming: false });
      } finally {
        setBusy(false);
      }
    },
    [busy, document, turns.length],
  );

  return (
    <main className="wrap">
      <header>
        <h1 >FinX</h1>
        <p className="sub">
          Ask about your loan agreement. Every answer is quoted from a clause and cited.
        </p>
      </header>

      {/* --- Upload ------------------------------------------------------ */}
      <section className="panel">
        <div className="row">
          <input
            ref={fileInput}
            type="file"
            accept=".pdf,.docx"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void handleUpload(file);
            }}
          />
          <button
            type="button"
            onClick={() => fileInput.current?.click()}
            disabled={uploading}
            
          >
            {uploading ? "Reading the document…" : "Upload agreement (.docx or .pdf)"}
          </button>
          {document && (
            <span >
              <strong>{document.title}</strong> · v{document.version} ·{" "}
              {document.clause_count} clauses
              {document.reuse_percent > 0 && (
                <span className="where">
                  {" "}
                  · {document.reuse_percent}% of embeddings reused
                </span>
              )}
            </span>
          )}
        </div>

        {uploadError && (
          <p className="error">
            {uploadError}
          </p>
        )}

        {document && document.warnings.length > 0 && (
          <details className="warnings">
            <summary >
              {document.warnings.length} note(s) from parsing this document
            </summary>
            <ul >
              {document.warnings.map((warning, i) => (
                <li key={i}>{warning}</li>
              ))}
            </ul>
          </details>
        )}
      </section>

      {/* --- Ask --------------------------------------------------------- */}
      {document && (
        <section className="panel">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void ask(question);
            }}
            className="row"
          >
            <input
              type="text"
              className="ask-input"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask about a charge, a clause, or a term…"
              disabled={busy}
              
            />
            <button
              type="submit"
              disabled={busy || !question.trim()}
              
            >
              {busy ? "Thinking…" : "Ask"}
            </button>
          </form>

          {turns.length === 0 && (
            <div className="suggestions">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  onClick={() => void ask(suggestion)}
                  className="ghost"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          )}
        </section>
      )}

      {/* --- Answers ----------------------------------------------------- */}
      {turns.map((turn, index) => (
        <article key={index} className="panel">
          <p className="q">{turn.question}</p>

          {turn.meta && (
            <p className="meta">
              v{turn.meta.version} · {turn.meta.considered} passages considered ·
              best match {turn.meta.confidence.toFixed(3)} · {turn.meta.provider}
            </p>
          )}

          {turn.fallback ? (
            <div className="fallback">
              <p className="fallback-notice">{turn.fallback.notice}</p>
              <p className="fallback-reason">{turn.fallback.reason}</p>
              <h4>In general terms</h4>
              <p>{turn.fallback.explanation}</p>
              <h4>For your specific loan</h4>
              <p>{turn.fallback.advisor}</p>
            </div>
          ) : turn.error ? (
            <p className="error">
              {turn.error}
            </p>
          ) : (
            <div className="answer-grid">
              <div className="answer">
                {turn.degraded && <p className="note">{turn.degraded}</p>}
                {turn.coverage.map((gap) => (
                  <p key={gap.superseded_by} className="coverage">
                    {gap.message}
                  </p>
                ))}
                {turn.answer || (turn.streaming ? "…" : "")}
                {turn.stripped.length > 0 && (
                  <p className="note">
                    {turn.stripped.length} sentence(s) were removed because no
                    retrieved clause supported them.
                  </p>
                )}
              </div>

              <aside className="sources">
                <h3>Sources</h3>
                <ol >
                  {turn.citations.map((citation) => (
                    <li
                      key={citation.chunk_id}
                      className={`cite ${citation.source === "rbi_corpus" ? "rbi" : ""}`}
                    >
                      <div >
                        [{citation.n}]{" "}
                        {citation.source === "rbi_corpus"
                          ? citation.circular_id ?? "RBI"
                          : `Clause ${citation.clause_number ?? citation.page}`}
                      </div>
                      <div className="where">
                        {citation.source === "rbi_corpus"
                          ? citation.title
                          : `${citation.title} · page ${citation.page}`}
                      </div>
                      <details >
                        <summary >
                          Read the text
                        </summary>
                        <p >
                          {citation.text}
                        </p>
                      </details>
                      {citation.source_url && (
                        <a
                          href={citation.source_url}
                          target="_blank"
                          rel="noreferrer"
                          
                        >
                          Open the circular
                        </a>
                      )}
                    </li>
                  ))}
                </ol>
              </aside>
            </div>
          )}
        </article>
      ))}
    </main>
  );
}
