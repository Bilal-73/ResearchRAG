/**
 * The chat surface, plus the right-hand inspector for the selected turn.
 *
 * The key UX decision: a blocked / refused / out-of-scope turn is a normal
 * message with a badge, not an error toast. The API returns 200 for those because
 * the system behaved correctly, and the UI has to agree — otherwise a safety
 * redirect looks like a crash.
 */
import { useEffect, useRef, useState } from 'react';
import { ApiError, NetworkError, api } from '../api/client';
import type { AnswerKind, ChatOptions, ChatResponse } from '../api/types';
import { AnswerText } from './AnswerText';
import { DiagnosticsPanel, GuardrailEvents } from './Diagnostics';
import { SourceList } from './SourceList';

export interface Turn {
  id: number;
  question: string;
  response: ChatResponse | null;
  error: { message: string; code?: string; hint?: string } | null;
  pending: boolean;
}

const KIND_BADGE: Record<AnswerKind, { label: string; cls: string; hint: string }> = {
  rag: { label: 'grounded answer', cls: 'ok', hint: 'Answered from retrieved excerpts with citations' },
  chitchat: { label: 'chit-chat', cls: '', hint: 'No retrieval ran' },
  out_of_scope: { label: 'out of scope', cls: 'warn', hint: 'Declined without retrieval' },
  not_found: { label: 'not in corpus', cls: 'warn', hint: 'Retrieval found nothing usable' },
  blocked: { label: 'blocked', cls: 'danger', hint: 'An input guardrail stopped this message' },
  safety: { label: 'safety response', cls: 'danger', hint: 'Crisis language detected; fixed reply, no LLM' },
  error: { label: 'error', cls: 'danger', hint: 'Something failed upstream' },
};

const EXAMPLES = [
  'What volatility models are compared in practice?',
  'What problem does the proposed order-execution method solve?',
  'Which baseline did the proposed approach outperform, and by how much?',
];

export function ChatView({
  options,
  conversationId,
  setConversationId,
  onTurnComplete,
}: {
  options: ChatOptions;
  conversationId: string | null;
  setConversationId: (id: string | null) => void;
  onTurnComplete: () => void;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  const [activeLabel, setActiveLabel] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' });
  }, [turns]);

  async function send(message: string) {
    const text = message.trim();
    if (!text || busy) return;

    const id = Date.now();
    setTurns((t) => [...t, { id, question: text, response: null, error: null, pending: true }]);
    setDraft('');
    setBusy(true);
    setSelected(id);

    try {
      const response = await api.chat(text, conversationId, options);
      setConversationId(response.conversation_id);
      setTurns((t) => t.map((turn) => (turn.id === id ? { ...turn, response, pending: false } : turn)));
    } catch (e) {
      let message = 'Something went wrong.';
      let code: string | undefined;
      let hint: string | undefined;
      if (e instanceof NetworkError) {
        message = `Cannot reach the API at ${api.baseUrl}.`;
        hint = 'Start it:  .venv\\Scripts\\python.exe scripts/run_api.py';
      } else if (e instanceof ApiError) {
        message = e.message;
        code = e.code;
        if (e.code === 'rate_limited' && e.retryAfterSeconds)
          hint = `Retry in about ${Math.ceil(e.retryAfterSeconds)}s.`;
        if (e.code === 'service_unavailable' || e.code === 'upstream_unavailable')
          hint = 'Check /health — Qdrant or the LLM is probably unreachable.';
        if (e.code === 'unauthorized') hint = 'Set VITE_API_KEY in .env.local to match the API_KEYS on the server.';
      }
      setTurns((t) =>
        t.map((turn) => (turn.id === id ? { ...turn, error: { message, code, hint }, pending: false } : turn)),
      );
    } finally {
      setBusy(false);
      onTurnComplete();
    }
  }

  async function reset() {
    if (conversationId) {
      try {
        await api.deleteConversation(conversationId);
      } catch {
        /* Forgetting a conversation the server already forgot is not worth surfacing. */
      }
    }
    setConversationId(null);
    setTurns([]);
    setSelected(null);
    onTurnComplete();
  }

  const current = turns.find((t) => t.id === selected) ?? null;

  return (
    <div className="pane-split">
      <div className="chat">
        <div className="chat-log" ref={logRef}>
          {turns.length === 0 && (
            <div className="empty">
              <div style={{ marginBottom: 14 }}>Ask something about the indexed papers.</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, maxWidth: 520, margin: '0 auto' }}>
                {EXAMPLES.map((q) => (
                  <button key={q} onClick={() => send(q)} style={{ textAlign: 'left' }}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {turns.map((turn) => {
            const kind = turn.response ? KIND_BADGE[turn.response.answer_kind] : null;
            return (
              <div key={turn.id} style={{ display: 'contents' }}>
                <div className="msg user">
                  <div className="bubble">{turn.question}</div>
                </div>

                <div
                  className="msg"
                  onClick={() => setSelected(turn.id)}
                  style={{ cursor: 'pointer', opacity: selected === turn.id || selected === null ? 1 : 0.75 }}
                >
                  {turn.pending && (
                    <div className="bubble row muted">
                      <span className="spin" /> thinking… (retrieval, grading and generation)
                    </div>
                  )}

                  {turn.error && (
                    <div className="banner danger" style={{ margin: 0 }}>
                      <div className="row">
                        {turn.error.code && <span className="badge danger">{turn.error.code}</span>}
                        <strong>Request failed</strong>
                      </div>
                      <div style={{ marginTop: 5 }}>{turn.error.message}</div>
                      {turn.error.hint && <pre>{turn.error.hint}</pre>}
                    </div>
                  )}

                  {turn.response && kind && (
                    <>
                      <div className="msg-meta">
                        <span className={`badge ${kind.cls}`} title={kind.hint}>
                          {kind.label}
                        </span>
                        {turn.response.block_kind && (
                          <span className="badge danger mono">{turn.response.block_kind}</span>
                        )}
                        {turn.response.diagnostics && (
                          <span className="badge">{turn.response.diagnostics.total_seconds.toFixed(1)}s</span>
                        )}
                        {turn.response.guardrail_events.length > 0 && (
                          <span className="badge warn">
                            {turn.response.guardrail_events.length} guardrail event
                            {turn.response.guardrail_events.length > 1 ? 's' : ''}
                          </span>
                        )}
                      </div>
                      <div className="bubble">
                        <AnswerText
                          text={turn.response.answer}
                          sources={turn.response.sources}
                          activeLabel={selected === turn.id ? activeLabel : null}
                          onCitationClick={(label) => {
                            setSelected(turn.id);
                            setActiveLabel(label);
                            document
                              .getElementById(`source-${label}`)
                              ?.scrollIntoView({ behavior: 'smooth', block: 'center' });
                          }}
                        />
                      </div>
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        <div className="composer">
          <textarea
            value={draft}
            placeholder={busy ? 'Waiting for the current answer…' : 'Ask about a method, result, dataset or limitation…'}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send(draft);
              }
            }}
            disabled={busy}
            rows={1}
          />
          <button className="primary" onClick={() => send(draft)} disabled={busy || !draft.trim()}>
            {busy ? '…' : 'Send'}
          </button>
          <button onClick={reset} disabled={busy || turns.length === 0} title="Delete this conversation's memory">
            New
          </button>
        </div>
      </div>

      <div className="pane">
        {!current?.response ? (
          <div className="empty">Send a message, or click a turn, to inspect how it was produced.</div>
        ) : (
          <>
            <div className="card">
              <h3>Guardrail events</h3>
              <GuardrailEvents events={current.response.guardrail_events} />
            </div>
            {current.response.diagnostics && <DiagnosticsPanel d={current.response.diagnostics} />}
            <SourceList sources={current.response.sources} activeLabel={activeLabel} onHover={setActiveLabel} />
          </>
        )}
      </div>
    </div>
  );
}
