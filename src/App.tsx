/**
 * App shell: tabs, the health strip, and the per-request pipeline toggles.
 *
 * The health strip polls /health rather than waiting for a request to fail,
 * because the two most common confusions with this stack — "the API isn't
 * running" and "Qdrant isn't running" — produce very different fixes, and the
 * component breakdown says which one you have before you type a question.
 */
import { useCallback, useEffect, useState } from 'react';
import { NetworkError, api } from './api/client';
import type { ChatOptions, ComponentStatus, HealthResponse } from './api/types';
import { ChatView } from './components/ChatView';
import { ContextView } from './components/ContextView';
import { CorpusView } from './components/CorpusView';
import { GuardrailsView } from './components/GuardrailsView';

type Tab = 'chat' | 'context' | 'corpus' | 'guardrails';

const TABS: { id: Tab; label: string }[] = [
  { id: 'chat', label: 'Chat' },
  { id: 'context', label: 'Context & Cache' },
  { id: 'corpus', label: 'Corpus' },
  { id: 'guardrails', label: 'Guardrails' },
];

const STATUS_CLASS: Record<ComponentStatus, string> = { ok: 'ok', degraded: 'warn', down: 'danger' };

const TOGGLES: { key: keyof ChatOptions; label: string; hint: string }[] = [
  { key: 'enable_query_rewrite', label: 'Query rewrite', hint: 'Resolve follow-ups into a standalone query' },
  { key: 'enable_hyde', label: 'HyDE', hint: 'Embed a hypothetical answer passage to improve recall' },
  { key: 'enable_reranker', label: 'Reranker', hint: 'Cross-encoder reorders the fused candidates' },
  { key: 'enable_planner', label: 'Planner', hint: 'Split multi-part questions, retrieve per part' },
  { key: 'enable_context_grading', label: 'Context grading', hint: 'Check evidence is sufficient; one corrective retry' },
  { key: 'enable_groundedness_check', label: 'Groundedness check', hint: 'Second LLM pass verifying every claim' },
];

function HealthStrip({ health, error }: { health: HealthResponse | null; error: string | null }) {
  if (error) {
    return (
      <span className="badge danger" title={error}>
        <span className="dot" /> API unreachable
      </span>
    );
  }
  if (!health)
    return (
      <span className="badge">
        <span className="spin" />
      </span>
    );

  return (
    <div className="row" style={{ gap: 5 }}>
      <span className={`badge ${STATUS_CLASS[health.status]}`} title={`API v${health.version}`}>
        <span className="dot" /> {health.status}
      </span>
      {Object.entries(health.components).map(([name, c]) => (
        <span key={name} className={`badge ${STATUS_CLASS[c.status]}`} title={`${name}: ${c.detail}`}>
          {name}
        </span>
      ))}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>('chat');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [options, setOptions] = useState<ChatOptions>({});
  const [showOptions, setShowOptions] = useState(false);
  // Bumped after each turn so the context/corpus panels pick up fresh numbers
  // without polling while you are not looking at them.
  const [refreshKey, setRefreshKey] = useState(0);

  const loadHealth = useCallback(async () => {
    try {
      setHealth(await api.health());
      setHealthError(null);
    } catch (e) {
      setHealth(null);
      setHealthError(e instanceof NetworkError ? e.message : (e as Error).message);
    }
  }, []);

  useEffect(() => {
    void loadHealth();
    const timer = setInterval(() => void loadHealth(), 15000);
    return () => clearInterval(timer);
  }, [loadHealth]);

  const bumpRefresh = useCallback(() => {
    setRefreshKey((k) => k + 1);
    void loadHealth();
  }, [loadHealth]);

  return (
    <div className="app">
      <header className="topbar">
        <h1>Research RAG</h1>
        <div className="tabs">
          {TABS.map((t) => (
            <button key={t.id} aria-selected={tab === t.id} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </div>

        <div className="spacer" />

        {conversationId && (
          <span className="badge mono tiny" title="Active conversation">
            {conversationId}
          </span>
        )}
        <button onClick={() => setShowOptions((s) => !s)} title="Per-request pipeline toggles">
          Pipeline {showOptions ? '▴' : '▾'}
        </button>
        <HealthStrip health={health} error={healthError} />
      </header>

      {showOptions && (
        <div style={{ padding: '10px 16px', background: 'var(--bg-sunken)', borderBottom: '1px solid var(--border)' }}>
          <div className="small muted" style={{ marginBottom: 7 }}>
            Sent with the next request as <code>options</code>. Unset means "use the server default" — useful for
            A/B-ing what each retrieval technique actually buys you.
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: '0 18px' }}>
            {TOGGLES.map((t) => {
              const value = options[t.key];
              return (
                <label key={t.key} className="toggle-row" title={t.hint}>
                  <span>{t.label}</span>
                  <select
                    style={{ width: 'auto' }}
                    value={value === undefined ? 'default' : String(value)}
                    onChange={(e) => {
                      const v = e.target.value;
                      setOptions((o) => {
                        const next = { ...o };
                        if (v === 'default') delete next[t.key];
                        else next[t.key] = v === 'true';
                        return next;
                      });
                    }}
                  >
                    <option value="default">default</option>
                    <option value="true">on</option>
                    <option value="false">off</option>
                  </select>
                </label>
              );
            })}
          </div>
        </div>
      )}

      <div className="body">
        {tab === 'chat' && (
          <ChatView
            options={options}
            conversationId={conversationId}
            setConversationId={setConversationId}
            onTurnComplete={bumpRefresh}
          />
        )}
        {tab === 'context' && <ContextView conversationId={conversationId} refreshKey={refreshKey} />}
        {tab === 'corpus' && <CorpusView refreshKey={refreshKey} />}
        {tab === 'guardrails' && <GuardrailsView />}
      </div>
    </div>
  );
}
