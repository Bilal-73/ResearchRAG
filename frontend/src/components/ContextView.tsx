/**
 * Context engineering, made observable: the Redis cache on one side, the agent's
 * conversation memory on the other.
 *
 * Why these two together — they are the two mechanisms that decide what the next
 * LLM call actually costs and contains:
 *   - the cache decides whether an embedding is recomputed or recalled;
 *   - the memory decides how much conversation the prompt carries, and the rolling
 *     summary is what keeps that flat instead of growing forever.
 *
 * The cold-vs-warm control exists because a hit rate is only meaningful against a
 * known starting point: reset, ask a question, ask it again, watch hits climb.
 */
import { useCallback, useEffect, useState } from 'react';
import { ApiError, NetworkError, api } from '../api/client';
import type { CacheResponse, MemoryResponse } from '../api/types';

function pct(value: number | null): string {
  return value == null ? '—' : `${Math.round(value * 100)}%`;
}

function CacheCard({ cache, onReset }: { cache: CacheResponse; onReset: () => void }) {
  const { stats } = cache;
  const total = stats.hits + stats.misses;

  return (
    <div className="card">
      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 10 }}>
        <h3 style={{ margin: 0 }}>Redis embedding cache</h3>
        <div className="row">
          <span className={`badge ${cache.enabled ? 'ok' : 'danger'}`}>
            <span className="dot" />
            {cache.enabled ? 'connected' : 'unavailable'}
          </span>
          <button onClick={onReset} disabled={!cache.enabled} title="Zero the counters; cached data is kept">
            Reset counters
          </button>
        </div>
      </div>

      {cache.note && <div className={`banner ${cache.enabled ? '' : 'danger'}`}>{cache.note}</div>}

      <div className="stat-grid">
        <div className="stat">
          <div className="label">Hit rate</div>
          <div className="value" style={{ color: stats.hit_rate != null && stats.hit_rate > 0 ? 'var(--ok)' : undefined }}>
            {pct(stats.hit_rate)}
          </div>
        </div>
        <div className="stat">
          <div className="label">Hits</div>
          <div className="value" style={{ color: 'var(--ok)' }}>{stats.hits}</div>
        </div>
        <div className="stat">
          <div className="label">Misses</div>
          <div className="value" style={{ color: 'var(--warn)' }}>{stats.misses}</div>
        </div>
        <div className="stat">
          <div className="label">Writes</div>
          <div className="value">{stats.sets}</div>
        </div>
      </div>

      {total > 0 && (
        <div style={{ marginTop: 11 }}>
          <div className="bar ok" style={{ height: 9 }}>
            <span style={{ width: `${(stats.hits / total) * 100}%` }} />
          </div>
          <div className="row tiny faint" style={{ justifyContent: 'space-between', marginTop: 4 }}>
            <span>{stats.hits} served from cache</span>
            <span>{stats.misses} recomputed via the embedding API</span>
          </div>
        </div>
      )}

      <div className="row tiny faint" style={{ marginTop: 11, gap: 14 }}>
        {stats.avg_get_ms != null && <span>avg read {stats.avg_get_ms.toFixed(2)}ms</span>}
        {stats.avg_set_ms != null && <span>avg write {stats.avg_set_ms.toFixed(2)}ms</span>}
        {stats.errors > 0 && <span style={{ color: 'var(--danger)' }}>{stats.errors} errors</span>}
        <span>TTL {Math.round(cache.ttl_seconds / 86400)}d</span>
      </div>

      {cache.namespaces.length > 0 && (
        <table className="data" style={{ marginTop: 13 }}>
          <thead>
            <tr>
              <th>Namespace</th>
              <th>What it holds</th>
              <th style={{ textAlign: 'right' }}>Keys</th>
            </tr>
          </thead>
          <tbody>
            {cache.namespaces.map((ns) => (
              <tr key={ns.name}>
                <td className="mono">{ns.name}</td>
                <td className="muted small">{ns.label}</td>
                <td className="num" style={{ textAlign: 'right' }}>{ns.keys}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {Object.keys(cache.server).length > 0 && (
        <details className="disclose" style={{ marginTop: 11 }}>
          <summary>Redis server · {String(cache.server.version ?? '?')}</summary>
          <table className="data">
            <tbody>
              {Object.entries(cache.server).map(([k, v]) => (
                <tr key={k}>
                  <td className="muted small">{k.replace(/_/g, ' ')}</td>
                  <td className="num">{String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}

      <div className="small muted" style={{ marginTop: 12, lineHeight: 1.5 }}>
        Embeddings are deterministic for a given text, so identical chunks and repeated questions are served
        from here instead of the embedding API. Counters are per API process and reset on restart; key counts
        come from Redis itself.
      </div>
    </div>
  );
}

function MemoryCard({ memory, conversationId }: { memory: MemoryResponse | null; conversationId: string | null }) {
  if (!conversationId) {
    return (
      <div className="card">
        <h3>Conversation memory</h3>
        <div className="small muted">No active conversation. Ask something in Chat first.</div>
      </div>
    );
  }
  if (!memory) {
    return (
      <div className="card">
        <h3>Conversation memory</h3>
        <div className="small muted">
          Nothing stored for <code>{conversationId}</code> yet.
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 10 }}>
        <h3 style={{ margin: 0 }}>Conversation memory</h3>
        <span className="badge">
          {memory.turns} / {memory.max_turns_kept} turns verbatim
        </span>
      </div>

      <div className="tiny faint mono" style={{ marginBottom: 11 }}>
        {memory.conversation_id}
      </div>

      {memory.focus_question && (
        <div className="small" style={{ marginBottom: 11 }}>
          <div className="tiny muted">Current focus (used to resolve "what about that?")</div>
          <div style={{ marginTop: 2 }}>{memory.focus_question}</div>
          {memory.focus_papers.length > 0 && (
            <div className="row tiny" style={{ marginTop: 5 }}>
              {memory.focus_papers.map((p) => (
                <span key={p} className="badge accent mono">{p}</span>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="small" style={{ marginBottom: 11 }}>
        <div className="tiny muted">Rolling summary (older turns folded in by an LLM)</div>
        {memory.summary ? (
          <div style={{ marginTop: 3 }}>{memory.summary}</div>
        ) : (
          <div className="faint" style={{ marginTop: 3 }}>
            Empty — nothing has aged out yet. It fills once the conversation passes {memory.max_turns_kept} turns,
            which is what keeps the prompt size flat.
          </div>
        )}
      </div>

      <details className="disclose" open>
        <summary>Verbatim history · {memory.history.length} messages</summary>
        {memory.history.map((m, i) => (
          <div key={i} style={{ marginBottom: 8 }}>
            <div className="row tiny" style={{ gap: 5 }}>
              <span className={`badge ${m.role === 'user' ? 'accent' : ''}`}>{m.role}</span>
              {m.intent && <span className="badge mono">{m.intent}</span>}
            </div>
            <div className="small muted" style={{ marginTop: 3 }}>
              {m.content.length > 300 ? `${m.content.slice(0, 300)}…` : m.content}
            </div>
          </div>
        ))}
      </details>

      <div className="small muted" style={{ marginTop: 10, lineHeight: 1.5 }}>
        Only PII-redacted text is ever stored here. This is exactly what the next turn's prompt is built from.
      </div>
    </div>
  );
}

export function ContextView({
  conversationId,
  refreshKey,
}: {
  conversationId: string | null;
  refreshKey: number;
}) {
  const [cache, setCache] = useState<CacheResponse | null>(null);
  const [memory, setMemory] = useState<MemoryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [auto, setAuto] = useState(false);

  const load = useCallback(async () => {
    try {
      setCache(await api.cache());
      setError(null);
    } catch (e) {
      setError(e instanceof NetworkError ? e.message : (e as Error).message);
      return;
    }
    if (!conversationId) {
      setMemory(null);
      return;
    }
    try {
      setMemory(await api.memory(conversationId));
    } catch (e) {
      // A 404 just means nothing is stored yet — not worth an error banner.
      if (e instanceof ApiError && e.status === 404) setMemory(null);
    }
  }, [conversationId]);

  useEffect(() => {
    void load();
  }, [load, refreshKey]);

  useEffect(() => {
    if (!auto) return;
    const timer = setInterval(() => void load(), 2000);
    return () => clearInterval(timer);
  }, [auto, load]);

  return (
    <div className="pane">
      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 12 }}>
        <div className="small muted">
          What the system caches, and what it remembers.
        </div>
        <div className="row">
          <label className="row tiny" style={{ gap: 5 }}>
            <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} />
            auto-refresh
          </label>
          <button onClick={() => void load()}>Refresh</button>
        </div>
      </div>

      {error && <div className="banner danger"><pre>{error}</pre></div>}

      {cache && (
        <CacheCard
          cache={cache}
          onReset={async () => {
            await api.resetCacheStats();
            void load();
          }}
        />
      )}

      <MemoryCard memory={memory} conversationId={conversationId} />

      <div className="card">
        <h3>Try it: cold vs warm</h3>
        <ol className="small muted" style={{ margin: 0, paddingLeft: 18, lineHeight: 1.7 }}>
          <li>Hit <strong>Reset counters</strong> above.</li>
          <li>Go to <strong>Chat</strong> and ask any question.</li>
          <li>Come back — misses went up, keys were written.</li>
          <li>Ask the <em>exact same</em> question again, then refresh here: hits climb.</li>
        </ol>
        <div className="small muted" style={{ marginTop: 9, lineHeight: 1.5 }}>
          Repeating a question will not hit 100%: query rewriting and HyDE generate fresh text each turn, so
          those embeddings are new every time. Only the deterministic parts recur — which makes the hit rate a
          decent readout of how much of your pipeline is actually deterministic.
        </div>
      </div>
    </div>
  );
}
