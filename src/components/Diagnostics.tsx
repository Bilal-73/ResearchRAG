/**
 * The agent's own account of a turn: route taken, why, and where the time went.
 *
 * `grounded === null` is rendered as "not checked" rather than as a pass. The
 * backend distinguishes "the checker ran and approved" from "the checker was
 * unavailable", and collapsing those two would turn an outage into a green tick.
 */
import type { Diagnostics as DiagnosticsData, GuardrailEvent } from '../api/types';

function GroundednessBadge({ grounded }: { grounded: boolean | null }) {
  if (grounded === null) return <span className="badge warn">groundedness not checked</span>;
  return grounded ? (
    <span className="badge ok">grounded</span>
  ) : (
    <span className="badge danger">unsupported claims</span>
  );
}

const ACTION_CLASS: Record<string, string> = {
  blocked: 'danger',
  dropped: 'danger',
  stripped: 'warn',
  flagged: 'warn',
  redacted: 'accent',
  redirected: 'accent',
  regenerated: 'warn',
  retry: 'warn',
  added: 'ok',
  split: 'purple',
};

export function GuardrailEvents({ events }: { events: GuardrailEvent[] }) {
  if (!events.length) {
    return (
      <div className="small muted">
        No guardrail fired this turn. (Guards that pass silently — rate limit, injection scan — leave no event.)
      </div>
    );
  }
  return (
    <>
      {events.map((e, i) => (
        <div key={i} className="trace-step" style={{ gridTemplateColumns: 'auto auto 1fr' }}>
          <span className="badge mono">{e.guardrail}</span>
          <span className={`badge ${ACTION_CLASS[e.action] ?? ''}`}>{e.action}</span>
          <span className="small muted">{e.detail}</span>
        </div>
      ))}
    </>
  );
}

export function DiagnosticsPanel({ d }: { d: DiagnosticsData }) {
  const slowest = Math.max(...d.trace.map((s) => s.seconds), 0.0001);

  return (
    <>
      <div className="card">
        <h3>Routing</h3>
        <div className="row" style={{ marginBottom: 8 }}>
          <span className="badge accent">{d.intent || 'n/a'}</span>
          {d.intent_source && (
            <span className="badge" title="How the intent was decided">
              via {d.intent_source}
              {d.intent_source === 'llm' && ` · ${Math.round(d.intent_confidence * 100)}%`}
            </span>
          )}
          <GroundednessBadge grounded={d.grounded} />
          {!d.citations_ok && <span className="badge danger">no valid citations</span>}
          {d.retry_count > 0 && (
            <span className="badge warn" title="The context grader judged the evidence weak and retried">
              {d.retry_count} corrective retr{d.retry_count === 1 ? 'y' : 'ies'}
            </span>
          )}
          <span className="badge">{d.total_seconds.toFixed(1)}s</span>
        </div>

        {d.explanation.map((line, i) => (
          <div key={i} className="small muted" style={{ marginBottom: 3 }}>
            {line.replace(/\*\*/g, '')}
          </div>
        ))}

        {d.route.length > 0 && (
          <div className="route" style={{ marginTop: 10 }}>
            {d.route.map((node, i) => (
              <span key={i} className="row" style={{ gap: 4 }}>
                {i > 0 && <span className="arrow">→</span>}
                <span className="node">{node}</span>
              </span>
            ))}
          </div>
        )}
      </div>

      {(d.standalone_query || d.sub_questions.length > 0) && (
        <div className="card">
          <h3>Query engineering</h3>
          {d.standalone_query && (
            <div className="small" style={{ marginBottom: 8 }}>
              <div className="tiny muted">Searched as (rewritten standalone query)</div>
              <div className="mono" style={{ marginTop: 2 }}>
                {d.standalone_query}
              </div>
            </div>
          )}
          {d.sub_questions.length > 0 && (
            <div className="small" style={{ marginBottom: 8 }}>
              <div className="tiny muted">Planner split this into {d.sub_questions.length} parts</div>
              <ol style={{ margin: '4px 0 0', paddingLeft: 18 }}>
                {d.sub_questions.map((q, i) => (
                  <li key={i} className="small">
                    {q}
                  </li>
                ))}
              </ol>
            </div>
          )}
          <div className="row tiny">
            {d.variants_used.map((v) => (
              <span key={v} className="badge purple" title="Query variant searched and fused with RRF">
                {v}
              </span>
            ))}
            {d.n_candidates > 0 && <span className="badge">{d.n_candidates} candidates fused</span>}
          </div>
          {!d.context_ok && d.context_reason && (
            <div className="banner" style={{ marginTop: 10, marginBottom: 0 }}>
              Context grader: {d.context_reason}
            </div>
          )}
        </div>
      )}

      {d.unsupported_claims.length > 0 && (
        <div className="card">
          <h3>Claims flagged as unsupported</h3>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {d.unsupported_claims.map((c, i) => (
              <li key={i} className="small">
                {c}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="card">
        <h3>Trace · {d.trace.length} steps</h3>
        {d.trace.map((step, i) => (
          <div key={i}>
            <div className="trace-step">
              <span className={`mono kind-${step.kind}`} title={step.kind}>
                {step.node}
              </span>
              <span className="muted">{step.detail}</span>
              <span className="secs">{step.seconds.toFixed(2)}s</span>
            </div>
            <div className="bar" style={{ marginBottom: 4 }}>
              <span style={{ width: `${(step.seconds / slowest) * 100}%` }} />
            </div>
          </div>
        ))}
        <div className="row tiny faint" style={{ marginTop: 8, gap: 12 }}>
          <span className="kind-llm">● llm</span>
          <span className="kind-rule">● rule</span>
          <span className="kind-search">● search</span>
          <span className="kind-static">● static</span>
          <span className="kind-mixed">● mixed</span>
        </div>
      </div>
    </>
  );
}
