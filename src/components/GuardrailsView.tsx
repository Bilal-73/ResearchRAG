/**
 * The guardrail catalogue, grouped by pipeline stage.
 *
 * This is rendered from `GET /v1/guardrails`, which the backend builds from the
 * single registry in `guardrails/registry.py` with each `enabled` flag read from
 * live settings. So a guardrail shown as "on" here really is on — the page cannot
 * drift from the code the way a hand-written feature list would.
 */
import { useEffect, useState } from 'react';
import { NetworkError, api } from '../api/client';
import type { GuardrailsResponse } from '../api/types';

export function GuardrailsView() {
  const [data, setData] = useState<GuardrailsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .guardrails()
      .then(setData)
      .catch((e) => setError(e instanceof NetworkError ? e.message : (e as Error).message));
  }, []);

  if (error) {
    return (
      <div className="pane">
        <div className="banner danger">
          <pre>{error}</pre>
        </div>
      </div>
    );
  }
  if (!data) {
    return (
      <div className="pane">
        <div className="empty">
          <span className="spin" /> loading…
        </div>
      </div>
    );
  }

  const enabled = data.guardrails.filter((g) => g.enabled).length;

  return (
    <div className="pane">
      <div className="card">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <h3 style={{ margin: 0 }}>Guardrails</h3>
          <span className={`badge ${enabled === data.guardrails.length ? 'ok' : 'warn'}`}>
            {enabled} of {data.guardrails.length} active
          </span>
        </div>
        <div className="small muted" style={{ marginTop: 9, lineHeight: 1.55 }}>
          Served from the backend's single guardrail registry, with each flag read from live settings — so this
          list cannot claim a guardrail is on while its setting is off.
        </div>
      </div>

      {data.stages.map((stage) => {
        const items = data.guardrails.filter((g) => g.stage === stage);
        if (!items.length) return null;
        return (
          <div key={stage} className="card">
            <h3>
              {data.stage_titles[stage] ?? stage} · {items.length}
            </h3>
            {items.map((g) => (
              <div
                key={g.id}
                style={{
                  padding: '9px 0',
                  borderBottom: '1px dashed var(--border)',
                }}
              >
                <div className="row" style={{ justifyContent: 'space-between' }}>
                  <div className="row">
                    <span className={`badge ${g.enabled ? 'ok' : 'danger'}`}>
                      <span className="dot" />
                      {g.enabled ? 'on' : 'off'}
                    </span>
                    <strong style={{ fontSize: 14 }}>{g.name}</strong>
                  </div>
                  <code className="tiny faint">{g.implementation}</code>
                </div>
                <div className="small muted" style={{ marginTop: 5, lineHeight: 1.5 }}>
                  {g.description}
                </div>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}
