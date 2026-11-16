/**
 * RAGAS scores for a single answer, shown when the request opted in.
 *
 * Two presentation rules that matter for honesty:
 *
 * 1. A `null` score is rendered as "—" with the reason, never as 0. The backend
 *    distinguishes "the metric computed and scored low" from "the metric could
 *    not be computed", and flattening those would invent a quality problem.
 * 2. Thresholds are shown as colour but the number is always visible. These are
 *    LLM-judged metrics with real variance; a green badge alone would imply more
 *    precision than they have.
 */
import type { AnswerScores as Scores } from '../api/types';

const METRICS: { key: 'faithfulness' | 'answer_relevancy'; label: string; hint: string }[] = [
  {
    key: 'faithfulness',
    label: 'Faithfulness',
    hint: 'Is every claim in the answer supported by the retrieved context? Low = hallucination.',
  },
  {
    key: 'answer_relevancy',
    label: 'Answer relevancy',
    hint: 'Does the answer address the question actually asked? Low = evasive or off-target.',
  },
];

function band(value: number): string {
  if (value >= 0.8) return 'ok';
  if (value >= 0.5) return 'warn';
  return 'danger';
}

export function AnswerScoresPanel({ scores }: { scores: Scores }) {
  const hasAny = METRICS.some((m) => scores[m.key] != null);

  return (
    <div className="card">
      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 10 }}>
        <h3 style={{ margin: 0 }}>RAGAS scores · this answer</h3>
        {scores.seconds > 0 && (
          <span className="badge" title="Added latency from scoring">
            +{scores.seconds.toFixed(1)}s
          </span>
        )}
      </div>

      {scores.error && !hasAny ? (
        <div className="small muted">{scores.error}</div>
      ) : (
        <>
          <div className="stat-grid">
            {METRICS.map((m) => {
              const value = scores[m.key];
              return (
                <div key={m.key} className="stat" title={m.hint}>
                  <div className="label">{m.label}</div>
                  <div
                    className="value"
                    style={{ color: value == null ? 'var(--text-faint)' : `var(--${band(value)})` }}
                  >
                    {value == null ? '—' : value.toFixed(2)}
                  </div>
                  {value != null && (
                    <div className={`bar ${band(value)}`} style={{ marginTop: 6 }}>
                      <span style={{ width: `${value * 100}%` }} />
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {scores.error && <div className="small muted" style={{ marginTop: 9 }}>{scores.error}</div>}

          <div className="small muted" style={{ marginTop: 11, lineHeight: 1.5 }}>
            Scored live by RAGAS against the excerpts this answer was given. Only reference-free metrics
            apply here — <code>context_precision</code> and <code>context_recall</code> need a gold answer,
            so they appear in the offline benchmark run, not on a real question.
          </div>
        </>
      )}
    </div>
  );
}
