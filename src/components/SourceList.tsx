/**
 * The sources behind an answer.
 *
 * Cited and merely-retrieved sources are visually separated on purpose: the model
 * was *given* all of them, but only referenced some. Presenting all of them as
 * "the sources" would overstate the evidence actually used.
 */
import type { Source } from '../api/types';

const CHUNK_ICON: Record<string, string> = { text: '¶', table: '▦', figure: '◫' };

function SourceCard({
  source,
  highlighted,
  onHover,
}: {
  source: Source;
  highlighted: boolean;
  onHover?: (label: string | null) => void;
}) {
  return (
    <div
      className={`source ${source.cited ? '' : 'uncited'} ${highlighted ? 'highlight' : ''}`}
      id={`source-${source.label}`}
      onMouseEnter={() => onHover?.(source.label)}
      onMouseLeave={() => onHover?.(null)}
    >
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div className="row" style={{ gap: 6 }}>
          <span className="badge accent">{source.label}</span>
          <span className="badge" title={`${source.chunk_type} chunk`}>
            {CHUNK_ICON[source.chunk_type] ?? '¶'} {source.chunk_type}
          </span>
          {!source.cited && <span className="badge faint">retrieved, not cited</span>}
        </div>
        {source.score != null && (
          <span className="tiny faint mono" title="Cross-encoder relevance score">
            {source.score.toFixed(2)}
          </span>
        )}
      </div>

      {source.paper_title && (
        <div style={{ marginTop: 6, fontWeight: 550, lineHeight: 1.35 }}>{source.paper_title}</div>
      )}

      <div className="tiny muted mono" style={{ marginTop: 4 }}>
        {source.paper_id}
        {source.section_id != null && ` › §${source.section_id}`} {source.section}
        {source.figure_no && ` › ${source.chunk_type === 'table' ? 'Table' : 'Figure'} ${source.figure_no}`}
      </div>

      {source.snippet && <div className="snippet">{source.snippet}</div>}

      <div className="row tiny" style={{ marginTop: 7, gap: 10 }}>
        {source.url && (
          <a href={source.url} target="_blank" rel="noreferrer">
            arXiv abstract ↗
          </a>
        )}
        {/* No PDF is stored locally — the corpus ships pre-parsed JSON — so this goes to arXiv. */}
        <a href={`https://arxiv.org/pdf/${source.paper_id}`} target="_blank" rel="noreferrer">
          PDF ↗
        </a>
      </div>
    </div>
  );
}

export function SourceList({
  sources,
  activeLabel,
  onHover,
}: {
  sources: Source[];
  activeLabel: string | null;
  onHover?: (label: string | null) => void;
}) {
  if (!sources.length) {
    return <div className="empty">No sources — this turn did not use retrieval.</div>;
  }
  const cited = sources.filter((s) => s.cited);
  const uncited = sources.filter((s) => !s.cited);

  return (
    <>
      <div className="card">
        <h3>
          Cited evidence · {cited.length} of {sources.length}
        </h3>
        {cited.length ? (
          cited.map((s) => (
            <SourceCard key={s.label} source={s} highlighted={activeLabel === s.label} onHover={onHover} />
          ))
        ) : (
          <div className="small muted">
            The answer referenced none of the retrieved excerpts — expected for a refusal, a red flag otherwise.
          </div>
        )}
      </div>

      {uncited.length > 0 && (
        <div className="card">
          <h3>Also retrieved · {uncited.length}</h3>
          <div className="small muted" style={{ marginBottom: 9 }}>
            Given to the model as context but not referenced in the answer.
          </div>
          {uncited.map((s) => (
            <SourceCard key={s.label} source={s} highlighted={activeLabel === s.label} onHover={onHover} />
          ))}
        </div>
      )}
    </>
  );
}
