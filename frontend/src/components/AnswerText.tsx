/**
 * Renders an answer with its [S#] markers turned into clickable chips.
 *
 * The backend guarantees every marker in a delivered answer points at a real
 * source (invalid ones are stripped server-side), so a marker we cannot resolve
 * means the source list and the text disagree — we render it inert rather than
 * silently dropping it, because hiding that would hide a real bug.
 *
 * Answer text is inserted as text nodes only, never as HTML: it contains
 * retrieved paper content, which is untrusted.
 */
import type { ReactNode } from 'react';

import type { Source } from '../api/types';

interface Props {
  text: string;
  sources: Source[];
  activeLabel: string | null;
  onCitationClick: (label: string) => void;
}

/** Matches "[S1]" and grouped forms like "[S1, S2]". */
const CITATION_GROUP = /\[((?:S\d+)(?:\s*,\s*S\d+)*)\]/g;

export function AnswerText({ text, sources, activeLabel, onCitationClick }: Props) {
  const known = new Set(sources.map((s) => s.label));
  const parts: ReactNode[] = [];
  let cursor = 0;
  let key = 0;

  for (const match of text.matchAll(CITATION_GROUP)) {
    const start = match.index ?? 0;
    if (start > cursor) parts.push(text.slice(cursor, start));

    const labels = match[1].split(',').map((l) => l.trim());
    parts.push(
      <span key={`c${key++}`}>
        {labels.map((label, i) => {
          const resolvable = known.has(label);
          return (
            <span key={label + i}>
              {i > 0 && <span className="faint"> </span>}
              {resolvable ? (
                <button
                  type="button"
                  className={`citation ${activeLabel === label ? 'active' : ''}`}
                  onClick={() => onCitationClick(label)}
                  title="Show this source"
                >
                  {label}
                </button>
              ) : (
                <span className="citation" style={{ opacity: 0.5, cursor: 'default' }} title="No matching source returned">
                  {label}
                </span>
              )}
            </span>
          );
        })}
      </span>,
    );
    cursor = start + match[0].length;
  }
  if (cursor < text.length) parts.push(text.slice(cursor));

  return <>{parts}</>;
}
