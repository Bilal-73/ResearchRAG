/**
 * What is actually indexed.
 *
 * This is the first thing to look at when an answer says "not reported in the
 * retrieved context": most of the time the corpus simply does not contain the
 * paper, which is a different problem from a retrieval failure.
 *
 * Note the PDF column. The Open RAG Benchmark ships papers as pre-parsed JSON —
 * sections, markdown tables, base64 figures — and no PDF is stored locally, so
 * these link out to arXiv rather than to a local file.
 */
import { useCallback, useEffect, useState } from 'react';
import { NetworkError, api } from '../api/client';
import type { CorpusResponse } from '../api/types';

export function CorpusView({ refreshKey }: { refreshKey: number }) {
  const [corpus, setCorpus] = useState<CorpusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [category, setCategory] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setCorpus(await api.corpus(200, 0, category || undefined));
      setError(null);
    } catch (e) {
      setError(e instanceof NetworkError ? e.message : (e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [category]);

  useEffect(() => {
    void load();
  }, [load, refreshKey]);

  if (error) {
    return (
      <div className="pane">
        <div className="banner danger">
          <pre>{error}</pre>
        </div>
      </div>
    );
  }
  if (!corpus) {
    return (
      <div className="pane">
        <div className="empty">
          <span className="spin" /> loading corpus…
        </div>
      </div>
    );
  }

  const categories = Object.entries(corpus.categories).sort((a, b) => b[1] - a[1]);
  const totalChunks = corpus.papers.reduce((sum, p) => sum + (p.n_chunks ?? 0), 0);

  return (
    <div className="pane">
      <div className="card">
        <h3>Indexed corpus</h3>
        <div className="stat-grid">
          <div className="stat">
            <div className="label">Papers</div>
            <div className="value">{corpus.n_papers}</div>
          </div>
          <div className="stat">
            <div className="label">Chunks</div>
            <div className="value">{totalChunks || '—'}</div>
          </div>
          <div className="stat">
            <div className="label">Categories</div>
            <div className="value">{categories.length}</div>
          </div>
        </div>
        <div className="small muted" style={{ marginTop: 11, whiteSpace: 'pre-wrap' }}>
          {corpus.description}
        </div>
      </div>

      {categories.length > 0 && (
        <div className="card">
          <h3>arXiv categories</h3>
          <div className="row">
            <button className={category === '' ? 'primary' : ''} onClick={() => setCategory('')}>
              all ({corpus.n_papers})
            </button>
            {categories.map(([cat, n]) => (
              <button key={cat} className={category === cat ? 'primary' : ''} onClick={() => setCategory(cat)}>
                {cat} ({n})
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="card">
        <div className="row" style={{ justifyContent: 'space-between', marginBottom: 10 }}>
          <h3 style={{ margin: 0 }}>
            Papers {category && <span className="faint">· filtered to {category}</span>}
          </h3>
          <div className="row">
            {loading && <span className="spin" />}
            <span className="badge">{corpus.total} shown</span>
          </div>
        </div>

        {corpus.papers.length === 0 ? (
          <div className="empty">
            Nothing indexed.
            <pre className="small" style={{ textAlign: 'left', marginTop: 12 }}>
{`.venv\\Scripts\\python.exe scripts/download_corpus.py --limit 50
.venv\\Scripts\\python.exe scripts/run_ingestion.py`}
            </pre>
          </div>
        ) : (
          <table className="data">
            <thead>
              <tr>
                <th>arXiv ID</th>
                <th>Title</th>
                <th>Categories</th>
                <th style={{ textAlign: 'right' }}>Chunks</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {corpus.papers.map((p) => (
                <tr key={p.paper_id}>
                  <td className="mono tiny">{p.paper_id}</td>
                  <td>
                    <div style={{ lineHeight: 1.4 }}>{p.title ?? <span className="faint">(no title)</span>}</div>
                    {p.authors.length > 0 && (
                      <div className="tiny faint" style={{ marginTop: 2 }}>
                        {p.authors.slice(0, 3).join(', ')}
                        {p.authors.length > 3 && ` +${p.authors.length - 3}`}
                        {p.published && ` · ${p.published.slice(0, 10)}`}
                      </div>
                    )}
                  </td>
                  <td>
                    <div className="row tiny" style={{ gap: 3 }}>
                      {p.categories.map((c) => (
                        <span key={c} className="badge mono">{c}</span>
                      ))}
                    </div>
                  </td>
                  <td className="num" style={{ textAlign: 'right' }}>{p.n_chunks ?? '—'}</td>
                  <td>
                    <div className="row tiny" style={{ gap: 7 }}>
                      {p.url && (
                        <a href={p.url} target="_blank" rel="noreferrer">
                          abs ↗
                        </a>
                      )}
                      {p.pdf_url && (
                        <a href={p.pdf_url} target="_blank" rel="noreferrer">
                          PDF ↗
                        </a>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        <div className="small muted" style={{ marginTop: 13, lineHeight: 1.55 }}>
          <strong>Where are the PDFs?</strong> Nowhere on disk. The Open RAG Benchmark ships each paper as
          pre-parsed JSON (sections, markdown tables, base64 figures) under{' '}
          <code>data/open_ragbench/corpus/&lt;arxiv-id&gt;.json</code>, which is what gets chunked and indexed.
          No PDF is downloaded or stored, so the links above go to arXiv.
        </div>
      </div>
    </div>
  );
}
