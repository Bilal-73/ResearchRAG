"""Corpus introspection: what can this system actually answer from?

A frontend needs this to set expectations before the user types anything —
showing "396 papers across cs.LG, stat.ME, …" prevents most out-of-scope
questions, which is cheaper than answering them.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from research_rag.api.dependencies import AppState, get_state, require_api_key
from research_rag.api.schemas import CorpusResponse, ErrorResponse, PaperSummary
from research_rag.models.response import arxiv_pdf_url, arxiv_url

router = APIRouter(prefix="/v1", tags=["corpus"])


@router.get("/corpus", response_model=CorpusResponse,
            responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
            summary="What is indexed")
async def corpus(
    limit: int = Query(25, ge=0, le=500, description="Papers per page; 0 returns counts only."),
    offset: int = Query(0, ge=0),
    category: str | None = Query(None, description="Filter to one arXiv category, e.g. 'cs.LG'."),
    state: AppState = Depends(get_state),
    api_key: str = Depends(require_api_key),
) -> CorpusResponse:
    info = state.corpus
    info.reload_manifest()
    all_ids = info.ingested_ids()

    if category:
        ids = [pid for pid in all_ids if category in info.categories_for(pid)]
    else:
        ids = all_ids

    papers = []
    for paper_id in ids[offset: offset + limit] if limit else []:
        entry = info.paper(paper_id) or {}
        papers.append(PaperSummary(
            paper_id=paper_id, title=entry.get("title"), categories=entry.get("categories", []),
            authors=entry.get("authors", [])[:8], published=entry.get("published"),
            n_chunks=entry.get("n_chunks"), url=arxiv_url(paper_id),
            pdf_url=arxiv_pdf_url(paper_id),
        ))

    return CorpusResponse(
        n_papers=len(all_ids), categories=info.category_counts(), description=info.describe(),
        papers=papers, total=len(ids), limit=limit, offset=offset,
    )


@router.get("/corpus/{paper_id}", response_model=PaperSummary,
            responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse},
                       503: {"model": ErrorResponse}},
            summary="One paper's metadata")
async def paper(
    paper_id: str,
    state: AppState = Depends(get_state),
    api_key: str = Depends(require_api_key),
) -> PaperSummary:
    entry = state.corpus.paper(paper_id)
    if not entry or not entry.get("ingested"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "paper_not_found",
                    "message": f"No indexed paper with id '{paper_id}'.",
                    "detail": {"paper_id": paper_id}},
        )
    return PaperSummary(
        paper_id=paper_id, title=entry.get("title"), categories=entry.get("categories", []),
        authors=entry.get("authors", [])[:8], published=entry.get("published"),
        n_chunks=entry.get("n_chunks"), url=arxiv_url(paper_id),
        pdf_url=arxiv_pdf_url(paper_id),
    )
