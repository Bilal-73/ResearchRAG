"""Reciprocal Rank Fusion across several ranked lists of chunks.

RRF scores each chunk by sum(weight / (k + rank)) over every list it appears
in, so a chunk that several query variants agree on rises to the top without
needing their raw scores to be comparable (dense cosine vs. sparse vs. a HyDE
vector are not). k=60 is the standard constant from the original RRF paper.
"""
from __future__ import annotations

from research_rag.models import Chunk


def rrf_fuse(rankings: list[list[Chunk]], weights: list[float] | None = None, k: int = 60) -> list[Chunk]:
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must have one entry per ranking")

    scores: dict[str, float] = {}
    first_seen: dict[str, Chunk] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + weight / (k + rank)
            first_seen.setdefault(chunk.id, chunk)

    order = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [first_seen[cid] for cid in order]
