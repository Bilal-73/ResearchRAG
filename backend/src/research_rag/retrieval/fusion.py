"""Reciprocal Rank Fusion across several ranked lists of chunks.

RRF scores each chunk by sum(weight / (k + rank)) over every list it appears
in, so a chunk that several query variants agree on rises to the top without
needing their raw scores to be comparable (dense cosine vs. sparse vs. a HyDE
vector are not). k=60 is the standard constant from the original RRF paper.

Items are matched by their `.id`, so this fuses LangChain `Document`s (the
retriever's currency) and `Chunk`s alike.
"""
from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def rrf_fuse(rankings: Sequence[Sequence[T]], weights: list[float] | None = None, k: int = 60) -> list[T]:
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must have one entry per ranking")

    scores: dict[str, float] = {}
    first_seen: dict[str, T] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            item_id = item.id  # type: ignore[attr-defined]
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)
            first_seen.setdefault(item_id, item)

    order = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [first_seen[cid] for cid in order]
