"""Hashing-based sparse encoder for the lexical half of hybrid search.

LangChain has no core interface for sparse vectors, so this is a small
standalone encoder the vector store holds next to its dense ``Embeddings``.
Swap it for SPLADE/BM25-over-corpus later — nothing outside the vector store
needs to change.

The hashing scheme must stay byte-for-byte stable: the indexed collection was
encoded with it, and a different scheme would silently break lexical matching.
"""
import hashlib
import re
from collections import Counter

SPARSE_VOCAB_SIZE = 30_000
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


class HashingSparseEncoder:
    def __init__(self, vocab_size: int = SPARSE_VOCAB_SIZE):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> dict[str, list]:
        """token -> stable index in [0, vocab_size), TF-weighted. Different tokens can
        hash into the same bucket (a collision) — Qdrant requires unique indices per
        point, so collisions are merged (counts summed) rather than left as duplicate
        index entries. Pure local computation: no API call, nothing to cache."""
        counts = Counter(_TOKEN_RE.findall(text.lower()))

        bucket_totals: dict[int, float] = {}
        for token, count in counts.items():
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.vocab_size
            bucket_totals[idx] = bucket_totals.get(idx, 0.0) + float(count)

        return {"indices": list(bucket_totals.keys()), "values": list(bucket_totals.values())}
