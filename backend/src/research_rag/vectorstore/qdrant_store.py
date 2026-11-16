"""Qdrant hybrid store as a LangChain ``VectorStore``.

One collection with a named dense vector and a named sparse vector, plus a flat
payload schema (``text``, ``paper_id``, ``section``, ``section_id``,
``chunk_type``, ``page_no``, ``figure_no``, ``image_path``, ``paper_title``)
that everything downstream (retrieval, citations) reads from.

Why not ``langchain_qdrant.QdrantVectorStore``: it nests metadata under a
``metadata`` payload key, which would force re-embedding the whole indexed
corpus. This class keeps the existing payload layout (so the live collection
keeps working) while exposing the standard LangChain surface —
``add_documents``, ``similarity_search``, ``as_retriever`` — so ingestion and
retrieval are plain LangChain calls.

The store owns both encoders: dense vectors come from its ``Embeddings``, sparse
vectors from its ``HashingSparseEncoder``; hybrid search fuses the two with
Qdrant's server-side RRF.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from research_rag.embeddings import HashingSparseEncoder

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
UPSERT_BATCH_SIZE = 64          # one embedding request per batch; well under Azure's limits


class QdrantHybridVectorStore(VectorStore):
    def __init__(self, client, collection_name: str, embedding: Embeddings,
                 sparse_encoder: HashingSparseEncoder | None = None):
        self._client = client
        self._collection_name = collection_name
        self._embedding = embedding
        self.sparse_encoder = sparse_encoder or HashingSparseEncoder()

    @classmethod
    def connect(cls, url: str, collection_name: str, embedding: Embeddings, api_key: str | None = None,
                sparse_encoder: HashingSparseEncoder | None = None) -> "QdrantHybridVectorStore":
        logger.debug(f"Connecting QdrantHybridVectorStore: url={url}, collection={collection_name}")
        from qdrant_client import QdrantClient

        return cls(QdrantClient(url=url, api_key=api_key), collection_name, embedding, sparse_encoder)

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    # ------------------------------------------------------------------ collection
    def create_collection(self, dense_dim: int | None = None) -> None:
        """Idempotent — safe to call on every ingestion run, not just the first.
        Without `dense_dim`, the dimension is probed with one embedding call."""
        from qdrant_client.models import Distance, SparseVectorParams, VectorParams

        if self._client.collection_exists(self._collection_name):
            logger.info(f"Qdrant collection '{self._collection_name}' already exists, skipping creation")
            return

        dense_dim = dense_dim or len(self._embedding.embed_query("dimension probe"))
        logger.info(f"Creating Qdrant collection '{self._collection_name}' (dense_dim={dense_dim})")
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={DENSE_VECTOR_NAME: VectorParams(size=dense_dim, distance=Distance.COSINE)},
            sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
        )
        logger.info(f"Collection '{self._collection_name}' created")

    # ------------------------------------------------------------------ write
    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Embed (dense + sparse) and upsert, in batches. Each document's `id` is the
        chunk id; it is mapped to a stable UUID point id, so re-ingesting a paper
        overwrites its points instead of duplicating them."""
        from qdrant_client.models import PointStruct, SparseVector

        ids = kwargs.get("ids") or [doc.id for doc in documents]
        if any(not i for i in ids):
            raise ValueError("Every document needs an id (the chunk id) to be upserted")

        logger.info(f"Upserting {len(documents)} document(s) into '{self._collection_name}' ...")
        t0 = time.monotonic()
        point_ids: list[str] = []
        for start in range(0, len(documents), UPSERT_BATCH_SIZE):
            batch = documents[start:start + UPSERT_BATCH_SIZE]
            batch_ids = ids[start:start + UPSERT_BATCH_SIZE]
            dense = self._embedding.embed_documents([doc.page_content for doc in batch])
            points = []
            for doc, chunk_id, vector in zip(batch, batch_ids, dense):
                sparse = self.sparse_encoder.encode(doc.page_content)
                point_id = _stable_point_id(chunk_id)
                points.append(PointStruct(
                    id=point_id,
                    vector={DENSE_VECTOR_NAME: vector,
                            SPARSE_VECTOR_NAME: SparseVector(indices=sparse["indices"], values=sparse["values"])},
                    payload=_to_payload(doc),
                ))
                point_ids.append(point_id)
            self._client.upsert(collection_name=self._collection_name, points=points)

        logger.info(f"Upserted {len(documents)} document(s) in {time.monotonic() - t0:.1f}s")
        return point_ids

    def add_texts(self, texts: Iterable[str], metadatas: list[dict] | None = None, *,
                  ids: list[str] | None = None, **kwargs: Any) -> list[str]:
        texts = list(texts)
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        return self.add_documents([Document(id=i, page_content=t, metadata=m)
                                   for i, t, m in zip(ids, texts, metadatas)])

    @classmethod
    def from_texts(cls, texts: list[str], embedding: Embeddings, metadatas: list[dict] | None = None, *,
                   url: str = "http://localhost:6333", collection_name: str = "research_rag",
                   api_key: str | None = None, **kwargs: Any) -> "QdrantHybridVectorStore":
        store = cls.connect(url, collection_name, embedding, api_key)
        store.create_collection()
        store.add_texts(texts, metadatas, **kwargs)
        return store

    # ------------------------------------------------------------------ search
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Hybrid (dense + sparse, RRF-fused) search for one query."""
        return self.hybrid_search_by_vector(self._embedding.embed_query(query), self.sparse_encoder.encode(query), k=k)

    def hybrid_search_by_vector(self, dense_vector: list[float], sparse_vector: dict[str, list],
                                k: int = 20) -> list[Document]:
        """Search with precomputed vectors — lets a multi-query retriever embed all of
        its variants in one API call. There is deliberately no filter parameter:
        relevance is decided by the system, never by a caller-supplied paper/section."""
        from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

        logger.info(f"Hybrid search: k={k}")
        t0 = time.monotonic()
        results = self._client.query_points(
            collection_name=self._collection_name,
            prefetch=[
                Prefetch(query=dense_vector, using=DENSE_VECTOR_NAME, limit=k),
                Prefetch(query=SparseVector(indices=sparse_vector["indices"], values=sparse_vector["values"]),
                         using=SPARSE_VECTOR_NAME, limit=k),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
        )
        logger.info(f"Hybrid search returned {len(results.points)} result(s) in {time.monotonic() - t0:.2f}s")
        return [_to_document(str(point.id), point.payload) for point in results.points]

    # ------------------------------------------------------------------ metadata
    def get_by_paper(self, paper_id: str) -> list[Document]:
        """Pure metadata scroll — no query vector needed."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        logger.info(f"Scrolling all chunks for paper_id={paper_id} ...")
        paper_filter = Filter(must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))])
        docs: list[Document] = []
        next_offset = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=self._collection_name, scroll_filter=paper_filter,
                limit=100, offset=next_offset, with_payload=True,
            )
            docs.extend(_to_document(str(point.id), point.payload) for point in points)
            if next_offset is None:
                break
        logger.info(f"Found {len(docs)} chunk(s) for paper_id={paper_id}")
        return docs

    def count_chunks(self) -> int:
        """Total chunk count across the whole collection — used to show corpus stats."""
        return self._client.count(collection_name=self._collection_name, exact=True).count

    def list_paper_ids(self) -> list[str]:
        """Distinct paper_ids currently ingested. Qdrant has no native DISTINCT, so this
        scrolls payloads (paper_id field only, cheap) and dedupes client-side."""
        paper_ids: set[str] = set()
        next_offset = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=self._collection_name, limit=200, offset=next_offset, with_payload=["paper_id"],
            )
            paper_ids.update(point.payload["paper_id"] for point in points)
            if next_offset is None:
                break
        return sorted(paper_ids)


def _to_payload(doc: Document) -> dict:
    """Flat payload: the text plus every metadata field except query-time ones."""
    metadata = {k: v for k, v in doc.metadata.items() if k not in ("relevance_score", "chunk_id")}
    return {"text": doc.page_content, **metadata}


def _to_document(point_id: str, payload: dict) -> Document:
    metadata = {k: v for k, v in payload.items() if k != "text"}
    metadata["chunk_id"] = point_id
    return Document(id=point_id, page_content=payload["text"], metadata=metadata)


def _stable_point_id(chunk_id: str) -> str:
    """Qdrant point IDs must be UUID or unsigned int — derive a stable UUID from the chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
