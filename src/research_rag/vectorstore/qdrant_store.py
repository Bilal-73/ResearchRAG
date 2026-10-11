"""Qdrant collection: dense + sparse named vectors, one payload schema."""
import logging
import time

from research_rag.models import Chunk
from .base import BaseVectorStore

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantVectorStore(BaseVectorStore):
    def __init__(self, url: str, collection_name: str, api_key: str | None = None):
        logger.debug(f"Initializing QdrantVectorStore: url={url}, collection={collection_name}")
        from qdrant_client import QdrantClient

        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection_name = collection_name

    def create_collection(self, dense_dim: int) -> None:
        """Idempotent — safe to call on every ingestion run, not just the first."""
        from qdrant_client.models import Distance, SparseVectorParams, VectorParams

        if self._client.collection_exists(self._collection_name):
            logger.info(f"Qdrant collection '{self._collection_name}' already exists, skipping creation")
            return

        logger.info(f"Creating Qdrant collection '{self._collection_name}' (dense_dim={dense_dim})")
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={DENSE_VECTOR_NAME: VectorParams(size=dense_dim, distance=Distance.COSINE)},
            sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
        )
        logger.info(f"Collection '{self._collection_name}' created")

    def upsert(self, chunks: list[Chunk]) -> None:
        from qdrant_client.models import PointStruct, SparseVector

        logger.info(f"Upserting {len(chunks)} chunk(s) into '{self._collection_name}' ...")
        t0 = time.monotonic()

        points = [
            PointStruct(
                id=_stable_point_id(chunk.id),
                vector={
                    DENSE_VECTOR_NAME: chunk.dense_vector,
                    SPARSE_VECTOR_NAME: SparseVector(
                        indices=chunk.sparse_vector["indices"],
                        values=chunk.sparse_vector["values"],
                    ),
                },
                payload=chunk.to_payload(),
            )
            for chunk in chunks
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)

        logger.info(f"Upserted {len(chunks)} chunk(s) in {time.monotonic() - t0:.1f}s")

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[str, list],
        limit: int = 20,
        filters: dict | None = None,
    ) -> list[Chunk]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue, SparseVector
        from qdrant_client.models import FusionQuery, Prefetch, Fusion

        logger.info(f"Hybrid search: limit={limit}, filters={filters}")
        t0 = time.monotonic()

        qdrant_filter = None
        if filters:
            qdrant_filter = Filter(must=[
                FieldCondition(key=key, match=MatchValue(value=value)) for key, value in filters.items()
            ])

        results = self._client.query_points(
            collection_name=self._collection_name,
            prefetch=[
                Prefetch(query=dense_vector, using=DENSE_VECTOR_NAME, limit=limit, filter=qdrant_filter),
                Prefetch(
                    query=SparseVector(indices=sparse_vector["indices"], values=sparse_vector["values"]),
                    using=SPARSE_VECTOR_NAME, limit=limit, filter=qdrant_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
        )

        elapsed = time.monotonic() - t0
        logger.info(f"Hybrid search returned {len(results.points)} result(s) in {elapsed:.2f}s")

        return [Chunk.from_payload(str(point.id), point.payload) for point in results.points]

    def get_chunks_by_paper(self, paper_id: str) -> list[Chunk]:
        """Pure metadata scroll — no query vector needed. Used by the golden QA
        generator to pull every chunk for a paper without running a search."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        logger.info(f"Scrolling all chunks for paper_id={paper_id} ...")
        paper_filter = Filter(must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))])

        chunks: list[Chunk] = []
        next_offset = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=paper_filter,
                limit=100,
                offset=next_offset,
                with_payload=True,
            )
            chunks.extend(Chunk.from_payload(str(point.id), point.payload) for point in points)
            if next_offset is None:
                break

        logger.info(f"Found {len(chunks)} chunk(s) for paper_id={paper_id}")
        return chunks

    def count_chunks(self) -> int:
        """Total chunk count across the whole collection — used by the UI to show corpus stats."""
        result = self._client.count(collection_name=self._collection_name, exact=True)
        return result.count

    def list_paper_ids(self) -> list[str]:
        """Distinct paper_ids currently ingested — used by the UI's corpus overview.
        Qdrant has no native DISTINCT, so this scrolls payloads (paper_id field only,
        cheap) and dedupes client-side."""
        paper_ids: set[str] = set()
        next_offset = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=self._collection_name,
                limit=200,
                offset=next_offset,
                with_payload=["paper_id"],
            )
            paper_ids.update(point.payload["paper_id"] for point in points)
            if next_offset is None:
                break
        return sorted(paper_ids)


def _stable_point_id(chunk_id: str) -> str:
    """Qdrant point IDs must be UUID or unsigned int — derive a stable UUID from the chunk id."""
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
