from abc import ABC, abstractmethod

from research_rag.models import Chunk


class BaseVectorStore(ABC):
    @abstractmethod
    def create_collection(self, dense_dim: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[str, list],
        limit: int = 20,
        filters: dict | None = None,
    ) -> list[Chunk]:
        raise NotImplementedError

    @abstractmethod
    def get_chunks_by_paper(self, paper_id: str) -> list[Chunk]:
        raise NotImplementedError

    @abstractmethod
    def count_chunks(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def list_paper_ids(self) -> list[str]:
        raise NotImplementedError
