from abc import ABC, abstractmethod

from research_rag.models import Chunk


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[Chunk]:
        raise NotImplementedError


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        raise NotImplementedError
