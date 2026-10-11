from abc import ABC, abstractmethod

from research_rag.models import Chunk, RAGResponse


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, chunks: list[Chunk]) -> RAGResponse:
        raise NotImplementedError
