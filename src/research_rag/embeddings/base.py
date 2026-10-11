from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def sparse_encode(self, text: str) -> dict[str, list]:
        """Returns {'indices': [...], 'values': [...]} for Qdrant's sparse vector format."""
        raise NotImplementedError
