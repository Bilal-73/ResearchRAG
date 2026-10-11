"""Abstract interface for the pluggable ingestion stage (figure captioning)."""
from abc import ABC, abstractmethod


class BaseCaptioner(ABC):
    @abstractmethod
    def caption(self, image_bytes: bytes, prompt: str) -> str:
        raise NotImplementedError
