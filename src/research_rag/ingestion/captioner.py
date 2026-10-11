"""Separate VLM used ONLY for figure/chart captioning — deliberately decoupled
from the embedding model and from the generation LLM.

Optional: only used when CAPTION_FIGURES=true. Without it, figure chunks are
built from the paper's own caption text, which the Open RAG Benchmark corpus
already provides — captioning adds detail (axis labels, trends, visible
numbers) at the cost of one VLM call per figure.

Default target: a self-hosted Qwen2.5-VL-7B model served behind an
OpenAI-compatible endpoint (Ollama or vLLM both work, no code changes needed —
only .env values change). This keeps figure captioning free and keeps paper
images off any third-party API.

---- Serving Qwen2.5-VL with Ollama (recommended — works on Windows/Mac/Linux) ----
    ollama pull qwen2.5vl:7b
    # Ollama runs as a background service automatically after install and
    # listens on http://localhost:11434 — nothing else to start manually.

    # .env:
    #   VLM_API_BASE=http://localhost:11434/v1
    #   VLM_API_KEY=ollama
    #   VLM_MODEL_NAME=qwen2.5vl:7b

---- Serving Qwen2.5-VL with vLLM (Linux/WSL2 only, needs a GPU, ~16GB VRAM) ----
    pip install vllm
    vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --limit-mm-per-prompt image=1

    # .env:
    #   VLM_API_BASE=http://localhost:8000/v1
    #   VLM_API_KEY=not-needed
    #   VLM_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
"""
import base64
import hashlib
import logging
import time

from research_rag.caching import RedisCache
from .base import BaseCaptioner

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 2
DEFAULT_TIMEOUT_SECONDS = 120


def _detect_image_mime_type(image_bytes: bytes) -> str:
    """Minimal magic-byte sniff — avoids the deprecated/removed stdlib imghdr module."""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


class VLMCaptioner(BaseCaptioner):
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model_name: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        cache: RedisCache | None = None,
    ):
        logger.debug(f"Initializing VLMCaptioner: api_base={api_base}, model={model_name}, "
                     f"max_retries={max_retries}, timeout={timeout}s, "
                     f"cache={'on' if cache and cache.enabled else 'off'}")
        from openai import OpenAI

        self._client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)
        self._model_name = model_name
        self._max_retries = max_retries
        self._cache = cache

    def caption(self, image_bytes: bytes, prompt: str) -> str:
        cache_namespace = f"caption:{self._model_name}"
        cache_key = hashlib.sha256(image_bytes + prompt.encode("utf-8")).hexdigest()
        if self._cache is not None:
            cached = self._cache.get(cache_namespace, cache_key)
            if cached is not None:
                logger.info("Figure caption served from cache (0 VLM call)")
                return cached

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = _detect_image_mime_type(image_bytes)

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                logger.debug(f"VLM call attempt {attempt + 1}/{self._max_retries + 1} "
                             f"(model={self._model_name}, image={len(image_bytes)} bytes, mime={mime_type})")
                t0 = time.monotonic()

                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
                        ],
                    }],
                    max_tokens=300,
                    temperature=0.0,
                )
                elapsed = time.monotonic() - t0
                logger.debug(f"VLM call succeeded in {elapsed:.1f}s")
                caption_text = response.choices[0].message.content.strip()
                if self._cache is not None:
                    self._cache.set(cache_namespace, cache_key, caption_text)
                return caption_text
            except Exception as error:  # noqa: BLE001
                last_error = error
                logger.warning(f"VLM call attempt {attempt + 1} failed: {error}")
                if attempt < self._max_retries:
                    backoff = 2 ** attempt
                    logger.debug(f"Retrying in {backoff}s ...")
                    time.sleep(backoff)

        logger.error(f"VLM captioning failed after {self._max_retries + 1} attempts")
        raise RuntimeError(f"VLM captioning failed after {self._max_retries + 1} attempts") from last_error
