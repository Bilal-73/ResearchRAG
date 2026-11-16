"""LangChain model factories — the one place Azure/OpenAI model objects are built.

The agent, ingestion, the RAGAS judge and the figure captioner all get their
models from here, so every component talks to the same deployments through the
same LangChain interfaces (``BaseChatModel`` / ``Embeddings``).
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


def build_chat_model(settings, *, max_retries: int = 0) -> BaseChatModel:
    """Azure chat deployment. ``max_retries=0`` by default: LLMClient owns the retry policy."""
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_chat_endpoint,
        api_key=settings.azure_openai_chat_api_key,
        api_version=settings.azure_openai_chat_api_version,
        azure_deployment=settings.azure_openai_chat_deployment,
        timeout=settings.llm_timeout_seconds,
        max_retries=max_retries,
        temperature=0.0,
        validate_base_url=False,
    )


def build_embeddings(settings) -> Embeddings:
    """Azure embedding deployment (its own resource, separate from chat)."""
    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_embedding_endpoint,
        api_key=settings.azure_openai_embedding_api_key,
        api_version=settings.azure_openai_embedding_api_version,
        azure_deployment=settings.azure_openai_embedding_deployment,
    )


def build_vision_model(api_base: str, api_key: str, model_name: str, timeout: float = 120.0) -> BaseChatModel:
    """Any OpenAI-compatible vision endpoint (Ollama / vLLM serving Qwen2.5-VL by default)."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(base_url=api_base, api_key=api_key, model=model_name, timeout=timeout,
                      max_retries=0, temperature=0.0, max_tokens=300)
