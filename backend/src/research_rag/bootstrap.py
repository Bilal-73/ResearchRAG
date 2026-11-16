"""Composition root: builds every object graph the app uses, from Settings.

This is the ONE place implementations are chosen, so swapping a component
(a different vector store, reranker or embedding model) is a change here and
nowhere else. Every component is a LangChain primitive — chat model,
``Embeddings``, ``VectorStore``, ``BaseRetriever``, ``BaseDocumentCompressor`` —
composed into LCEL chains and orchestrated by the LangGraph agent. The API, the CLI scripts and the evaluation harness all call into it, so
they are guaranteed to be running the same system.
"""
from __future__ import annotations

import logging
from pathlib import Path

from config.settings import get_settings

from research_rag.agents import AgentNodes, IntentClassifier, QuestionPlanner, ResearchRAGAgent, build_checkpointer
from research_rag.caching import RedisCache
from research_rag.corpus import CorpusInfo
from research_rag.embeddings import CachedEmbeddings, HashingSparseEncoder
from research_rag.generation import AnswerGenerator, CitationPromptBuilder
from research_rag.guardrails import (
    Canary, DualUseDetector, GuardrailRegistry, InputGuardrails, OutputGuardrails, PIIGuard,
    PromptInjectionGuard, SelfHarmDetector, build_rate_limiter,
)
from research_rag.llm import LLMClient, build_chat_model, build_embeddings
from research_rag.rag_system import RAGSystem
from research_rag.retrieval import AdvancedRetriever, CrossEncoderReranker, HyDEGenerator, QueryRewriter
from research_rag.vectorstore import QdrantHybridVectorStore

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build_cache(settings):
    if not settings.redis_cache_enabled:
        logger.info("Redis cache disabled via settings (REDIS_CACHE_ENABLED=false)")
        return None
    return RedisCache(url=settings.redis_url)


def _build_llm(settings) -> LLMClient:
    return LLMClient(
        build_chat_model(settings),
        max_retries=settings.llm_max_retries,
        max_concurrency=settings.llm_max_concurrency,
    )


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_index_components(settings=None) -> dict:
    """Embeddings + vector store + corpus view. Everything ingestion needs and nothing more —
    notably NOT the cross-encoder, which takes seconds to load and is irrelevant to indexing."""
    settings = settings or get_settings()
    cache = _build_cache(settings)
    embeddings = CachedEmbeddings(
        build_embeddings(settings), cache,
        namespace=f"embed:{settings.azure_openai_embedding_deployment}",   # same keys as before LangChain
    )
    vector_store = QdrantHybridVectorStore.connect(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
        embedding=embeddings,
        api_key=settings.qdrant_api_key,
        sparse_encoder=HashingSparseEncoder(),
    )
    corpus = CorpusInfo(vector_store, manifest_path=_resolve(settings.corpus_manifest_path))
    return dict(settings=settings, cache=cache, embeddings=embeddings, vector_store=vector_store, corpus=corpus)


def build_query_components(settings=None) -> dict:
    """Everything needed to answer questions (no ingestion machinery)."""
    settings = settings or get_settings()
    c = build_index_components(settings)
    cache, embeddings, vector_store, corpus = c["cache"], c["embeddings"], c["vector_store"], c["corpus"]
    llm = _build_llm(settings)
    canary = Canary()
    rewriter, hyde = QueryRewriter(llm), HyDEGenerator(llm)
    retriever = AdvancedRetriever(
        vector_store=vector_store, rewriter=rewriter, hyde=hyde,
        use_rewrite=settings.enable_query_rewrite, use_hyde=settings.enable_hyde,
        k=settings.retrieval_top_k,
    )
    reranker = CrossEncoderReranker(model_name=settings.reranker_model, top_n=settings.rerank_top_k)
    generator = AnswerGenerator(llm, CitationPromptBuilder(canary), corpus)
    return dict(settings=settings, cache=cache, embeddings=embeddings, vector_store=vector_store, llm=llm,
                corpus=corpus, canary=canary, rewriter=rewriter, hyde=hyde, retriever=retriever,
                reranker=reranker, generator=generator)


def build_agent(settings=None):
    """The LangGraph agent plus the handles the API needs.

    Returns (agent, guardrail_registry, rate_limiter, llm, corpus_info, cache).
    Build ONCE and reuse — loading the cross-encoder is the slow part.
    """
    settings = settings or get_settings()
    logger.info("Building conversational agent (LangGraph) ...")
    c = build_query_components(settings)
    llm, corpus = c["llm"], c["corpus"]

    pii = PIIGuard(backend=settings.pii_backend, enabled=settings.pii_redaction_enabled)
    injection = PromptInjectionGuard(enabled=settings.injection_guard_enabled)
    rate_limiter = build_rate_limiter(settings)
    input_guard = InputGuardrails(
        rate_limiter, pii, injection,
        SelfHarmDetector(settings.self_harm_guard_enabled),
        DualUseDetector(settings.hazard_guard_enabled),
        max_input_chars=settings.max_input_chars,
    )
    defaults = {
        "enable_query_rewrite": settings.enable_query_rewrite, "enable_hyde": settings.enable_hyde,
        "enable_reranker": settings.enable_reranker, "enable_planner": settings.enable_planner,
        "max_sub_questions": settings.max_sub_questions,
        "enable_context_grading": settings.enable_context_grading,
        "enable_groundedness_check": settings.enable_groundedness_check,
        "max_retrieval_retries": settings.max_retrieval_retries,
        "memory_max_turns": settings.memory_max_turns,
        "retrieval_top_k": settings.retrieval_top_k, "rerank_top_k": settings.rerank_top_k,
    }
    nodes = AgentNodes(
        llm=llm, input_guard=input_guard,
        output_guard=OutputGuardrails(pii, c["canary"], settings.reference_check_enabled),
        injection=injection,
        classifier=IntentClassifier(llm, corpus, min_confidence=settings.classifier_min_confidence),
        planner=QuestionPlanner(llm, settings.max_sub_questions),
        rewriter=c["rewriter"], hyde=c["hyde"], retriever=c["retriever"], reranker=c["reranker"],
        generator=c["generator"], corpus=corpus, defaults=defaults,
    )
    agent = ResearchRAGAgent(
        nodes,
        build_checkpointer(settings.checkpoint_backend, str(_resolve(settings.checkpoint_path))),
        defaults,
    )
    logger.info("Agent built successfully.")
    return agent, GuardrailRegistry(settings), rate_limiter, llm, corpus, c["cache"]


def build_rag_system(settings=None) -> RAGSystem:
    """Facade over the agent, used by evaluation and one-shot queries."""
    settings = settings or get_settings()
    agent, _registry, _limiter, _llm, _corpus, _cache = build_agent(settings)
    components = build_query_components(settings)
    return RAGSystem(agent, retriever=components["retriever"], reranker=components["reranker"])


def build_ingestion(settings=None):
    """Corpus reader + chunker + pipeline. Returns (pipeline, corpus, index_components)."""
    from research_rag.ingestion import (
        IngestionPipeline, RAGBenchCorpus, SectionChunker, TiktokenCounter, VLMCaptioner,
    )

    settings = settings or get_settings()
    c = build_index_components(settings)

    captioner = None
    if settings.caption_figures:
        if not (settings.vlm_api_base and settings.vlm_model_name):
            raise ValueError("CAPTION_FIGURES=true requires VLM_API_BASE and VLM_MODEL_NAME in .env")
        captioner = VLMCaptioner(
            api_base=settings.vlm_api_base, api_key=settings.vlm_api_key or "none",
            model_name=settings.vlm_model_name, cache=c["cache"],
        )
        logger.info(f"Figure captioning ON via {settings.vlm_model_name}")

    corpus = RAGBenchCorpus(_resolve(settings.open_ragbench_dir))
    chunker = SectionChunker(
        max_tokens=settings.chunk_max_tokens,
        overlap_paragraphs=settings.chunk_overlap_paragraphs,
        token_counter=TiktokenCounter(),
        skip_reference_sections=settings.skip_reference_sections,
        captioner=captioner,
    )
    pipeline = IngestionPipeline(
        corpus=corpus, chunker=chunker, vector_store=c["vector_store"],
        manifest_path=_resolve(settings.corpus_manifest_path),
        load_images=settings.caption_figures,
    )
    return pipeline, corpus, c
