"""
Single source of truth for configuration. Every other module receives its
config via constructor injection from a Settings instance — nothing else
reads os.environ directly.
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- Azure OpenAI: embeddings (can be a different resource than chat) ----
    azure_openai_embedding_endpoint: str
    azure_openai_embedding_api_key: str
    azure_openai_embedding_api_version: str = "2023-05-15"
    azure_openai_embedding_deployment: str

    # ---- Azure OpenAI: chat/generation (can be a different resource than embeddings) ----
    azure_openai_chat_endpoint: str
    azure_openai_chat_api_key: str
    azure_openai_chat_api_version: str = "2025-01-01-preview"
    azure_openai_chat_deployment: str

    # ---- Separate VLM: figure captioning only (optional — see CAPTION_FIGURES) ----
    vlm_api_base: str | None = None
    vlm_api_key: str | None = None
    vlm_model_name: str | None = None

    # ---- Qdrant ----
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "arxiv_papers"

    # ---- Corpus: Open RAGBench (vectara/open_ragbench, arXiv split) ----
    hf_dataset_repo: str = "vectara/open_ragbench"
    hf_dataset_subdir: str = "pdf/arxiv"
    open_ragbench_dir: str = "data/open_ragbench"      # local mirror: queries/qrels/answers/pdf_urls + corpus/*.json
    corpus_manifest_path: str = "data/arxiv_manifest.json"  # written by ingestion: paper_id -> title, categories ...

    # ---- Ingestion ----
    chunk_max_tokens: int = 400            # text chunks are packed up to this many tokens (cl100k_base)
    chunk_overlap_paragraphs: int = 1      # paragraphs repeated between consecutive text chunks
    caption_figures: bool = False          # True = VLM caption per figure (slow, needs VLM_*); False = figure caption text only
    skip_reference_sections: bool = True   # bibliography sections add noise to retrieval

    # ---- Retrieval / generation knobs ----
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---- Advanced retrieval techniques (each can be switched off independently) ----
    enable_query_rewrite: bool = True      # standalone / expanded query rewriting
    enable_hyde: bool = True               # Hypothetical Document Embeddings
    enable_reranker: bool = True           # cross-encoder reranking
    enable_planner: bool = True            # split multi-part questions into sub-questions, retrieve per part
    max_sub_questions: int = 3
    enable_context_grading: bool = True    # agentic "is this context enough?" check + one corrective retry
    enable_groundedness_check: bool = True # post-generation "is every claim supported?" check
    max_retrieval_retries: int = 1         # corrective retrieval loops (bounded so the graph can never spin)

    # ---- Agent state & memory ----
    memory_max_turns: int = 6              # turns kept verbatim; older turns are folded into a rolling summary
    checkpoint_backend: Literal["memory", "sqlite"] = "memory"
    checkpoint_path: str = "data/state/checkpoints.sqlite"
    classifier_min_confidence: float = 0.5 # below this, the intent defaults to "new_question" (the safe path)

    # ---- Guardrails ----
    pii_redaction_enabled: bool = True
    pii_backend: Literal["regex", "presidio"] = "regex"   # presidio needs optional extra deps, see requirements-extras.txt
    injection_guard_enabled: bool = True
    self_harm_guard_enabled: bool = True   # crisis-language short-circuit (any public chatbot meets this)
    hazard_guard_enabled: bool = True      # dual-use requests (weapons synthesis, pathogen enhancement ...)
    reference_check_enabled: bool = True   # arXiv IDs / DOIs in an answer must come from the retrieved sources
    max_input_chars: int = 1500

    # ---- Rate limiting ----
    rate_limit_per_minute: int = 10        # per API client
    rate_limit_per_hour: int = 100         # per API client
    global_rate_limit_per_minute: int = 60 # across every client (protects the Azure quota)
    llm_max_retries: int = 4               # retries on 429 / 5xx / timeouts, with exponential backoff
    llm_max_concurrency: int = 4           # simultaneous in-flight LLM calls
    llm_timeout_seconds: float = 60.0

    # ---- Inference-time evaluation (RAGAS) ----
    # Scoring a live answer costs extra LLM calls and seconds of latency, so it is
    # opt-in per request. This flag only controls whether the API will HONOUR that
    # opt-in; it is never on for every request.
    enable_inference_scoring: bool = True

    # ---- HTTP API ----
    api_keys: str = ""                     # comma-separated; empty = no auth (local development only)
    cors_origins: str = "*"                # comma-separated origins allowed to call the API from a browser
    trust_forwarded_for: bool = False      # use X-Forwarded-For for client identity (only behind a trusted proxy)

    # ---- Logging ----
    log_level: str = "INFO"  # DEBUG for maximum verbosity, INFO for normal use

    # ---- Redis cache (optional — caches embeddings + VLM captions) ----
    # If unset or unreachable, caching is silently disabled and everything
    # still works, just without the speedup on repeated runs.
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_enabled: bool = True

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — safe to call repeatedly across the app."""
    return Settings()
