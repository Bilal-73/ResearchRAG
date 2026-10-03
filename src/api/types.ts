/**
 * Mirrors the backend's Pydantic schemas (src/research_rag/api/schemas.py).
 *
 * Kept hand-written rather than generated so the shapes stay readable, but it is
 * a contract: if a field here disagrees with the API, the API wins. `GET /openapi.json`
 * is the source of truth.
 */

/** How the answer was produced. `rag` is the only grounded path — everything else
 *  is a deliberate non-answer and must not be rendered as if it were sourced. */
export type AnswerKind =
  | 'rag'
  | 'chitchat'
  | 'out_of_scope'
  | 'not_found'
  | 'blocked'
  | 'safety'
  | 'error';

export type BlockKind =
  | ''
  | 'rate_limit'
  | 'too_long'
  | 'empty'
  | 'self_harm'
  | 'dual_use'
  | 'injection';

export type ChunkType = 'text' | 'table' | 'figure';

export interface Source {
  label: string;
  paper_id: string;
  paper_title: string | null;
  section: string;
  section_id: number | null;
  page_no: number | null;
  chunk_type: ChunkType;
  figure_no: string | null;
  snippet: string;
  score: number | null;
  /** True when the answer actually references this source inline. */
  cited: boolean;
  url: string | null;
}

export interface GuardrailEvent {
  guardrail: string;
  action: string;
  detail: string;
}

export interface TraceStep {
  node: string;
  role: string;
  kind: string;
  detail: string;
  seconds: number;
}

export interface Diagnostics {
  intent: string;
  intent_confidence: number;
  intent_reason: string;
  intent_source: string;
  standalone_query: string;
  sub_questions: string[];
  variants_used: string[];
  n_candidates: number;
  retry_count: number;
  context_ok: boolean;
  context_reason: string;
  citations_ok: boolean;
  /** null means the groundedness check did not run or was unavailable — NOT "passed". */
  grounded: boolean | null;
  unsupported_claims: string[];
  route: string[];
  explanation: string[];
  trace: TraceStep[];
  total_seconds: number;
}

export interface ChatResponse {
  answer: string;
  answer_kind: AnswerKind;
  conversation_id: string;
  sources: Source[];
  blocked: boolean;
  block_kind: BlockKind;
  guardrail_events: GuardrailEvent[];
  diagnostics: Diagnostics | null;
}

export interface ChatOptions {
  enable_query_rewrite?: boolean;
  enable_hyde?: boolean;
  enable_reranker?: boolean;
  enable_planner?: boolean;
  enable_context_grading?: boolean;
  enable_groundedness_check?: boolean;
}

export interface PaperSummary {
  paper_id: string;
  title: string | null;
  categories: string[];
  authors: string[];
  published: string | null;
  n_chunks: number | null;
  url: string | null;
  /** The corpus ships pre-parsed JSON, so no PDF is stored locally — this points at arXiv. */
  pdf_url: string | null;
}

export interface CorpusResponse {
  n_papers: number;
  categories: Record<string, number>;
  description: string;
  papers: PaperSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface CacheStats {
  hits: number;
  misses: number;
  sets: number;
  errors: number;
  lookups: number;
  hit_rate: number | null;
  avg_get_ms: number | null;
  avg_set_ms: number | null;
  get_seconds: number;
  set_seconds: number;
}

export interface CacheNamespace {
  name: string;
  label: string;
  keys: number;
}

export interface CacheResponse {
  enabled: boolean;
  url: string;
  ttl_seconds: number;
  key_prefix: string;
  stats: CacheStats;
  namespaces: CacheNamespace[];
  server: Record<string, unknown>;
  note: string | null;
}

export interface MemoryTurn {
  role: 'user' | 'assistant' | '';
  content: string;
  intent: string;
}

export interface MemoryResponse {
  conversation_id: string;
  turns: number;
  max_turns_kept: number;
  summary: string;
  focus_question: string;
  focus_papers: string[];
  history: MemoryTurn[];
}

export interface GuardrailInfo {
  id: string;
  name: string;
  stage: string;
  description: string;
  implementation: string;
  enabled: boolean;
}

export interface GuardrailsResponse {
  stages: string[];
  stage_titles: Record<string, string>;
  guardrails: GuardrailInfo[];
}

export type ComponentStatus = 'ok' | 'degraded' | 'down';

export interface HealthResponse {
  status: ComponentStatus;
  version: string;
  components: Record<string, { status: ComponentStatus; detail: string }>;
  n_papers: number;
}

/** Every 4xx/5xx body from the API has this shape. */
export interface ApiErrorBody {
  error: string;
  message: string;
  detail?: Record<string, unknown>;
  retry_after_seconds?: number;
}
