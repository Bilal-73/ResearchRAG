/**
 * The single place this app talks to the API.
 *
 * Two behaviours worth knowing, both mirroring deliberate backend decisions:
 *
 * 1. A refused, redirected or safety-blocked turn returns **HTTP 200**. It is not
 *    an error — the request was valid and the system behaved correctly. So only
 *    genuine 4xx/5xx throw here, and the UI branches on `answer_kind` for the rest.
 * 2. Every error body is `{error, message, ...}`, so `ApiError` carries a stable
 *    machine code alongside a message that is safe to show a user. We never render
 *    a raw status number at someone.
 */
import type {
  ApiErrorBody,
  CacheResponse,
  ChatOptions,
  ChatResponse,
  CorpusResponse,
  GuardrailsResponse,
  HealthResponse,
  MemoryResponse,
} from './types';

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '');
const API_KEY = import.meta.env.VITE_API_KEY ?? '';

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly retryAfterSeconds?: number;
  readonly detail?: Record<string, unknown>;

  constructor(status: number, body: ApiErrorBody) {
    super(body.message || `Request failed (${status})`);
    this.name = 'ApiError';
    this.status = status;
    this.code = body.error || 'error';
    this.retryAfterSeconds = body.retry_after_seconds;
    this.detail = body.detail;
  }

  /** Whether retrying the same request could plausibly succeed. */
  get isTransient(): boolean {
    return this.status === 429 || this.status === 503;
  }
}

/** Raised when the API cannot be reached at all — a different problem from an API error,
 *  and the fix is different too (start the server vs. fix the request). */
export class NetworkError extends Error {
  readonly baseUrl: string;

  constructor(baseUrl: string, cause?: unknown) {
    super(
      `Cannot reach the API at ${baseUrl}. Is it running?\n` +
        `Start it with:  .venv\\Scripts\\python.exe scripts/run_api.py`,
    );
    this.name = 'NetworkError';
    this.baseUrl = baseUrl;
    this.cause = cause;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(init.headers as Record<string, string> | undefined),
  };
  if (API_KEY) headers['X-API-Key'] = API_KEY;

  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, { ...init, headers });
  } catch (cause) {
    // fetch() rejects on DNS/connection/CORS failures — never on an HTTP error status.
    throw new NetworkError(BASE_URL, cause);
  }

  if (response.status === 204) return undefined as T;

  const text = await response.text();
  const parsed: unknown = text ? JSON.parse(text) : {};

  if (!response.ok) {
    throw new ApiError(response.status, parsed as ApiErrorBody);
  }
  return parsed as T;
}

export const api = {
  baseUrl: BASE_URL,
  hasApiKey: Boolean(API_KEY),

  health: () => request<HealthResponse>('/health'),

  chat: (message: string, conversationId: string | null, options?: ChatOptions) =>
    request<ChatResponse>('/v1/chat', {
      method: 'POST',
      body: JSON.stringify({
        message,
        conversation_id: conversationId,
        options: options && Object.keys(options).length ? options : undefined,
      }),
    }),

  deleteConversation: (conversationId: string) =>
    request<void>(`/v1/chat/${encodeURIComponent(conversationId)}`, { method: 'DELETE' }),

  memory: (conversationId: string) =>
    request<MemoryResponse>(`/v1/chat/${encodeURIComponent(conversationId)}/memory`),

  corpus: (limit = 50, offset = 0, category?: string) => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (category) params.set('category', category);
    return request<CorpusResponse>(`/v1/corpus?${params}`);
  },

  cache: () => request<CacheResponse>('/v1/cache'),

  resetCacheStats: () => request<void>('/v1/cache/reset-stats', { method: 'POST' }),

  guardrails: () => request<GuardrailsResponse>('/v1/guardrails'),
};
