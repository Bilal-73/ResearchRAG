/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the Research RAG API. Default: http://127.0.0.1:8000 */
  readonly VITE_API_BASE_URL?: string;
  /** Sent as X-API-Key. Only needed when the server has API_KEYS set. */
  readonly VITE_API_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
