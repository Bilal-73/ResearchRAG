# Security policy

## Reporting a vulnerability

Please **do not** report security problems in public issues.

Use GitHub's private reporting instead: open the repository's **Security** tab and choose
**Report a vulnerability**. Include what you found, how to reproduce it, and its impact. You should get
an acknowledgement within a few days.

This covers, for example:

- a way to bypass the input guardrails (prompt injection, dual-use screening, PII redaction)
- a way to make the API return content that is not grounded in the indexed papers, without that being reported
- authentication or rate-limit bypass
- leakage of secrets, other users' conversation memory, or unredacted PII

## Supported versions

Only the latest commit on `main` is supported.

## Running this project safely

This is a research and demonstration project. The defaults are tuned for local development, **not**
for exposure to the internet:

| Default | Why it is unsafe in public | Fix |
|---|---|---|
| `API_KEYS` empty means no authentication | Anyone can spend your Azure quota | Set `API_KEYS` |
| `CORS_ORIGINS=*` | Any website can call your API from a visitor's browser | Set it to your frontend's exact origin |
| `VITE_API_KEY` is compiled into the frontend bundle | The key is readable by every visitor | Put a server-side proxy in front of the API |
| Qdrant and Redis run without passwords | Anyone on the network can read or modify the index and cache | Do not expose ports 6333, 6334 or 6379. Set `QDRANT_API_KEY` and Redis auth. |
| In-memory rate limiting | Limits are per process | Put a gateway or shared limiter in front of multiple workers |

See the production checklist in [docs/CONFIGURATION.md](docs/CONFIGURATION.md#production-checklist).

## Secrets

No secrets are committed to this repository. `.env` and `.env.local` are git-ignored, and the
`.env.example` files contain placeholders only. If you ever commit a key by mistake, **rotate it
immediately**. Removing it from the latest commit does not remove it from git history.
