# Documentation

Repository-wide documentation for Research RAG. Each application also has its own detailed README:
[backend/README.md](../backend/README.md) and [frontend/README.md](../frontend/README.md).

## Read in this order

| # | Document | Answers |
|---|---|---|
| 1 | [GETTING_STARTED.md](GETTING_STARTED.md) | How do I run the whole thing from a fresh clone, and how do I know each step worked? |
| 2 | [ARCHITECTURE.md](ARCHITECTURE.md) | What are the moving parts, how does a request flow through them, where does each concern live? |
| 3 | [PORTS_AND_SERVICES.md](PORTS_AND_SERVICES.md) | Which process listens on which port, which are required, and how do I change them? |
| 4 | [TECH_STACK.md](TECH_STACK.md) | Which libraries, models and services are used, and why each one was chosen. |
| 5 | [CONFIGURATION.md](CONFIGURATION.md) | Every environment variable for the backend and the frontend, grouped by purpose. |
| 6 | [DOS_AND_DONTS.md](DOS_AND_DONTS.md) | The rules this codebase follows: what to do and what never to do. |
| 7 | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Symptom → cause → fix, for both apps and the infrastructure. |
| 8 | [ROADMAP.md](ROADMAP.md) | Known limitations and the improvements that can be built next. |

## Other material

| File | What it is |
|---|---|
| [architecture-diagram.html](architecture-diagram.html) | Interactive architecture diagram (dark/light, export to PNG/SVG). Download it and open it in a browser, because GitHub shows HTML as source. |
| [backend/docs/SYSTEM_GUIDE.md](../backend/docs/SYSTEM_GUIDE.md) | The long-form design story: journey, decisions, bugs found, RAGAS wiring, evaluation results. |
| [backend/docs/RESPONSIBLE_AI.md](../backend/docs/RESPONSIBLE_AI.md) | Safety posture, risks considered, and how each guardrail mitigates them. |
| [backend/docs/TEST_QUESTIONS.md](../backend/docs/TEST_QUESTIONS.md) | Questions known to work on a 50-paper index, grouped by the capability they exercise. |
| [backend/docs/DEMO_SCRIPT.md](../backend/docs/DEMO_SCRIPT.md) | A ~10-minute walkthrough script for demonstrating the system. |
| [backend/docs/research-rag-deck.html](../backend/docs/research-rag-deck.html) | Slide deck (HTML). |
