# Contributing

Thanks for your interest in Research RAG. Issues, ideas and pull requests are welcome.

## Before you start

1. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) to see where your change belongs.
2. Read [docs/DOS_AND_DONTS.md](docs/DOS_AND_DONTS.md). It is short and lists the rules reviews check for.
3. For anything larger than a small fix, open an issue first so the approach can be agreed before you
   write code. [docs/ROADMAP.md](docs/ROADMAP.md) lists ideas that are already welcome.

## Development setup

Follow [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md). For backend work you also need:

```bash
pip install -r requirements-dev.txt
```

## Workflow

1. Fork the repo and create a branch from `main`: `feature/<short-name>`, `fix/<short-name>` or `docs/<short-name>`.
2. Make the change, with tests.
3. Run the checks below locally.
4. Open a pull request. The template has a checklist. CI must pass before merge.

## Checks

| Area | Command (run inside the folder) | Must |
|---|---|---|
| Backend | `python -m pytest tests -q` | pass. It is offline and needs no keys. |
| Frontend | `npm run lint` | report no errors |
| Frontend | `npm run build` | succeed (it type-checks with `tsc -b`) |

## Conventions

- **Python:** type hints on public functions, 4-space indent, imports grouped stdlib → third-party → local.
  New settings go in `config/settings.py` **and** `.env.example`.
- **TypeScript:** strict mode, HTTP only through `src/api/client.ts`, types in `src/api/types.ts`.
- **API changes:** update `schemas.py` and `types.ts` in the same pull request, and the endpoint table in
  `backend/README.md`.
- **Docs:** if you change what a module does, update the `README.md` in its folder.
- **Commits:** imperative mood, and say why ("Fix reranker dropping table chunks when …").

## Reporting bugs

Open an issue with:

- what you did (the request body, or the steps in the UI)
- what you expected and what happened
- the `X-Request-ID` from the response, and the matching backend log lines
- the output of `GET /health`

Never paste API keys, `.env` contents or private data into an issue.

## Security issues

Do **not** open a public issue. See [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
