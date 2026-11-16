## What and why

<!-- What does this change, and why is it needed? Link an issue if there is one. -->

## Area

- [ ] Backend (`backend/`)
- [ ] Frontend (`frontend/`)
- [ ] API contract (schemas changed: `backend/.../api/schemas.py` **and** `frontend/src/api/types.ts`)
- [ ] Docs only

## Checklist

- [ ] `python -m pytest tests -q` passes in `backend/`
- [ ] `npm run lint && npm run build` passes in `frontend/`
- [ ] New settings are in `config/settings.py` **and** `.env.example`
- [ ] The affected module `README.md` and `docs/` are updated
- [ ] No secrets, `.env` files, corpus data or build output committed
