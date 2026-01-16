# Repository Guidelines

## Project Structure & Module Organization

- `agent_bus/`: main Python package
  - `peer_server.py`: MCP server (FastMCP) tools: `topic_*`, `cursor_reset`, `sync`
  - `db.py`: SQLite (WAL) storage, schema/versioning, sync/cursor logic
  - `cli.py`: admin CLI (`agent-bus cli ...`)
  - `web/`: optional FastAPI web UI + Jinja templates in `agent_bus/web/templates/`
- `tests/`: pytest suite (`test_*.py`)
- `spec.md`: protocol + schema spec (treat as the source of truth when changing behavior)

## Build, Test, and Development Commands

This repo targets Python 3.12+ and uses `uv` for local development:

```bash
uv sync                 # install runtime deps
uv sync --dev           # install dev deps (pytest, ruff, web extra)
uv run agent-bus        # run MCP server over stdio
uv sync --extra web     # install Web UI deps
uv run agent-bus serve  # run Web UI (default http://127.0.0.1:8080)
uv run ruff format      # format
uv run ruff check       # lint
uv run pytest           # tests
```

## Coding Style & Naming Conventions

- Format/lint with Ruff (`ruff format`, `ruff check`). Line length is 100.
- Prefer type hints and small, focused modules.
- Naming: `snake_case` (functions/vars), `PascalCase` (types), `SCREAMING_SNAKE_CASE` (constants).

## Testing Guidelines

- Framework: pytest (some tests use `anyio` to exercise the stdio MCP server).
- Keep tests hermetic: use `tmp_path` and point `AGENT_BUS_DB` at a temp SQLite file.
- Name tests `tests/test_*.py` and prefer unit tests for `AgentBusDB` plus a small number of end-to-end smoke tests.

## Commit & Pull Request Guidelines

Commit history uses lightweight Conventional-ish prefixes (e.g. `feat(web): ...`, `fix(web): ...`, `docs: ...`, `chore: ...`) and component prefixes like `web: ...` / `peer: ...` / `cli: ...`. Follow one of those patterns.

PRs should include: a short description, how to test, and screenshots for Web UI changes. Run `uv run ruff check` and `uv run pytest` before requesting review.

## Security & Configuration Tips

- Web UI has no auth; keep it bound to localhost unless you add protections.
- DB schema is versioned; a mismatch requires wiping the DB (see `agent-bus cli db wipe --yes`).
- Key env vars: `AGENT_BUS_DB`, `AGENT_BUS_MAX_*`, `AGENT_BUS_POLL_*`.
