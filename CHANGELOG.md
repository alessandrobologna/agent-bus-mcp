# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - Unreleased

### Breaking Changes

- The browser UI served by `agent-bus serve` is now a packaged React SPA workbench instead of the
  older Jinja/HTMX interface.
- The legacy template partials and browser-only HTML fragment routes have been removed. The web
  surface now centers on the SPA shell plus `/api/*` JSON and SSE endpoints.

### Added

- Added a tabbed SPA workbench for browsing topics, opening multiple threads at once, searching
  from the sidebar, and following live updates through SSE-backed invalidation.
- Added frontend unit and Playwright smoke coverage, and package builds now include the compiled
  frontend bundle automatically in wheels and sdists.

### Changed

- The reusable `agent-bus-workflows` skill now spells out safer review-loop defaults, including
  `sync(wait_seconds=0)` for backlog catch-up and an explicit summarize-and-confirm stop before an
  agent starts implementing findings the user did not yet approve.
- `/api/stream/topics` now uses a lightweight `topics_version` invalidation check instead of
  rebuilding full topic summaries on every poll, which reduces SQLite contention and makes live
  topic-list updates cheaper under active browser sessions.

### Fixed

- `agent-bus serve` now exits on the first `Ctrl+C` even when browser tabs still hold active SSE
  topic streams open. This local-dev shutdown path now prioritizes returning control to the shell
  over gracefully draining long-lived browser stream connections.
- `/api/stream/topics` and `/api/stream/topics/{topic_id}` now treat transient SQLite
  `DBBusyError` conditions as non-fatal while polling, keeping the SSE stream alive with normal
  heartbeat behavior instead of crashing with traceback-level noise.

### Upgrade

- Packaged installs already include the built frontend bundle. If you run from a source checkout,
  rebuild the frontend before starting the browser UI:

  ```bash
  pnpm --dir frontend build
  uv run agent-bus serve
  ```

- To preview this release explicitly with `uvx`, run:

  ```bash
  uvx --from agent-bus-mcp==0.4.0 agent-bus serve
  ```

## [0.3.1]

### Breaking Changes

- Topic participant names are now reserved for the lifetime of the topic. `topic_join()` rejects duplicate
  `agent_name` values with `AGENT_NAME_IN_USE` instead of allowing multiple clients to share the same name.
- Reconnecting clients are now expected to persist and reuse the returned `reclaim_token` if they want to
  keep the same identity after a restart or reconnect.
- The dialog protocol `spec_version` is now `v6.3`.

### Changed

- `topic_join()` now returns `reclaim_token` in both `structuredContent` and plain-text output so text-only
  clients can persist reconnect state.
- `delete_topic()` now removes durable agent-name reservations together with the rest of the topic-owned
  state.
- `sync.max_items` metadata now exposes the configured cap while preserving the longstanding default of
  `min(20, AGENT_BUS_MAX_SYNC_ITEMS)`.
- `agent-bus --version` and `agent-bus cli --version` now report the installed package version.
- `ping()` now returns `package_version`, and the optional Web UI footer now shows the runtime package
  version instead of a stale hardcoded string.

### Added

- The repo now includes a reusable `agent-bus-workflows` workflow skill asset for Agent Bus collaboration,
  handoffs, and reviewer/implementer loops.

### Upgrade

- If your agents often join topics with generic names like `codex` or `claude`, update them to:
  - choose semantic names up front when possible
  - handle `AGENT_NAME_IN_USE` by selecting a suggested fallback name or by retrying with the original
    `reclaim_token`
- If your client persists Agent Bus state, store the last successful `agent_name` and `reclaim_token`
  together per topic so reconnects can reclaim the same identity cleanly.
- If you use `uvx`, you can preview the new release explicitly with:

  ```bash
  uvx --from agent-bus-mcp==0.3.1 agent-bus --help
  ```

## [0.2.1]

### Fixed

- Fixed the optional Web UI on current FastAPI/Starlette builds. `agent-bus serve` in `0.2.0`
  could start successfully but fail with `500 Internal Server Error` on `GET /` because the web
  routes still used the old `TemplateResponse("template.html", context)` calling convention.
- Added a regression test covering the topic list page and a topic detail page render so the Web
  UI `serve` path stays covered in future releases.

### Upgrade

- If you already installed `0.2.0` and use the Web UI, upgrade to `0.2.1`.
- To force `uvx` to refresh a cached `0.2.0` environment, run:

  ```bash
  uvx --refresh-package agent-bus-mcp --from "agent-bus-mcp[web]==0.2.1" agent-bus serve
  ```

## [0.2.0]

### Breaking Changes

- Semantic indexing now uses `fastembed` in the Rust core instead of the older raw ONNX/tokenizer path.
- Existing semantic embeddings from `0.1.x` are treated as stale and must be rebuilt once after upgrading.
- The old ONNX/tokenizer override environment variables are no longer part of the supported config surface:
  `AGENT_BUS_EMBEDDING_ONNX_FILE`, `AGENT_BUS_EMBEDDING_TOKENIZER_FILE`,
  `AGENT_BUS_EMBEDDING_ONNX_PATH`, and `AGENT_BUS_EMBEDDING_TOKENIZER_PATH`.
- `AGENT_BUS_EMBEDDING_MODEL=intfloat/e5-small-v2` is no longer accepted. Use
  `intfloat/multilingual-e5-small` instead.

### Changed

- Added a SQLite-backed leader lease for the background embedding worker so CLI indexing and background indexing coordinate against the same DB lock.
- Unified CLI and background embedding work behind the shared `index_message_rows()` / `_index_message()` pipeline.
- `agent-bus cli embeddings index` now fails explicitly when indexing fails instead of reporting a successful run with only per-row errors.
- Tool metadata for `topic_create` and `topic_join` is more explicit for agent clients, including exported schema guidance for the create-then-join flow.

### Migration

1. Upgrade the package to `0.2.0`.

   PyPI / uvx:

   ```bash
   uvx --from agent-bus-mcp==0.2.0 agent-bus --help
   ```

   Local checkout:

   ```bash
   uv sync
   uv run maturin develop
   ```

2. If you set embedding-related environment variables, update them before restarting the server:

   - Keep using `AGENT_BUS_EMBEDDING_MODEL` to select the model.
   - Use `AGENT_BUS_EMBEDDING_CACHE_DIR` or `FASTEMBED_CACHE_DIR` to control the local model cache.
   - Stop using the removed ONNX/tokenizer file and path override variables listed above.

3. If you previously used `AGENT_BUS_EMBEDDING_MODEL=intfloat/e5-small-v2`, change it to:

   ```bash
   export AGENT_BUS_EMBEDDING_MODEL=intfloat/multilingual-e5-small
   ```

4. Rebuild semantic embeddings for existing messages once after upgrading:

   ```bash
   uvx --from agent-bus-mcp==0.2.0 agent-bus cli embeddings index
   # or from a local checkout:
   uv run agent-bus cli embeddings index
   ```

5. Expect the first semantic run to download the selected `fastembed` model into the local cache.

6. If hybrid search still reports partial or missing semantic coverage, rerun the indexing command until it completes cleanly.

7. If you are upgrading from a much older database layout and hit a schema mismatch, wipe the DB and recreate it:

   ```bash
   agent-bus cli db wipe --yes
   ```
