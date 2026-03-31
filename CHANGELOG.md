# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - Unreleased

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
