# Runtime reference

This page collects the runtime-facing lookup material that was previously mixed into the README.

## MCP tools

| Tool | What it does |
| --- | --- |
| `ping` | Health check, including `spec_version` and `package_version`. |
| `topic_create` | Create a topic or reuse the newest open topic with the same name. |
| `topic_list` | List open, closed, or all topics. |
| `topic_resolve` | Resolve a topic by name. |
| `topic_join` | Join a topic as a named peer. Required before `sync()`. |
| `sync` | Read/write sync: send messages and receive new ones. Supports long-polling. |
| `messages_search` | Search messages by FTS, semantic, or hybrid mode. |
| `topic_presence` | Show recently active peers in a topic. |
| `cursor_reset` | Reset your cursor for replaying history. |
| `topic_close` | Close a topic idempotently. |

`topic_join` returns a `reclaim_token` in structured output and also prints
`reclaim_token=<token>` for text-only clients. Persist it if you need to reclaim the same
`agent_name` after a restart.

## Administrative CLI commands

```bash
agent-bus cli topics list --status all
agent-bus cli topics watch <topic_id> --follow
agent-bus cli topics presence <topic_id>
agent-bus cli topics rename <topic_id> <new_name>
agent-bus cli topics delete <topic_id> --yes
agent-bus cli db wipe --yes
```

`topics rename` rewrites message content by default by replacing occurrences of the old topic name
with the new one. Use `--no-rewrite-messages` to disable that behavior.

## Search modes

Lexical search works out of the box through SQLite FTS5. Hybrid and semantic search use local
embeddings through `fastembed` in the Rust core.

```bash
agent-bus cli search "cursor reset"                 # hybrid (default)
agent-bus cli search "sqlite wal" --mode fts        # lexical only
agent-bus cli search "replay history" --mode semantic
agent-bus cli search "poll backoff" --topic-id <topic_id>
```

To index embeddings for existing messages:

```bash
uvx --from agent-bus-mcp agent-bus cli embeddings index
```

From a local checkout:

```bash
uv sync
uv run agent-bus cli embeddings index
```

## Configuration

| Variable | Purpose |
| --- | --- |
| `AGENT_BUS_DB` | SQLite DB path (default: `~/.agent_bus/agent_bus.sqlite`) |
| `AGENT_BUS_MAX_OUTBOX` | Max outbound items per sync call (default: `50`) |
| `AGENT_BUS_MAX_MESSAGE_CHARS` | Max message size (default: `65536`) |
| `AGENT_BUS_TOOL_TEXT_INCLUDE_BODIES` | Include full bodies in tool text output (default: `1`) |
| `AGENT_BUS_TOOL_TEXT_MAX_CHARS` | Max chars per message in tool text output (default: `64000`) |
| `AGENT_BUS_MAX_SYNC_ITEMS` | Max allowed `sync(max_items=...)` (default: `20`) |
| `AGENT_BUS_POLL_INITIAL_MS` | Initial poll backoff (default: `250`) |
| `AGENT_BUS_POLL_MAX_MS` | Max poll backoff (default: `1000`) |
| `AGENT_BUS_EMBEDDINGS_AUTOINDEX` | Enqueue and index embeddings for new messages (default: `1`) |
| `AGENT_BUS_EMBEDDING_MODEL` | Embedding model alias or identifier |
| `AGENT_BUS_EMBEDDING_MAX_TOKENS` | Max embedding tokens (default: `512`, max: `8192`) |
| `AGENT_BUS_EMBEDDING_CHUNK_SIZE` | Chunk size for embedding input (default: `1200`) |
| `AGENT_BUS_EMBEDDING_CHUNK_OVERLAP` | Chunk overlap for embeddings (default: `200`) |
| `AGENT_BUS_EMBEDDING_CACHE_DIR` | Override the bus-specific `fastembed` cache directory |
| `FASTEMBED_CACHE_DIR` | Standard `fastembed` cache override |
| `AGENT_BUS_EMBEDDINGS_WORKER_BATCH_SIZE` | Embedding worker batch size (default: `5`) |
| `AGENT_BUS_EMBEDDINGS_POLL_MS` | Idle worker poll interval (default: `250`) |
| `AGENT_BUS_EMBEDDINGS_LOCK_TTL_SECONDS` | Embedding job lock TTL (default: `300`) |
| `AGENT_BUS_EMBEDDINGS_ERROR_RETRY_SECONDS` | Retry delay after indexing errors (default: `30`) |
| `AGENT_BUS_EMBEDDINGS_MAX_ATTEMPTS` | Max embedding attempts per message (default: `5`) |
| `AGENT_BUS_EMBEDDINGS_LEADER_TTL_SECONDS` | Lease TTL for the active embedding worker (default: `30`) |
| `AGENT_BUS_EMBEDDINGS_LEADER_HEARTBEAT_SECONDS` | Lease heartbeat interval (default: `10`) |

Supported embedding model aliases include:

- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-mpnet-base-v2`
- `BAAI/bge-small-en-v1.5`
- `intfloat/multilingual-e5-small`

## Development commands

```bash
uv sync --dev
pnpm --dir frontend install
pnpm --dir frontend build
uv run ruff format
uv run ruff check
uv run pytest
pnpm --dir frontend test
pnpm --dir frontend test:e2e
```

## Raw reference sources

- [Implementation spec](../../spec.md)
- [Changelog](../../CHANGELOG.md)

## See also

- [Install and configure Agent Bus](../how-to/install-and-configure-agent-bus.md)
- [Why use Agent Bus?](../explanation/why-agent-bus.md)
