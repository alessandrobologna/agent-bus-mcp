# Configuration reference

Use this reference for environment variables that control storage paths, sync limits, tool text
output, polling, and embeddings.

Use this reference when you need to:

- point multiple clients at the same SQLite database
- tune sync, message, or tool text limits
- adjust poll timings or embedding worker behavior

## Storage and limits

| Variable | Purpose |
| --- | --- |
| `AGENT_BUS_DB` | SQLite DB path (default: `~/.agent_bus/agent_bus.sqlite`) |
| `AGENT_BUS_MAX_OUTBOX` | Max outbound items per sync call (default: `50`) |
| `AGENT_BUS_MAX_MESSAGE_CHARS` | Max message size (default: `65536`) |
| `AGENT_BUS_MAX_SYNC_ITEMS` | Max allowed `sync(max_items=...)` (default: `20`) |

## Tool text output

| Variable | Purpose |
| --- | --- |
| `AGENT_BUS_TOOL_TEXT_INCLUDE_BODIES` | Include full bodies in tool text output (default: `1`) |
| `AGENT_BUS_TOOL_TEXT_MAX_CHARS` | Max chars per message in tool text output (default: `64000`) |

## Polling

| Variable | Purpose |
| --- | --- |
| `AGENT_BUS_POLL_INITIAL_MS` | Initial poll backoff (default: `250`) |
| `AGENT_BUS_POLL_MAX_MS` | Max poll backoff (default: `1000`) |

## Embeddings

| Variable | Purpose |
| --- | --- |
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

## See also

- [Runtime reference](runtime-reference.md)
- [Search and embeddings reference](search-and-embeddings-reference.md)
- [Implementation spec](../../spec.md)
