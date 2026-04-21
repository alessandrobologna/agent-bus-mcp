# Search and embeddings reference

Use this reference for exact search modes, indexing commands, and embedding behavior.

Use this reference when you need to:

- choose between `fts`, `hybrid`, and `semantic`
- backfill embeddings for existing history
- understand which search modes need embeddings and which do not

## Search modes

- `fts`: lexical search through SQLite FTS5
- `hybrid`: the default mode, combining lexical and vector recall
- `semantic`: vector-only matching through local embeddings

```bash
agent-bus cli search "cursor reset"                 # hybrid (default)
agent-bus cli search "sqlite wal" --mode fts        # lexical only
agent-bus cli search "replay history" --mode semantic
agent-bus cli search "poll backoff" --topic-id <topic_id>
```

FTS works without embeddings. Hybrid and semantic search improve after vectors are indexed.

## Backfill embeddings for existing messages

From a published package:

```bash
uvx --from agent-bus-mcp agent-bus cli embeddings index
```

From a local checkout:

```bash
uv sync
uv run agent-bus cli embeddings index
```

Supported embedding model aliases include:

- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-mpnet-base-v2`
- `BAAI/bge-small-en-v1.5`
- `intfloat/multilingual-e5-small`

## See also

- [Configuration reference](configuration-reference.md)
- [Runtime reference](runtime-reference.md)
- [Implementation spec](../../spec.md)
- [Changelog](../../CHANGELOG.md)
