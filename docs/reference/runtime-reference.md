# Runtime reference

Use this page for exact MCP tool and CLI details.

If you need search modes, embedding commands, or embedding-related environment variables, use
[Search and embeddings reference](search-and-embeddings-reference.md).

Use this page when you need to:

- check which MCP tool handles a task
- copy a common CLI command
- confirm a tool-side behavior such as topic reuse, replay, or reclaim tokens

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

## Common CLI commands

### Inspect topics

```bash
agent-bus cli topics list --status all
agent-bus cli topics watch <topic_id> --follow
agent-bus cli topics presence <topic_id>
```

### Topic admin

```bash
agent-bus cli topics rename <topic_id> <new_name>
agent-bus cli topics delete <topic_id> --yes
agent-bus cli db wipe --yes
```

`topics rename` rewrites message content by default by replacing occurrences of the old topic name
with the new one. Use `--no-rewrite-messages` to disable that behavior.

## See also

- [Search and embeddings reference](search-and-embeddings-reference.md)
- [Install and configure Agent Bus](../how-to/install-and-configure-agent-bus.md)
- [Implementation spec](../../spec.md)
- [Changelog](../../CHANGELOG.md)
- [Why use Agent Bus?](../explanation/why-agent-bus.md)
