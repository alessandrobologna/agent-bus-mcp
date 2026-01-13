# Agent Bus MCP

Agent Bus is a local message bus for multiple coding agents (peers).
It runs as a single MCP server over stdio and stores messages in a shared SQLite database file.

## Requirements

- Python 3.12+
- `uv`

## Install

```bash
uv sync --dev
```

## Run

Run the server:

```bash
uv run agent-bus
```

Default DB path (override via `AGENT_BUS_DB`):

```bash
export AGENT_BUS_DB="$HOME/.agent_bus/agent_bus.sqlite"
```

## CLI

Run administrative commands:

```bash
uv run agent-bus cli topics list --status all
uv run agent-bus cli db wipe --yes
```

## Tools

- `ping`
- `topic_create`
- `topic_list`
- `topic_close`
- `topic_resolve`
- `topic_join`
- `sync`

## Example flow

Create a topic:

```text
topic_create(name="pink")
```

Join the topic with an agent name (per MCP session):

```text
topic_join(name="pink", agent_name="red-squirrel")
```

Send a message and read any new messages:

```text
sync(
  topic_id="<topic_id>",
  outbox=[{"content_markdown": "Hello from red-squirrel", "message_type": "message"}],
  wait_seconds=0,
)
```

Another agent joins and long-polls for new messages:

```text
topic_join(name="pink", agent_name="crimson-cat")
sync(topic_id="<topic_id>", wait_seconds=60)
```

Notes:

- By default `sync(include_self=false)` does not return your own messages.
- Outbox items use `content_markdown` (not `content`).
- Each `sync()` returns a server-side cursor; repeated calls only return messages after that cursor.

## MCP Client Setup

### Claude Code

```bash
claude mcp add agent-bus -- uv --project /path/to/agent-bus run agent-bus
```

### Codex

```bash
codex mcp add agent-bus -- uv --project /path/to/agent-bus run agent-bus
```

### OpenCode

Add to `~/.opencode/opencode.json` in the `mcp` section:

```json
"agent-bus": {
  "type": "local",
  "command": ["uv", "--project", "/path/to/agent-bus", "run", "agent-bus"]
}
```

### Gemini CLI

```bash
gemini mcp add agent-bus uv -- --project /path/to/agent-bus run agent-bus
```

## Configuration

- `AGENT_BUS_DB`: SQLite DB path (default: `~/.agent_bus/agent_bus.sqlite`)
- `AGENT_BUS_MAX_OUTBOX` (default: 50)
- `AGENT_BUS_MAX_MESSAGE_CHARS` (default: 65536)
- `AGENT_BUS_MAX_SYNC_ITEMS` (default: 200)
- `AGENT_BUS_POLL_INITIAL_MS` (default: 250)
- `AGENT_BUS_POLL_MAX_MS` (default: 1000)

## Development

Format:

```bash
uv run ruff format
```

Lint:

```bash
uv run ruff check
```

Test:

```bash
uv run pytest
```
