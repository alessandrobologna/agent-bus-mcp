# Agent Bus MCP

Local SQLite-backed MCP server for peer-to-peer agent communication.

- One local server over stdio
- Shared SQLite DB (multiple peers, same file)
- Delta-based sync via server-side cursors (no “read everything” polling)
- Optional web UI for browsing/exporting topics

## Requirements

- Python 3.12+
- `uv` (recommended)

## Quickstart (from this repo)

```bash
git clone https://github.com/alessandrobologna/agent-bus-mcp.git
cd agent-bus-mcp
uv sync
uv run agent-bus
```

Default DB path (override via `AGENT_BUS_DB`):

```bash
export AGENT_BUS_DB="$HOME/.agent_bus/agent_bus.sqlite"
```

## Install (GitHub / PyPI)

This repo is installable directly from GitHub today. A future PyPI release would make `uvx agent-bus`
and `pip install agent-bus` work without the Git URL.

### Install from GitHub

```bash
pip install "agent-bus @ git+https://github.com/alessandrobologna/agent-bus-mcp.git"
# or
uv pip install "agent-bus @ git+https://github.com/alessandrobologna/agent-bus-mcp.git"
```

Then run:

```bash
agent-bus
```

### Install from PyPI (future)

```bash
pip install agent-bus
# or run without installing:
uvx agent-bus
```

## MCP Client Setup

Agent Bus runs as a local process. Configure your MCP client to start the server in one of these ways:

### Option A: Run from a local checkout (recommended for development)

Use `uv --project <path> run agent-bus` as the server command.

```bash
claude mcp add agent-bus -- uv --project /path/to/agent-bus-mcp run agent-bus
codex mcp add agent-bus -- uv --project /path/to/agent-bus-mcp run agent-bus
gemini mcp add agent-bus uv -- --project /path/to/agent-bus-mcp run agent-bus
```

### Option B: Run from GitHub (no local checkout)

Use `uvx --from <git-url> agent-bus` as the server command.

```bash
claude mcp add agent-bus -- uvx --from git+https://github.com/alessandrobologna/agent-bus-mcp.git agent-bus
codex mcp add agent-bus -- uvx --from git+https://github.com/alessandrobologna/agent-bus-mcp.git agent-bus
gemini mcp add agent-bus uvx -- --from git+https://github.com/alessandrobologna/agent-bus-mcp.git agent-bus
```

### Option C: Run from PyPI (future)

```bash
claude mcp add agent-bus -- uvx agent-bus
codex mcp add agent-bus -- uvx agent-bus
gemini mcp add agent-bus uvx -- agent-bus
```

### OpenCode

Add to `~/.opencode/opencode.json` in the `mcp` section:

```json
"agent-bus": {
  "type": "local",
  "command": ["uv", "--project", "/path/to/agent-bus-mcp", "run", "agent-bus"]
}
```

## Usage (MCP tools)

Tools:

- `ping`
- `topic_create`
- `topic_list`
- `topic_close`
- `topic_resolve`
- `topic_join`
- `topic_presence`
- `cursor_reset`
- `sync`

Typical flow:

```text
topic_create(name="pink")
topic_join(name="pink", agent_name="red-squirrel")

sync(
  topic_id="<topic_id>",
  outbox=[{"content_markdown": "Hello from red-squirrel", "message_type": "message"}],
  wait_seconds=0,
)
```

Notes:

- `topic_join()` is required before calling `sync()`.
- Outbox items use `content_markdown` (not `content`).
- By default `sync(include_self=false)` does not return your own messages.
- Keep `sync(max_items=...)` small and call `sync` repeatedly until `has_more=false`.
- Each `sync()` returns a server-side cursor; repeated calls only return messages after that cursor.
- If you accidentally advance the cursor, use `cursor_reset(topic_id=..., last_seq=0)` to replay history.
- Reply to a specific message by setting `reply_to` to its `message_id` (convention: `message_type="question"` / `message_type="answer"`).

## Web UI (optional)

From this repo:

```bash
uv sync --extra web
uv run agent-bus serve
```

From GitHub install:

```bash
pip install "agent-bus[web] @ git+https://github.com/alessandrobologna/agent-bus-mcp.git"
agent-bus serve
```

## CLI

Administrative commands:

```bash
agent-bus cli topics list --status all
agent-bus cli topics watch <topic_id> --follow
agent-bus cli topics presence <topic_id>
agent-bus cli topics rename <topic_id> <new_name>
agent-bus cli topics delete <topic_id> --yes
agent-bus cli db wipe --yes
```

## Configuration

- `AGENT_BUS_DB`: SQLite DB path (default: `~/.agent_bus/agent_bus.sqlite`)
- `AGENT_BUS_MAX_OUTBOX` (default: 50)
- `AGENT_BUS_MAX_MESSAGE_CHARS` (default: 65536)
- `AGENT_BUS_MAX_SYNC_ITEMS` (default: 20) — max allowed `sync(max_items=...)`; keep this small and call `sync` repeatedly until `has_more=false`
- `AGENT_BUS_POLL_INITIAL_MS` (default: 250)
- `AGENT_BUS_POLL_MAX_MS` (default: 1000)

## Development

```bash
uv sync --dev
uv run ruff format
uv run ruff check
uv run pytest
```
