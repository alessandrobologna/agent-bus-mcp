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

The server uses this DB path:

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
- `ask`
- `ask_poll`
- `ask_cancel`
- `question_mark_answered`
- `pending_list`
- `answer`

## Example flow

Create a topic:

```text
topic_create(name="pink")
```

Join the topic with an agent name (per MCP session):

```text
topic_join(name="pink", agent_name="red-squirrel")
```

Ask a question:

```text
ask(topic_id="<topic_id>", question="What is the purpose of this repo?")
```

To enqueue without waiting, set `wait_seconds=0`.

Another agent joins and answers:

```text
topic_join(name="pink", agent_name="crimson-cat")
pending_list(topic_id="<topic_id>")
answer(topic_id="<topic_id>", responses=[...])
```

To list pending questions without waiting, set `wait_seconds=0`.

Poll for answers:

```text
ask_poll(topic_id="<topic_id>", question_id="<question_id>")
```

Close the question when you're done (so it won't be offered to answerers anymore):

```text
question_mark_answered(topic_id="<topic_id>", question_id="<question_id>")
```

## Optional system instruction (for follow-ups)

If a tool result contains `FOLLOW_UP_REQUIRED`, you can configure your agent with a rule like:

```text
If a tool result contains FOLLOW_UP_REQUIRED, you must:
- call ask() with exactly ONE suggested follow-up question, unless
- onboarding is complete; in that case output NO_FOLLOWUP_NEEDED plus a 3-5 bullet summary.
```

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
- `AGENT_BUS_MAX_QUESTION_CHARS` (default: 8000)
- `AGENT_BUS_MAX_ANSWER_CHARS` (default: 65536)
- `AGENT_BUS_MAX_PUBLISH_BATCH` (default: 50)
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
