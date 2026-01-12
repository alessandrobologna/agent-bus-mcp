# Agent Bus MCP

Agent Bus is a local message bus for two coding agents.
It runs as two separate MCP servers (teacher and student) over stdio.
Both servers read and write a shared SQLite database file.

## Requirements

- Python 3.12+
- `uv`

## Install

```bash
uv sync --dev
```

## Run

Run the teacher server:

```bash
uv run agent-bus-teacher
```

Run the student server:

```bash
uv run agent-bus-student
```

Both servers use the same DB path:

```bash
export AGENT_BUS_DB="$HOME/.agent_bus/agent_bus.sqlite"
```

## Tools

Teacher server:

- `ping`
- `topic_create`
- `topic_list`
- `topic_close`
- `teacher_drain`
- `teacher_publish`

Student server:

- `ping`
- `topic_list`
- `topic_resolve`
- `ask`
- `ask_poll`
- `ask_cancel`

## Example flow

Create a topic on the teacher server:

```text
topic_create(name="pink")
```

Ask a question on the student server:

```text
topic_resolve(name="pink")
ask(topic_id="<topic_id>", question="What is the purpose of this repo?", wait_seconds=0)
```

Answer on the teacher server:

```text
teacher_drain(topic_id="<topic_id>")
teacher_publish(topic_id="<topic_id>", responses=[...])
```

Poll on the student server:

```text
ask_poll(topic_id="<topic_id>", question_id="<question_id>")
```

## Student system instruction

Configure the student agent with a rule like:

```text
If a tool result contains FOLLOW_UP_REQUIRED, you must:
- call ask() with exactly ONE suggested follow-up question, unless
- onboarding is complete; in that case output NO_FOLLOWUP_NEEDED plus a 3-5 bullet summary.
```

## Configuration

- `AGENT_BUS_DB`: SQLite DB path (default: `~/.agent_bus/agent_bus.sqlite`)
- `AGENT_BUS_MAX_QUESTION_CHARS` (default: 8000)
- `AGENT_BUS_MAX_ANSWER_CHARS` (default: 65536)
- `AGENT_BUS_MAX_TEACHER_NOTES_CHARS` (default: 16384)
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
