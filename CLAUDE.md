# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent Bus is a local SQLite-backed MCP (Model Context Protocol) server for peer-to-peer agent communication. It enables multiple AI agents to communicate through a shared message bus with delta-based sync via server-side cursors.

## Common Commands

```bash
# Install dependencies
uv sync              # Core dependencies only
uv sync --dev        # With dev dependencies (pytest, ruff)
uv sync --extra web  # With web UI dependencies

# Run the MCP server (stdio transport)
uv run agent-bus

# Run the web UI
uv run agent-bus serve

# Linting and formatting
uv run ruff format
uv run ruff check

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_db.py

# Run a specific test
uv run pytest tests/test_db.py::test_topic_create_reuse_newest_open

# CLI administrative commands
uv run agent-bus cli topics list --status all
uv run agent-bus cli topics watch <topic_id> --follow
uv run agent-bus cli db wipe --yes
```

## Architecture

### Core Components

- **`agent_bus/peer_server.py`**: MCP server implementation using FastMCP. Defines all MCP tools (`ping`, `topic_create`, `topic_join`, `sync`, etc.). The `sync()` tool is the primary read/write interface with delta-based polling and server-side cursors.

- **`agent_bus/db.py`**: SQLite database layer (`AgentBusDB` class). Handles all persistence: topics, messages, cursors, and sequences. Uses WAL mode for concurrent access. Key method is `sync_once()` which handles message sending and receiving in a single transaction.

- **`agent_bus/models.py`**: Frozen dataclasses for `Topic`, `Message`, and `Cursor`.

- **`agent_bus/common.py`**: Shared utilities including `tool_ok()`/`tool_error()` for MCP response formatting, error codes, and environment variable helpers.

- **`agent_bus/cli.py`**: Click-based CLI for administrative operations (list topics, watch messages, export, delete).

- **`agent_bus/entrypoint.py`**: Main entry point. Runs MCP server by default; `serve` subcommand starts web UI; `cli` subcommand runs admin commands.

- **`agent_bus/web/server.py`**: Optional FastAPI web server with HTMX-powered UI for browsing topics and messages.

### Key Patterns

- **Server-side cursors**: Each agent has a cursor per topic tracking `last_seq`. The `sync()` tool advances this cursor, enabling delta-based message retrieval without re-fetching history.

- **Idempotent sends**: Use `client_message_id` to make message sends idempotent for retry safety.

- **Per-session join**: Agents must call `topic_join()` before `sync()`. Join state is in-memory (`_joined_agent_names` dict in peer_server.py).

- **Presence tracking**: Derived from cursor `updated_at` timestamps - calling `sync()` updates presence.

### Database Schema (v6)

Tables: `meta`, `topics`, `topic_seq`, `messages`, `cursors`

- Topics can be "open" or "closed"
- Messages have sequential `seq` numbers per topic
- Cursors track `(topic_id, agent_name, last_seq)`

### Environment Variables

- `AGENT_BUS_DB`: SQLite path (default: `~/.agent_bus/agent_bus.sqlite`)
- `AGENT_BUS_MAX_SYNC_ITEMS`: Max messages per sync (default: 20)
- `AGENT_BUS_MAX_MESSAGE_CHARS`: Max message length (default: 65536)
