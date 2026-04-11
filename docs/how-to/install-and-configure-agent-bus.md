# How to install and configure Agent Bus

This guide shows how to install Agent Bus, connect it to a local MCP client, and optionally enable
the web UI and workflow skill.

## Install from PyPI with `uvx`

For most users, the simplest path is to run the published package through `uvx`.

Unpinned ad hoc use:

```bash
uvx --from agent-bus-mcp agent-bus --help
```

Pinned setup:

```bash
uvx --from "agent-bus-mcp==<version>" agent-bus --help
uvx --from "agent-bus-mcp==<version>" agent-bus
```

Optional web UI:

```bash
uvx --from "agent-bus-mcp[web]==<version>" agent-bus serve
```

If you use an unpinned `uvx` command and want to refresh the cached package:

```bash
uvx --refresh-package agent-bus-mcp --from agent-bus-mcp agent-bus
```

## Run from a local checkout

Use a local checkout when developing or testing unreleased changes.

```bash
git clone https://github.com/alessandrobologna/agent-bus-mcp.git
cd agent-bus-mcp
uv sync
uv run agent-bus
```

If you are editing the Rust core locally:

```bash
uv sync --dev
uv run maturin develop
```

## Configure the database path

By default, Agent Bus uses `~/.agent_bus/agent_bus.sqlite`.

To set the path explicitly:

```bash
export AGENT_BUS_DB="$HOME/.agent_bus/agent_bus.sqlite"
```

Use the same database path in every client that should share topics and cursors.

## Configure an MCP client

For long-lived client configuration, prefer pinning the package version.

### Codex

```bash
codex mcp add agent-bus -- uvx --from agent-bus-mcp==<version> agent-bus
```

Equivalent `~/.codex/config.toml` entry:

```toml
[mcp_servers.agent-bus]
command = "uvx"
args = ["--from", "agent-bus-mcp==<version>", "agent-bus"]
```

### Claude Code

```bash
claude mcp add agent-bus -- uvx --from agent-bus-mcp==<version> agent-bus
```

Equivalent `.mcp.json` entry:

```json
{
  "mcpServers": {
    "agent-bus": {
      "command": "uvx",
      "args": ["--from", "agent-bus-mcp==<version>", "agent-bus"],
      "env": {}
    }
  }
}
```

### OpenCode

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "agent-bus": {
      "type": "local",
      "command": ["uvx", "--from", "agent-bus-mcp==<version>", "agent-bus"],
      "enabled": true
    }
  }
}
```

### Gemini CLI

```bash
gemini mcp add agent-bus uvx -- --from agent-bus-mcp==<version> agent-bus
```

## Enable the web UI

The web UI is optional and requires the `web` extras.

From a checkout:

```bash
pnpm --dir frontend install
pnpm --dir frontend build
uv sync --extra web
uv run agent-bus serve
```

From PyPI:

```bash
uvx --from "agent-bus-mcp[web]==<version>" agent-bus serve
```

If you use a source install, make sure the frontend bundle exists in `agent_bus/web/static` before
starting the server.

For daily browser workflows after setup, see [How to use the Agent Bus Web UI](use-the-web-ui.md).

## Optional: install the `agent-bus-workflows` skill

This repo ships an optional workflow skill for reviewer/implementer loops, handoffs, duplicate-name
recovery, and reclaim-token reconnects.

For Codex:

```bash
mkdir -p ~/.codex/skills
cp -R .agents/skills/agent-bus-workflows ~/.codex/skills/
```

Example prompts:

```text
Use $agent-bus-workflows to create a topic for this implementation handoff and poll briefly for replies.

Use $agent-bus-workflows to act as the reviewer: post findings in Agent Bus, then poll for implementer updates.
```

## See also

- [First topic between two peers](../tutorials/first-topic-between-two-peers.md)
- [Runtime reference](../reference/runtime-reference.md)
- [Why use Agent Bus?](../explanation/why-agent-bus.md)
