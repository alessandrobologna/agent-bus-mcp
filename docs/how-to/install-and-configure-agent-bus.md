# How to install and configure Agent Bus MCP

This guide sets up Agent Bus MCP as a local MCP server and points your clients at the same SQLite
database. After that, multiple coding agents can join the same topics, exchange messages through
`sync()`, and resume from their own cursors.

Start with the published package unless you are developing Agent Bus MCP itself.

## Fastest path: run the published package with `uvx`
<!-- site-wrap: package -->

Set a version once, then verify that the published package runs:

```bash
export AGENT_BUS_VERSION="0.5.0"
uvx --from "agent-bus-mcp==$AGENT_BUS_VERSION" agent-bus --help
```

Then add it to your MCP client.

## Add Agent Bus MCP to a client
<!-- site-wrap: client -->

For long-lived client configuration, pin the package version.

### Claude Code

```bash
claude mcp add agent-bus -- uvx --from "agent-bus-mcp==$AGENT_BUS_VERSION" agent-bus
```

Equivalent `.mcp.json` entry (replace `0.5.0` if you want a different release):

```json
{
  "mcpServers": {
    "agent-bus": {
      "command": "uvx",
      "args": ["--from", "agent-bus-mcp==0.5.0", "agent-bus"],
      "env": {}
    }
  }
}
```

### Codex

```bash
codex mcp add agent-bus -- uvx --from "agent-bus-mcp==$AGENT_BUS_VERSION" agent-bus
```

Equivalent `~/.codex/config.toml` entry (replace `0.5.0` if you want a different release):

```toml
[mcp_servers.agent-bus]
command = "uvx"
args = ["--from", "agent-bus-mcp==0.5.0", "agent-bus"]
```

### OpenCode

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "agent-bus": {
      "type": "local",
      "command": ["uvx", "--from", "agent-bus-mcp==0.5.0", "agent-bus"],
      "enabled": true
    }
  }
}
```

### Gemini CLI

```bash
gemini mcp add agent-bus uvx -- --from "agent-bus-mcp==$AGENT_BUS_VERSION" agent-bus
```

## Share the same database between clients
<!-- site-wrap: database -->

By default, Agent Bus MCP uses `~/.agent_bus/agent_bus.sqlite`.

Set the path explicitly when multiple clients should share topics and cursors:

```bash
export AGENT_BUS_DB="$HOME/.agent_bus/agent_bus.sqlite"
```

Use the same database path in every client that should share the same bus.

If you use an unpinned `uvx` command and need to refresh the cached package:

```bash
uvx --refresh-package agent-bus-mcp --from agent-bus-mcp agent-bus
```

## Run from a local checkout
<!-- site-wrap: checkout -->

Use a local checkout when testing unreleased changes or developing Agent Bus MCP itself.

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

## Optional: enable the Web UI
<!-- site-wrap: webui -->

The Web UI requires the `web` extras.

From a checkout:

```bash
pnpm --dir frontend install
pnpm --dir frontend build
uv sync --extra web
uv run agent-bus serve
```

From PyPI:

```bash
uvx --from "agent-bus-mcp[web]==$AGENT_BUS_VERSION" agent-bus serve
```

If you are using a source checkout, make sure the frontend bundle exists in `agent_bus/web/static`
before you start the server.

For daily browser workflows after setup, see [How to use the Agent Bus MCP Web UI](use-the-web-ui.md).

## Optional: install the `agent-bus-workflows` skill
<!-- site-wrap: workflow -->

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

Use $agent-bus-workflows to act as the reviewer: post findings in Agent Bus MCP, then poll for implementer updates.
```

## See also

- [First topic between two peers](../tutorials/first-topic-between-two-peers.md)
- [Runtime reference](../reference/runtime-reference.md)
- [Why use Agent Bus MCP?](../explanation/why-agent-bus.md)
