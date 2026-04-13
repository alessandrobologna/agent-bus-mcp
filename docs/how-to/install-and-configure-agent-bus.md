# How to install and configure Agent Bus

This guide gets Agent Bus running in a local MCP client. Start with the published package unless
you are developing Agent Bus itself.

## Fastest path: run the published package with `uvx`

Check that the published package runs:

```bash
uvx --from "agent-bus-mcp==<version>" agent-bus --help
```

Then add it to your MCP client.

## Add Agent Bus to a client

For long-lived client configuration, pin the package version.

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

## Share the same database between clients

By default, Agent Bus uses `~/.agent_bus/agent_bus.sqlite`.

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

Use a local checkout when testing unreleased changes or developing Agent Bus itself.

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
uvx --from "agent-bus-mcp[web]==<version>" agent-bus serve
```

If you are using a source checkout, make sure the frontend bundle exists in `agent_bus/web/static`
before you start the server.

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
