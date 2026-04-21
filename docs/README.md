# Agent Bus MCP docs

Agent Bus MCP helps multiple local coding agents coordinate on the same task.

Instead of pasting summaries between tools or losing the thread when a process restarts, give every
agent the same durable topic stream. Agent Bus MCP stores messages, peer identities, and read
cursors in SQLite, exposes them through MCP, and lets you inspect or search the history from the
CLI or local Web UI.

These docs follow the [Diataxis](https://diataxis.fr/) split:

- **Tutorial** walks through one complete handoff before any setup details.
- **How-to guides** cover install, shared databases, and the Web UI.
- **Reference** lists exact tools, commands, configuration, and search behavior.
- **Why & fit** explains the design, when it helps, and when simpler coordination is enough.

If you already know what you need, jump straight to the
[Runtime reference](reference/runtime-reference.md), the
[Configuration reference](reference/configuration-reference.md), or the raw
[implementation spec](../spec.md).
