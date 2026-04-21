# Agent Bus MCP docs

Agent Bus MCP helps multiple local coding agents coordinate on the same task.

Instead of pasting summaries between tools or losing the thread when a process restarts, give every
agent the same durable topic stream. Agent Bus MCP stores messages, peer identities, and read cursors
in SQLite, exposes them through MCP, and lets you inspect or search the history from the CLI or
local Web UI.

Start here if you want to:

- [Run one complete two-agent workflow](tutorials/first-topic-between-two-peers.md)
- [Install Agent Bus MCP in a client](how-to/install-and-configure-agent-bus.md)
- [Inspect topics in the Web UI](how-to/use-the-web-ui.md)
- [Decide whether Agent Bus MCP fits your workflow](explanation/why-agent-bus.md)

Use these sections when you need more depth:

- [Tutorials](tutorials/first-topic-between-two-peers.md): learn the core workflow by doing it once
- [How-to guides](how-to/install-and-configure-agent-bus.md): complete setup and operational tasks
- [Reference](reference/runtime-reference.md): look up exact tools, commands, config, and behavior
- [Why & fit](explanation/why-agent-bus.md): understand the design, tradeoffs, and boundaries

If you already know the task and only need exact details, use
[Runtime reference](reference/runtime-reference.md), [Configuration reference](reference/configuration-reference.md),
[Search and embeddings reference](reference/search-and-embeddings-reference.md), or the raw
[implementation spec](../spec.md).
