# Agent Bus docs

Agent Bus gives MCP-capable agents and tools a shared local thread. It stores topics, messages, and
cursors in SQLite so work can continue across restarts without a hosted service.

Start here if you want to:

- [Install Agent Bus in a client](how-to/install-and-configure-agent-bus.md)
- [Run one complete two-agent workflow](tutorials/first-topic-between-two-peers.md)
- [Inspect topics in the Web UI](how-to/use-the-web-ui.md)
- [Decide whether Agent Bus fits your workflow](explanation/why-agent-bus.md)

Use these sections when you need more depth:

- [Tutorials](tutorials/first-topic-between-two-peers.md): work through a full handoff from start to finish
- [How-to guides](how-to/install-and-configure-agent-bus.md): complete setup and operational tasks
- [Reference](reference/runtime-reference.md): check tool names, commands, environment variables, and exact behavior
- [Explanation](explanation/why-agent-bus.md): understand the design and where it fits best

If you already know the task and only need exact details, use
[Runtime reference](reference/runtime-reference.md) or the raw [implementation spec](../spec.md).
