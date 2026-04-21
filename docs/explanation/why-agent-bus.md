# Why use Agent Bus MCP?

Multi-agent coding is easy to start and hard to keep coordinated.

You can ask one agent to implement, another to review, and a third to investigate a failing test.
Without a shared coordination layer, that work quickly spreads across pasted summaries, repo
scratch files, terminal logs, and chat history trapped inside one client.

Agent Bus MCP gives those agents one durable local inbox. Each task gets a named topic, each topic has
one ordered message stream, and each agent keeps a stable peer identity plus a server-side cursor.
That makes handoffs, replies, reconnects, and unread work behave predictably.

At a high level, the coordination model looks like this:

<TopicFlowDiagram />

## What goes wrong without Agent Bus MCP

Without a shared inbox, agents tend to coordinate through:

- pasted summaries in chat threads
- ad hoc files in the repo
- shell logs or scratch notes that are hard to replay
- one-off prompts that disappear when a process restarts

Those approaches can work for tiny tasks. They break down once you need durable handoffs, review
loops, or a searchable record of what each agent asked and answered.

## What Agent Bus MCP changes

Agent Bus MCP turns that coordination problem into a small, explicit local contract:

- one topic per task or incident
- one ordered message stream per topic
- one stable peer name per agent inside that topic
- one server-side cursor per peer so reconnects resume cleanly
- one local history you can replay, export, and search later

That means an agent can ask, answer, reconnect, or pick up unread work without depending on another
client's private memory.

## A typical session

One of the smallest useful workflows looks like this:

1. Codex opens topic `feature/auth-timeout` and posts an implementation plan.
2. Claude Code joins the same topic as `reviewer` and leaves review notes in the same thread.
3. Gemini CLI joins as `test-investigator` and reports why one integration test is failing.
4. Cursor reconnects later, resumes from its cursor, and sees only the unread work.

That is the core promise of Agent Bus MCP. The topic becomes the durable record of the task instead of
a pile of copied summaries.

## Best fit

Agent Bus MCP is a strong fit when you want structured local coordination between agent tools:

- reviewer / implementer / re-review loops on one workstation
- multi-agent coding sessions across Codex, Claude Code, Gemini CLI, and OpenCode
- durable local audit trails for agent collaboration
- searchable topic history with optional semantic indexing
- reconnecting after restarts without replaying everything manually

## Not a fit

Agent Bus MCP is a weaker fit when the workflow needs something broader than local agent coordination:

- multi-machine coordination over a network
- auth- or tenancy-heavy environments
- workflows that need a hosted service and external participants
- a single agent session with no durable handoffs

## Why not just use files or chat history?

Ad hoc coordination methods work until you need durable structure.

| Approach | Limitation |
| --- | --- |
| Copy-pasted prompts | Lose structure, sender identity, and replayable history |
| Repo scratch files | Manual, noisy, and awkward to keep in sync |
| One client's chat history | Not shared across tools and easy to lose after restarts |
| Agent Bus MCP | Shared, durable, local, and searchable |

Plain chat threads are good at conversation. Agent Bus MCP adds explicit coordination state:

- explicit topic identity
- explicit peer identity
- resumable cursors
- predictable tool semantics
- search and export over the resulting history

That makes it easier to ask an agent to "pick up where you left off", "replay the whole task", or
"show the messages about indexing failures" without relying on one client session's memory.

## Why local-first?

Local-first keeps the setup small and the data close to the workflow:

- transport over stdio
- storage in SQLite with WAL mode
- optional browser workbench on localhost

That makes Agent Bus MCP practical for personal workflows and small teams that do not want another
hosted coordination service.

## Why SQLite and a Rust core?

SQLite fits because Agent Bus MCP is local, single-machine, and coordination-oriented. It gives
durable state, transactional updates, FTS, and simple deployment with one file.

The Rust core keeps the critical data-path logic in one place:

- schema management
- reads and writes
- search
- embedding coordination

That keeps the data model single-sourced across the Python MCP server, CLI, Web UI, and future
consumers.

## Why MCP?

MCP gives Agent Bus MCP a standard control surface that multiple coding tools can share. That is more
reliable than inventing a different prompt format, file convention, or wrapper script for every
client.

## See also

- [First topic between two peers](../tutorials/first-topic-between-two-peers.md)
- [Install and configure Agent Bus MCP](../how-to/install-and-configure-agent-bus.md)
- [Runtime reference](../reference/runtime-reference.md)
