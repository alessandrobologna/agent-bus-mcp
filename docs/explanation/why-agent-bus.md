# Why use Agent Bus?

Agent Bus exists for a narrow but common problem: multiple local coding agents often need to
coordinate, and the usual tools for doing that do not hold up well.

Without a shared bus, agents tend to coordinate through:

- pasted summaries in chat threads
- ad hoc files in the repo
- shell logs or scratch notes that are hard to replay
- one-off prompts that disappear when a process restarts

Those approaches work for tiny tasks, but they break down when you want durable handoffs,
reviewer/implementer loops, or a searchable record of what each agent asked and answered.

## What Agent Bus gives you

Agent Bus turns that coordination problem into a local messaging system with a small, explicit
contract.

### Topics instead of implicit threads

Each conversation happens in a named topic. That keeps coordination explicit:

- one topic per task or incident
- one ordered message stream per topic
- one place to replay what happened later

### Stable peer identity

Agents join with a human-friendly `agent_name`, and the name is reserved for the life of the topic.
That matters because coordination is easier to follow when "reviewer", "implementer", or "codex
backend" keeps the same meaning across reconnects.

### Server-side cursors

Each peer has a server-side cursor per topic. Clients do not need to keep re-reading full history or
guess where they left off.

This makes it easier to:

- replay from the beginning when needed
- long-polling for new work
- resuming after a restart without inventing a client-side checkpoint format

### One local dependency surface

Agent Bus stays local:

- transport: stdio
- storage: SQLite in WAL mode
- optional UI: local browser workbench

That keeps setup lightweight and makes it practical to use in personal or small-team workflows
without another network service.

## Why SQLite and a Rust core?

SQLite is a good fit because the bus is local, single-machine, and mostly coordination-oriented.
It gives durable state, transactional updates, FTS, and simple deployment with one file.

The Rust core exists because the critical data-path logic belongs in one place:

- schema management
- reads and writes
- search
- embedding coordination

Keeping that logic in one core avoids re-implementing DB behavior across multiple Python and future
Rust consumers.

## Where Agent Bus fits

Agent Bus is strongest when you want structured local coordination between agent tools. It is not a
full remote collaboration service.

Good fits:

- reviewer / implementer / re-review loops
- multi-agent coding sessions on one workstation
- durable local audit trails for agent collaboration
- searchable topic history with optional semantic indexing

Poor fits:

- multi-machine coordination over a network
- auth- or tenancy-heavy environments
- workflows that need a hosted service and external participants

## Why not just use chat?

Plain chat threads are good at conversation but poor at replayable systems coordination.

Agent Bus adds:

- explicit topic identity
- explicit peer identity
- resumable cursors
- predictable tool semantics
- search and export over the resulting history

That makes it easier to ask an agent to "pick up where you left off", "replay the whole task", or
"show the messages about indexing failures" without relying on one client session's memory.

## See also

- [First topic between two peers](../tutorials/first-topic-between-two-peers.md)
- [Install and configure Agent Bus](../how-to/install-and-configure-agent-bus.md)
- [Runtime reference](../reference/runtime-reference.md)
