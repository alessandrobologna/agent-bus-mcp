# First topic between two peers

This tutorial walks through the smallest useful Agent Bus MCP workflow: one agent hands off work,
another agent replies, and both keep the history in one durable local topic.

Use it when you want to see why Agent Bus MCP matters before reading the lower-level primitives.

## Before you start

- Agent Bus MCP installed in two local MCP clients
- two agent sessions, for example Codex and Claude Code
- a versioned `uvx` command or local checkout already working

If you still need setup help, use [Install and configure Agent Bus MCP](../how-to/install-and-configure-agent-bus.md).

## Step 1: connect Agent Bus MCP in both clients

Make sure both clients can reach the same local Agent Bus MCP runtime.

Example:

```bash
export AGENT_BUS_VERSION="0.4.3"
codex mcp add agent-bus -- uvx --from "agent-bus-mcp==$AGENT_BUS_VERSION" agent-bus
claude mcp add agent-bus -- uvx --from "agent-bus-mcp==$AGENT_BUS_VERSION" agent-bus
```

After this step, both clients should list `agent-bus` as an MCP server. If you set
`AGENT_BUS_DB`, both clients should point at the same database path.

## Step 2: create a topic from the first agent

In the first agent, ask for a reusable topic and an explicit peer name.

Example prompt:

```text
Create or reuse an Agent Bus MCP topic named `tutorial-demo`, join it as `agent-a`, and tell me the topic_id.
```

You should now have a `topic_id`, and `agent-a` should be joined to an open topic.

## Step 3: join from the second agent

In the second agent, join the same topic with a different name.

Example prompt:

```text
Join Agent Bus MCP topic `tutorial-demo` as `agent-b`.
```

At this point, both agents should be in the same topic. If the name is already taken, Agent Bus MCP
should reject it instead of silently renaming the peer.

## Step 4: send one message from the first agent

Now create one small, visible task for the second agent.

Example prompt in the first agent:

```text
Send the message `Please confirm that you can read this topic and reply with one sentence describing what Agent Bus MCP stores.` to the topic.
```

Success looks like one new message in the topic stream from `agent-a`.

## Step 5: read and answer from the second agent

In the second agent, ask it to sync and answer any unread questions.

Example prompt:

```text
Sync topic `tutorial-demo`, read any unread messages, and reply in the same topic with one sentence about what Agent Bus MCP stores.
```

After this step, `agent-b` should receive the message from `agent-a` and post a reply in the same
topic.

## Step 6: replay or long-poll from the first agent

Ask the first agent to either replay from the beginning or long-poll for new messages.

Example prompts:

```text
Replay topic `tutorial-demo` from the beginning.
```

```text
Long-poll topic `tutorial-demo` for new messages and print them as they arrive.
```

Replay should return the ordered conversation. Long-poll should wait for new activity instead of
forcing repeated full reads.

## What you just learned

You used the core Agent Bus MCP workflow:

- topic creation and reuse
- stable peer identity through `topic_join`
- send and receive through `sync()`
- replay and long-poll without ad hoc state management in the client

## See also

- [Install and configure Agent Bus MCP](../how-to/install-and-configure-agent-bus.md)
- [Runtime reference](../reference/runtime-reference.md)
- [Why use Agent Bus MCP?](../explanation/why-agent-bus.md)
