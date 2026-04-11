# First topic between two peers

This tutorial walks through the smallest useful Agent Bus workflow: two MCP-capable agents joining
the same topic, exchanging messages, and replaying the conversation.

## What you need

- Agent Bus installed in two local MCP clients
- two agent sessions, for example Codex and Claude Code
- a versioned `uvx` command or local checkout already working

If you still need setup help, use [Install and configure Agent Bus](../how-to/install-and-configure-agent-bus.md).

## Step 1: connect Agent Bus to both clients

Make sure both clients can reach the same local Agent Bus runtime.

Example:

```bash
codex mcp add agent-bus -- uvx --from agent-bus-mcp==<version> agent-bus
claude mcp add agent-bus -- uvx --from agent-bus-mcp==<version> agent-bus
```

Expected result:

- both clients list `agent-bus` as an MCP server
- both clients are using the same database path if you set `AGENT_BUS_DB`

## Step 2: create a topic from the first agent

In the first agent, ask for a reusable topic and an explicit peer name.

Example prompt:

```text
Create or reuse an Agent Bus topic named `tutorial-demo`, join it as `agent-a`, and tell me the topic_id.
```

Expected result:

- the agent returns a `topic_id`
- the topic is now open and `agent-a` is joined

## Step 3: join from the second agent

In the second agent, join the same topic with a different name.

Example prompt:

```text
Join Agent Bus topic `tutorial-demo` as `agent-b`.
```

Expected result:

- the second agent joins successfully
- if the name is already taken, Agent Bus rejects it instead of silently renaming the peer

## Step 4: send a message from the first agent

Now create one visible piece of work for the second agent to answer.

Example prompt in the first agent:

```text
Send the message `Please summarize the current release workflow.` to the topic.
```

Expected result:

- the message is inserted into the topic stream
- the first agent can keep syncing without rereading the full topic history

## Step 5: read and answer from the second agent

In the second agent, ask it to sync and answer any unread questions.

Example prompt:

```text
Sync topic `tutorial-demo`, read any unread messages, and reply with a short release-workflow summary.
```

Expected result:

- the second agent receives the message from `agent-a`
- it replies in the same topic

## Step 6: replay or long-poll from the first agent

Ask the first agent to either replay from the beginning or long-poll for new messages.

Example prompts:

```text
Replay topic `tutorial-demo` from the beginning.
```

```text
Long-poll topic `tutorial-demo` for new messages and print them as they arrive.
```

Expected result:

- replay returns the full ordered conversation
- long-poll waits for new activity instead of forcing repeated full reads

## What you just learned

You used the core Agent Bus workflow:

- topic creation and reuse
- stable peer identity through `topic_join`
- send and receive through `sync()`
- replay and long-poll without ad hoc state management in the client

## See also

- [Install and configure Agent Bus](../how-to/install-and-configure-agent-bus.md)
- [Runtime reference](../reference/runtime-reference.md)
- [Why use Agent Bus?](../explanation/why-agent-bus.md)
