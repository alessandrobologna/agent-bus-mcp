# Agent Bus MCP - Implementation Spec (Peer Dialog Mode, stdio, single machine)

> Version: v6.0 (dialog mode)  
> Transport: stdio (local)  
> Storage: shared SQLite (WAL)

## 0) Goal

Enable multiple coding agents on the same machine to collaborate **as peers** by exchanging messages through a shared **agent bus**.

Key properties:

- Peers identify themselves with a human-friendly `agent_name` (e.g., `red-squirrel`) via `topic_join(...)`.
- All peer communication happens via a single **delta-based** `sync()` tool.
- `sync()` is **read/write**: a peer can send outgoing messages and receive new incoming messages in the same call.
- Per-peer cursors are stored **server-side** (in SQLite) keyed by `(topic_id, agent_name)`.

## 1) Non-goals

- No network transport, auth, or security hardening (local-only).
- No server-side LLM calls (messages are authored by peers).
- No streaming/push required (long-polling is sufficient).

## 2) High-level architecture

### 2.1 One MCP server

Implement a single MCP server exposing peer tools over stdio.

Multiple client processes may spawn multiple local server processes; all coordination happens via the shared SQLite DB.

### 2.2 Bus concepts

- Topic: a named communication lane (e.g., `"agent-bus"`).
- Message: an entry in a topic’s ordered message stream.
- Cursor: per peer `(topic_id, agent_name)` state indicating the last seen message sequence.

### 2.3 Peer identity (`agent_name`)

Each agent session joins a topic with `topic_join(agent_name=..., ...)`.

Rules:

- `agent_name` is stored in-memory per MCP server process for the current session.
- Cursor state is stored durably in SQLite and is keyed by `(topic_id, agent_name)`.
- Clients must call `topic_join(...)` again after a server restart.

## 3) Data model (SQLite)

### 3.0 Schema versioning

The DB must include a `meta` table with:

- `schema_version`: `"6"`

If the DB has a missing/mismatched schema version, tools must fail with `DB_SCHEMA_MISMATCH` and instruct the user to wipe the DB.

### 3.1 Tables

#### `meta`

| column | type | notes |
|---|---|---|
| key | TEXT PK | |
| value | TEXT | |

#### `topics`

| column | type | notes |
|---|---|---|
| topic_id | TEXT PK | short UUID |
| name | TEXT | human-friendly |
| created_at | REAL | unix seconds |
| status | TEXT | `open` / `closed` |
| closed_at | REAL NULL | unix seconds |
| close_reason | TEXT NULL | optional |
| metadata_json | TEXT NULL | optional |

Indexes:

- `idx_topics_name_status_created_at` on `(name, status, created_at)`

#### `topic_seq`

Tracks the next per-topic sequence value.

| column | type | notes |
|---|---|---|
| topic_id | TEXT PK | |
| next_seq | INTEGER | next sequence to allocate (starts at 1) |
| updated_at | REAL | unix seconds |

#### `messages`

| column | type | notes |
|---|---|---|
| message_id | TEXT PK | short UUID |
| topic_id | TEXT | |
| seq | INTEGER | per-topic monotonic sequence |
| sender | TEXT | `agent_name` |
| message_type | TEXT | e.g. `message`, `question`, `event`, `resolved` |
| reply_to | TEXT NULL | message_id (same topic) |
| content_markdown | TEXT | message body |
| metadata_json | TEXT NULL | optional JSON |
| client_message_id | TEXT NULL | optional idempotency key |
| created_at | REAL | unix seconds |

Indexes / constraints:

- Unique: `(topic_id, seq)`
- Unique: `(topic_id, sender, client_message_id)` when `client_message_id` is not null (idempotent sends)
- `idx_messages_topic_seq` on `(topic_id, seq)`
- `idx_messages_topic_reply_to` on `(topic_id, reply_to)`

#### `cursors`

Per-peer server-side cursor state.

| column | type | notes |
|---|---|---|
| topic_id | TEXT | |
| agent_name | TEXT | |
| last_seq | INTEGER | last acknowledged seq (starts at 0) |
| updated_at | REAL | unix seconds |

Constraints:

- Primary key: `(topic_id, agent_name)`

## 4) MCP server behavior

### 4.1 Error codes

- topic not found -> `TOPIC_NOT_FOUND`
- topic closed (when writing) -> `TOPIC_CLOSED`
- invalid args -> `INVALID_ARGUMENT`
- db busy/locked -> `DB_BUSY`
- db schema mismatch -> `DB_SCHEMA_MISMATCH`
- not joined to topic -> `AGENT_NOT_JOINED`

### 4.2 Tool list

- `ping`
- `topic_create`
- `topic_list`
- `topic_resolve`
- `topic_close`
- `topic_join`
- `topic_presence`
- `sync`

### 4.3 `sync()` semantics

`sync()` is the single read/write tool for peer communication.

Inputs:

- `topic_id: string` (required)
- `outbox?: OutgoingMessage[]` (optional)
- `max_items?: int = 50`
- `include_self?: bool = false`
- `wait_seconds?: int = 60` (`0` = return immediately)
- `auto_advance?: bool = true`
- `ack_through?: int` (optional; only meaningful when `auto_advance=false`)

OutgoingMessage object (each `outbox` item):

- `content_markdown: string` (required; note: this field name is `content_markdown`, not `content`)
- `message_type?: string` (optional; default: `"message"`)
- `reply_to?: string | null` (optional; `message_id` in the same topic)
- `metadata?: object | null` (optional)
- `client_message_id?: string | null` (optional; idempotency key per `(topic_id, sender, client_message_id)`)

Cursor semantics:

- The server maintains `cursors.last_seq` for `(topic_id, agent_name)`.
- If `auto_advance=true`, the server advances `last_seq` to the highest `seq` among messages returned in `received`.
- If `auto_advance=false`, the server advances `last_seq` only when `ack_through` is provided.
- `ack_through` must be `>= 0` and must not exceed the current highest `seq` in the topic.

Long-polling:

- If there are no new messages to return and `wait_seconds > 0`, the server should long-poll until at least one message
  is available or the timeout expires.

Output:

- `received`: messages ordered by `seq` (oldest first)
- `sent`: records for successfully inserted outgoing messages (with assigned `seq`)
- `cursor`: the updated server-side cursor value
- `status`: `ready` / `timeout` / `empty`

Recommended conventions:

- Use `message_type="question"` for messages that should be answered.
- Reply with `message_type="answer"` and set `reply_to` to the question’s `message_id`.
- When asked to "check for messages", clients should `sync()` and then reply to any new questions they can answer.

### 4.4 `topic_presence()` semantics

List peers that have been active recently on a topic.

Inputs:

- `topic_id: string` (required)
- `window_seconds?: int = 300` (required > 0)
- `limit?: int = 200` (required > 0)

Presence source:

- Presence is derived from `cursors.updated_at`, and `sync_once()` always touches `updated_at` on every `sync()`.

Output:

- `peers`: list of `{agent_name,last_seq,updated_at,age_seconds}`
