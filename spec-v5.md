# Agent Bus MCP - Implementation Spec (Peer Mode, stdio, single machine)

> Version: v5.0 (peer mode)  
> Transport: stdio (local)  
> Storage: shared SQLite (WAL)

## 0) Goal

Enable multiple coding agents on the same machine to collaborate **as peers** by exchanging structured Q/A through a shared **agent bus** backed by SQLite.

Key properties:

- Any peer can create topics, ask questions, and answer questions.
- Peers identify themselves with a human-friendly `agent_name` (e.g., `red-squirrel`).
- The server must **forbid answering your own questions** (to prevent accidental self-dialog loops).
- A question can have **multiple answers** (from different peers).
- The asking peer can **close** a question when done so it stops being offered to answerers.

## 1) Non-goals

- No network transport, auth, or security hardening (local-only).
- No server-side LLM calls (answers are authored by peers).
- No streaming/push required (polling is fine).
- No durable identity across restarts (identity is set by tools, not CLI flags).

## 2) High-level architecture

### 2.1 One MCP server implementation

Implement one MCP server exposing peer tools over stdio.

In practice, multiple client processes may spawn separate local server processes; all coordination happens via the shared SQLite DB.

### 2.2 Bus concepts

- Topic: a named communication lane on the bus (e.g., `"pink"`).
- `topic_id`: unique identifier for a topic.
- Question: a peer message queued on a topic.
- Answer: a peer response linked to a question.

### 2.3 Peer identity (`agent_name`)

Each agent session **joins** a topic with an `agent_name` via `topic_join(...)`.

Rules:

- `agent_name` is stored in-memory (per MCP server process) for the current session.
- `asked_by` / `answered_by` fields in the DB are set to this name.
- If the server process restarts, the agent must call `topic_join(...)` again.

## 3) Repository layout / packaging

### 3.1 Entry point (console script)

- `agent-bus` -> runs the peer MCP server over stdio  
  - `agent-bus cli ...` -> admin CLI (DB wipe, topic listing)

### 3.2 Dependencies

- Python 3.12+
- `mcp` Python SDK (FastMCP)
- `click` (CLI)
- Standard library: `sqlite3`, `uuid`, `time`, `json`

## 4) Data model (SQLite)

### 4.0 Schema versioning

The DB must include a `meta` table with a required key:

- `schema_version`: `"5"`

If the DB is missing a schema version or has a different schema version, tools must fail with `DB_SCHEMA_MISMATCH` and instruct the user to wipe the DB.

### 4.1 Tables

#### `meta`

| column | type | notes |
|---|---|---|
| key | TEXT PK | |
| value | TEXT | |

#### `topics`

| column | type | notes |
|---|---|---|
| topic_id | TEXT PK | short UUID (10-16 chars ok) |
| name | TEXT | human-friendly (e.g., `"pink"`) |
| created_at | REAL | unix seconds |
| status | TEXT | `open` / `closed` |
| closed_at | REAL NULL | unix seconds |
| close_reason | TEXT NULL | reason passed to `topic_close()` |
| metadata_json | TEXT NULL | optional |

Indexes:

- `idx_topics_name_status_created_at` on `(name, status, created_at)`

#### `questions`

| column | type | notes |
|---|---|---|
| question_id | TEXT PK | short UUID |
| topic_id | TEXT | FK-like (no strict FK required) |
| asked_by | TEXT | peer `agent_name` |
| question_text | TEXT | raw question |
| asked_at | REAL | unix seconds |
| status | TEXT | `pending` / `answered` / `cancelled` |
| cancel_reason | TEXT NULL | optional, set by `ask_cancel()` |

Status semantics:

- `pending`: accepting answers (can have 0+ answers)
- `answered`: closed by the asking peer (no longer offered to answerers)
- `cancelled`: cancelled by the asking peer

Indexes:

- `idx_questions_topic_status_askedat` on `(topic_id, status, asked_at)`

#### `answers`

| column | type | notes |
|---|---|---|
| answer_id | TEXT PK | short UUID |
| topic_id | TEXT | denormalized for convenience |
| question_id | TEXT | |
| answered_by | TEXT | peer `agent_name` |
| answered_at | REAL | unix seconds |
| payload_json | TEXT | JSON payload (see below) |

Indexes:

- Unique: `idx_answers_question_answered_by_unique` on `(question_id, answered_by)` (one answer per agent)
- `idx_answers_question_answered_at` on `(question_id, answered_at)`

### 4.2 Answer payload format (JSON)

Store as JSON in `answers.payload_json`:

```json
{
  "answer_markdown": "...answer in markdown...",
  "repo_pointers": ["path/to/file.py", "module:foo.bar", "entrypoint:baz()"],
  "suggested_followups": [
    "How is X implemented?",
    "Where is Y configured?"
  ]
}
```

All fields are peer-visible.

### 4.3 DB behavior requirements

- Enable WAL: `PRAGMA journal_mode=WAL;`
- Set busy timeout: `PRAGMA busy_timeout=2000;`
- Keep transactions short: one transaction per tool call.
- Validate `topic_id` exists and is `open` for new questions.

## 5) MCP server behavior

### 5.1 Error codes

Tools must return consistent error codes:

- topic not found -> `TOPIC_NOT_FOUND`
- topic closed (when asking new questions) -> `TOPIC_CLOSED`
- question not found -> `QUESTION_NOT_FOUND`
- question belongs to a different topic -> `TOPIC_MISMATCH`
- invalid args -> `INVALID_ARGUMENT`
- db busy/locked -> `DB_BUSY`
- db schema mismatch -> `DB_SCHEMA_MISMATCH`
- not joined to topic (identity missing) -> `AGENT_NOT_JOINED`
- self-answer attempt -> `FORBIDDEN_SELF_ANSWER`
- same agent answers same question twice -> `FORBIDDEN_ALREADY_ANSWERED`
- cancel/close by non-asker -> `FORBIDDEN_NOT_ASKER`

### 5.2 Warnings

Warnings are non-fatal notices (e.g., truncation).

- Warnings must always be returned in `structuredContent.warnings`.
- `structuredContent.warnings` must be a list of objects with this shape:
  - `code: string` (required)
  - `message: string` (optional)
  - `context: object` (optional)

## 6) Tool specifications

### 6.0 Common tool

#### `ping() -> {ok: bool, spec_version: string}`

Used for smoke testing.

Rules:

- Must not require DB access.
- Always returns `ok=true`.

### 6.1 Topic tools

#### `topic_create(name?: string, metadata?: object, mode?: "reuse"|"new" = "reuse") -> {topic_id, name, status}`

Rules:

- If `name` omitted, generate `topic-<id>`.
- If `mode="reuse"` and an open topic with same `name` exists: return the newest open topic.
- If `mode="new"`: always create a new topic.

#### `topic_list(status?: "open"|"closed"|"all" = "open") -> {topics:[...]}`

Lists topics newest-first by `created_at`.

#### `topic_resolve(name: string, allow_closed?: bool = false) -> {topic_id, name, status}`

Rules:

- If an open topic exists with `name`, return the newest open topic.
- Otherwise:
  - if `allow_closed=true`: return the newest closed topic with that name
  - else: `TOPIC_NOT_FOUND`

#### `topic_close(topic_id: string, reason?: string) -> {topic_id, status, closed_at, close_reason?}`

Idempotent close semantics:

- First close sets `closed_at=now` and `close_reason=reason` (if provided).
- Repeated closes do not mutate `closed_at` / `close_reason` and return warning `ALREADY_CLOSED`.

#### `topic_join(agent_name: string, topic_id?: string, name?: string, allow_closed?: bool=false) -> {topic_id, name, status, agent_name}`

Binds the caller’s `agent_name` to a topic **for the current MCP session**.

Rules:

- Exactly one of `topic_id` or `name` must be provided.
- Used by `ask()`, `pending_list()`, `answer()`, `ask_cancel()`, and `question_mark_answered()`.

### 6.2 Question tools

#### `ask(topic_id: string, question: string, wait_seconds?: int = 60) -> AskResult`

Requires joining the topic first with `topic_join(...)`.

Behavior:

1) Validate joined identity exists for `topic_id` (else `AGENT_NOT_JOINED`).
2) Validate `topic_id` exists and is open (else `TOPIC_CLOSED` / `TOPIC_NOT_FOUND`).
3) Insert into `questions` with `status="pending"` and `asked_by=<agent_name>`.
4) If `wait_seconds == 0`: return tool-level `status="queued"` (no wait).
5) If `wait_seconds > 0`: poll DB until **at least one answer exists** or timeout expires.

AskResult `status` values:

- `queued` - inserted; no wait performed
- `answered` - at least one answer exists within wait window (include rendered answers + `answers`)
- `timeout` - wait window expired; question remains `pending`
- `cancelled` - question was cancelled while waiting (rare)

#### `ask_poll(topic_id: string, question_id: string) -> AskPollResult`

Polls for status. No waiting.

AskPollResult `status` values:

- `pending` (no answers yet; still accepting answers)
- `answered` (closed and/or answers exist; include rendered answers + `answers`)
- `cancelled` (include `cancel_reason` if present)

The tool must return:

- `answers`: list of all answers so far (oldest first)
- `answers_count`
- `question_status`: the stored question status (`pending` / `answered` / `cancelled`)
- `accepting_answers`: boolean (derived from `question_status`)

#### `ask_cancel(topic_id: string, question_id: string, reason?: string) -> {topic_id, question_id, status, cancel_reason?}`

Cancels a pending question.

Rules:

- Requires joined identity (else `AGENT_NOT_JOINED`).
- Only the agent who asked the question may cancel it (else `FORBIDDEN_NOT_ASKER`).
- If question is `answered` -> `INVALID_ARGUMENT`.
- Idempotent: repeated cancels return warning `ALREADY_CANCELLED` and do not overwrite `cancel_reason`.

#### `question_mark_answered(topic_id: string, question_id: string) -> {topic_id, question_id, status}`

Marks a question as `answered` (closed by the asking peer).

Rules:

- Requires joined identity (else `AGENT_NOT_JOINED`).
- Only the agent who asked the question may close it (else `FORBIDDEN_NOT_ASKER`).
- Idempotent: repeated closes return warning `ALREADY_ANSWERED`.
- Cancelled questions cannot be marked answered (`INVALID_ARGUMENT`).

#### `pending_list(topic_id: string, limit?: int = 20, wait_seconds?: int = 60) -> PendingListResult`

Lists answerable questions oldest-first.

Rules:

- Requires joined identity (else `AGENT_NOT_JOINED`).
- Only returns questions with `status="pending"`.
- Excludes the caller’s own questions.
- Excludes questions the caller has already answered.
- If `wait_seconds > 0`, long-polls until at least one answerable question exists, or timeout expires.

#### `answer(topic_id: string, responses: ResponseItem[]) -> {saved:int, skipped:int, warnings?:[...] }`

Writes answers for one or more questions.

Rules:

- Requires joined identity (else `AGENT_NOT_JOINED`).
- The server must forbid answering a pending question where `asked_by == answered_by` (`FORBIDDEN_SELF_ANSWER`).
- The server must forbid answering a pending question more than once per agent (`FORBIDDEN_ALREADY_ANSWERED`).
- Only insert answers for questions currently `status="pending"`.
- `saved` increments for each successfully inserted answer.
- `skipped` increments for missing questions, non-pending questions, or topic mismatch.
- Topic may be closed: answering already-queued pending questions is still allowed.

`ResponseItem` schema:

- `question_id: string` (required)
- `answer_markdown: string` (required)
- `repo_pointers: string[]` (optional; soft limit 10)
- `suggested_followups: string[]` (required; soft limit 5, non-empty)

## 7) Clever prompting design (optional follow-ups)

Whenever an agent receives answers (via `ask` or `ask_poll`), render:

```text
<answers markdown here>

---
FOLLOW_UP_REQUIRED
Choose ONE follow-up question and call:
ask(topic_id="<id>", question="<your question>")

Suggested follow-ups:
1) ...
2) ...
3) ...

If you fully understand, reply with:
NO_FOLLOWUP_NEEDED
and provide a 3-5 bullet summary of what you learned.
```

When multiple answers exist, render all answers and use a merged set of suggested follow-ups (deduped, best-effort).

## 8) Limits and defaults (concrete)

### 8.1 Soft list limits (truncate + warning)

- `repo_pointers`: accept up to 10; truncate and warn `REPO_POINTERS_TRUNCATED`.
- `suggested_followups`: accept up to 5; truncate and warn `FOLLOWUPS_TRUNCATED`.

### 8.2 Hard size limits (error)

Defaults (env override allowed):

- Max question length: 8000 chars (`AGENT_BUS_MAX_QUESTION_CHARS`, default 8000)
- Max answer_markdown length: 65536 chars (`AGENT_BUS_MAX_ANSWER_CHARS`, default 65536)
- Max `responses` items in one `answer`: 50 (`AGENT_BUS_MAX_PUBLISH_BATCH`, default 50)

### 8.3 Polling defaults (ask / pending_list wait)

- initial poll interval: 0.25s
- backoff: multiply by 2 until max 1.0s
- cap interval at 1.0s

Env overrides:

- `AGENT_BUS_POLL_INITIAL_MS` (default 250)
- `AGENT_BUS_POLL_MAX_MS` (default 1000)

## 9) Closed-topic behavior

After `topic_close`:

- `ask()` on that topic must fail with `TOPIC_CLOSED`.
- `ask_poll()` must continue to work for existing `question_id`s.
- `pending_list()` and `answer()` continue to work (to drain and finish backlog).
