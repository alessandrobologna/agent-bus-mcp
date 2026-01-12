# Agent Bus MCP - Implementation Spec (stdio, single machine)

> Version: v3.1 (edge cases locked down)  
> Transport: stdio (local)  
> Storage: shared SQLite (WAL)

## 0) Goal

Enable two separate coding agents running on the same machine (e.g., Codex + Claude Code, or two instances of either) to collaborate without manual copy/paste by exchanging structured Q/A through a shared **agent bus**.

- The teacher agent has deep repo context and answers questions.
- The student agent is onboarding and asks iterative questions.
- The system should encourage self-propelling follow-up questioning via carefully formatted responses ("clever prompting").

## 1) Non-goals

- No network transport, auth, or security hardening (local-only).
- No server-side LLM calls (the teacher agent authors answers).
- No streaming/push required (polling is fine).

## 2) High-level architecture

### 2.1 Two MCP servers (role-separated)

Implement two distinct MCP servers (from the agent's perspective):

1) Teacher Agent-Bus MCP server: exposes teacher-only tools
2) Student Agent-Bus MCP server: exposes student-only tools

Both servers:

- run over stdio
- read/write to the same shared SQLite database file

This ensures:

- the student cannot invoke teacher tools because the student server never exposes them
- both agents can exchange messages without sharing a single stdio pipe

### 2.2 Bus concepts

- Topic: a named communication lane on the bus (e.g., "pink").
- topic_id: unique identifier for a topic.
- Question: a student message queued on a topic.
- Answer: teacher response linked to a question.

### 2.3 Shared persistence

Use a local SQLite DB with WAL enabled:

- default path: `~/.agent_bus/agent_bus.sqlite`
- configurable via env var: `AGENT_BUS_DB`

SQLite is sufficient for multiple local processes with WAL and short transactions.

---

## 3) Repository layout / packaging

### 3.1 Project structure

```text
agent_bus/
  agent_bus/
    __init__.py
    db.py
    models.py
    prompts.py
    teacher_server.py
    student_server.py
    common.py
  pyproject.toml
  README.md
  tests/
    test_db.py
    test_tools_smoke.py
```

### 3.2 Entry points (console scripts)

Provide two executables:

- `agent-bus-teacher` -> runs teacher MCP server over stdio
- `agent-bus-student` -> runs student MCP server over stdio

(Alternative acceptable approach: single script with `MCP_ROLE=teacher|student`.)

### 3.3 Dependencies

- Python 3.12+
- `mcp` Python SDK (FastMCP)
- Standard library: `sqlite3`, `uuid`, `time`, `json`

---

## 4) Data model (SQLite)

### 4.1 Tables

#### `topics`

| column | type | notes |
|---|---|---|
| topic_id | TEXT PK | short UUID (10-16 chars ok) |
| name | TEXT | human-friendly (e.g., "pink") |
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
| asked_by | TEXT | `student` (or instance id) |
| question_text | TEXT | raw question |
| asked_at | REAL | unix seconds |
| status | TEXT | `pending` / `answered` / `cancelled` |
| cancel_reason | TEXT NULL | optional, set by `ask_cancel()` |
| answered_at | REAL NULL | unix seconds |
| answered_by | TEXT NULL | `teacher` (or instance id) |
| answer_payload_json | TEXT NULL | JSON payload (see below) |

Indexes:

- `idx_questions_topic_status_askedat` on `(topic_id, status, asked_at)`

### 4.2 Answer payload format (JSON)

Store as JSON in `answer_payload_json`:

```json
{
  "answer_markdown": "...teacher answer in markdown...",
  "repo_pointers": ["path/to/file.py", "module:foo.bar", "entrypoint:baz()"],
  "suggested_followups": [
    "How is X implemented?",
    "Where is Y configured?"
  ],
  "teacher_notes": "optional internal notes"
}
```

Visibility rule:

- `teacher_notes` is teacher/internal-only. Student-facing tools (`ask`, `ask_poll`) must strip `teacher_notes` from any returned `answer_payload`.
- If you want student-visible extra context, add a separate field such as `teacher_addendum` (explicitly student-facing).

The server must also render a student-facing text from this payload (see section 7).

### 4.3 DB behavior requirements

- Enable WAL: `PRAGMA journal_mode=WAL;`
- Set busy timeout: `PRAGMA busy_timeout=2000;`
- Keep transactions short: one transaction per tool call (may update multiple rows).
- Validate `topic_id` exists and is `open` for new questions.

---

## 5) MCP server behavior

### 5.1 Common behavior

Both servers must:

- expose a `ping()` tool for smoke testing
- provide consistent error codes:
  - topic not found -> `TOPIC_NOT_FOUND`
  - topic closed (when asking new questions) -> `TOPIC_CLOSED`
  - question not found -> `QUESTION_NOT_FOUND`
  - question belongs to a different topic -> `TOPIC_MISMATCH`
  - invalid args -> `INVALID_ARGUMENT`
  - db busy/locked -> `DB_BUSY`
- return both:
  - a human-readable text result (for LLM consumption)
  - and `structuredContent` whenever possible

### 5.2 Warnings

Warnings are non-fatal notices (e.g., truncation).

- Warnings must always be returned in `structuredContent.warnings`.
- `structuredContent.warnings` must be a list of objects with this shape:
  - `code: string` (required)
  - `message: string` (optional, short human description)
  - `context: object` (optional, tool-specific details)
- Teacher-facing tools MAY also mention warnings in the human-readable text.
- Student-facing answer rendering MUST NOT include warnings (keep answers clean).

Example:

```json
{
  "warnings": [
    {
      "code": "FOLLOWUPS_TRUNCATED",
      "context": {
        "question_id": "9f2c1a",
        "original_count": 7,
        "kept_count": 5
      }
    }
  ]
}
```

---

## 6) Tool specifications

### 6.0 Common MCP tools (both servers)

#### 6.0.1 `ping() -> {ok: bool, role: "teacher"|"student", spec_version: string}`

Used for smoke testing.

Rules:

- Must not require DB access.
- Always returns `ok=true`.

### 6.1 Teacher MCP tools

#### 6.1.1 `topic_create(name?: string, metadata?: object, mode?: "reuse"|"new" = "reuse") -> {topic_id, name, status}`

Creates a new bus topic.

Rules:

- If `name` omitted, generate `topic-<id>`.
- If `mode="reuse"` and an open topic with same `name` exists: return the newest open topic (max `created_at`).
- If `mode="new"`: always create a new topic, even if an open topic with same `name` exists.

#### 6.1.2 `topic_list(status?: "open"|"closed"|"all" = "open") -> {topics:[...]}`

Lists topics.

Response shape:

- `topics` is a list of:
  - `topic_id: string`
  - `name: string`
  - `status: "open"|"closed"`
  - `created_at: number` (unix seconds)
  - `closed_at: number|null` (unix seconds)
  - `close_reason: string|null`
  - `metadata: object|null` (parsed from `metadata_json`)

Ordering:

- Newest-first by `created_at` (descending).

#### 6.1.3 `topic_close(topic_id: string, reason?: string) -> {topic_id, status, closed_at, close_reason?}`

Closes a topic.

Error behavior:

- If `topic_id` does not exist -> `TOPIC_NOT_FOUND`

Persistence rules:

- Always set `status="closed"` and `closed_at=now` on the first close.
- Set `close_reason=reason` only when provided on the first close (otherwise leave as NULL).
- Idempotent behavior:
  - If already closed, DO NOT change `closed_at`.
  - If already closed, DO NOT overwrite `close_reason` (ignore any new reason).
  - Return the existing `closed_at` / `close_reason`.
  - Add a warning `ALREADY_CLOSED` when called on an already closed topic.

#### 6.1.4 `teacher_drain(topic_id: string, limit?: int = 20) -> {pending:[{question_id, question_text, asked_at}]}`

Returns pending questions oldest-first.

Error behavior:

- If `topic_id` does not exist -> `TOPIC_NOT_FOUND`

Closed-topic behavior:

- Works even when topic is closed (so the teacher can finish answering already-queued questions).

Exclusions:

- Cancelled questions are not returned.

#### 6.1.5 `teacher_publish(topic_id: string, responses: ResponseItem[]) -> {saved:int, skipped:int, warnings?:[...]}`

Writes answers for one or more questions.

`ResponseItem` schema:

- `question_id: string` (required)
- `answer_markdown: string` (required)
- `repo_pointers: string[]` (optional; soft limit 10)
- `suggested_followups: string[]` (required; soft limit 5)
- `teacher_notes: string` (optional; internal-only)

Rules:

- If `topic_id` does not exist -> `TOPIC_NOT_FOUND`
- Only update questions currently `status="pending"`.
- `saved` increments for each successfully answered.
- `skipped` increments for missing questions, non-pending questions, or topic mismatch.
- Topic may be closed: teacher may still answer already-queued questions.

Limits policy (see section 8): inputs exceeding soft limits are truncated and a warning is returned.

Transaction rule:

- Use one transaction per tool call. Updating multiple questions inside the call is permitted.
- On unexpected DB error: rollback and return error. Partial commits are not allowed.

---

### 6.2 Student MCP tools

#### 6.2.1 `topic_list(status?: "open"|"closed"|"all" = "open") -> {topics:[...]}`

Lists topics for discovery.

Response shape and ordering are the same as teacher `topic_list()`.

#### 6.2.2 `topic_resolve(name: string, allow_closed?: bool = false) -> {topic_id, name, status}`

Resolves a topic by name.

Rules:

- If an open topic exists with `name`, return the newest open topic.
- If no open topic exists:
  - if `allow_closed=true`: return the newest closed topic with that name
  - else: return `TOPIC_NOT_FOUND`

#### 6.2.3 `ask(topic_id: string, question: string, wait_seconds?: int = 0) -> AskResult`

Creates a question and optionally waits for an answer.

Behavior:

1) Validate `topic_id` exists and is open (else `TOPIC_CLOSED` / `TOPIC_NOT_FOUND`).
2) Insert into `questions` with `status="pending"`.
3) If `wait_seconds == 0`: return tool-level `status="queued"`.
4) If `wait_seconds > 0`: poll DB until answered or timeout expires.
5) If answered: return tool-level `status="answered"` and include rendered student-facing answer + `answer_payload` (teacher_notes stripped).
6) If timeout expires: return tool-level `status="timeout"` (question remains pending).

AskResult `status` values:

- `queued` - inserted; no wait performed
- `answered` - answered within wait window
- `timeout` - wait window expired; still pending

#### 6.2.4 `ask_poll(topic_id: string, question_id: string) -> AskPollResult`

Polls for status. No waiting.

Rules:

- If `question_id` does not exist -> `QUESTION_NOT_FOUND`
- If question exists but belongs to another `topic_id` -> `TOPIC_MISMATCH`
- Otherwise return DB-derived status:

AskPollResult `status` values:

- `pending` - not answered yet
- `answered` - answer available (include rendered student-facing answer + `answer_payload` with teacher_notes stripped)
- `cancelled` - cancelled (include `cancel_reason` if present)

Notes:

- `ask_poll()` never returns `queued` (it reflects DB state).
- `ask_poll()` never returns `timeout` (no waiting is performed).

#### 6.2.5 `ask_cancel(topic_id: string, question_id: string, reason?: string) -> {topic_id, question_id, status, cancel_reason?}`

Cancels a pending question.

Error behavior:

- If `question_id` does not exist -> `QUESTION_NOT_FOUND`
- If question exists but belongs to another `topic_id` -> `TOPIC_MISMATCH`
- If question is `answered` -> `INVALID_ARGUMENT`

Idempotency:

- If question is already `cancelled`, return success with `status="cancelled"` (no-op).
- Do not overwrite `cancel_reason` on subsequent cancels; ignore any new reason.
- Add a warning `ALREADY_CANCELLED` when called on an already cancelled question.

State transition:

- Only transitions `pending` -> `cancelled`.

---

## 7) Clever prompting design (self-propelling follow-ups)

### 7.1 Student-facing rendering contract

Whenever the student receives an answered response (via `ask` or `ask_poll`), render:

```text
<teacher answer markdown here>

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

Notes:

- The `FOLLOW_UP_REQUIRED` and `NO_FOLLOWUP_NEEDED` tokens are deliberately machine-salient.
- Suggested follow-ups come from the teacher payload, but are rendered at most 5.

### 7.2 Student agent system rule (configure in student agent UI)

If a tool result contains `FOLLOW_UP_REQUIRED`, the student must:

- call `ask()` with exactly ONE suggested follow-up question, unless
- onboarding is complete; in that case output `NO_FOLLOWUP_NEEDED` plus a 3-5 bullet summary.

---

## 8) Limits and defaults (concrete)

### 8.1 Soft list limits (truncate + warning)

These are enforced by truncation + warnings (not hard errors), to keep agents resilient.

- `repo_pointers`: accept up to 10; if more provided, truncate to first 10 and add warning `REPO_POINTERS_TRUNCATED`.
- `suggested_followups`: accept up to 5; if more provided, truncate to first 5 and add warning `FOLLOWUPS_TRUNCATED`.

### 8.2 Hard size limits (error)

These return `INVALID_ARGUMENT` if exceeded.

Default limits (can be overridden by env vars if desired):

- Max question length: 8000 characters (`AGENT_BUS_MAX_QUESTION_CHARS`, default 8000)
- Max answer_markdown length: 65536 characters (`AGENT_BUS_MAX_ANSWER_CHARS`, default 65536)
- Max teacher_notes length: 16384 characters (`AGENT_BUS_MAX_TEACHER_NOTES_CHARS`, default 16384)
- Max `responses` items in one `teacher_publish`: 50 (`AGENT_BUS_MAX_PUBLISH_BATCH`, default 50)

### 8.3 Polling defaults (ask wait)

Default polling behavior for `ask(wait_seconds > 0)`:

- initial poll interval: 0.25s
- backoff: multiply by 2 until max 1.0s
- cap interval at 1.0s
- stop at `wait_seconds` deadline

Env overrides (optional):

- `AGENT_BUS_POLL_INITIAL_MS` (default 250)
- `AGENT_BUS_POLL_MAX_MS` (default 1000)

---

## 9) Closed-topic behavior (unambiguous)

After `topic_close`:

- Student:
  - `ask()` on that topic must fail with `TOPIC_CLOSED`.
  - `ask_poll()` must continue to work for existing `question_id`s.
  - `topic_resolve(name)` will not return the closed topic unless `allow_closed=true`.

- Teacher:
  - `teacher_drain()` continues to work (so teacher can finish queued questions).
  - `teacher_publish()` continues to work for already-queued `pending` questions.

Rationale: closing a topic prevents new traffic but allows draining existing backlog.

---

## 10) Decision log / Clarifications (answers to open items)

This section exists to remove ambiguity for implementers.

### 10.1 Status naming (DB vs tool statuses)

- DB `questions.status` is one of: `pending`, `answered`, `cancelled`.
- Tool-level statuses:
  - `ask()` returns: `queued` / `answered` / `timeout`
  - `ask_poll()` returns: `pending` / `answered` / `cancelled`
- `ask_poll()` never returns `queued` (it reflects DB state) and never returns `timeout` (no waiting is performed).
- Mapping:
  - `ask(): queued` means "DB row inserted with `pending`" (ack only).
  - `ask(): timeout` means "DB row still `pending` at deadline".

### 10.2 Transaction guidance

- Rule is one transaction per tool call, not "one row per tool call".
- `teacher_publish(responses=[...])` should update multiple rows within one transaction and commit once.
- On unexpected DB error: rollback and return error. Partial commits are not allowed.

### 10.3 topic_close persistence semantics

- Always set `topics.status="closed"` and `topics.closed_at=now` on first close.
- Set `topics.close_reason=reason` only if provided on first close; otherwise leave NULL.
- Idempotent on repeated calls: do not change closed_at/close_reason.

### 10.4 Duplicate topic names / resolution rules

- `topic_create(mode="new")` may produce multiple open topics with the same `name`.
- `topic_create(mode="reuse")` returns the newest open topic matching `name` (max `created_at`).
- `topic_resolve(name, allow_closed=false)` returns the newest open topic; if none open:
  - `allow_closed=false` -> `TOPIC_NOT_FOUND`
  - `allow_closed=true` -> newest closed topic

### 10.5 Limits enforcement (followups, repo pointers, payload sizes)

- Lists above soft limits are truncated and warnings returned.
- Payloads above hard size limits return `INVALID_ARGUMENT`.
- Student-facing answer rendering never shows warnings.

### 10.6 Closed-topic behavior for drain/publish/poll

- After close:
  - teacher may still `teacher_drain` and `teacher_publish` to finish existing pending questions
  - student may still `ask_poll` for existing questions
  - student may not `ask` new questions

### 10.7 Cancelled status in scope

- `cancelled` is in scope.
- Student tool `ask_cancel(topic_id, question_id, reason?)` is required.
- Only pending questions may be cancelled.
- Cancelled questions do not appear in `teacher_drain()`.

### 10.8 ask_cancel error and idempotency

- Unknown question_id -> `QUESTION_NOT_FOUND`.
- topic mismatch -> `TOPIC_MISMATCH`.
- already cancelled -> success no-op + warning `ALREADY_CANCELLED`.
- answered -> `INVALID_ARGUMENT`.

---

## 11) Testing plan

### 11.1 Unit tests (DB)

- create topic, resolve by name
- insert question, drain pending
- publish, verify answer_payload_json stored
- cancel pending question; ensure drain excludes it and poll shows cancelled

### 11.2 Integration smoke test (two processes)

Verify:

- teacher process sees questions from student process
- student poll returns rendered FOLLOW_UP_REQUIRED
- teacher tools absent from student `tools/list`, and vice versa

---

## 12) Deliverables checklist

1) `agent-bus-teacher` stdio MCP server, exposing teacher tool set
2) `agent-bus-student` stdio MCP server, exposing student tool set
3) Shared SQLite DB implementation with WAL and indexes
4) Response rendering implementing FOLLOW_UP_REQUIRED + NO_FOLLOWUP_NEEDED
5) README with:
   - how to configure each agent to launch the correct server command
   - recommended student system instruction (copy/paste)
   - example flows
