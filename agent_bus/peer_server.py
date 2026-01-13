from __future__ import annotations

import time
from typing import Any, Literal, cast

from mcp.server.fastmcp import FastMCP

from agent_bus.common import (
    ErrorCode,
    ToolWarning,
    WarningCode,
    env_int,
    tool_error,
    tool_ok,
    truncate_list,
)
from agent_bus.db import (
    AgentBusDB,
    AlreadyAnsweredError,
    DBBusyError,
    QuestionNotFoundError,
    SchemaMismatchError,
    SelfAnswerForbiddenError,
    TopicClosedError,
    TopicMismatchError,
    TopicNotFoundError,
)
from agent_bus.render import render_answers

SPEC_VERSION = "v5.0"

CLIENT_SUGGESTED_MAX_ATTEMPTS = 3
CLIENT_SUGGESTED_DELAY_SECONDS = 10

db = AgentBusDB()
mcp = FastMCP(name="agent-bus")

# In-memory (per server process) mapping of joined topic_id -> agent_name.
# This is intentionally ephemeral: clients must call topic_join() again after a server restart.
_joined_agent_names: dict[str, str] = {}


def _normalize_agent_name(agent_name: str) -> str:
    return agent_name.strip()


def _validate_agent_name(agent_name: object) -> str | None:
    if not isinstance(agent_name, str) or not agent_name.strip():
        return "agent_name must be a non-empty string"
    normalized = agent_name.strip()
    if len(normalized) > 64:
        return "agent_name must be <= 64 characters"
    if any(c in normalized for c in ("\n", "\r", "\0")):
        return "agent_name must not contain control characters"
    return None


def _agent_name_for_topic(topic_id: str) -> str | None:
    return _joined_agent_names.get(topic_id)


def _schema_mismatch_result(e: SchemaMismatchError) -> Any:
    return tool_error(code=ErrorCode.DB_SCHEMA_MISMATCH, message=str(e))


@mcp.tool(description="Health check for the Agent Bus peer MCP server.")
def ping() -> Any:
    """Health check for the Agent Bus peer MCP server."""
    return tool_ok(text="pong", structured={"ok": True, "spec_version": SPEC_VERSION})


@mcp.tool(description="Create a topic (or reuse an existing open topic).")
def topic_create(
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
    mode: Literal["reuse", "new"] = "reuse",
) -> Any:
    """Create a topic (or reuse an existing open topic).

    mode:
    - reuse: return newest open topic with the same name
    - new: always create a new open topic
    """
    if metadata is not None and not isinstance(metadata, dict):
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message="metadata must be an object")
    try:
        topic = db.topic_create(name=name, metadata=metadata, mode=mode)
    except SchemaMismatchError as e:
        return _schema_mismatch_result(e)
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")
    text = f'Topic: name="{topic.name}", topic_id="{topic.topic_id}", status="{topic.status}"'
    return tool_ok(
        text=text,
        structured={"topic_id": topic.topic_id, "name": topic.name, "status": topic.status},
    )


@mcp.tool(description="List topics in the shared Agent Bus DB.")
def topic_list(status: Literal["open", "closed", "all"] = "open") -> Any:
    """List topics in the shared Agent Bus DB."""
    try:
        topics = db.topic_list(status=status)
    except SchemaMismatchError as e:
        return _schema_mismatch_result(e)
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")

    structured_topics = [
        {
            "topic_id": t.topic_id,
            "name": t.name,
            "status": t.status,
            "created_at": t.created_at,
            "closed_at": t.closed_at,
            "close_reason": t.close_reason,
            "metadata": t.metadata,
        }
        for t in topics
    ]
    lines = [f"Topics ({status}): {len(structured_topics)}"]
    for t in structured_topics[:20]:
        lines.append(f"- {t['name']} ({t['topic_id']}) status={t['status']}")
    if len(structured_topics) > 20:
        lines.append(f"... ({len(structured_topics) - 20} more)")
    return tool_ok(text="\n".join(lines), structured={"topics": structured_topics})


@mcp.tool(description="Close a topic (idempotent).")
def topic_close(topic_id: str, reason: str | None = None) -> Any:
    """Close a topic (idempotent)."""
    try:
        topic, already_closed = db.topic_close(topic_id=topic_id, reason=reason)
    except SchemaMismatchError as e:
        return _schema_mismatch_result(e)
    except TopicNotFoundError:
        return tool_error(code=ErrorCode.TOPIC_NOT_FOUND, message="Topic not found.")
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")

    warnings: list[ToolWarning] = []
    if already_closed:
        warnings.append(
            ToolWarning(code=str(WarningCode.ALREADY_CLOSED), context={"topic_id": topic_id})
        )

    text = f'Topic closed: topic_id="{topic.topic_id}" closed_at={topic.closed_at}'
    if topic.close_reason:
        text += f' close_reason="{topic.close_reason}"'
    return tool_ok(
        text=text,
        structured={
            "topic_id": topic.topic_id,
            "status": topic.status,
            "closed_at": topic.closed_at,
            "close_reason": topic.close_reason,
        },
        warnings=warnings or None,
    )


@mcp.tool(description="Resolve a topic by name.")
def topic_resolve(name: str, allow_closed: bool = False) -> Any:
    """Resolve a topic by name.

    Returns the newest open topic with this name. If allow_closed is true and no open topic exists,
    returns the newest closed topic with this name.
    """
    if not isinstance(name, str) or not name:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="name must be a non-empty string"
        )
    try:
        topic = db.topic_resolve(name=name, allow_closed=allow_closed)
    except SchemaMismatchError as e:
        return _schema_mismatch_result(e)
    except TopicNotFoundError:
        return tool_error(code=ErrorCode.TOPIC_NOT_FOUND, message="Topic not found.")
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")
    return tool_ok(
        text=f'Topic resolved: name="{topic.name}" topic_id="{topic.topic_id}" status="{topic.status}"',
        structured={"topic_id": topic.topic_id, "name": topic.name, "status": topic.status},
    )


@mcp.tool(description="Join a topic as a named peer (per MCP session).")
def topic_join(
    agent_name: str,
    *,
    topic_id: str | None = None,
    name: str | None = None,
    allow_closed: bool = False,
) -> Any:
    """Join a topic as a named peer (in-memory per server process).

    The Agent Bus DB stores `asked_by` / `answered_by` as this name, and the server uses it to:
    - avoid returning your own questions in `pending_list()`
    - forbid answering your own questions in `answer()`

    Exactly one of `topic_id` or `name` must be provided.
    """
    err = _validate_agent_name(agent_name)
    if err:
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message=err)

    if topic_id and name:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="Provide exactly one of topic_id or name"
        )
    if not topic_id and not name:
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message="Provide topic_id or name")

    try:
        if topic_id:
            if not isinstance(topic_id, str) or not topic_id:
                return tool_error(
                    code=ErrorCode.INVALID_ARGUMENT, message="topic_id must be a non-empty string"
                )
            topic = db.get_topic(topic_id=topic_id)
        else:
            if not isinstance(name, str) or not name:
                return tool_error(
                    code=ErrorCode.INVALID_ARGUMENT, message="name must be a non-empty string"
                )
            topic = db.topic_resolve(name=cast(str, name), allow_closed=allow_closed)
    except SchemaMismatchError as e:
        return _schema_mismatch_result(e)
    except TopicNotFoundError:
        return tool_error(code=ErrorCode.TOPIC_NOT_FOUND, message="Topic not found.")
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")

    normalized = _normalize_agent_name(agent_name)
    _joined_agent_names[topic.topic_id] = normalized

    text = f'Joined topic "{topic.name}" ({topic.topic_id}) as "{normalized}".'
    return tool_ok(
        text=text,
        structured={
            "topic_id": topic.topic_id,
            "name": topic.name,
            "status": topic.status,
            "agent_name": normalized,
        },
    )


@mcp.tool(description="Ask a question on a topic and optionally wait for the first answer.")
def ask(topic_id: str, question: str, wait_seconds: int = 60) -> Any:
    """Queue a question on a topic and optionally wait for an answer.

    Requires joining the topic first via `topic_join()`.

    Tool-level status values:
    - queued: inserted; no wait performed
    - answered: answered within wait window
    - timeout: wait window expired; question remains pending

    To enqueue without waiting, set wait_seconds=0.
    """
    try:
        max_question_chars = env_int("AGENT_BUS_MAX_QUESTION_CHARS", default=8000, min_value=1)
        poll_initial_ms = env_int("AGENT_BUS_POLL_INITIAL_MS", default=250, min_value=1)
        poll_max_ms = env_int("AGENT_BUS_POLL_MAX_MS", default=1000, min_value=1)
    except ValueError as e:  # pragma: no cover
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message=str(e))

    if not isinstance(topic_id, str) or not topic_id:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="topic_id must be a non-empty string"
        )
    agent_name = _agent_name_for_topic(topic_id)
    if agent_name is None:
        return tool_error(
            code=ErrorCode.AGENT_NOT_JOINED,
            message="Not joined to topic. Call topic_join() first.",
        )

    if not isinstance(question, str) or not question:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="question must be a non-empty string"
        )
    if len(question) > max_question_chars:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT,
            message=f"question exceeds max length ({max_question_chars})",
        )
    if not isinstance(wait_seconds, int):
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message="wait_seconds must be an int")
    if wait_seconds < 0:
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message="wait_seconds must be >= 0")

    try:
        topic = db.get_topic(topic_id=topic_id)
        q = db.question_create(topic_id=topic_id, asked_by=agent_name, question_text=question)
    except SchemaMismatchError as e:
        return _schema_mismatch_result(e)
    except TopicNotFoundError:
        return tool_error(code=ErrorCode.TOPIC_NOT_FOUND, message="Topic not found.")
    except TopicClosedError:
        return tool_error(code=ErrorCode.TOPIC_CLOSED, message="Topic is closed.")
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")

    if wait_seconds == 0:
        next_action = {
            "tool": "ask_poll",
            "args": {"topic_id": topic_id, "question_id": q.question_id},
        }
        client_plan = {
            "max_attempts": CLIENT_SUGGESTED_MAX_ATTEMPTS,
            "delay_seconds": CLIENT_SUGGESTED_DELAY_SECONDS,
        }

        text = (
            f'Question queued on topic "{topic.name}" ({topic.topic_id}).\n'
            f"question_id={q.question_id}\n"
            "\n"
            "---\n"
            "POLL_FOR_ANSWER\n"
            f'Call ask_poll(topic_id="{topic_id}", question_id="{q.question_id}") '
            f"up to {CLIENT_SUGGESTED_MAX_ATTEMPTS} times "
            f"(e.g. every {CLIENT_SUGGESTED_DELAY_SECONDS}s). "
            "Stop after the last attempt if still pending.\n"
        )
        return tool_ok(
            text=text,
            structured={
                "question_id": q.question_id,
                "status": "queued",
                "asked_by": agent_name,
                "next_action": next_action,
                "client_plan": client_plan,
                "instructions": (
                    "Poll for an answer by calling ask_poll up to max_attempts times. "
                    "If still pending after max_attempts, stop and wait for user direction."
                ),
            },
        )

    deadline = time.monotonic() + wait_seconds
    interval_s = poll_initial_ms / 1000.0
    max_interval_s = poll_max_ms / 1000.0

    while True:
        try:
            current = db.question_get(question_id=q.question_id)
        except DBBusyError:
            current = None

        if current is not None and current.status == "cancelled":
            return tool_ok(
                text=f"Cancelled.\nquestion_id={q.question_id}",
                structured={
                    "question_id": q.question_id,
                    "status": "cancelled",
                    "cancel_reason": current.cancel_reason,
                },
            )

        try:
            answers = db.answers_list(question_id=q.question_id)
        except DBBusyError:
            answers = []

        if answers:
            rendered = render_answers(topic_id=topic_id, answers=answers)
            structured_answers = [
                {
                    "answer_id": a.answer_id,
                    "answered_by": a.answered_by,
                    "answered_at": a.answered_at,
                    "payload": a.payload,
                }
                for a in answers
            ]
            return tool_ok(
                text=rendered,
                structured={
                    "question_id": q.question_id,
                    "status": "answered",
                    "question_status": current.status if current is not None else None,
                    "accepting_answers": (current.status == "pending")
                    if current is not None
                    else None,
                    "answers": structured_answers,
                    "answers_count": len(structured_answers),
                },
            )

        if time.monotonic() >= deadline:
            break

        time.sleep(interval_s)
        interval_s = min(interval_s * 2, max_interval_s)

    next_action = {
        "tool": "ask_poll",
        "args": {"topic_id": topic_id, "question_id": q.question_id},
    }
    client_plan = {
        "max_attempts": CLIENT_SUGGESTED_MAX_ATTEMPTS,
        "delay_seconds": CLIENT_SUGGESTED_DELAY_SECONDS,
    }
    text = (
        f"No answer yet.\nquestion_id={q.question_id}\n"
        "\n"
        "---\n"
        "POLL_FOR_ANSWER\n"
        f'Call ask_poll(topic_id="{topic_id}", question_id="{q.question_id}") '
        f"up to {CLIENT_SUGGESTED_MAX_ATTEMPTS} times "
        f"(e.g. every {CLIENT_SUGGESTED_DELAY_SECONDS}s). "
        "Stop after the last attempt if still pending.\n"
    )
    return tool_ok(
        text=text,
        structured={
            "question_id": q.question_id,
            "status": "timeout",
            "next_action": next_action,
            "client_plan": client_plan,
            "instructions": (
                "Poll for an answer by calling ask_poll up to max_attempts times. "
                "If still pending after max_attempts, stop and wait for user direction."
            ),
        },
    )


@mcp.tool(description="Poll for answers to a previously asked question.")
def ask_poll(topic_id: str, question_id: str) -> Any:
    """Poll the status of a question without waiting."""
    if not isinstance(topic_id, str) or not topic_id:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="topic_id must be a non-empty string"
        )
    if not isinstance(question_id, str) or not question_id:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="question_id must be a non-empty string"
        )

    try:
        q = db.question_get(question_id=question_id)
        answers = db.answers_list(question_id=question_id)
    except SchemaMismatchError as e:
        return _schema_mismatch_result(e)
    except QuestionNotFoundError:
        return tool_error(code=ErrorCode.QUESTION_NOT_FOUND, message="Question not found.")
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")

    if q.topic_id != topic_id:
        return tool_error(
            code=ErrorCode.TOPIC_MISMATCH, message="Question belongs to a different topic."
        )

    structured_answers = [
        {
            "answer_id": a.answer_id,
            "answered_by": a.answered_by,
            "answered_at": a.answered_at,
            "payload": a.payload,
        }
        for a in answers
    ]

    if q.status == "cancelled":
        structured: dict[str, Any] = {
            "question_id": q.question_id,
            "status": "cancelled",
            "question_status": q.status,
            "accepting_answers": False,
            "answers": structured_answers,
            "answers_count": len(structured_answers),
        }
        if q.cancel_reason:
            structured["cancel_reason"] = q.cancel_reason
        text = f"Cancelled.\nquestion_id={q.question_id}"
        if q.cancel_reason:
            text += f'\nreason="{q.cancel_reason}"'
        return tool_ok(text=text, structured=structured)

    if structured_answers:
        rendered = render_answers(topic_id=topic_id, answers=answers)
        return tool_ok(
            text=rendered,
            structured={
                "question_id": q.question_id,
                "status": "answered",
                "question_status": q.status,
                "accepting_answers": q.status == "pending",
                "answers": structured_answers,
                "answers_count": len(structured_answers),
            },
        )

    if q.status == "answered":
        return tool_ok(
            text=f"Closed.\nquestion_id={q.question_id}",
            structured={
                "question_id": q.question_id,
                "status": "answered",
                "question_status": q.status,
                "accepting_answers": False,
                "answers": [],
                "answers_count": 0,
            },
        )

    return tool_ok(
        text=f"Pending.\nquestion_id={q.question_id}",
        structured={
            "question_id": q.question_id,
            "status": "pending",
            "question_status": q.status,
            "accepting_answers": True,
            "answers": [],
            "answers_count": 0,
        },
    )


@mcp.tool(description="Cancel your own pending question (idempotent).")
def ask_cancel(topic_id: str, question_id: str, reason: str | None = None) -> Any:
    """Cancel your own pending question (idempotent).

    Requires joining the topic first via `topic_join()`.
    Returns success if already cancelled (no-op) and includes warning ALREADY_CANCELLED.
    """
    if not isinstance(topic_id, str) or not topic_id:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="topic_id must be a non-empty string"
        )
    agent_name = _agent_name_for_topic(topic_id)
    if agent_name is None:
        return tool_error(
            code=ErrorCode.AGENT_NOT_JOINED,
            message="Not joined to topic. Call topic_join() first.",
        )

    if not isinstance(question_id, str) or not question_id:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="question_id must be a non-empty string"
        )
    if reason is not None and not isinstance(reason, str):
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message="reason must be a string")

    try:
        existing = db.question_get(question_id=question_id)
    except SchemaMismatchError as e:
        return _schema_mismatch_result(e)
    except QuestionNotFoundError:
        return tool_error(code=ErrorCode.QUESTION_NOT_FOUND, message="Question not found.")
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")

    if existing.topic_id != topic_id:
        return tool_error(
            code=ErrorCode.TOPIC_MISMATCH, message="Question belongs to a different topic."
        )
    if existing.asked_by != agent_name:
        return tool_error(
            code=ErrorCode.FORBIDDEN_NOT_ASKER,
            message="Only the asking agent can cancel this question.",
        )

    try:
        q, already_cancelled = db.question_cancel(
            topic_id=topic_id, question_id=question_id, reason=reason
        )
    except SchemaMismatchError as e:
        return _schema_mismatch_result(e)
    except QuestionNotFoundError:
        return tool_error(code=ErrorCode.QUESTION_NOT_FOUND, message="Question not found.")
    except TopicMismatchError:
        return tool_error(
            code=ErrorCode.TOPIC_MISMATCH, message="Question belongs to a different topic."
        )
    except ValueError:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="Cannot cancel answered question."
        )
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")

    warnings: list[ToolWarning] = []
    if already_cancelled:
        warnings.append(
            ToolWarning(
                code=str(WarningCode.ALREADY_CANCELLED), context={"question_id": question_id}
            )
        )

    structured: dict[str, Any] = {
        "topic_id": topic_id,
        "question_id": question_id,
        "status": q.status,
    }
    if q.cancel_reason:
        structured["cancel_reason"] = q.cancel_reason

    text = f"Cancelled.\nquestion_id={question_id}"
    return tool_ok(text=text, structured=structured, warnings=warnings or None)


@mcp.tool(description="Mark your own question as answered (stop offering it to other peers).")
def question_mark_answered(topic_id: str, question_id: str) -> Any:
    """Mark your own question as answered (idempotent).

    Requires joining the topic first via `topic_join()`.
    Returns success if already answered (no-op) and includes warning ALREADY_ANSWERED.
    """
    if not isinstance(topic_id, str) or not topic_id:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="topic_id must be a non-empty string"
        )
    agent_name = _agent_name_for_topic(topic_id)
    if agent_name is None:
        return tool_error(
            code=ErrorCode.AGENT_NOT_JOINED,
            message="Not joined to topic. Call topic_join() first.",
        )

    if not isinstance(question_id, str) or not question_id:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="question_id must be a non-empty string"
        )

    try:
        existing = db.question_get(question_id=question_id)
    except SchemaMismatchError as e:
        return _schema_mismatch_result(e)
    except QuestionNotFoundError:
        return tool_error(code=ErrorCode.QUESTION_NOT_FOUND, message="Question not found.")
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")

    if existing.topic_id != topic_id:
        return tool_error(
            code=ErrorCode.TOPIC_MISMATCH, message="Question belongs to a different topic."
        )
    if existing.asked_by != agent_name:
        return tool_error(
            code=ErrorCode.FORBIDDEN_NOT_ASKER,
            message="Only the asking agent can mark this question as answered.",
        )

    try:
        q, already_answered = db.question_mark_answered(topic_id=topic_id, question_id=question_id)
    except SchemaMismatchError as e:
        return _schema_mismatch_result(e)
    except QuestionNotFoundError:
        return tool_error(code=ErrorCode.QUESTION_NOT_FOUND, message="Question not found.")
    except TopicMismatchError:
        return tool_error(
            code=ErrorCode.TOPIC_MISMATCH, message="Question belongs to a different topic."
        )
    except ValueError:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="Cannot mark cancelled question as answered."
        )
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")

    warnings: list[ToolWarning] = []
    if already_answered:
        warnings.append(
            ToolWarning(
                code=str(WarningCode.ALREADY_ANSWERED), context={"question_id": question_id}
            )
        )

    text = f"Marked answered.\nquestion_id={question_id}"
    return tool_ok(
        text=text,
        structured={"topic_id": topic_id, "question_id": question_id, "status": q.status},
        warnings=warnings or None,
    )


@mcp.tool(
    description="List pending questions you can answer (excluding your own and ones you've already answered)."
)
def pending_list(topic_id: str, limit: int = 20, wait_seconds: int = 60) -> Any:
    """Return pending questions for a topic, oldest first (excluding your own).

    If wait_seconds > 0, long-polls until at least one pending question is available,
    or until the timeout expires.
    """
    if not isinstance(topic_id, str) or not topic_id:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="topic_id must be a non-empty string"
        )
    agent_name = _agent_name_for_topic(topic_id)
    if agent_name is None:
        return tool_error(
            code=ErrorCode.AGENT_NOT_JOINED,
            message="Not joined to topic. Call topic_join() first.",
        )
    if limit <= 0:
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message="limit must be > 0")
    if not isinstance(wait_seconds, int):
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message="wait_seconds must be an int")
    if wait_seconds < 0:
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message="wait_seconds must be >= 0")

    try:
        poll_initial_ms = env_int("AGENT_BUS_POLL_INITIAL_MS", default=250, min_value=1)
        poll_max_ms = env_int("AGENT_BUS_POLL_MAX_MS", default=1000, min_value=1)
    except ValueError as e:  # pragma: no cover
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message=str(e))

    try:
        topic = db.get_topic(topic_id=topic_id)
    except SchemaMismatchError as e:
        return _schema_mismatch_result(e)
    except TopicNotFoundError:
        return tool_error(code=ErrorCode.TOPIC_NOT_FOUND, message="Topic not found.")
    except DBBusyError:
        # If we're long-polling, treat DB locks as transient and keep trying.
        if wait_seconds == 0:
            return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")
        topic = None

    deadline = time.monotonic() + wait_seconds
    interval_s = poll_initial_ms / 1000.0
    max_interval_s = poll_max_ms / 1000.0

    pending = []
    while True:
        if topic is None:
            try:
                topic = db.get_topic(topic_id=topic_id)
            except SchemaMismatchError as e:
                return _schema_mismatch_result(e)
            except TopicNotFoundError:
                return tool_error(code=ErrorCode.TOPIC_NOT_FOUND, message="Topic not found.")
            except DBBusyError:
                if time.monotonic() >= deadline:
                    topic = None
                    break
                time.sleep(interval_s)
                interval_s = min(interval_s * 2, max_interval_s)
                continue

        try:
            pending = db.question_list_answerable(
                topic_id=topic_id, limit=limit, agent_name=agent_name
            )
        except SchemaMismatchError as e:
            return _schema_mismatch_result(e)
        except DBBusyError:
            if wait_seconds == 0:
                return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")
            pending = []

        if pending or wait_seconds == 0:
            break

        if time.monotonic() >= deadline:
            break

        time.sleep(interval_s)
        interval_s = min(interval_s * 2, max_interval_s)

    structured_pending = [
        {
            "question_id": q.question_id,
            "question_text": q.question_text,
            "asked_at": q.asked_at,
            "asked_by": q.asked_by,
        }
        for q in pending
    ]

    status: Literal["ready", "timeout", "empty"]
    if structured_pending:
        status = "ready"
    elif wait_seconds > 0:
        status = "timeout"
    else:
        status = "empty"

    if not structured_pending:
        if topic is None:  # pragma: no cover
            return tool_ok(
                text="No pending questions.",
                structured={"pending": [], "status": status},
            )
        text = f'No pending questions for topic "{topic.name}" ({topic.topic_id}).'
        return tool_ok(text=text, structured={"pending": [], "status": status})

    assert topic is not None  # for type checkers
    lines = [f'Pending questions for topic "{topic.name}" ({topic.topic_id}):', ""]
    for i, q in enumerate(structured_pending, start=1):
        lines.append(f"{i}) [{q['question_id']}] ({q['asked_by']}) {q['question_text']}")
    return tool_ok(
        text="\n".join(lines), structured={"pending": structured_pending, "status": status}
    )


@mcp.tool(description="Publish answers for one or more pending questions.")
def answer(topic_id: str, responses: list[dict[str, Any]]) -> Any:
    """Answer one or more pending questions.

    Requires joining the topic first via `topic_join()`.

    Applies soft truncation for repo_pointers (10) and suggested_followups (5) with warnings.

    Defaults:
    - If repo_pointers is omitted, it is treated as [].
    - suggested_followups is required and must be non-empty.
    """
    agent_name = _agent_name_for_topic(topic_id)
    if agent_name is None:
        return tool_error(
            code=ErrorCode.AGENT_NOT_JOINED,
            message="Not joined to topic. Call topic_join() first.",
        )

    max_batch = env_int("AGENT_BUS_MAX_PUBLISH_BATCH", default=50, min_value=1)
    max_answer_chars = env_int("AGENT_BUS_MAX_ANSWER_CHARS", default=65536, min_value=1)

    if not isinstance(responses, list):
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message="responses must be a list")
    if len(responses) > max_batch:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT,
            message=f"responses must have at most {max_batch} items",
        )

    warnings: list[ToolWarning] = []
    items: list[tuple[str, dict[str, Any]]] = []

    for idx, item in enumerate(responses):
        if not isinstance(item, dict):
            return tool_error(
                code=ErrorCode.INVALID_ARGUMENT, message=f"responses[{idx}] must be an object"
            )

        question_id = item.get("question_id")
        answer_markdown = item.get("answer_markdown")
        repo_pointers = item.get("repo_pointers") or []
        suggested_followups = item.get("suggested_followups")

        if not isinstance(question_id, str) or not question_id:
            return tool_error(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"responses[{idx}].question_id must be a non-empty string",
            )
        if not isinstance(answer_markdown, str):
            return tool_error(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"responses[{idx}].answer_markdown must be a string",
            )
        if len(answer_markdown) > max_answer_chars:
            return tool_error(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"responses[{idx}].answer_markdown exceeds max length ({max_answer_chars})",
            )

        if not isinstance(repo_pointers, list) or not all(
            isinstance(p, str) for p in repo_pointers
        ):
            return tool_error(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"responses[{idx}].repo_pointers must be a list of strings",
            )

        if not isinstance(suggested_followups, list) or not all(
            isinstance(f, str) for f in suggested_followups
        ):
            return tool_error(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"responses[{idx}].suggested_followups must be a list of strings",
            )
        if not suggested_followups:
            return tool_error(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"responses[{idx}].suggested_followups must not be empty",
            )

        repo_pointers, warn = truncate_list(
            repo_pointers,
            max_items=10,
            warning_code=WarningCode.REPO_POINTERS_TRUNCATED,
            context={"question_id": question_id},
        )
        if warn:
            warnings.append(warn)

        suggested_followups, warn = truncate_list(
            suggested_followups,
            max_items=5,
            warning_code=WarningCode.FOLLOWUPS_TRUNCATED,
            context={"question_id": question_id},
        )
        if warn:
            warnings.append(warn)

        payload: dict[str, Any] = {
            "answer_markdown": answer_markdown,
            "repo_pointers": repo_pointers,
            "suggested_followups": suggested_followups,
        }
        items.append((cast(str, question_id), payload))

    question_ids = [qid for qid, _ in items]
    if len(set(question_ids)) != len(question_ids):
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT,
            message="responses contains duplicate question_id(s)",
        )

    try:
        saved, skipped = db.answer_insert_batch(
            topic_id=topic_id, answered_by=agent_name, items=items
        )
    except SchemaMismatchError as e:
        return _schema_mismatch_result(e)
    except TopicNotFoundError:
        return tool_error(code=ErrorCode.TOPIC_NOT_FOUND, message="Topic not found.")
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")
    except SelfAnswerForbiddenError as e:
        return tool_error(
            code=ErrorCode.FORBIDDEN_SELF_ANSWER,
            message="Cannot answer your own question.",
            structured={"forbidden_question_ids": e.question_ids},
        )
    except AlreadyAnsweredError as e:
        return tool_error(
            code=ErrorCode.FORBIDDEN_ALREADY_ANSWERED,
            message="Cannot answer the same question more than once.",
            structured={"already_answered_question_ids": e.question_ids},
        )

    text_lines = [f"Saved answers: {saved}", f"Skipped: {skipped}"]
    if warnings:
        text_lines.append(f"Warnings: {len(warnings)}")

    next_action = {
        "tool": "pending_list",
        "args": {"topic_id": topic_id, "limit": 20, "wait_seconds": 60},
    }
    client_plan = {"max_attempts": CLIENT_SUGGESTED_MAX_ATTEMPTS}

    text = (
        "\n".join(text_lines)
        + "\n"
        + "\n"
        + "---\n"
        + "CHECK_FOR_MORE_QUESTIONS\n"
        + f'Call pending_list(topic_id="{topic_id}") up to {client_plan["max_attempts"]} times '
        + "to see if more questions arrive. Stop after the last attempt.\n"
    )

    return tool_ok(
        text=text,
        structured={
            "saved": saved,
            "skipped": skipped,
            "next_action": next_action,
            "client_plan": client_plan,
            "instructions": (
                "After answering, check for more pending questions by calling pending_list. "
                "Do this up to max_attempts times, then stop."
            ),
        },
        warnings=warnings or None,
    )


def main() -> None:
    mcp.run(transport="stdio")
