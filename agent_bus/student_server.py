from __future__ import annotations

import time
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from agent_bus.common import (
    ErrorCode,
    ToolWarning,
    WarningCode,
    env_int,
    tool_error,
    tool_ok,
)
from agent_bus.db import (
    AgentBusDB,
    DBBusyError,
    QuestionNotFoundError,
    TopicClosedError,
    TopicMismatchError,
    TopicNotFoundError,
)
from agent_bus.render import render_student_answer, strip_teacher_notes

SPEC_VERSION = "v3.1"
ROLE: Literal["student"] = "student"

db = AgentBusDB()
mcp = FastMCP(name="agent-bus-student")


@mcp.tool()
def ping() -> Any:
    return tool_ok(
        text="pong",
        structured={"ok": True, "role": ROLE, "spec_version": SPEC_VERSION},
    )


@mcp.tool()
def topic_list(status: Literal["open", "closed", "all"] = "open") -> Any:
    try:
        topics = db.topic_list(status=status)
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


@mcp.tool()
def topic_resolve(name: str, allow_closed: bool = False) -> Any:
    if not isinstance(name, str) or not name:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="name must be a non-empty string"
        )
    try:
        topic = db.topic_resolve(name=name, allow_closed=allow_closed)
    except TopicNotFoundError:
        return tool_error(code=ErrorCode.TOPIC_NOT_FOUND, message="Topic not found.")
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")
    return tool_ok(
        text=f'Topic resolved: name="{topic.name}" topic_id="{topic.topic_id}" status="{topic.status}"',
        structured={"topic_id": topic.topic_id, "name": topic.name, "status": topic.status},
    )


@mcp.tool()
def ask(topic_id: str, question: str, wait_seconds: int = 0) -> Any:
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
    if not isinstance(question, str) or not question:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="question must be a non-empty string"
        )
    if len(question) > max_question_chars:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT,
            message=f"question exceeds max length ({max_question_chars})",
        )
    if wait_seconds < 0:
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message="wait_seconds must be >= 0")

    try:
        topic = db.get_topic(topic_id=topic_id)
        q = db.question_create(topic_id=topic_id, asked_by="student", question_text=question)
    except TopicNotFoundError:
        return tool_error(code=ErrorCode.TOPIC_NOT_FOUND, message="Topic not found.")
    except TopicClosedError:
        return tool_error(code=ErrorCode.TOPIC_CLOSED, message="Topic is closed.")
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")

    if wait_seconds == 0:
        text = f'Question queued on topic "{topic.name}" ({topic.topic_id}).\nquestion_id={q.question_id}'
        return tool_ok(text=text, structured={"question_id": q.question_id, "status": "queued"})

    deadline = time.monotonic() + wait_seconds
    interval_s = poll_initial_ms / 1000.0
    max_interval_s = poll_max_ms / 1000.0

    while True:
        try:
            current = db.question_get(question_id=q.question_id)
        except DBBusyError:
            current = None

        if current is not None and current.status == "answered":
            payload = strip_teacher_notes(current.answer_payload) or {}
            answer_md = str(payload.get("answer_markdown", ""))
            followups = payload.get("suggested_followups") or []
            if not isinstance(followups, list):
                followups = []
            followups = [f for f in followups if isinstance(f, str)]

            rendered = render_student_answer(
                topic_id=topic_id,
                answer_markdown=answer_md,
                suggested_followups=followups,
            )
            return tool_ok(
                text=rendered,
                structured={
                    "question_id": q.question_id,
                    "status": "answered",
                    "answer_payload": payload,
                },
            )

        if time.monotonic() >= deadline:
            break

        time.sleep(interval_s)
        interval_s = min(interval_s * 2, max_interval_s)

    text = f"No answer yet.\nquestion_id={q.question_id}"
    return tool_ok(text=text, structured={"question_id": q.question_id, "status": "timeout"})


@mcp.tool()
def ask_poll(topic_id: str, question_id: str) -> Any:
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
    except QuestionNotFoundError:
        return tool_error(code=ErrorCode.QUESTION_NOT_FOUND, message="Question not found.")
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")

    if q.topic_id != topic_id:
        return tool_error(
            code=ErrorCode.TOPIC_MISMATCH, message="Question belongs to a different topic."
        )

    if q.status == "pending":
        return tool_ok(
            text=f"Pending.\nquestion_id={q.question_id}",
            structured={"question_id": q.question_id, "status": "pending"},
        )

    if q.status == "cancelled":
        structured: dict[str, Any] = {"question_id": q.question_id, "status": "cancelled"}
        if q.cancel_reason:
            structured["cancel_reason"] = q.cancel_reason
        text = f"Cancelled.\nquestion_id={q.question_id}"
        if q.cancel_reason:
            text += f'\nreason="{q.cancel_reason}"'
        return tool_ok(text=text, structured=structured)

    payload = strip_teacher_notes(q.answer_payload) or {}
    answer_md = str(payload.get("answer_markdown", ""))
    followups = payload.get("suggested_followups") or []
    if not isinstance(followups, list):
        followups = []
    followups = [f for f in followups if isinstance(f, str)]

    rendered = render_student_answer(
        topic_id=topic_id,
        answer_markdown=answer_md,
        suggested_followups=followups,
    )
    return tool_ok(
        text=rendered,
        structured={"question_id": q.question_id, "status": "answered", "answer_payload": payload},
    )


@mcp.tool()
def ask_cancel(topic_id: str, question_id: str, reason: str | None = None) -> Any:
    if not isinstance(topic_id, str) or not topic_id:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="topic_id must be a non-empty string"
        )
    if not isinstance(question_id, str) or not question_id:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="question_id must be a non-empty string"
        )
    if reason is not None and not isinstance(reason, str):
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message="reason must be a string")

    try:
        q, already_cancelled = db.question_cancel(
            topic_id=topic_id, question_id=question_id, reason=reason
        )
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


def main() -> None:
    mcp.run(transport="stdio")
