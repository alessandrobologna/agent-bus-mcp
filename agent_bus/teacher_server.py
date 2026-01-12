from __future__ import annotations

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
from agent_bus.db import AgentBusDB, DBBusyError, TopicNotFoundError

SPEC_VERSION = "v3.1"
ROLE: Literal["teacher"] = "teacher"

db = AgentBusDB()
mcp = FastMCP(name="agent-bus-teacher")


@mcp.tool()
def ping() -> Any:
    return tool_ok(
        text="pong",
        structured={"ok": True, "role": ROLE, "spec_version": SPEC_VERSION},
    )


@mcp.tool()
def topic_create(
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
    mode: Literal["reuse", "new"] = "reuse",
) -> Any:
    if metadata is not None and not isinstance(metadata, dict):
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message="metadata must be an object")
    try:
        topic = db.topic_create(name=name, metadata=metadata, mode=mode)
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")
    text = f'Topic: name="{topic.name}", topic_id="{topic.topic_id}", status="{topic.status}"'
    return tool_ok(
        text=text,
        structured={"topic_id": topic.topic_id, "name": topic.name, "status": topic.status},
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
def topic_close(topic_id: str, reason: str | None = None) -> Any:
    try:
        topic, already_closed = db.topic_close(topic_id=topic_id, reason=reason)
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


@mcp.tool()
def teacher_drain(topic_id: str, limit: int = 20) -> Any:
    if limit <= 0:
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message="limit must be > 0")
    try:
        topic = db.get_topic(topic_id=topic_id)
        pending = db.teacher_drain(topic_id=topic_id, limit=limit)
    except TopicNotFoundError:
        return tool_error(code=ErrorCode.TOPIC_NOT_FOUND, message="Topic not found.")
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")

    structured_pending = [
        {"question_id": q.question_id, "question_text": q.question_text, "asked_at": q.asked_at}
        for q in pending
    ]

    if not structured_pending:
        text = f'No pending questions for topic "{topic.name}" ({topic.topic_id}).'
        return tool_ok(text=text, structured={"pending": []})

    lines = [f'Pending questions for topic "{topic.name}" ({topic.topic_id}):', ""]
    for i, q in enumerate(structured_pending, start=1):
        lines.append(f"{i}) [{q['question_id']}] {q['question_text']}")
    return tool_ok(text="\n".join(lines), structured={"pending": structured_pending})


@mcp.tool()
def teacher_publish(topic_id: str, responses: list[dict[str, Any]]) -> Any:
    max_batch = env_int("AGENT_BUS_MAX_PUBLISH_BATCH", default=50, min_value=1)
    max_answer_chars = env_int("AGENT_BUS_MAX_ANSWER_CHARS", default=65536, min_value=1)
    max_teacher_notes_chars = env_int(
        "AGENT_BUS_MAX_TEACHER_NOTES_CHARS", default=16384, min_value=0
    )

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
        teacher_notes = item.get("teacher_notes")

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

        if teacher_notes is not None:
            if not isinstance(teacher_notes, str):
                return tool_error(
                    code=ErrorCode.INVALID_ARGUMENT,
                    message=f"responses[{idx}].teacher_notes must be a string",
                )
            if len(teacher_notes) > max_teacher_notes_chars:
                return tool_error(
                    code=ErrorCode.INVALID_ARGUMENT,
                    message=f"responses[{idx}].teacher_notes exceeds max length ({max_teacher_notes_chars})",
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
        if teacher_notes:
            payload["teacher_notes"] = teacher_notes

        items.append((cast(str, question_id), payload))

    try:
        saved, skipped = db.teacher_publish_batch(
            topic_id=topic_id, answered_by="teacher", items=items
        )
    except TopicNotFoundError:
        return tool_error(code=ErrorCode.TOPIC_NOT_FOUND, message="Topic not found.")
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")

    text_lines = [f"Saved answers: {saved}", f"Skipped: {skipped}"]
    if warnings:
        text_lines.append(f"Warnings: {len(warnings)}")
    return tool_ok(
        text="\n".join(text_lines),
        structured={"saved": saved, "skipped": skipped},
        warnings=warnings or None,
    )


def main() -> None:
    mcp.run(transport="stdio")
