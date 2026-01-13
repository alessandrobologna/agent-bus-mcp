from __future__ import annotations

import time
from typing import Annotated, Any, Literal, cast

from mcp.server.fastmcp import FastMCP
from pydantic import Field

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
    SchemaMismatchError,
    TopicClosedError,
    TopicNotFoundError,
)

SPEC_VERSION = "v6.0"

db = AgentBusDB()
mcp = FastMCP(
    name="agent-bus",
    instructions=(
        "Join a topic with topic_join(agent_name=..., topic_id=...|name=...), then use sync() to "
        "read/write messages. Outbox items require content_markdown. Use reply_to to respond to a "
        "specific message. Convention: message_type='question' for questions and "
        "message_type='answer' for replies."
    ),
)

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


@mcp.tool(description="Health check for the Agent Bus dialog MCP server.")
def ping() -> Any:
    """Health check for the Agent Bus dialog MCP server."""
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
    """Resolve a topic by name."""
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

    Requires joining before calling `sync()`.
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


@mcp.tool(description="Sync messages on a topic (delta-based, read/write, server-side cursor).")
def sync(
    topic_id: str,
    *,
    outbox: Annotated[
        list[dict[str, Any]] | None,
        Field(
            description=(
                "Outgoing messages to send. Each item is an object with: "
                "content_markdown (string, required), message_type (string, optional, default "
                '"message"), reply_to (string|null), metadata (object|null), client_message_id '
                "(string|null, optional idempotency key)."
            ),
            json_schema_extra={
                "examples": [
                    [
                        {
                            "content_markdown": "Hello from red-squirrel",
                            "message_type": "message",
                            "reply_to": None,
                            "metadata": {"kind": "greeting"},
                            "client_message_id": "msg-001",
                        }
                    ]
                ]
            },
        ),
    ] = None,
    max_items: int = 50,
    include_self: bool = False,
    wait_seconds: int = 60,
    auto_advance: bool = True,
    ack_through: int | None = None,
) -> Any:
    """Read/write sync against a topic message stream.

    Requires joining the topic first via `topic_join()`.
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

    if outbox is None:
        outbox = []
    if not isinstance(outbox, list):
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message="outbox must be a list")

    if not isinstance(max_items, int) or max_items <= 0:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="max_items must be a positive int"
        )
    if not isinstance(wait_seconds, int) or wait_seconds < 0:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="wait_seconds must be an int >= 0"
        )
    if not isinstance(include_self, bool):
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message="include_self must be a bool")
    if not isinstance(auto_advance, bool):
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message="auto_advance must be a bool")
    if ack_through is not None and (not isinstance(ack_through, int) or ack_through < 0):
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT, message="ack_through must be an int >= 0"
        )
    if ack_through is not None and auto_advance:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT,
            message="ack_through requires auto_advance=false",
        )

    try:
        max_outbox = env_int("AGENT_BUS_MAX_OUTBOX", default=50, min_value=0)
        max_message_chars = env_int("AGENT_BUS_MAX_MESSAGE_CHARS", default=65536, min_value=1)
        max_sync_items = env_int("AGENT_BUS_MAX_SYNC_ITEMS", default=200, min_value=1)
        poll_initial_ms = env_int("AGENT_BUS_POLL_INITIAL_MS", default=250, min_value=1)
        poll_max_ms = env_int("AGENT_BUS_POLL_MAX_MS", default=1000, min_value=1)
    except ValueError as e:  # pragma: no cover
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message=str(e))

    if max_items > max_sync_items:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT,
            message=f"max_items must be <= {max_sync_items}",
        )
    if len(outbox) > max_outbox:
        return tool_error(
            code=ErrorCode.INVALID_ARGUMENT,
            message=f"outbox must have at most {max_outbox} items",
        )

    sanitized: list[dict[str, Any]] = []
    for idx, item in enumerate(outbox):
        if not isinstance(item, dict):
            return tool_error(
                code=ErrorCode.INVALID_ARGUMENT, message=f"outbox[{idx}] must be an object"
            )

        content = item.get("content_markdown")
        if not isinstance(content, str) or not content:
            return tool_error(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"outbox[{idx}].content_markdown must be a non-empty string",
            )
        if len(content) > max_message_chars:
            return tool_error(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"outbox[{idx}].content_markdown exceeds max length ({max_message_chars})",
            )

        message_type = item.get("message_type", "message")
        if not isinstance(message_type, str) or not message_type.strip():
            return tool_error(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"outbox[{idx}].message_type must be a non-empty string",
            )
        message_type = message_type.strip()
        if len(message_type) > 32:
            return tool_error(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"outbox[{idx}].message_type must be <= 32 characters",
            )

        reply_to = item.get("reply_to")
        if reply_to is not None and (not isinstance(reply_to, str) or not reply_to):
            return tool_error(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"outbox[{idx}].reply_to must be a non-empty string or null",
            )

        metadata = item.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            return tool_error(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"outbox[{idx}].metadata must be an object",
            )

        client_message_id = item.get("client_message_id")
        if client_message_id is not None and (
            not isinstance(client_message_id, str) or not client_message_id.strip()
        ):
            return tool_error(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"outbox[{idx}].client_message_id must be a non-empty string or null",
            )
        if isinstance(client_message_id, str):
            client_message_id = client_message_id.strip()
            if len(client_message_id) > 128:
                return tool_error(
                    code=ErrorCode.INVALID_ARGUMENT,
                    message=f"outbox[{idx}].client_message_id must be <= 128 characters",
                )

        sanitized.append(
            {
                "content_markdown": content,
                "message_type": message_type,
                "reply_to": reply_to,
                "metadata": metadata,
                "client_message_id": client_message_id,
            }
        )

    sent: list[tuple[Any, bool]] = []
    received = []
    cursor = None
    has_more = False

    try:
        sent, received, cursor, has_more = db.sync_once(
            topic_id=topic_id,
            agent_name=agent_name,
            outbox=sanitized,
            max_items=max_items,
            include_self=include_self,
            auto_advance=auto_advance,
            ack_through=ack_through,
        )
    except SchemaMismatchError as e:
        return _schema_mismatch_result(e)
    except TopicNotFoundError:
        return tool_error(code=ErrorCode.TOPIC_NOT_FOUND, message="Topic not found.")
    except TopicClosedError:
        return tool_error(code=ErrorCode.TOPIC_CLOSED, message="Topic is closed.")
    except DBBusyError:
        return tool_error(code=ErrorCode.DB_BUSY, message="Database is busy.")
    except ValueError as e:
        return tool_error(code=ErrorCode.INVALID_ARGUMENT, message=str(e))

    deadline = time.monotonic() + wait_seconds
    interval_s = poll_initial_ms / 1000.0
    max_interval_s = poll_max_ms / 1000.0

    while not received and wait_seconds > 0 and time.monotonic() < deadline:
        time.sleep(interval_s)
        interval_s = min(interval_s * 2, max_interval_s)
        try:
            _, received, cursor, has_more = db.sync_once(
                topic_id=topic_id,
                agent_name=agent_name,
                outbox=[],
                max_items=max_items,
                include_self=include_self,
                auto_advance=auto_advance,
                ack_through=None,
            )
        except DBBusyError:
            continue
        except SchemaMismatchError as e:
            return _schema_mismatch_result(e)
        except TopicNotFoundError:
            return tool_error(code=ErrorCode.TOPIC_NOT_FOUND, message="Topic not found.")
        except ValueError as e:
            return tool_error(code=ErrorCode.INVALID_ARGUMENT, message=str(e))

    status: Literal["ready", "timeout", "empty"]
    if received:
        status = "ready"
    elif wait_seconds > 0:
        status = "timeout"
    else:
        status = "empty"

    def _msg_struct(m: Any) -> dict[str, Any]:
        return {
            "message_id": m.message_id,
            "topic_id": m.topic_id,
            "seq": m.seq,
            "sender": m.sender,
            "message_type": m.message_type,
            "reply_to": m.reply_to,
            "content_markdown": m.content_markdown,
            "metadata": m.metadata,
            "client_message_id": m.client_message_id,
            "created_at": m.created_at,
        }

    structured_sent = [{"message": _msg_struct(m), "duplicate": dup} for m, dup in sent]
    structured_received = [_msg_struct(m) for m in received]

    assert cursor is not None
    structured = {
        "topic_id": topic_id,
        "agent_name": agent_name,
        "status": status,
        "cursor": {"last_seq": cursor.last_seq, "updated_at": cursor.updated_at},
        "sent": structured_sent,
        "received": structured_received,
        "received_count": len(structured_received),
        "has_more": has_more,
    }

    lines = [
        f"Sync: status={status} received={len(structured_received)} sent={len(structured_sent)} cursor={cursor.last_seq}"
    ]
    for m in structured_received[:20]:
        preview = m["content_markdown"].splitlines()[0][:80]
        lines.append(f"[{m['seq']}] {m['sender']}: {preview}")
    if len(structured_received) > 20:
        lines.append(f"... ({len(structured_received) - 20} more)")

    return tool_ok(text="\n".join(lines), structured=structured)


def main() -> None:
    mcp.run(transport="stdio")
