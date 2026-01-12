from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from mcp.types import CallToolResult, TextContent


class ErrorCode(StrEnum):
    TOPIC_NOT_FOUND = "TOPIC_NOT_FOUND"
    TOPIC_CLOSED = "TOPIC_CLOSED"
    QUESTION_NOT_FOUND = "QUESTION_NOT_FOUND"
    TOPIC_MISMATCH = "TOPIC_MISMATCH"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    DB_BUSY = "DB_BUSY"


class WarningCode(StrEnum):
    ALREADY_CLOSED = "ALREADY_CLOSED"
    ALREADY_CANCELLED = "ALREADY_CANCELLED"
    REPO_POINTERS_TRUNCATED = "REPO_POINTERS_TRUNCATED"
    FOLLOWUPS_TRUNCATED = "FOLLOWUPS_TRUNCATED"


@dataclass(frozen=True, slots=True)
class ToolWarning:
    code: str
    message: str | None = None
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code}
        if self.message is not None:
            out["message"] = self.message
        if self.context is not None:
            out["context"] = self.context
        return out


def now() -> float:
    return time.time()


def env_int(name: str, *, default: int, min_value: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as e:  # pragma: no cover
            raise ValueError(f"{name} must be an int") from e
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    return value


def env_str(name: str, *, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def json_loads(data: str) -> Any:
    return json.loads(data)


def tool_ok(
    *,
    text: str,
    structured: dict[str, Any] | None = None,
    warnings: list[ToolWarning] | None = None,
) -> CallToolResult:
    payload: dict[str, Any] = {} if structured is None else dict(structured)
    if warnings:
        payload["warnings"] = [w.to_dict() for w in warnings]
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent=payload,
    )


def tool_error(
    *,
    code: ErrorCode,
    message: str,
    structured: dict[str, Any] | None = None,
) -> CallToolResult:
    payload: dict[str, Any] = {"error": {"code": str(code), "message": message}}
    if structured:
        payload.update(structured)
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        structuredContent=payload,
        isError=True,
    )


def truncate_list(
    items: list[Any],
    *,
    max_items: int,
    warning_code: WarningCode,
    context: dict[str, Any] | None = None,
) -> tuple[list[Any], ToolWarning | None]:
    if len(items) <= max_items:
        return items, None
    truncated = items[:max_items]
    warn = ToolWarning(
        code=str(warning_code),
        context={"original_count": len(items), "kept_count": len(truncated), **(context or {})},
    )
    return truncated, warn
