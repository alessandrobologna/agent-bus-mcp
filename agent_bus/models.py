from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Topic:
    topic_id: str
    name: str
    status: str
    created_at: float
    closed_at: float | None
    close_reason: str | None
    metadata: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class Message:
    message_id: str
    topic_id: str
    seq: int
    sender: str
    message_type: str
    reply_to: str | None
    content_markdown: str
    metadata: dict[str, Any] | None
    client_message_id: str | None
    created_at: float


@dataclass(frozen=True, slots=True)
class Cursor:
    topic_id: str
    agent_name: str
    last_seq: int
    updated_at: float
