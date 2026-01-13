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
class Question:
    question_id: str
    topic_id: str
    asked_by: str
    question_text: str
    asked_at: float
    status: str
    cancel_reason: str | None


@dataclass(frozen=True, slots=True)
class Answer:
    answer_id: str
    topic_id: str
    question_id: str
    answered_by: str
    answered_at: float
    payload: dict[str, Any]
