from __future__ import annotations

import time
from typing import Any, Literal

from agent_bus._core import (  # type: ignore[import-not-found]
    CoreDb,
    DBBusyError,
    SchemaMismatchError,
    TopicClosedError,
    TopicMismatchError,
    TopicNotFoundError,
)
from agent_bus.common import json_dumps, json_loads
from agent_bus.models import Cursor, Message, Topic

TopicStatus = Literal["open", "closed"]

SCHEMA_VERSION = "6"

__all__ = [
    "AgentBusDB",
    "DBBusyError",
    "SchemaMismatchError",
    "TopicClosedError",
    "TopicMismatchError",
    "TopicNotFoundError",
]


class AgentBusDB:
    def __init__(self, *, path: str | None = None) -> None:
        self._core = CoreDb(path)

    @property
    def path(self) -> str:
        return self._core.path

    @property
    def fts_available(self) -> bool:
        return bool(self._core.fts_available)

    def search_messages_fts(
        self,
        *,
        query: str,
        topic_id: str | None = None,
        sender: str | None = None,
        message_type: str | None = None,
        limit: int = 20,
        include_content: bool = False,
    ) -> list[dict[str, Any]]:
        return self._core.search_messages_fts(
            query, topic_id, sender, message_type, limit, include_content
        )

    def get_message_by_id(self, *, message_id: str) -> Message:
        data = self._core.get_message_by_id(message_id)
        return _message_from_dict(data)

    def upsert_embeddings(
        self,
        *,
        message_id: str,
        model: str,
        topic_id: str,
        content_hash: str,
        chunk_size: int,
        chunk_overlap: int,
        dims: int,
        chunks: list[dict[str, Any]],
    ) -> None:
        self._core.upsert_embeddings(
            message_id,
            model,
            topic_id,
            content_hash,
            chunk_size,
            chunk_overlap,
            dims,
            chunks,
        )

    def get_embedding_state(
        self,
        *,
        message_id: str,
        model: str,
    ) -> dict[str, Any] | None:
        return self._core.get_embedding_state(message_id, model)

    def list_messages_for_embedding(
        self,
        *,
        topic_id: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        return self._core.list_messages_for_embedding(topic_id, limit)

    def enqueue_embedding_jobs(
        self,
        *,
        jobs: list[tuple[str, str]],
        model: str,
    ) -> int:
        return int(self._core.enqueue_embedding_jobs(jobs, model))

    def claim_embedding_jobs(
        self,
        *,
        model: str,
        limit: int,
        worker_id: str,
        lock_ttl_seconds: int,
        error_retry_seconds: int,
        max_attempts: int,
    ) -> list[dict[str, Any]]:
        return self._core.claim_embedding_jobs(
            model,
            limit,
            worker_id,
            lock_ttl_seconds,
            error_retry_seconds,
            max_attempts,
        )

    def complete_embedding_job(self, *, message_id: str, model: str) -> None:
        self._core.complete_embedding_job(message_id, model)

    def fail_embedding_job(self, *, message_id: str, model: str, error: str) -> None:
        self._core.fail_embedding_job(message_id, model, error)

    def list_chunk_embedding_candidates(
        self,
        *,
        model: str,
        topic_id: str | None = None,
        message_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return self._core.list_chunk_embedding_candidates(model, topic_id, message_ids)

    def get_topic(self, *, topic_id: str) -> Topic:
        data = self._core.get_topic(topic_id)
        return _topic_from_dict(data)

    def topic_create(
        self,
        *,
        name: str | None,
        metadata: dict[str, Any] | None,
        mode: Literal["reuse", "new"],
    ) -> Topic:
        metadata_json = json_dumps(metadata) if metadata is not None else None
        data = self._core.topic_create(name, metadata_json, mode, now())
        return _topic_from_dict(data)

    def topic_list(self, *, status: Literal["open", "closed", "all"]) -> list[Topic]:
        items = self._core.topic_list(status)
        return [_topic_from_dict(i) for i in items]

    def topic_list_with_counts(
        self, *, status: Literal["open", "closed", "all"], limit: int
    ) -> list[dict[str, Any]]:
        rows = self._core.topic_list_with_counts(status, limit)
        out: list[dict[str, Any]] = []
        for r in rows:
            metadata_json = r.pop("metadata_json", None)
            metadata = None if metadata_json is None else json_loads(metadata_json)
            r["metadata"] = metadata
            out.append(r)
        return out

    def topic_close(self, *, topic_id: str, reason: str | None) -> tuple[Topic, bool]:
        data, already = self._core.topic_close(topic_id, reason)
        return _topic_from_dict(data), bool(already)

    def delete_topic(self, *, topic_id: str) -> bool:
        return bool(self._core.delete_topic(topic_id))

    def topic_rename(
        self,
        *,
        topic_id: str,
        new_name: str,
        rewrite_messages: bool = True,
    ) -> tuple[Topic, bool, int]:
        data, unchanged, rewritten = self._core.topic_rename(topic_id, new_name, rewrite_messages)
        return _topic_from_dict(data), bool(unchanged), int(rewritten)

    def delete_message(self, *, message_id: str) -> int:
        return int(self._core.delete_message(message_id))

    def delete_messages_batch(self, *, topic_id: str, message_ids: list[str]) -> list[str]:
        return list(self._core.delete_messages_batch(topic_id, message_ids))

    def topic_resolve(self, *, name: str, allow_closed: bool) -> Topic:
        data = self._core.topic_resolve(name, allow_closed)
        return _topic_from_dict(data)

    def sync_once(
        self,
        *,
        topic_id: str,
        agent_name: str,
        outbox: list[dict[str, Any]],
        max_items: int,
        include_self: bool,
        auto_advance: bool,
        ack_through: int | None,
    ) -> tuple[list[tuple[Message, bool]], list[Message], Cursor, bool]:
        payload = _serialize_outbox(outbox)
        sent, received, cursor, has_more = self._core.sync_once(
            topic_id,
            agent_name,
            payload,
            max_items,
            include_self,
            auto_advance,
            ack_through,
            now(),
        )
        sent_out = [(_message_from_dict(m), bool(is_dup)) for m, is_dup in sent]
        received_out = [_message_from_dict(m) for m in received]
        return sent_out, received_out, _cursor_from_dict(cursor), bool(has_more)

    def cursor_set(self, *, topic_id: str, agent_name: str, last_seq: int) -> Cursor:
        data = self._core.cursor_set(topic_id, agent_name, last_seq)
        return _cursor_from_dict(data)

    def get_presence(
        self,
        *,
        topic_id: str,
        window_seconds: int = 300,
        limit: int = 200,
    ) -> list[Cursor]:
        rows = self._core.get_presence(topic_id, window_seconds, limit, now())
        return [_cursor_from_dict(r) for r in rows]

    def get_messages(
        self,
        *,
        topic_id: str,
        after_seq: int = 0,
        before_seq: int | None = None,
        limit: int = 100,
    ) -> list[Message]:
        rows = self._core.get_messages(topic_id, after_seq, before_seq, limit)
        return [_message_from_dict(r) for r in rows]

    def get_latest_messages(
        self,
        *,
        topic_id: str,
        limit: int = 10,
    ) -> list[Message]:
        rows = self._core.get_latest_messages(topic_id, limit)
        return [_message_from_dict(r) for r in rows]

    def get_senders_by_message_ids(self, message_ids: list[str]) -> dict[str, str]:
        return dict(self._core.get_senders_by_message_ids(message_ids))


def _serialize_outbox(outbox: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in outbox:
        metadata = item.get("metadata")
        metadata_json = json_dumps(metadata) if metadata is not None else None
        payload.append(
            {
                "content_markdown": item["content_markdown"],
                "message_type": item["message_type"],
                "reply_to": item.get("reply_to"),
                "metadata_json": metadata_json,
                "client_message_id": item.get("client_message_id"),
            }
        )
    return payload


def _topic_from_dict(data: dict[str, Any]) -> Topic:
    metadata_json = data.get("metadata_json")
    metadata = None if metadata_json is None else json_loads(metadata_json)
    return Topic(
        topic_id=data["topic_id"],
        name=data["name"],
        status=data["status"],
        created_at=data["created_at"],
        closed_at=data["closed_at"],
        close_reason=data["close_reason"],
        metadata=metadata,
    )


def _message_from_dict(data: dict[str, Any]) -> Message:
    metadata_json = data.get("metadata_json")
    metadata = None if metadata_json is None else json_loads(metadata_json)
    return Message(
        message_id=data["message_id"],
        topic_id=data["topic_id"],
        seq=int(data["seq"]),
        sender=data["sender"],
        message_type=data["message_type"],
        reply_to=data["reply_to"],
        content_markdown=data["content_markdown"],
        metadata=metadata,
        client_message_id=data["client_message_id"],
        created_at=data["created_at"],
    )


def _cursor_from_dict(data: dict[str, Any]) -> Cursor:
    return Cursor(
        topic_id=data["topic_id"],
        agent_name=data["agent_name"],
        last_seq=int(data["last_seq"]),
        updated_at=data["updated_at"],
    )


def now() -> float:
    return time.time()
