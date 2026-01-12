from __future__ import annotations

import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast

from agent_bus.common import json_dumps, json_loads, now
from agent_bus.models import Question, Topic

TopicStatus = Literal["open", "closed"]
QuestionStatus = Literal["pending", "answered", "cancelled"]


class DBBusyError(RuntimeError):
    pass


class TopicNotFoundError(RuntimeError):
    pass


class TopicClosedError(RuntimeError):
    pass


class QuestionNotFoundError(RuntimeError):
    pass


class TopicMismatchError(RuntimeError):
    pass


def _default_db_path() -> str:
    return str(Path("~/.agent_bus/agent_bus.sqlite").expanduser())


def _ensure_parent_dir(path: str) -> None:
    p = Path(path)
    if p.name == ":memory:":
        return
    p.parent.mkdir(parents=True, exist_ok=True)


def new_id(*, length: int = 10) -> str:
    return uuid.uuid4().hex[:length]


class AgentBusDB:
    def __init__(self, *, path: str | None = None) -> None:
        self.path = path or os.environ.get("AGENT_BUS_DB") or _default_db_path()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        _ensure_parent_dir(self.path)
        try:
            conn = sqlite3.connect(self.path, timeout=2.0)
        except sqlite3.OperationalError as e:  # pragma: no cover
            msg = str(e).lower()
            if "locked" in msg or "busy" in msg:
                raise DBBusyError(str(e)) from e
            raise
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=2000;")
        self._ensure_schema(conn)
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS topics (
              topic_id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              created_at REAL NOT NULL,
              status TEXT NOT NULL,
              closed_at REAL NULL,
              close_reason TEXT NULL,
              metadata_json TEXT NULL
            );

            CREATE TABLE IF NOT EXISTS questions (
              question_id TEXT PRIMARY KEY,
              topic_id TEXT NOT NULL,
              asked_by TEXT NOT NULL,
              question_text TEXT NOT NULL,
              asked_at REAL NOT NULL,
              status TEXT NOT NULL,
              cancel_reason TEXT NULL,
              answered_at REAL NULL,
              answered_by TEXT NULL,
              answer_payload_json TEXT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_topics_name_status_created_at
              ON topics(name, status, created_at);

            CREATE INDEX IF NOT EXISTS idx_questions_topic_status_askedat
              ON questions(topic_id, status, asked_at);
            """
        )

    def get_topic(self, *, topic_id: str) -> Topic:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT topic_id, name, status, created_at, closed_at, close_reason, metadata_json
                FROM topics
                WHERE topic_id = ?
                """,
                (topic_id,),
            ).fetchone()
            if row is None:
                raise TopicNotFoundError(topic_id)
            return _topic_from_row(row)

    def topic_create(
        self,
        *,
        name: str | None,
        metadata: dict[str, Any] | None,
        mode: Literal["reuse", "new"],
    ) -> Topic:
        topic_id = new_id()
        topic_name = name or f"topic-{topic_id}"
        created_at = now()
        metadata_json = json_dumps(metadata) if metadata is not None else None

        with self.connect() as conn:
            if mode == "reuse":
                row = conn.execute(
                    """
                    SELECT topic_id, name, status, created_at, closed_at, close_reason, metadata_json
                    FROM topics
                    WHERE name = ? AND status = 'open'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (topic_name,),
                ).fetchone()
                if row is not None:
                    return _topic_from_row(row)

            with conn:
                conn.execute(
                    """
                    INSERT INTO topics(topic_id, name, created_at, status, closed_at, close_reason, metadata_json)
                    VALUES (?, ?, ?, 'open', NULL, NULL, ?)
                    """,
                    (topic_id, topic_name, created_at, metadata_json),
                )
            return Topic(
                topic_id=topic_id,
                name=topic_name,
                status="open",
                created_at=created_at,
                closed_at=None,
                close_reason=None,
                metadata=metadata,
            )

    def topic_list(self, *, status: Literal["open", "closed", "all"]) -> list[Topic]:
        where = ""
        params: tuple[Any, ...] = ()
        if status in ("open", "closed"):
            where = "WHERE status = ?"
            params = (status,)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT topic_id, name, status, created_at, closed_at, close_reason, metadata_json
                FROM topics
                {where}
                ORDER BY created_at DESC
                """,
                params,
            ).fetchall()
        return [_topic_from_row(r) for r in rows]

    def topic_close(self, *, topic_id: str, reason: str | None) -> tuple[Topic, bool]:
        with self.connect() as conn, conn:
            row = conn.execute(
                """
                SELECT topic_id, name, status, created_at, closed_at, close_reason, metadata_json
                FROM topics
                WHERE topic_id = ?
                """,
                (topic_id,),
            ).fetchone()
            if row is None:
                raise TopicNotFoundError(topic_id)

            existing = _topic_from_row(row)
            if existing.status == "closed":
                return existing, True

            closed_at = now()
            close_reason = reason if reason else None
            conn.execute(
                """
                UPDATE topics
                SET status = 'closed', closed_at = ?, close_reason = ?
                WHERE topic_id = ?
                """,
                (closed_at, close_reason, topic_id),
            )
            updated = Topic(
                topic_id=existing.topic_id,
                name=existing.name,
                status="closed",
                created_at=existing.created_at,
                closed_at=closed_at,
                close_reason=close_reason,
                metadata=existing.metadata,
            )
            return updated, False

    def topic_resolve(self, *, name: str, allow_closed: bool) -> Topic:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT topic_id, name, status, created_at, closed_at, close_reason, metadata_json
                FROM topics
                WHERE name = ? AND status = 'open'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (name,),
            ).fetchone()
            if row is not None:
                return _topic_from_row(row)

            if not allow_closed:
                raise TopicNotFoundError(name)

            row = conn.execute(
                """
                SELECT topic_id, name, status, created_at, closed_at, close_reason, metadata_json
                FROM topics
                WHERE name = ? AND status = 'closed'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (name,),
            ).fetchone()
            if row is None:
                raise TopicNotFoundError(name)
            return _topic_from_row(row)

    def question_create(self, *, topic_id: str, asked_by: str, question_text: str) -> Question:
        question_id = new_id()
        asked_at = now()
        with self.connect() as conn, conn:
            row = conn.execute(
                """
                SELECT status
                FROM topics
                WHERE topic_id = ?
                """,
                (topic_id,),
            ).fetchone()
            if row is None:
                raise TopicNotFoundError(topic_id)
            if row["status"] != "open":
                raise TopicClosedError(topic_id)

            conn.execute(
                """
                INSERT INTO questions(
                  question_id, topic_id, asked_by, question_text, asked_at,
                  status, cancel_reason, answered_at, answered_by, answer_payload_json
                )
                VALUES (?, ?, ?, ?, ?, 'pending', NULL, NULL, NULL, NULL)
                """,
                (question_id, topic_id, asked_by, question_text, asked_at),
            )

            return Question(
                question_id=question_id,
                topic_id=topic_id,
                asked_by=asked_by,
                question_text=question_text,
                asked_at=asked_at,
                status="pending",
                cancel_reason=None,
                answered_at=None,
                answered_by=None,
                answer_payload=None,
            )

    def teacher_drain(self, *, topic_id: str, limit: int) -> list[Question]:
        with self.connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM topics WHERE topic_id = ?",
                (topic_id,),
            ).fetchone()
            if exists is None:
                raise TopicNotFoundError(topic_id)
            rows = conn.execute(
                """
                SELECT question_id, topic_id, asked_by, question_text, asked_at, status,
                       cancel_reason, answered_at, answered_by, answer_payload_json
                FROM questions
                WHERE topic_id = ? AND status = 'pending'
                ORDER BY asked_at ASC
                LIMIT ?
                """,
                (topic_id, limit),
            ).fetchall()
        return [_question_from_row(r) for r in rows]

    def question_get(self, *, question_id: str) -> Question:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT question_id, topic_id, asked_by, question_text, asked_at, status,
                       cancel_reason, answered_at, answered_by, answer_payload_json
                FROM questions
                WHERE question_id = ?
                """,
                (question_id,),
            ).fetchone()
            if row is None:
                raise QuestionNotFoundError(question_id)
            return _question_from_row(row)

    def question_cancel(
        self, *, topic_id: str, question_id: str, reason: str | None
    ) -> tuple[Question, bool]:
        with self.connect() as conn, conn:
            row = conn.execute(
                """
                SELECT question_id, topic_id, asked_by, question_text, asked_at, status,
                       cancel_reason, answered_at, answered_by, answer_payload_json
                FROM questions
                WHERE question_id = ?
                """,
                (question_id,),
            ).fetchone()
            if row is None:
                raise QuestionNotFoundError(question_id)
            q = _question_from_row(row)
            if q.topic_id != topic_id:
                raise TopicMismatchError(question_id)
            if q.status == "answered":
                raise ValueError("Cannot cancel answered question")
            if q.status == "cancelled":
                return q, True

            cancel_reason = reason if reason else None
            conn.execute(
                """
                UPDATE questions
                SET status = 'cancelled', cancel_reason = ?
                WHERE question_id = ?
                """,
                (cancel_reason, question_id),
            )
            updated = Question(
                question_id=q.question_id,
                topic_id=q.topic_id,
                asked_by=q.asked_by,
                question_text=q.question_text,
                asked_at=q.asked_at,
                status="cancelled",
                cancel_reason=cancel_reason,
                answered_at=q.answered_at,
                answered_by=q.answered_by,
                answer_payload=q.answer_payload,
            )
            return updated, False

    def teacher_publish_one(
        self,
        *,
        topic_id: str,
        question_id: str,
        answered_by: str,
        answer_payload: dict[str, Any],
    ) -> bool:
        answer_payload_json = json_dumps(answer_payload)
        answered_at = now()
        with self.connect() as conn, conn:
            exists = conn.execute(
                "SELECT 1 FROM topics WHERE topic_id = ?",
                (topic_id,),
            ).fetchone()
            if exists is None:
                raise TopicNotFoundError(topic_id)
            cur = conn.execute(
                """
                UPDATE questions
                SET status = 'answered', answered_at = ?, answered_by = ?, answer_payload_json = ?
                WHERE question_id = ? AND topic_id = ? AND status = 'pending'
                """,
                (answered_at, answered_by, answer_payload_json, question_id, topic_id),
            )
            return cast(int, cur.rowcount) == 1

    def teacher_publish_batch(
        self,
        *,
        topic_id: str,
        answered_by: str,
        items: list[tuple[str, dict[str, Any]]],
    ) -> tuple[int, int]:
        answered_at = now()
        with self.connect() as conn, conn:
            exists = conn.execute(
                "SELECT 1 FROM topics WHERE topic_id = ?",
                (topic_id,),
            ).fetchone()
            if exists is None:
                raise TopicNotFoundError(topic_id)

            saved = 0
            skipped = 0
            for question_id, payload in items:
                payload_json = json_dumps(payload)
                cur = conn.execute(
                    """
                    UPDATE questions
                    SET status = 'answered', answered_at = ?, answered_by = ?, answer_payload_json = ?
                    WHERE question_id = ? AND topic_id = ? AND status = 'pending'
                    """,
                    (answered_at, answered_by, payload_json, question_id, topic_id),
                )
                if cast(int, cur.rowcount) == 1:
                    saved += 1
                else:
                    skipped += 1
            return saved, skipped


def _topic_from_row(row: sqlite3.Row) -> Topic:
    metadata_json = row["metadata_json"]
    metadata = None if metadata_json is None else cast(dict[str, Any], json_loads(metadata_json))
    return Topic(
        topic_id=row["topic_id"],
        name=row["name"],
        status=row["status"],
        created_at=row["created_at"],
        closed_at=row["closed_at"],
        close_reason=row["close_reason"],
        metadata=metadata,
    )


def _question_from_row(row: sqlite3.Row) -> Question:
    payload_json = row["answer_payload_json"]
    payload = None if payload_json is None else cast(dict[str, Any], json_loads(payload_json))
    return Question(
        question_id=row["question_id"],
        topic_id=row["topic_id"],
        asked_by=row["asked_by"],
        question_text=row["question_text"],
        asked_at=row["asked_at"],
        status=row["status"],
        cancel_reason=row["cancel_reason"],
        answered_at=row["answered_at"],
        answered_by=row["answered_by"],
        answer_payload=payload,
    )
