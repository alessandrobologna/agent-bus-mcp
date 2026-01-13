from __future__ import annotations

import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast

from agent_bus.common import json_dumps, json_loads, now
from agent_bus.models import Answer, Question, Topic

TopicStatus = Literal["open", "closed"]
QuestionStatus = Literal["pending", "answered", "cancelled"]

SCHEMA_VERSION = "5"


class DBBusyError(RuntimeError):
    pass


class SchemaMismatchError(RuntimeError):
    pass


class TopicNotFoundError(RuntimeError):
    pass


class TopicClosedError(RuntimeError):
    pass


class QuestionNotFoundError(RuntimeError):
    pass


class TopicMismatchError(RuntimeError):
    pass


class SelfAnswerForbiddenError(RuntimeError):
    def __init__(self, question_ids: list[str]) -> None:
        self.question_ids = question_ids
        super().__init__("Cannot answer own question(s)")


class AlreadyAnsweredError(RuntimeError):
    def __init__(self, question_ids: list[str]) -> None:
        self.question_ids = question_ids
        super().__init__("Already answered question(s)")


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
        raw_path = path or os.environ.get("AGENT_BUS_DB") or _default_db_path()
        if raw_path != ":memory:":
            raw_path = str(Path(raw_path).expanduser())
        self.path = raw_path

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
        tables = {
            cast(str, r["name"])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
            ).fetchall()
        }
        if "meta" not in tables:
            if tables:
                raise SchemaMismatchError(
                    "Database schema is outdated (missing schema version). "
                    "Wipe it with `agent-bus cli db wipe --yes` or delete the file at $AGENT_BUS_DB."
                )
            conn.executescript(
                f"""
                CREATE TABLE meta (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL
                );

                INSERT INTO meta(key, value)
                VALUES ('schema_version', '{SCHEMA_VERSION}');
                """
            )
        else:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'",
            ).fetchone()
            if row is None or cast(str, row["value"]) != SCHEMA_VERSION:
                raise SchemaMismatchError(
                    "Database schema version mismatch. "
                    "Wipe it with `agent-bus cli db wipe --yes` or delete the file at $AGENT_BUS_DB."
                )

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
              cancel_reason TEXT NULL
            );

            CREATE TABLE IF NOT EXISTS answers (
              answer_id TEXT PRIMARY KEY,
              topic_id TEXT NOT NULL,
              question_id TEXT NOT NULL,
              answered_by TEXT NOT NULL,
              answered_at REAL NOT NULL,
              payload_json TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_answers_question_answered_by_unique
              ON answers(question_id, answered_by);

            CREATE INDEX IF NOT EXISTS idx_topics_name_status_created_at
              ON topics(name, status, created_at);

            CREATE INDEX IF NOT EXISTS idx_questions_topic_status_askedat
              ON questions(topic_id, status, asked_at);

            CREATE INDEX IF NOT EXISTS idx_answers_question_answered_at
              ON answers(question_id, answered_at);
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

    def topic_list_with_counts(
        self, *, status: Literal["open", "closed", "all"], limit: int
    ) -> list[dict[str, Any]]:
        where = ""
        params: list[Any] = []
        if status in ("open", "closed"):
            where = "WHERE t.status = ?"
            params.append(status)
        params.append(limit)

        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                  t.topic_id,
                  t.name,
                  t.status,
                  t.created_at,
                  t.closed_at,
                  t.close_reason,
                  t.metadata_json,
                  COUNT(q.question_id) AS total_count,
                  COALESCE(SUM(CASE WHEN q.status = 'pending' THEN 1 ELSE 0 END), 0) AS pending_count,
                  COALESCE(SUM(CASE WHEN q.status = 'answered' THEN 1 ELSE 0 END), 0) AS answered_count,
                  COALESCE(SUM(CASE WHEN q.status = 'cancelled' THEN 1 ELSE 0 END), 0) AS cancelled_count
                FROM topics t
                LEFT JOIN questions q ON q.topic_id = t.topic_id
                {where}
                GROUP BY t.topic_id
                ORDER BY t.created_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()

        out: list[dict[str, Any]] = []
        for r in rows:
            metadata_json = r["metadata_json"]
            metadata = (
                None if metadata_json is None else cast(dict[str, Any], json_loads(metadata_json))
            )
            out.append(
                {
                    "topic_id": r["topic_id"],
                    "name": r["name"],
                    "status": r["status"],
                    "created_at": r["created_at"],
                    "closed_at": r["closed_at"],
                    "close_reason": r["close_reason"],
                    "metadata": metadata,
                    "counts": {
                        "total": cast(int, r["total_count"]),
                        "pending": cast(int, r["pending_count"]),
                        "answered": cast(int, r["answered_count"]),
                        "cancelled": cast(int, r["cancelled_count"]),
                    },
                }
            )
        return out

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
                  status, cancel_reason
                )
                VALUES (?, ?, ?, ?, ?, 'pending', NULL)
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
            )

    def question_list_answerable(
        self, *, topic_id: str, limit: int, agent_name: str
    ) -> list[Question]:
        with self.connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM topics WHERE topic_id = ?",
                (topic_id,),
            ).fetchone()
            if exists is None:
                raise TopicNotFoundError(topic_id)

            rows = conn.execute(
                """
                SELECT
                  q.question_id,
                  q.topic_id,
                  q.asked_by,
                  q.question_text,
                  q.asked_at,
                  q.status,
                  q.cancel_reason
                FROM questions q
                WHERE
                  q.topic_id = ?
                  AND q.status = 'pending'
                  AND q.asked_by <> ?
                  AND NOT EXISTS (
                    SELECT 1 FROM answers a
                    WHERE a.question_id = q.question_id AND a.answered_by = ?
                  )
                ORDER BY asked_at ASC
                LIMIT ?
                """,
                (topic_id, agent_name, agent_name, limit),
            ).fetchall()
        return [_question_from_row(r) for r in rows]

    def question_get(self, *, question_id: str) -> Question:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT question_id, topic_id, asked_by, question_text, asked_at, status,
                       cancel_reason
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
                       cancel_reason
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
            )
            return updated, False

    def question_mark_answered(self, *, topic_id: str, question_id: str) -> tuple[Question, bool]:
        with self.connect() as conn, conn:
            row = conn.execute(
                """
                SELECT question_id, topic_id, asked_by, question_text, asked_at, status,
                       cancel_reason
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
            if q.status == "cancelled":
                raise ValueError("Cannot mark cancelled question as answered")
            if q.status == "answered":
                return q, True

            conn.execute(
                """
                UPDATE questions
                SET status = 'answered'
                WHERE question_id = ? AND topic_id = ? AND status = 'pending'
                """,
                (question_id, topic_id),
            )
            updated = Question(
                question_id=q.question_id,
                topic_id=q.topic_id,
                asked_by=q.asked_by,
                question_text=q.question_text,
                asked_at=q.asked_at,
                status="answered",
                cancel_reason=q.cancel_reason,
            )
            return updated, False

    def answer_insert_batch(
        self,
        *,
        topic_id: str,
        answered_by: str,
        items: list[tuple[str, dict[str, Any]]],
    ) -> tuple[int, int]:
        answered_at = now()
        if not items:
            return 0, 0

        question_ids = [qid for qid, _ in items]
        placeholders = ",".join("?" for _ in question_ids)

        with self.connect() as conn, conn:
            exists = conn.execute(
                "SELECT 1 FROM topics WHERE topic_id = ?",
                (topic_id,),
            ).fetchone()
            if exists is None:
                raise TopicNotFoundError(topic_id)

            rows = conn.execute(
                f"""
                SELECT question_id, asked_by, status
                FROM questions
                WHERE topic_id = ? AND question_id IN ({placeholders})
                """,
                (topic_id, *question_ids),
            ).fetchall()

            pending_rows = [r for r in rows if cast(str, r["status"]) == "pending"]
            pending_ids = [cast(str, r["question_id"]) for r in pending_rows]

            forbidden = [
                cast(str, r["question_id"])
                for r in pending_rows
                if cast(str, r["asked_by"]) == answered_by
            ]
            if forbidden:
                raise SelfAnswerForbiddenError(forbidden)

            if pending_ids:
                pending_placeholders = ",".join("?" for _ in pending_ids)
                already_rows = conn.execute(
                    f"""
                    SELECT question_id
                    FROM answers
                    WHERE answered_by = ? AND question_id IN ({pending_placeholders})
                    """,
                    (answered_by, *pending_ids),
                ).fetchall()
                already = [cast(str, r["question_id"]) for r in already_rows]
                if already:
                    raise AlreadyAnsweredError(already)

            pending_by_id = {cast(str, r["question_id"]): r for r in pending_rows}

            saved = 0
            skipped = 0
            for question_id, payload in items:
                row = pending_by_id.get(question_id)
                if row is None:
                    skipped += 1
                    continue

                payload_json = json_dumps(payload)
                answer_id = new_id()
                try:
                    conn.execute(
                        """
                        INSERT INTO answers(
                          answer_id, topic_id, question_id, answered_by, answered_at, payload_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (answer_id, topic_id, question_id, answered_by, answered_at, payload_json),
                    )
                except sqlite3.IntegrityError as e:
                    raise AlreadyAnsweredError([question_id]) from e

                saved += 1
            return saved, skipped

    def answers_list(self, *, question_id: str) -> list[Answer]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT answer_id, topic_id, question_id, answered_by, answered_at, payload_json
                FROM answers
                WHERE question_id = ?
                ORDER BY answered_at ASC, answer_id ASC
                """,
                (question_id,),
            ).fetchall()
        return [_answer_from_row(r) for r in rows]


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
    return Question(
        question_id=row["question_id"],
        topic_id=row["topic_id"],
        asked_by=row["asked_by"],
        question_text=row["question_text"],
        asked_at=row["asked_at"],
        status=row["status"],
        cancel_reason=row["cancel_reason"],
    )


def _answer_from_row(row: sqlite3.Row) -> Answer:
    payload = cast(dict[str, Any], json_loads(cast(str, row["payload_json"])))
    return Answer(
        answer_id=row["answer_id"],
        topic_id=row["topic_id"],
        question_id=row["question_id"],
        answered_by=row["answered_by"],
        answered_at=row["answered_at"],
        payload=payload,
    )
