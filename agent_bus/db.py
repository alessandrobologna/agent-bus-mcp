from __future__ import annotations

import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast

from agent_bus.common import json_dumps, json_loads, now
from agent_bus.models import Cursor, Message, Topic

TopicStatus = Literal["open", "closed"]

SCHEMA_VERSION = "6"


class DBBusyError(RuntimeError):
    pass


class SchemaMismatchError(RuntimeError):
    pass


class TopicNotFoundError(RuntimeError):
    pass


class TopicClosedError(RuntimeError):
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

            CREATE TABLE IF NOT EXISTS topic_seq (
              topic_id TEXT PRIMARY KEY,
              next_seq INTEGER NOT NULL,
              updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
              message_id TEXT PRIMARY KEY,
              topic_id TEXT NOT NULL,
              seq INTEGER NOT NULL,
              sender TEXT NOT NULL,
              message_type TEXT NOT NULL,
              reply_to TEXT NULL,
              content_markdown TEXT NOT NULL,
              metadata_json TEXT NULL,
              client_message_id TEXT NULL,
              created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cursors (
              topic_id TEXT NOT NULL,
              agent_name TEXT NOT NULL,
              last_seq INTEGER NOT NULL,
              updated_at REAL NOT NULL,
              PRIMARY KEY(topic_id, agent_name)
            );

            CREATE INDEX IF NOT EXISTS idx_topics_name_status_created_at
              ON topics(name, status, created_at);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_topic_seq_unique
              ON messages(topic_id, seq);

            CREATE INDEX IF NOT EXISTS idx_messages_topic_seq
              ON messages(topic_id, seq);

            CREATE INDEX IF NOT EXISTS idx_messages_topic_reply_to
              ON messages(topic_id, reply_to);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_topic_sender_client_id_unique
              ON messages(topic_id, sender, client_message_id)
              WHERE client_message_id IS NOT NULL;
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
                  COUNT(m.message_id) AS message_count,
                  COALESCE(MAX(m.seq), 0) AS last_seq
                FROM topics t
                LEFT JOIN messages m ON m.topic_id = t.topic_id
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
                        "messages": cast(int, r["message_count"]),
                        "last_seq": cast(int, r["last_seq"]),
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

    def delete_topic(self, *, topic_id: str) -> bool:
        """Delete a topic and all related data (messages, cursors, sequences).

        Returns True if the topic was deleted, False if it didn't exist.
        """
        with self.connect() as conn, conn:
            # Check if topic exists
            row = conn.execute(
                "SELECT topic_id FROM topics WHERE topic_id = ?",
                (topic_id,),
            ).fetchone()
            if row is None:
                return False

            # Delete in order: messages, cursors, topic_seq, topics
            conn.execute("DELETE FROM messages WHERE topic_id = ?", (topic_id,))
            conn.execute("DELETE FROM cursors WHERE topic_id = ?", (topic_id,))
            conn.execute("DELETE FROM topic_seq WHERE topic_id = ?", (topic_id,))
            conn.execute("DELETE FROM topics WHERE topic_id = ?", (topic_id,))
            return True

    def delete_message(self, *, message_id: str) -> int:
        """Delete a message and all its replies (cascade).

        Returns the count of deleted messages.
        """
        with self.connect() as conn, conn:
            # Find the topic_id for this message
            row = conn.execute(
                "SELECT topic_id FROM messages WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if row is None:
                return 0

            topic_id = row["topic_id"]

            # Find all messages in the cascade chain using recursive CTE
            # This finds the message and all messages that reply to it (transitively)
            cascade_ids = conn.execute(
                """
                WITH RECURSIVE cascade(mid) AS (
                    SELECT message_id FROM messages WHERE message_id = ?
                    UNION ALL
                    SELECT m.message_id FROM messages m
                    JOIN cascade c ON m.reply_to = c.mid
                    WHERE m.topic_id = ?
                )
                SELECT mid FROM cascade
                """,
                (message_id, topic_id),
            ).fetchall()

            ids_to_delete = [r["mid"] for r in cascade_ids]
            if not ids_to_delete:
                return 0

            # Delete all messages in the cascade
            placeholders = ",".join("?" for _ in ids_to_delete)
            conn.execute(
                f"DELETE FROM messages WHERE message_id IN ({placeholders})",
                tuple(ids_to_delete),
            )
            return len(ids_to_delete)

    def delete_messages_batch(self, *, topic_id: str, message_ids: list[str]) -> list[str]:
        """Delete multiple messages and their reply chains (cascade) within one topic.

        Returns the list of deleted message IDs (including replies).
        """
        if not message_ids:
            return []

        placeholders = ",".join("?" for _ in message_ids)
        with self.connect() as conn, conn:
            rows = conn.execute(
                f"""
                WITH RECURSIVE cascade(mid) AS (
                    SELECT message_id
                    FROM messages
                    WHERE topic_id = ? AND message_id IN ({placeholders})
                    UNION ALL
                    SELECT m.message_id
                    FROM messages m
                    JOIN cascade c ON m.reply_to = c.mid
                    WHERE m.topic_id = ?
                )
                SELECT mid FROM cascade
                """,
                (topic_id, *message_ids, topic_id),
            ).fetchall()

            ids_to_delete = [r["mid"] for r in rows]
            if not ids_to_delete:
                return []

            delete_placeholders = ",".join("?" for _ in ids_to_delete)
            conn.execute(
                f"DELETE FROM messages WHERE message_id IN ({delete_placeholders})",
                tuple(ids_to_delete),
            )

        return ids_to_delete

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
        """Sync once (no long-polling).

        Returns:
          - sent: list of (message, is_duplicate)
          - received: messages since cursor (subject to include_self + limit)
          - cursor: current server-side cursor after any advancement
          - has_more: whether more messages exist beyond `received`
        """
        updated_at = now()
        created_at = updated_at

        with self.connect() as conn, conn:
            topic_row = conn.execute(
                "SELECT status FROM topics WHERE topic_id = ?",
                (topic_id,),
            ).fetchone()
            if topic_row is None:
                raise TopicNotFoundError(topic_id)
            if outbox and cast(str, topic_row["status"]) != "open":
                raise TopicClosedError(topic_id)

            conn.execute(
                """
                INSERT OR IGNORE INTO cursors(topic_id, agent_name, last_seq, updated_at)
                VALUES (?, ?, 0, ?)
                """,
                (topic_id, agent_name, updated_at),
            )
            cursor_row = conn.execute(
                """
                SELECT topic_id, agent_name, last_seq, updated_at
                FROM cursors
                WHERE topic_id = ? AND agent_name = ?
                """,
                (topic_id, agent_name),
            ).fetchone()
            assert cursor_row is not None  # inserted above
            cursor = _cursor_from_row(cursor_row)

            conn.execute(
                """
                INSERT OR IGNORE INTO topic_seq(topic_id, next_seq, updated_at)
                VALUES (?, 1, ?)
                """,
                (topic_id, updated_at),
            )
            next_seq_row = conn.execute(
                "SELECT next_seq FROM topic_seq WHERE topic_id = ?",
                (topic_id,),
            ).fetchone()
            assert next_seq_row is not None
            next_seq = cast(int, next_seq_row["next_seq"])

            sent: list[tuple[Message, bool]] = []
            for item in outbox:
                client_message_id = item.get("client_message_id")
                if client_message_id is not None:
                    existing = conn.execute(
                        """
                        SELECT
                          message_id, topic_id, seq, sender, message_type, reply_to,
                          content_markdown, metadata_json, client_message_id, created_at
                        FROM messages
                        WHERE topic_id = ? AND sender = ? AND client_message_id = ?
                        """,
                        (topic_id, agent_name, client_message_id),
                    ).fetchone()
                    if existing is not None:
                        sent.append((_message_from_row(existing), True))
                        continue

                message_id = new_id()
                seq = next_seq
                next_seq += 1

                metadata = item.get("metadata")
                metadata_json = json_dumps(metadata) if metadata is not None else None

                try:
                    conn.execute(
                        """
                        INSERT INTO messages(
                          message_id, topic_id, seq, sender, message_type, reply_to,
                          content_markdown, metadata_json, client_message_id, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            message_id,
                            topic_id,
                            seq,
                            agent_name,
                            item["message_type"],
                            item.get("reply_to"),
                            item["content_markdown"],
                            metadata_json,
                            client_message_id,
                            created_at,
                        ),
                    )
                except sqlite3.IntegrityError:
                    if client_message_id is None:
                        raise
                    existing = conn.execute(
                        """
                        SELECT
                          message_id, topic_id, seq, sender, message_type, reply_to,
                          content_markdown, metadata_json, client_message_id, created_at
                        FROM messages
                        WHERE topic_id = ? AND sender = ? AND client_message_id = ?
                        """,
                        (topic_id, agent_name, client_message_id),
                    ).fetchone()
                    assert existing is not None
                    sent.append((_message_from_row(existing), True))
                    continue

                msg = Message(
                    message_id=message_id,
                    topic_id=topic_id,
                    seq=seq,
                    sender=agent_name,
                    message_type=item["message_type"],
                    reply_to=item.get("reply_to"),
                    content_markdown=item["content_markdown"],
                    metadata=metadata,
                    client_message_id=client_message_id,
                    created_at=created_at,
                )
                sent.append((msg, False))

            conn.execute(
                """
                UPDATE topic_seq
                SET next_seq = ?, updated_at = ?
                WHERE topic_id = ?
                """,
                (next_seq, updated_at, topic_id),
            )

            if not auto_advance and ack_through is not None:
                if ack_through < 0:
                    raise ValueError("ack_through must be >= 0")
                max_seq = next_seq - 1
                if ack_through > max_seq:
                    raise ValueError("ack_through exceeds latest message seq")

                if ack_through > cursor.last_seq:
                    conn.execute(
                        """
                        UPDATE cursors
                        SET last_seq = ?, updated_at = ?
                        WHERE topic_id = ? AND agent_name = ?
                        """,
                        (ack_through, updated_at, topic_id, agent_name),
                    )
                    cursor = Cursor(
                        topic_id=cursor.topic_id,
                        agent_name=cursor.agent_name,
                        last_seq=ack_through,
                        updated_at=updated_at,
                    )

            params: list[Any] = [topic_id, cursor.last_seq]
            where = "topic_id = ? AND seq > ?"
            if not include_self:
                where += " AND sender <> ?"
                params.append(agent_name)
            params.append(max_items + 1)

            rows = conn.execute(
                f"""
                SELECT
                  message_id, topic_id, seq, sender, message_type, reply_to,
                  content_markdown, metadata_json, client_message_id, created_at
                FROM messages
                WHERE {where}
                ORDER BY seq ASC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()

            has_more = len(rows) > max_items
            visible = rows[:max_items]
            received = [_message_from_row(r) for r in visible]

            if auto_advance and received:
                new_last_seq = max(m.seq for m in received)
                if new_last_seq > cursor.last_seq:
                    conn.execute(
                        """
                        UPDATE cursors
                        SET last_seq = ?, updated_at = ?
                        WHERE topic_id = ? AND agent_name = ?
                        """,
                        (new_last_seq, updated_at, topic_id, agent_name),
                    )
                    cursor = Cursor(
                        topic_id=cursor.topic_id,
                        agent_name=cursor.agent_name,
                        last_seq=new_last_seq,
                        updated_at=updated_at,
                    )

            if cursor.updated_at != updated_at:
                conn.execute(
                    """
                    UPDATE cursors
                    SET updated_at = ?
                    WHERE topic_id = ? AND agent_name = ?
                    """,
                    (updated_at, topic_id, agent_name),
                )
                cursor = Cursor(
                    topic_id=cursor.topic_id,
                    agent_name=cursor.agent_name,
                    last_seq=cursor.last_seq,
                    updated_at=updated_at,
                )

            return sent, received, cursor, has_more

    def get_presence(
        self,
        *,
        topic_id: str,
        window_seconds: int = 300,
        limit: int = 200,
    ) -> list[Cursor]:
        """Return cursors active within the given time window (seconds)."""
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if limit <= 0:
            raise ValueError("limit must be > 0")

        cutoff = now() - window_seconds
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT 1 FROM topics WHERE topic_id = ?",
                (topic_id,),
            ).fetchone()
            if existing is None:
                raise TopicNotFoundError(topic_id)

            rows = conn.execute(
                """
                SELECT topic_id, agent_name, last_seq, updated_at
                FROM cursors
                WHERE topic_id = ? AND updated_at >= ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (topic_id, cutoff, limit),
            ).fetchall()
        return [_cursor_from_row(r) for r in rows]

    def get_messages(
        self,
        *,
        topic_id: str,
        after_seq: int = 0,
        limit: int = 100,
    ) -> list[Message]:
        """Fetch messages from a topic after a given sequence number."""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  message_id, topic_id, seq, sender, message_type, reply_to,
                  content_markdown, metadata_json, client_message_id, created_at
                FROM messages
                WHERE topic_id = ? AND seq > ?
                ORDER BY seq ASC
                LIMIT ?
                """,
                (topic_id, after_seq, limit),
            ).fetchall()
        return [_message_from_row(r) for r in rows]

    def get_latest_messages(
        self,
        *,
        topic_id: str,
        limit: int = 10,
    ) -> list[Message]:
        """Fetch the last N messages from a topic efficiently."""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  message_id, topic_id, seq, sender, message_type, reply_to,
                  content_markdown, metadata_json, client_message_id, created_at
                FROM messages
                WHERE topic_id = ?
                ORDER BY seq DESC
                LIMIT ?
                """,
                (topic_id, limit),
            ).fetchall()
        # Return in chronological order
        return [_message_from_row(r) for r in reversed(rows)]

    def get_senders_by_message_ids(self, message_ids: list[str]) -> dict[str, str]:
        """Lookup sender names for a list of message IDs.

        Returns a dict mapping message_id -> sender.
        """
        if not message_ids:
            return {}
        placeholders = ",".join("?" for _ in message_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT message_id, sender
                FROM messages
                WHERE message_id IN ({placeholders})
                """,
                tuple(message_ids),
            ).fetchall()
        return {row["message_id"]: row["sender"] for row in rows}


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


def _message_from_row(row: sqlite3.Row) -> Message:
    metadata_json = row["metadata_json"]
    metadata = None if metadata_json is None else cast(dict[str, Any], json_loads(metadata_json))
    return Message(
        message_id=row["message_id"],
        topic_id=row["topic_id"],
        seq=cast(int, row["seq"]),
        sender=row["sender"],
        message_type=row["message_type"],
        reply_to=row["reply_to"],
        content_markdown=row["content_markdown"],
        metadata=metadata,
        client_message_id=row["client_message_id"],
        created_at=row["created_at"],
    )


def _cursor_from_row(row: sqlite3.Row) -> Cursor:
    return Cursor(
        topic_id=row["topic_id"],
        agent_name=row["agent_name"],
        last_seq=cast(int, row["last_seq"]),
        updated_at=row["updated_at"],
    )
