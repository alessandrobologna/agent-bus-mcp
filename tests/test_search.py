from __future__ import annotations

import pytest

from agent_bus.db import AgentBusDB
from agent_bus.search import search_messages


def test_search_messages_fts_basic(tmp_path) -> None:
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    db.sync_once(
        topic_id=t.topic_id,
        agent_name="a",
        outbox=[{"content_markdown": "hello world", "message_type": "message"}],
        max_items=10,
        include_self=True,
        auto_advance=True,
        ack_through=None,
    )

    try:
        results = db.search_messages_fts(query="hello", topic_id=t.topic_id, limit=10)
    except RuntimeError:
        pytest.skip("FTS5 not available on this SQLite build")

    assert len(results) == 1
    assert results[0]["topic_id"] == t.topic_id
    assert results[0]["sender"] == "a"


def test_search_messages_hybrid_falls_back_without_embeddings(tmp_path) -> None:
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    db.sync_once(
        topic_id=t.topic_id,
        agent_name="a",
        outbox=[{"content_markdown": "cursor reset replays history", "message_type": "message"}],
        max_items=10,
        include_self=True,
        auto_advance=True,
        ack_through=None,
    )

    try:
        results, _warnings = search_messages(
            db, query="cursor reset", topic_id=t.topic_id, mode="hybrid"
        )
    except RuntimeError:
        pytest.skip("FTS5 not available on this SQLite build")

    assert results
