from __future__ import annotations

import pytest

import agent_bus.search as search_module
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


def test_search_messages_hybrid_keeps_fts_snippet_when_semantic_excerpt_omits_query(
    tmp_path, monkeypatch
) -> None:
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    topic = db.topic_create(name="pink", metadata=None, mode="new")

    db.sync_once(
        topic_id=topic.topic_id,
        agent_name="a",
        outbox=[
            {
                "content_markdown": "Batch event shape details first.\n\nLater we mention lambda adapter wiring.",
                "message_type": "message",
            }
        ],
        max_items=10,
        include_self=True,
        auto_advance=True,
        ack_through=None,
    )

    try:
        fts_results = db.search_messages_fts(query="lambda", topic_id=topic.topic_id, limit=10)
    except RuntimeError:
        pytest.skip("FTS5 not available on this SQLite build")

    assert fts_results
    message = db.get_latest_messages(topic_id=topic.topic_id, limit=1)[0]

    monkeypatch.setattr(
        search_module,
        "_semantic_best_by_message",
        lambda *args, **kwargs: {
            message.message_id: {
                "topic_id": topic.topic_id,
                "topic_name": topic.name,
                "message_id": message.message_id,
                "seq": message.seq,
                "sender": message.sender,
                "message_type": message.message_type,
                "created_at": message.created_at,
                "start_char": 0,
                "end_char": 24,
                "semantic_score": 0.9,
            }
        },
    )

    results, _warnings = search_messages(db, query="lambda", topic_id=topic.topic_id, mode="hybrid")

    assert results
    assert "\ue000lambda\ue001" in results[0]["snippet"].lower()


def test_snippet_contains_query_terms_ignores_fts_operators() -> None:
    assert not search_module._snippet_contains_query_terms(
        "batch or stream handling",
        query="lambda OR adapter",
    )
    assert search_module._snippet_contains_query_terms(
        "lambda adapter wiring",
        query="lambda OR adapter",
    )
