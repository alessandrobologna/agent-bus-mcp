"""Tests for message pagination (before_seq parameter)."""

from __future__ import annotations

from pathlib import Path

from agent_bus.db import AgentBusDB


def test_get_messages_before_seq(tmp_path: Path) -> None:
    """Test fetching messages before a specific sequence number."""
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)

    # Create topic with 10 messages
    t = db.topic_create(name="test-pagination", metadata=None, mode="new")
    for i in range(10):
        db.sync_once(
            topic_id=t.topic_id,
            agent_name="agent",
            outbox=[{"content_markdown": f"Message {i + 1}", "message_type": "message"}],
            max_items=10,
            include_self=True,
            auto_advance=True,
            ack_through=None,
        )

    # Get messages before seq 8 (should return 1-7)
    messages = db.get_messages(topic_id=t.topic_id, before_seq=8, limit=100)
    assert len(messages) == 7
    assert messages[0].seq == 1
    assert messages[-1].seq == 7

    # Get last 3 messages before seq 8 (should return 5, 6, 7)
    messages = db.get_messages(topic_id=t.topic_id, before_seq=8, limit=3)
    assert len(messages) == 3
    assert messages[0].seq == 5
    assert messages[-1].seq == 7

    # Get messages before seq 1 (should return empty)
    messages = db.get_messages(topic_id=t.topic_id, before_seq=1, limit=100)
    assert len(messages) == 0


def test_get_messages_before_seq_with_after_seq(tmp_path: Path) -> None:
    """Test fetching messages in a range (both before_seq and after_seq)."""
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)

    # Create topic with 10 messages
    t = db.topic_create(name="test-range", metadata=None, mode="new")
    for i in range(10):
        db.sync_once(
            topic_id=t.topic_id,
            agent_name="agent",
            outbox=[{"content_markdown": f"Message {i + 1}", "message_type": "message"}],
            max_items=10,
            include_self=True,
            auto_advance=True,
            ack_through=None,
        )

    # Get messages in range (3, 8) -> should return 4, 5, 6, 7
    messages = db.get_messages(topic_id=t.topic_id, after_seq=3, before_seq=8, limit=100)
    assert len(messages) == 4
    assert messages[0].seq == 4
    assert messages[-1].seq == 7


def test_get_messages_after_seq_only(tmp_path: Path) -> None:
    """Test that after_seq only still works (backwards compatibility)."""
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)

    # Create topic with 5 messages
    t = db.topic_create(name="test-after", metadata=None, mode="new")
    for i in range(5):
        db.sync_once(
            topic_id=t.topic_id,
            agent_name="agent",
            outbox=[{"content_markdown": f"Message {i + 1}", "message_type": "message"}],
            max_items=10,
            include_self=True,
            auto_advance=True,
            ack_through=None,
        )

    # Get messages after seq 2 (should return 3, 4, 5)
    messages = db.get_messages(topic_id=t.topic_id, after_seq=2, limit=100)
    assert len(messages) == 3
    assert messages[0].seq == 3
    assert messages[-1].seq == 5


def test_get_messages_returns_ascending_order(tmp_path: Path) -> None:
    """Test that messages are always returned in ascending seq order."""
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)

    # Create topic with 20 messages
    t = db.topic_create(name="test-order", metadata=None, mode="new")
    for i in range(20):
        db.sync_once(
            topic_id=t.topic_id,
            agent_name="agent",
            outbox=[{"content_markdown": f"Message {i + 1}", "message_type": "message"}],
            max_items=10,
            include_self=True,
            auto_advance=True,
            ack_through=None,
        )

    # Get last 5 messages before seq 15
    messages = db.get_messages(topic_id=t.topic_id, before_seq=15, limit=5)
    assert len(messages) == 5
    # Should be in ascending order: 10, 11, 12, 13, 14
    seqs = [m.seq for m in messages]
    assert seqs == [10, 11, 12, 13, 14]
