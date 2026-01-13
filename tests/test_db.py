from __future__ import annotations

import pytest

import agent_bus.db as db_mod
from agent_bus.db import (
    AgentBusDB,
    TopicClosedError,
    TopicNotFoundError,
)


def test_topic_create_reuse_newest_open(tmp_path):
    db_path = tmp_path / "bus.sqlite"
    db = AgentBusDB(path=str(db_path))

    t2 = db.topic_create(name="pink", metadata={"a": 1}, mode="new")

    reused = db.topic_create(name="pink", metadata={"ignored": True}, mode="reuse")

    assert reused.topic_id == t2.topic_id
    assert reused.name == "pink"
    assert reused.status == "open"


def test_topic_resolve_closed_requires_flag(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))

    t = db.topic_create(name="pink", metadata=None, mode="new")
    db.topic_close(topic_id=t.topic_id, reason="done")

    try:
        db.topic_resolve(name="pink", allow_closed=False)
    except TopicNotFoundError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected TopicNotFoundError")

    resolved = db.topic_resolve(name="pink", allow_closed=True)
    assert resolved.topic_id == t.topic_id
    assert resolved.status == "closed"


def test_topic_close_idempotent(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    closed1, already1 = db.topic_close(topic_id=t.topic_id, reason="first")
    assert already1 is False
    assert closed1.status == "closed"
    assert closed1.close_reason == "first"
    assert closed1.closed_at is not None

    closed2, already2 = db.topic_close(topic_id=t.topic_id, reason="second")
    assert already2 is True
    assert closed2.closed_at == closed1.closed_at
    assert closed2.close_reason == "first"


def test_sync_once_write_rejects_closed_topic(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")
    db.topic_close(topic_id=t.topic_id, reason=None)

    with pytest.raises(TopicClosedError):
        db.sync_once(
            topic_id=t.topic_id,
            agent_name="a",
            outbox=[
                {
                    "content_markdown": "hi",
                    "message_type": "message",
                    "reply_to": None,
                    "metadata": None,
                    "client_message_id": None,
                }
            ],
            max_items=50,
            include_self=False,
            auto_advance=True,
            ack_through=None,
        )

    sent, received, cursor, has_more = db.sync_once(
        topic_id=t.topic_id,
        agent_name="a",
        outbox=[],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )
    assert sent == []
    assert received == []
    assert cursor.topic_id == t.topic_id
    assert has_more is False


def test_sync_once_send_and_receive(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    outbox = [
        {
            "content_markdown": "hello",
            "message_type": "message",
            "reply_to": None,
            "metadata": {"k": "v"},
            "client_message_id": None,
        }
    ]
    sent, received_a, cursor_a, has_more_a = db.sync_once(
        topic_id=t.topic_id,
        agent_name="a",
        outbox=outbox,
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )
    assert len(sent) == 1
    assert sent[0][1] is False
    assert received_a == []
    assert cursor_a.last_seq == 0
    assert has_more_a is False

    sent_b, received_b, cursor_b, has_more_b = db.sync_once(
        topic_id=t.topic_id,
        agent_name="b",
        outbox=[],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )
    assert sent_b == []
    assert has_more_b is False
    assert [m.content_markdown for m in received_b] == ["hello"]
    assert received_b[0].sender == "a"
    assert cursor_b.last_seq == received_b[0].seq

    _, received_b2, cursor_b2, has_more_b2 = db.sync_once(
        topic_id=t.topic_id,
        agent_name="b",
        outbox=[],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )
    assert received_b2 == []
    assert has_more_b2 is False
    assert cursor_b2.last_seq == cursor_b.last_seq


def test_sync_once_client_message_id_is_idempotent(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    outbox = [
        {
            "content_markdown": "hello",
            "message_type": "message",
            "reply_to": None,
            "metadata": None,
            "client_message_id": "msg-1",
        }
    ]

    sent1, _, _, _ = db.sync_once(
        topic_id=t.topic_id,
        agent_name="a",
        outbox=outbox,
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )
    sent2, _, _, _ = db.sync_once(
        topic_id=t.topic_id,
        agent_name="a",
        outbox=outbox,
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )

    assert len(sent1) == 1
    assert len(sent2) == 1
    assert sent1[0][1] is False
    assert sent2[0][1] is True
    assert sent1[0][0].message_id == sent2[0][0].message_id
    assert sent1[0][0].seq == sent2[0][0].seq


def test_sync_once_ack_through_rejects_future_seq(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    db.sync_once(
        topic_id=t.topic_id,
        agent_name="a",
        outbox=[
            {
                "content_markdown": "hello",
                "message_type": "message",
                "reply_to": None,
                "metadata": None,
                "client_message_id": None,
            }
        ],
        max_items=50,
        include_self=True,
        auto_advance=True,
        ack_through=None,
    )

    with pytest.raises(ValueError, match="exceeds latest message seq"):
        db.sync_once(
            topic_id=t.topic_id,
            agent_name="b",
            outbox=[],
            max_items=50,
            include_self=False,
            auto_advance=False,
            ack_through=999,
        )


def test_get_messages_returns_messages_after_seq(tmp_path):
    """Test that get_messages returns messages after a given sequence number."""
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    # Send 5 messages
    for i in range(5):
        db.sync_once(
            topic_id=t.topic_id,
            agent_name="a",
            outbox=[
                {
                    "content_markdown": f"message {i + 1}",
                    "message_type": "message",
                    "reply_to": None,
                    "metadata": None,
                    "client_message_id": None,
                }
            ],
            max_items=50,
            include_self=True,
            auto_advance=True,
            ack_through=None,
        )

    # Get all messages
    all_msgs = db.get_messages(topic_id=t.topic_id, after_seq=0)
    assert len(all_msgs) == 5
    assert [m.content_markdown for m in all_msgs] == [
        "message 1",
        "message 2",
        "message 3",
        "message 4",
        "message 5",
    ]

    # Get messages after seq 2
    after_2 = db.get_messages(topic_id=t.topic_id, after_seq=2)
    assert len(after_2) == 3
    assert [m.content_markdown for m in after_2] == [
        "message 3",
        "message 4",
        "message 5",
    ]

    # Get messages with limit
    limited = db.get_messages(topic_id=t.topic_id, after_seq=0, limit=2)
    assert len(limited) == 2
    assert [m.content_markdown for m in limited] == ["message 1", "message 2"]


def test_get_presence_returns_active_cursors(tmp_path):
    """Test that get_presence returns active cursors within the window."""
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    # Sync as agent 'a'
    db.sync_once(
        topic_id=t.topic_id,
        agent_name="a",
        outbox=[],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )

    # Presence should show agent 'a'
    presence = db.get_presence(topic_id=t.topic_id, window_seconds=300)
    assert len(presence) == 1
    assert presence[0].agent_name == "a"

    # Sync as agent 'b'
    db.sync_once(
        topic_id=t.topic_id,
        agent_name="b",
        outbox=[],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )

    # Presence should show both, newest first ('b' then 'a')
    presence2 = db.get_presence(topic_id=t.topic_id, window_seconds=300)
    assert len(presence2) == 2
    assert presence2[0].agent_name == "b"
    assert presence2[1].agent_name == "a"


def test_get_presence_topic_not_found(tmp_path):
    """Test that get_presence raises TopicNotFoundError for non-existent topics."""
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    with pytest.raises(TopicNotFoundError):
        db.get_presence(topic_id="ghost123")


def test_get_messages_empty_topic(tmp_path):
    """Test that get_messages returns empty list for topic with no messages."""
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    msgs = db.get_messages(topic_id=t.topic_id, after_seq=0)
    assert msgs == []


def test_sync_once_touches_cursor_timestamp(monkeypatch, tmp_path):
    times = iter([1000.0, 1001.0, 1002.0])
    monkeypatch.setattr(db_mod, "now", lambda: next(times))

    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    _, _, cursor1, _ = db.sync_once(
        topic_id=t.topic_id,
        agent_name="a",
        outbox=[],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )
    _, _, cursor2, _ = db.sync_once(
        topic_id=t.topic_id,
        agent_name="a",
        outbox=[],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )

    assert cursor1.updated_at == 1001.0
    assert cursor2.updated_at == 1002.0


def test_get_presence_filters_by_window(monkeypatch, tmp_path):
    times = iter([1000.0, 1001.0, 1002.0, 1003.0])
    monkeypatch.setattr(db_mod, "now", lambda: next(times))

    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    db.sync_once(
        topic_id=t.topic_id,
        agent_name="a",
        outbox=[],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )
    db.sync_once(
        topic_id=t.topic_id,
        agent_name="b",
        outbox=[],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )

    active = db.get_presence(topic_id=t.topic_id, window_seconds=1)
    assert [c.agent_name for c in active] == ["b"]


def test_get_senders_by_message_ids(tmp_path):
    """Test that get_senders_by_message_ids returns correct sender mapping."""
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    # Send messages from two different agents
    sent_a, _, _, _ = db.sync_once(
        topic_id=t.topic_id,
        agent_name="alice",
        outbox=[{"content_markdown": "hello from alice", "message_type": "message"}],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )
    sent_b, _, _, _ = db.sync_once(
        topic_id=t.topic_id,
        agent_name="bob",
        outbox=[{"content_markdown": "hello from bob", "message_type": "message"}],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )

    msg_a = sent_a[0][0]
    msg_b = sent_b[0][0]

    # Test basic mapping
    result = db.get_senders_by_message_ids([msg_a.message_id, msg_b.message_id])
    assert result[msg_a.message_id] == "alice"
    assert result[msg_b.message_id] == "bob"

    # Test with missing IDs - they should be omitted from result
    result_partial = db.get_senders_by_message_ids([msg_a.message_id, "nonexistent-id"])
    assert result_partial[msg_a.message_id] == "alice"
    assert "nonexistent-id" not in result_partial

    # Test empty list
    result_empty = db.get_senders_by_message_ids([])
    assert result_empty == {}


def test_delete_topic_removes_all_data(tmp_path):
    """Test that delete_topic removes topic and all related data."""
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    # Add messages and sync to create cursor
    db.sync_once(
        topic_id=t.topic_id,
        agent_name="alice",
        outbox=[{"content_markdown": "hello", "message_type": "message"}],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )

    # Verify data exists
    messages = db.get_messages(topic_id=t.topic_id, after_seq=0, limit=100)
    assert len(messages) == 1

    # Delete the topic
    deleted = db.delete_topic(topic_id=t.topic_id)
    assert deleted is True

    # Verify all data is gone
    with pytest.raises(TopicNotFoundError):
        db.get_topic(topic_id=t.topic_id)

    # Second delete should return False
    deleted_again = db.delete_topic(topic_id=t.topic_id)
    assert deleted_again is False


def test_delete_topic_nonexistent_returns_false(tmp_path):
    """Test that deleting a nonexistent topic returns False."""
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    deleted = db.delete_topic(topic_id="nonexistent-id")
    assert deleted is False


def test_delete_message_cascades_to_replies(tmp_path):
    """Test that delete_message cascades to all replies."""
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    # Create a chain: msg1 -> msg2 (reply to msg1) -> msg3 (reply to msg2)
    sent1, _, _, _ = db.sync_once(
        topic_id=t.topic_id,
        agent_name="alice",
        outbox=[{"content_markdown": "msg1", "message_type": "message"}],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )
    msg1_id = sent1[0][0].message_id

    sent2, _, _, _ = db.sync_once(
        topic_id=t.topic_id,
        agent_name="bob",
        outbox=[
            {
                "content_markdown": "msg2 reply to msg1",
                "message_type": "message",
                "reply_to": msg1_id,
            }
        ],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )
    msg2_id = sent2[0][0].message_id

    db.sync_once(
        topic_id=t.topic_id,
        agent_name="charlie",
        outbox=[
            {
                "content_markdown": "msg3 reply to msg2",
                "message_type": "message",
                "reply_to": msg2_id,
            }
        ],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )

    # Verify we have 3 messages
    messages = db.get_messages(topic_id=t.topic_id, after_seq=0, limit=100)
    assert len(messages) == 3

    # Delete msg1 - should cascade to msg2 and msg3
    deleted_count = db.delete_message(message_id=msg1_id)
    assert deleted_count == 3

    # Verify all messages are gone
    messages = db.get_messages(topic_id=t.topic_id, after_seq=0, limit=100)
    assert len(messages) == 0


def test_delete_message_no_replies(tmp_path):
    """Test deleting a message with no replies."""
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    sent, _, _, _ = db.sync_once(
        topic_id=t.topic_id,
        agent_name="alice",
        outbox=[{"content_markdown": "standalone msg", "message_type": "message"}],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )
    msg_id = sent[0][0].message_id

    deleted_count = db.delete_message(message_id=msg_id)
    assert deleted_count == 1

    messages = db.get_messages(topic_id=t.topic_id, after_seq=0, limit=100)
    assert len(messages) == 0


def test_delete_message_nonexistent_returns_zero(tmp_path):
    """Test deleting a nonexistent message returns 0."""
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    deleted_count = db.delete_message(message_id="nonexistent-id")
    assert deleted_count == 0


def test_delete_messages_batch(tmp_path):
    """Test batch deletion of multiple messages with cascades."""
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    # Create msg1 with a reply, and msg3 standalone
    sent1, _, _, _ = db.sync_once(
        topic_id=t.topic_id,
        agent_name="alice",
        outbox=[{"content_markdown": "msg1", "message_type": "message"}],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )
    msg1_id = sent1[0][0].message_id

    sent2, _, _, _ = db.sync_once(
        topic_id=t.topic_id,
        agent_name="bob",
        outbox=[
            {
                "content_markdown": "msg2 reply to msg1",
                "message_type": "message",
                "reply_to": msg1_id,
            }
        ],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )
    msg2_id = sent2[0][0].message_id

    sent3, _, _, _ = db.sync_once(
        topic_id=t.topic_id,
        agent_name="charlie",
        outbox=[{"content_markdown": "msg3 standalone", "message_type": "message"}],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )
    msg3_id = sent3[0][0].message_id

    # Batch delete msg1 and msg3 - should cascade msg1 -> msg2
    deleted_ids = db.delete_messages_batch(topic_id=t.topic_id, message_ids=[msg1_id, msg3_id])
    assert set(deleted_ids) == {msg1_id, msg2_id, msg3_id}

    messages = db.get_messages(topic_id=t.topic_id, after_seq=0, limit=100)
    assert len(messages) == 0


def test_delete_messages_batch_empty_list(tmp_path):
    """Test batch deletion with empty list returns 0."""
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    deleted_ids = db.delete_messages_batch(topic_id="does-not-matter", message_ids=[])
    assert deleted_ids == []


def test_delete_messages_batch_does_not_cross_topics(tmp_path):
    """Test batch deletion cannot delete messages from another topic."""
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t1 = db.topic_create(name="t1", metadata=None, mode="new")
    t2 = db.topic_create(name="t2", metadata=None, mode="new")

    sent1, _, _, _ = db.sync_once(
        topic_id=t1.topic_id,
        agent_name="alice",
        outbox=[{"content_markdown": "t1 msg", "message_type": "message"}],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )
    msg_t1_id = sent1[0][0].message_id

    sent2, _, _, _ = db.sync_once(
        topic_id=t2.topic_id,
        agent_name="bob",
        outbox=[{"content_markdown": "t2 msg", "message_type": "message"}],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )
    msg_t2_id = sent2[0][0].message_id

    deleted_ids = db.delete_messages_batch(topic_id=t1.topic_id, message_ids=[msg_t2_id])
    assert deleted_ids == []

    messages_t1 = db.get_messages(topic_id=t1.topic_id, after_seq=0, limit=100)
    assert [m.message_id for m in messages_t1] == [msg_t1_id]

    messages_t2 = db.get_messages(topic_id=t2.topic_id, after_seq=0, limit=100)
    assert [m.message_id for m in messages_t2] == [msg_t2_id]
