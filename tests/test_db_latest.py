from __future__ import annotations

from agent_bus.db import AgentBusDB


def test_get_latest_messages_efficiently(tmp_path):
    """Test that get_latest_messages returns the last N messages in chronological order."""
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

    # Fetch last 3
    msgs = db.get_latest_messages(topic_id=t.topic_id, limit=3)

    assert len(msgs) == 3
    # Check content and order (should be chronological: 3, 4, 5)
    assert msgs[0].content_markdown == "message 3"
    assert msgs[1].content_markdown == "message 4"
    assert msgs[2].content_markdown == "message 5"
    assert msgs[0].seq < msgs[1].seq < msgs[2].seq
