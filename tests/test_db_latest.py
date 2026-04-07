from __future__ import annotations

from agent_bus import db as db_module
from agent_bus.db import AgentBusDB


def install_fake_now(monkeypatch, *, start: float = 1_700_000_000.0, step: float = 1.0) -> None:
    current = {"value": start - step}

    def fake_now() -> float:
        current["value"] += step
        return current["value"]

    monkeypatch.setattr(db_module, "now", fake_now)


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


def test_topic_list_with_counts_includes_last_updated_at(tmp_path, monkeypatch):
    install_fake_now(monkeypatch)
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    first = db.topic_create(name="first", metadata=None, mode="new")
    second = db.topic_create(name="second", metadata=None, mode="new")

    db.sync_once(
        topic_id=first.topic_id,
        agent_name="reviewer",
        outbox=[
            {
                "content_markdown": "latest update",
                "message_type": "message",
                "reply_to": None,
                "metadata": None,
                "client_message_id": None,
            }
        ],
        max_items=20,
        include_self=True,
        auto_advance=True,
        ack_through=None,
    )

    topics = db.topic_list_with_counts(status="all", sort="last_updated_desc", limit=10)
    by_id = {topic["topic_id"]: topic for topic in topics}

    assert by_id[first.topic_id]["counts"]["messages"] == 1
    assert by_id[first.topic_id]["last_message_at"] is not None
    assert by_id[first.topic_id]["last_updated_at"] >= by_id[first.topic_id]["created_at"]
    assert by_id[second.topic_id]["counts"]["messages"] == 0
    assert by_id[second.topic_id]["last_message_at"] is None
    assert by_id[second.topic_id]["last_updated_at"] == by_id[second.topic_id]["created_at"]


def test_topic_list_with_counts_sorts_updated_older_topic_first(tmp_path, monkeypatch):
    install_fake_now(monkeypatch)
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    oldest = db.topic_create(name="oldest", metadata=None, mode="new")
    db.topic_create(name="middle", metadata=None, mode="new")
    newest = db.topic_create(name="newest", metadata=None, mode="new")

    db.sync_once(
        topic_id=oldest.topic_id,
        agent_name="reviewer",
        outbox=[
            {
                "content_markdown": "fresh update",
                "message_type": "message",
                "reply_to": None,
                "metadata": None,
                "client_message_id": None,
            }
        ],
        max_items=20,
        include_self=True,
        auto_advance=True,
        ack_through=None,
    )

    topics = db.topic_list_with_counts(status="all", sort="last_updated_desc", limit=2)

    assert [topic["topic_id"] for topic in topics] == [oldest.topic_id, newest.topic_id]


def test_topic_get_with_counts_returns_specific_topic_summary(tmp_path, monkeypatch):
    install_fake_now(monkeypatch)
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    topic = db.topic_create(name="specific", metadata=None, mode="new")
    db.topic_create(name="later", metadata=None, mode="new")
    seed = {
        "content_markdown": "hello",
        "message_type": "message",
        "reply_to": None,
        "metadata": None,
        "client_message_id": None,
    }
    db.sync_once(
        topic_id=topic.topic_id,
        agent_name="reviewer",
        outbox=[seed],
        max_items=20,
        include_self=True,
        auto_advance=True,
        ack_through=None,
    )

    summary = db.topic_get_with_counts(topic_id=topic.topic_id)

    assert summary["topic_id"] == topic.topic_id
    assert summary["counts"]["messages"] == 1
    assert summary["counts"]["last_seq"] == 1
