from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_bus.cli import cli
from agent_bus.db import AgentBusDB


def test_cli_topics_list_json_includes_counts(tmp_path: Path) -> None:
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)

    t1 = db.topic_create(name="pink", metadata=None, mode="new")
    t2 = db.topic_create(name="blue", metadata=None, mode="new")

    db.sync_once(
        topic_id=t1.topic_id,
        agent_name="a",
        outbox=[
            {
                "content_markdown": "m1",
                "message_type": "message",
                "reply_to": None,
                "metadata": None,
                "client_message_id": None,
            },
            {
                "content_markdown": "m2",
                "message_type": "message",
                "reply_to": None,
                "metadata": None,
                "client_message_id": None,
            },
        ],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )
    db.sync_once(
        topic_id=t2.topic_id,
        agent_name="a",
        outbox=[
            {
                "content_markdown": "m3",
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

    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["--db-path", db_path, "topics", "list", "--status", "all", "--json"],
    )
    assert res.exit_code == 0, res.output

    payload = json.loads(res.output)
    topics = {t["name"]: t for t in payload["topics"]}

    assert topics["pink"]["counts"] == {
        "messages": 2,
        "last_seq": 2,
    }
    assert topics["blue"]["counts"] == {
        "messages": 1,
        "last_seq": 1,
    }


def test_cli_db_wipe_deletes_db_file(tmp_path: Path) -> None:
    db_path = tmp_path / "bus.sqlite"
    db = AgentBusDB(path=str(db_path))
    db.topic_create(name="pink", metadata=None, mode="new")

    assert db_path.exists()

    runner = CliRunner()
    res = runner.invoke(cli, ["--db-path", str(db_path), "db", "wipe", "--yes"])
    assert res.exit_code == 0, res.output
    assert not db_path.exists()


def test_cli_topics_watch_shows_recent_messages(tmp_path: Path) -> None:
    """Test that topics watch shows recent messages."""
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)

    t = db.topic_create(name="test-topic", metadata=None, mode="new")

    # Send some messages from different senders
    for sender, content in [("alice", "Hello!"), ("bob", "Hi there!"), ("alice", "How are you?")]:
        db.sync_once(
            topic_id=t.topic_id,
            agent_name=sender,
            outbox=[
                {
                    "content_markdown": content,
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

    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["--db-path", db_path, "topics", "watch", t.topic_id, "-n", "10"],
    )
    assert res.exit_code == 0, res.output

    # Check that all messages are shown
    assert "test-topic" in res.output
    assert "Hello!" in res.output
    assert "Hi there!" in res.output
    assert "How are you?" in res.output
    assert "alice" in res.output
    assert "bob" in res.output


def test_cli_topics_watch_topic_not_found(tmp_path: Path) -> None:
    """Test that topics watch returns error for non-existent topic."""
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)
    # Create a topic just to initialize the DB
    db.topic_create(name="other", metadata=None, mode="new")

    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["--db-path", db_path, "topics", "watch", "nonexistent123"],
    )
    assert res.exit_code != 0
    assert "Topic not found" in res.output


def test_cli_topics_watch_last_n_messages(tmp_path: Path) -> None:
    """Test that -n flag limits the number of messages shown."""
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)

    t = db.topic_create(name="test-topic", metadata=None, mode="new")

    # Send 5 messages
    for i in range(5):
        db.sync_once(
            topic_id=t.topic_id,
            agent_name="sender",
            outbox=[
                {
                    "content_markdown": f"Message number {i + 1}",
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

    runner = CliRunner()
    # Only show last 2 messages
    res = runner.invoke(
        cli,
        ["--db-path", db_path, "topics", "watch", t.topic_id, "-n", "2"],
    )
    assert res.exit_code == 0, res.output

    # Should only show messages 4 and 5
    assert "Message number 4" in res.output
    assert "Message number 5" in res.output
    # Should NOT show earlier messages
    assert "Message number 1" not in res.output
    assert "Message number 2" not in res.output
    assert "Message number 3" not in res.output


def test_cli_topics_presence_list_active_peers(tmp_path: Path) -> None:
    """Test that topics presence lists active peers."""
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)

    t = db.topic_create(name="test-topic", metadata=None, mode="new")

    # Sync as multiple agents
    for agent in ["alice", "bob"]:
        db.sync_once(
            topic_id=t.topic_id,
            agent_name=agent,
            outbox=[],
            max_items=50,
            include_self=False,
            auto_advance=True,
            ack_through=None,
        )

    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["--db-path", db_path, "topics", "presence", t.topic_id],
    )
    assert res.exit_code == 0, res.output

    assert "alice" in res.output
    assert "bob" in res.output
    assert "Active peers" in res.output


def test_cli_topics_presence_json_output(tmp_path: Path) -> None:
    """Test that topics presence --json returns valid JSON."""
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)

    t = db.topic_create(name="test-topic", metadata=None, mode="new")
    db.sync_once(
        topic_id=t.topic_id,
        agent_name="tester",
        outbox=[],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )

    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["--db-path", db_path, "topics", "presence", t.topic_id, "--json"],
    )
    assert res.exit_code == 0, res.output

    payload = json.loads(res.output)
    assert payload["topic_id"] == t.topic_id
    assert any(p["agent_name"] == "tester" for p in payload["peers"])


def test_cli_topics_presence_lists_active_peers(tmp_path: Path) -> None:
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)
    t = db.topic_create(name="test-topic", metadata=None, mode="new")

    db.sync_once(
        topic_id=t.topic_id,
        agent_name="alice",
        outbox=[],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )

    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["--db-path", db_path, "topics", "presence", t.topic_id, "--window", "300"],
    )
    assert res.exit_code == 0, res.output
    assert "alice" in res.output


def test_cli_topics_presence_json(tmp_path: Path) -> None:
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)
    t = db.topic_create(name="test-topic", metadata=None, mode="new")

    db.sync_once(
        topic_id=t.topic_id,
        agent_name="alice",
        outbox=[],
        max_items=50,
        include_self=False,
        auto_advance=True,
        ack_through=None,
    )

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "--db-path",
            db_path,
            "topics",
            "presence",
            t.topic_id,
            "--window",
            "300",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["topic_id"] == t.topic_id
    assert payload["window_seconds"] == 300
    assert payload["peers"][0]["agent_name"] == "alice"


def test_cli_topics_rename(tmp_path: Path) -> None:
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)
    t = db.topic_create(name="old", metadata=None, mode="new")

    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["--db-path", db_path, "topics", "rename", t.topic_id, "new-name"],
    )
    assert res.exit_code == 0, res.output

    updated = db.get_topic(topic_id=t.topic_id)
    assert updated.name == "new-name"


def test_cli_topics_delete(tmp_path: Path) -> None:
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)
    t = db.topic_create(name="to-delete", metadata=None, mode="new")

    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["--db-path", db_path, "topics", "delete", t.topic_id, "--yes"],
    )
    assert res.exit_code == 0, res.output

    from agent_bus.db import TopicNotFoundError

    with pytest.raises(TopicNotFoundError):
        db.get_topic(topic_id=t.topic_id)
