from __future__ import annotations

import json
import re
from pathlib import Path

from click.testing import CliRunner

from agent_bus.cli import cli
from agent_bus.db import AgentBusDB


def test_cli_topics_export_markdown(tmp_path: Path) -> None:
    """Test exporting a topic as markdown."""
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)

    # Create topic with messages
    t = db.topic_create(name="test-export", metadata=None, mode="new")
    db.sync_once(
        topic_id=t.topic_id,
        agent_name="alice",
        outbox=[{"content_markdown": "Hello world!", "message_type": "message"}],
        max_items=10,
        include_self=True,
        auto_advance=True,
        ack_through=None,
    )
    db.sync_once(
        topic_id=t.topic_id,
        agent_name="bob",
        outbox=[{"content_markdown": "Hi Alice!", "message_type": "message"}],
        max_items=10,
        include_self=True,
        auto_advance=True,
        ack_through=None,
    )

    runner = CliRunner()
    res = runner.invoke(cli, ["--db-path", db_path, "topics", "export", t.topic_id])

    assert res.exit_code == 0
    assert "# test-export" in res.output
    assert "### [1] alice" in res.output
    assert "Hello world!" in res.output
    assert "### [2] bob" in res.output
    assert "Hi Alice!" in res.output


def test_cli_topics_export_json(tmp_path: Path) -> None:
    """Test exporting a topic as JSON."""
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)

    t = db.topic_create(name="test-json", metadata=None, mode="new")
    db.sync_once(
        topic_id=t.topic_id,
        agent_name="alice",
        outbox=[{"content_markdown": "Test message", "message_type": "question"}],
        max_items=10,
        include_self=True,
        auto_advance=True,
        ack_through=None,
    )

    runner = CliRunner()
    res = runner.invoke(cli, ["--db-path", db_path, "topics", "export", t.topic_id, "-f", "json"])

    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["topic_name"] == "test-json"
    assert data["message_count"] == 1
    assert len(data["messages"]) == 1
    assert data["messages"][0]["sender"] == "alice"
    assert data["messages"][0]["content_markdown"] == "Test message"
    assert data["messages"][0]["message_type"] == "question"


def test_cli_topics_export_to_file(tmp_path: Path) -> None:
    """Test exporting to a file."""
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)

    t = db.topic_create(name="test-file", metadata=None, mode="new")
    db.sync_once(
        topic_id=t.topic_id,
        agent_name="agent",
        outbox=[{"content_markdown": "File content", "message_type": "message"}],
        max_items=10,
        include_self=True,
        auto_advance=True,
        ack_through=None,
    )

    output_file = tmp_path / "export.md"
    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["--db-path", db_path, "topics", "export", t.topic_id, "-o", str(output_file)],
    )

    assert res.exit_code == 0
    assert "Exported 1 messages" in res.output
    assert output_file.exists()
    content = output_file.read_text()
    assert "# test-file" in content
    assert "File content" in content


def test_cli_topics_export_with_metadata(tmp_path: Path) -> None:
    """Test exporting with metadata included."""
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)

    t = db.topic_create(name="test-meta", metadata=None, mode="new")
    db.sync_once(
        topic_id=t.topic_id,
        agent_name="agent",
        outbox=[{"content_markdown": "With metadata", "message_type": "message"}],
        max_items=10,
        include_self=True,
        auto_advance=True,
        ack_through=None,
    )

    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["--db-path", db_path, "topics", "export", t.topic_id, "--include-metadata"],
    )

    assert res.exit_code == 0
    # Should have timestamp in format YYYY-MM-DD HH:MM:SS
    assert "### [1] agent" in res.output
    # Check that there's a timestamp (asterisk-wrapped)
    assert re.search(r"\*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\*", res.output)


def test_cli_topics_export_with_reply(tmp_path: Path) -> None:
    """Test exporting with reply threading."""
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)

    t = db.topic_create(name="test-reply", metadata=None, mode="new")

    # First message
    sent1, _, _, _ = db.sync_once(
        topic_id=t.topic_id,
        agent_name="alice",
        outbox=[{"content_markdown": "Original question", "message_type": "question"}],
        max_items=10,
        include_self=True,
        auto_advance=True,
        ack_through=None,
    )
    msg1_id = sent1[0][0].message_id  # sent is list of (Message, is_duplicate)

    # Reply to first message
    db.sync_once(
        topic_id=t.topic_id,
        agent_name="bob",
        outbox=[
            {
                "content_markdown": "This is my answer",
                "message_type": "answer",
                "reply_to": msg1_id,
            }
        ],
        max_items=10,
        include_self=True,
        auto_advance=True,
        ack_through=None,
    )

    runner = CliRunner()
    res = runner.invoke(cli, ["--db-path", db_path, "topics", "export", t.topic_id])

    assert res.exit_code == 0
    assert "*↩︎ reply to alice (#1)*" in res.output
    assert "This is my answer" in res.output


def test_cli_topics_export_not_found(tmp_path: Path) -> None:
    """Test exporting a non-existent topic."""
    db_path = str(tmp_path / "bus.sqlite")
    # Initialize db
    AgentBusDB(path=db_path)

    runner = CliRunner()
    res = runner.invoke(cli, ["--db-path", db_path, "topics", "export", "nonexistent"])

    assert res.exit_code != 0
    assert "Topic not found" in res.output


def test_cli_topics_export_empty(tmp_path: Path) -> None:
    """Test exporting a topic with no messages."""
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)

    t = db.topic_create(name="empty-topic", metadata=None, mode="new")

    runner = CliRunner()
    res = runner.invoke(cli, ["--db-path", db_path, "topics", "export", t.topic_id])

    assert res.exit_code == 0
    assert "No messages to export" in res.output


def test_cli_topics_export_after_seq(tmp_path: Path) -> None:
    """Test exporting only messages after a specific sequence number."""
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)

    t = db.topic_create(name="test-delta", metadata=None, mode="new")

    # Create 3 messages
    for i in range(3):
        db.sync_once(
            topic_id=t.topic_id,
            agent_name="agent",
            outbox=[{"content_markdown": f"Message {i + 1}", "message_type": "message"}],
            max_items=10,
            include_self=True,
            auto_advance=True,
            ack_through=None,
        )

    runner = CliRunner()

    # Export only messages after seq 1 (should get messages 2 and 3)
    res = runner.invoke(
        cli,
        ["--db-path", db_path, "topics", "export", t.topic_id, "--after-seq", "1"],
    )

    assert res.exit_code == 0
    assert "**Messages:** 2" in res.output
    assert "Message 1" not in res.output
    assert "Message 2" in res.output
    assert "Message 3" in res.output
