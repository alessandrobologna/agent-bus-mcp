from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from agent_bus.cli import cli
from agent_bus.db import AgentBusDB


def test_cli_topics_watch_tail_alias(tmp_path: Path) -> None:
    """Test that --tail is accepted as a flag."""
    runner = CliRunner()
    res = runner.invoke(cli, ["topics", "watch", "--help"])
    assert res.exit_code == 0
    assert "--tail" in res.output
    assert "Wait for new messages" in res.output


def test_cli_topics_watch_last_still_works(tmp_path: Path) -> None:
    """Ensure -n/--last still works after the changes."""
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)

    t = db.topic_create(name="test-topic", metadata=None, mode="new")
    db.sync_once(
        topic_id=t.topic_id,
        agent_name="a",
        outbox=[{"content_markdown": "m1", "message_type": "m"}],
        max_items=10,
        include_self=True,
        auto_advance=True,
        ack_through=None,
    )

    runner = CliRunner()
    res = runner.invoke(cli, ["--db-path", db_path, "topics", "watch", t.topic_id, "-n", "1"])
    assert res.exit_code == 0
    assert "m1" in res.output
