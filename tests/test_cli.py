from __future__ import annotations

import json
from pathlib import Path

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
