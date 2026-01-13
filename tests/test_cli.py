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

    q1 = db.question_create(topic_id=t1.topic_id, asked_by="a", question_text="q1")
    q2 = db.question_create(topic_id=t1.topic_id, asked_by="a", question_text="q2")
    db.question_create(topic_id=t2.topic_id, asked_by="a", question_text="q3")

    db.answer_insert_batch(
        topic_id=t1.topic_id,
        answered_by="b",
        items=[(q1.question_id, {"answer_markdown": "a", "suggested_followups": ["x"]})],
    )
    db.question_mark_answered(topic_id=t1.topic_id, question_id=q1.question_id)
    db.question_cancel(topic_id=t1.topic_id, question_id=q2.question_id, reason="nvm")

    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["--db-path", db_path, "topics", "list", "--status", "all", "--json"],
    )
    assert res.exit_code == 0, res.output

    payload = json.loads(res.output)
    topics = {t["name"]: t for t in payload["topics"]}

    assert topics["pink"]["counts"] == {
        "total": 2,
        "pending": 0,
        "answered": 1,
        "cancelled": 1,
    }
    assert topics["blue"]["counts"] == {
        "total": 1,
        "pending": 1,
        "answered": 0,
        "cancelled": 0,
    }


def test_cli_db_wipe_deletes_db_file(tmp_path: Path) -> None:
    db_path = tmp_path / "bus.sqlite"
    db = AgentBusDB(path=str(db_path))
    t = db.topic_create(name="pink", metadata=None, mode="new")
    db.question_create(topic_id=t.topic_id, asked_by="a", question_text="q1")

    assert db_path.exists()

    runner = CliRunner()
    res = runner.invoke(cli, ["--db-path", str(db_path), "db", "wipe", "--yes"])
    assert res.exit_code == 0, res.output
    assert not db_path.exists()
