from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_bus.cli import cli
from agent_bus.db import AgentBusDB


def test_cli_search_json(tmp_path: Path) -> None:
    db_path = str(tmp_path / "bus.sqlite")
    db = AgentBusDB(path=db_path)
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

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "--db-path",
            db_path,
            "search",
            "hello",
            "--mode",
            "fts",
            "--json",
            "--topic-id",
            t.topic_id,
        ],
    )
    if res.exit_code != 0 and "FTS5" in res.output:
        pytest.skip("FTS5 not available on this SQLite build")

    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["results"]
    assert payload["results"][0]["topic_id"] == t.topic_id
