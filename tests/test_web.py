from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from agent_bus.web import server as web_server


def test_web_topics_list_renders(tmp_path: Path) -> None:
    web_server.init_db(str(tmp_path / "bus.sqlite"))
    db = web_server.get_db()
    db.topic_create(name="pink", metadata=None, mode="new")

    with TestClient(web_server.app) as client:
        res = client.get("/")

    assert res.status_code == 200
    assert "pink" in res.text


def test_web_topic_detail_renders(tmp_path: Path) -> None:
    web_server.init_db(str(tmp_path / "bus.sqlite"))
    db = web_server.get_db()
    topic = db.topic_create(name="pink", metadata=None, mode="new")
    db.sync_once(
        topic_id=topic.topic_id,
        agent_name="alice",
        outbox=[{"content_markdown": "hello from the web test", "message_type": "message"}],
        max_items=10,
        include_self=True,
        auto_advance=True,
        ack_through=None,
    )

    with TestClient(web_server.app) as client:
        res = client.get(f"/topics/{topic.topic_id}")

    assert res.status_code == 200
    assert "hello from the web test" in res.text
