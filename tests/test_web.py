from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from agent_bus import db as db_module
from agent_bus import search as search_module
from agent_bus.web import server as web_server


def prepare_static_bundle(tmp_path: Path, monkeypatch) -> None:
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text(
        "<!doctype html><html><body><div id='root'></div></body></html>",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_server, "STATIC_DIR", static_dir)
    monkeypatch.setattr(web_server, "SPA_INDEX", static_dir / "index.html")


def seed_topic_message(db, topic_id: str, sender: str, content: str) -> None:
    db.sync_once(
        topic_id=topic_id,
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
        max_items=20,
        include_self=True,
        auto_advance=True,
        ack_through=None,
    )


def install_fake_now(monkeypatch, *, start: float = 1_700_000_000.0, step: float = 1.0) -> None:
    current = {"value": start - step}

    def fake_now() -> float:
        current["value"] += step
        return current["value"]

    monkeypatch.setattr(db_module, "now", fake_now)


def test_spa_shell_routes_serve_index(tmp_path: Path, monkeypatch) -> None:
    prepare_static_bundle(tmp_path, monkeypatch)
    install_fake_now(monkeypatch)
    web_server.init_db(str(tmp_path / "bus.sqlite"))
    db = web_server.get_db()
    topic = db.topic_create(name="pink", metadata=None, mode="new")

    with TestClient(web_server.app) as client:
        root = client.get("/")
        topic_page = client.get(f"/topics/{topic.topic_id}")

    assert root.status_code == 200
    assert topic_page.status_code == 200
    assert "<div id='root'></div>" in root.text
    assert "<div id='root'></div>" in topic_page.text


def test_api_topics_returns_last_updated_sorting(tmp_path: Path, monkeypatch) -> None:
    prepare_static_bundle(tmp_path, monkeypatch)
    install_fake_now(monkeypatch)
    web_server.init_db(str(tmp_path / "bus.sqlite"))
    db = web_server.get_db()
    first = db.topic_create(name="first", metadata=None, mode="new")
    db.topic_create(name="second", metadata=None, mode="new")
    seed_topic_message(db, first.topic_id, "alice", "latest note")

    with TestClient(web_server.app) as client:
        res = client.get("/api/topics?sort=last_updated_desc&status=all")

    assert res.status_code == 200
    payload = res.json()
    assert [topic["name"] for topic in payload["topics"][:2]] == ["first", "second"]
    assert payload["topics"][0]["last_updated_at"] >= payload["topics"][1]["last_updated_at"]


def test_api_topics_last_updated_sort_includes_older_recently_updated_topic(
    tmp_path: Path, monkeypatch
) -> None:
    prepare_static_bundle(tmp_path, monkeypatch)
    install_fake_now(monkeypatch)
    web_server.init_db(str(tmp_path / "bus.sqlite"))
    db = web_server.get_db()
    oldest = db.topic_create(name="oldest", metadata=None, mode="new")
    db.topic_create(name="middle", metadata=None, mode="new")
    newest = db.topic_create(name="newest", metadata=None, mode="new")
    seed_topic_message(db, oldest.topic_id, "alice", "fresh update")

    with TestClient(web_server.app) as client:
        res = client.get("/api/topics?sort=last_updated_desc&status=all&limit=2")

    assert res.status_code == 200
    payload = res.json()
    assert [topic["topic_id"] for topic in payload["topics"]] == [oldest.topic_id, newest.topic_id]


def test_api_topic_detail_with_focus_returns_context_window(tmp_path: Path, monkeypatch) -> None:
    prepare_static_bundle(tmp_path, monkeypatch)
    install_fake_now(monkeypatch)
    web_server.init_db(str(tmp_path / "bus.sqlite"))
    db = web_server.get_db()
    topic = db.topic_create(name="pink", metadata=None, mode="new")
    for index in range(3):
        seed_topic_message(db, topic.topic_id, "alice", f"message {index}")

    focused = db.get_latest_messages(topic_id=topic.topic_id, limit=1)[0]

    with TestClient(web_server.app) as client:
        res = client.get(f"/api/topics/{topic.topic_id}?focus={focused.message_id}")

    assert res.status_code == 200
    payload = res.json()
    assert payload["context_mode"] is True
    assert payload["focus_message_id"] == focused.message_id
    assert any(message["message_id"] == focused.message_id for message in payload["messages"])


def test_api_global_search_returns_json(tmp_path: Path, monkeypatch) -> None:
    prepare_static_bundle(tmp_path, monkeypatch)
    install_fake_now(monkeypatch)
    web_server.init_db(str(tmp_path / "bus.sqlite"))
    db = web_server.get_db()
    topic = db.topic_create(name="violet", metadata=None, mode="new")

    def fake_search_messages(db_obj, **kwargs):
        assert db_obj is db
        assert kwargs["query"] == "handoff"
        assert kwargs["mode"] == "semantic"
        assert kwargs["topic_id"] is None
        return (
            [
                {
                    "topic_id": topic.topic_id,
                    "topic_name": topic.name,
                    "message_id": "msg-123",
                    "seq": 4,
                    "sender": "reviewer",
                    "message_type": "message",
                    "semantic_score": 0.812,
                    "snippet": "handoff summary",
                }
            ],
            [],
        )

    monkeypatch.setattr(search_module, "search_messages", fake_search_messages)

    with TestClient(web_server.app) as client:
        res = client.get("/api/search?q=handoff&mode=semantic")

    assert res.status_code == 200
    payload = res.json()
    assert payload["results"][0]["topic_name"] == "violet"
    assert payload["results"][0]["snippet"] == "handoff summary"


def test_api_delete_routes_remove_messages_and_topics(tmp_path: Path, monkeypatch) -> None:
    prepare_static_bundle(tmp_path, monkeypatch)
    install_fake_now(monkeypatch)
    web_server.init_db(str(tmp_path / "bus.sqlite"))
    db = web_server.get_db()
    topic = db.topic_create(name="pink", metadata=None, mode="new")
    seed_topic_message(db, topic.topic_id, "alice", "one")
    seed_topic_message(db, topic.topic_id, "alice", "two")
    messages = db.get_messages(topic_id=topic.topic_id, limit=20)

    with TestClient(web_server.app) as client:
        delete_messages_res = client.request(
            "DELETE",
            f"/api/topics/{topic.topic_id}/messages",
            json={"message_ids": [messages[0].message_id]},
        )
        delete_topic_res = client.delete(f"/api/topics/{topic.topic_id}")

    assert delete_messages_res.status_code == 200
    assert delete_messages_res.json()["deleted_count"] == 1
    assert delete_topic_res.status_code == 200
    assert delete_topic_res.json()["deleted"] is True


def test_api_topics_stream_uses_sse_framing(tmp_path: Path, monkeypatch) -> None:
    prepare_static_bundle(tmp_path, monkeypatch)
    install_fake_now(monkeypatch)
    web_server.init_db(str(tmp_path / "bus.sqlite"))
    db = web_server.get_db()
    db.topic_create(name="pink", metadata=None, mode="new")

    class FakeRequest:
        async def is_disconnected(self) -> bool:
            return False

    response = asyncio.run(web_server.api_topics_stream(FakeRequest()))
    first_chunk = asyncio.run(response.body_iterator.__anext__()).decode("utf-8")

    assert response.media_type == "text/event-stream"
    assert response.headers["Cache-Control"] == "no-cache"
    assert "event: topics.invalidate" in first_chunk


def test_topics_signature_changes_after_topic_rename(tmp_path: Path, monkeypatch) -> None:
    prepare_static_bundle(tmp_path, monkeypatch)
    install_fake_now(monkeypatch)
    web_server.init_db(str(tmp_path / "bus.sqlite"))
    db = web_server.get_db()
    topic = db.topic_create(name="before", metadata=None, mode="new")

    before = web_server.topics_signature(db)
    db.topic_rename(topic_id=topic.topic_id, new_name="after", rewrite_messages=False)
    after = web_server.topics_signature(db)

    assert before != after
    assert any(item[0] == topic.topic_id and item[1] == "after" for item in after)


def test_topic_stream_state_changes_after_non_tail_message_delete(
    tmp_path: Path, monkeypatch
) -> None:
    prepare_static_bundle(tmp_path, monkeypatch)
    install_fake_now(monkeypatch)
    web_server.init_db(str(tmp_path / "bus.sqlite"))
    db = web_server.get_db()
    topic = db.topic_create(name="pink", metadata=None, mode="new")
    seed_topic_message(db, topic.topic_id, "alice", "one")
    seed_topic_message(db, topic.topic_id, "alice", "two")
    before = web_server.topic_stream_state(db, topic_id=topic.topic_id)

    messages = db.get_messages(topic_id=topic.topic_id, limit=20)
    db.delete_messages_batch(topic_id=topic.topic_id, message_ids=[messages[0].message_id])
    after = web_server.topic_stream_state(db, topic_id=topic.topic_id)

    assert before["last_seq"] == after["last_seq"] == messages[-1].seq
    assert before["message_count"] == 2
    assert after["message_count"] == 1
    assert before != after


def test_api_topic_export_uses_iso_timestamps(tmp_path: Path, monkeypatch) -> None:
    prepare_static_bundle(tmp_path, monkeypatch)
    install_fake_now(monkeypatch)
    web_server.init_db(str(tmp_path / "bus.sqlite"))
    db = web_server.get_db()
    topic = db.topic_create(name="alpha", metadata=None, mode="new")
    seed_topic_message(db, topic.topic_id, "alice", "hello")

    with TestClient(web_server.app) as client:
        res = client.get(f"/api/topics/{topic.topic_id}/export")

    assert res.status_code == 200
    expected = datetime.fromtimestamp(1_700_000_001, UTC).isoformat().replace("+00:00", "Z")
    assert expected in res.text
