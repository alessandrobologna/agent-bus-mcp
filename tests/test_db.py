from __future__ import annotations

from agent_bus.db import AgentBusDB, QuestionNotFoundError, TopicClosedError, TopicNotFoundError


def test_topic_create_reuse_newest_open(tmp_path):
    db_path = tmp_path / "bus.sqlite"
    db = AgentBusDB(path=str(db_path))

    t2 = db.topic_create(name="pink", metadata={"a": 1}, mode="new")

    reused = db.topic_create(name="pink", metadata={"ignored": True}, mode="reuse")

    assert reused.topic_id == t2.topic_id
    assert reused.name == "pink"
    assert reused.status == "open"


def test_topic_resolve_closed_requires_flag(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))

    t = db.topic_create(name="pink", metadata=None, mode="new")
    db.topic_close(topic_id=t.topic_id, reason="done")

    try:
        db.topic_resolve(name="pink", allow_closed=False)
    except TopicNotFoundError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected TopicNotFoundError")

    resolved = db.topic_resolve(name="pink", allow_closed=True)
    assert resolved.topic_id == t.topic_id
    assert resolved.status == "closed"


def test_topic_close_idempotent(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    closed1, already1 = db.topic_close(topic_id=t.topic_id, reason="first")
    assert already1 is False
    assert closed1.status == "closed"
    assert closed1.close_reason == "first"
    assert closed1.closed_at is not None

    closed2, already2 = db.topic_close(topic_id=t.topic_id, reason="second")
    assert already2 is True
    assert closed2.closed_at == closed1.closed_at
    assert closed2.close_reason == "first"


def test_question_create_topic_closed(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")
    db.topic_close(topic_id=t.topic_id, reason=None)

    try:
        db.question_create(topic_id=t.topic_id, asked_by="student", question_text="hi")
    except TopicClosedError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected TopicClosedError")


def test_teacher_drain_excludes_cancelled(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    q1 = db.question_create(topic_id=t.topic_id, asked_by="student", question_text="q1")
    q2 = db.question_create(topic_id=t.topic_id, asked_by="student", question_text="q2")

    db.question_cancel(topic_id=t.topic_id, question_id=q1.question_id, reason="nvm")

    pending = db.teacher_drain(topic_id=t.topic_id, limit=20)
    assert [q.question_id for q in pending] == [q2.question_id]


def test_teacher_publish_one_updates_pending(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")
    q = db.question_create(topic_id=t.topic_id, asked_by="student", question_text="q1")

    saved = db.teacher_publish_one(
        topic_id=t.topic_id,
        question_id=q.question_id,
        answered_by="teacher",
        answer_payload={"answer_markdown": "hi", "suggested_followups": ["x"]},
    )
    assert saved is True

    q2 = db.question_get(question_id=q.question_id)
    assert q2.status == "answered"
    assert q2.answer_payload is not None
    assert q2.answer_payload["answer_markdown"] == "hi"


def test_question_cancel_missing_raises(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    try:
        db.question_cancel(topic_id=t.topic_id, question_id="missing", reason=None)
    except QuestionNotFoundError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected QuestionNotFoundError")
