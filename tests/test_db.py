from __future__ import annotations

from agent_bus.db import (
    AgentBusDB,
    AlreadyAnsweredError,
    QuestionNotFoundError,
    SelfAnswerForbiddenError,
    TopicClosedError,
    TopicNotFoundError,
)


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
        db.question_create(topic_id=t.topic_id, asked_by="a", question_text="hi")
    except TopicClosedError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected TopicClosedError")


def test_question_list_answerable_excludes_cancelled_and_already_answered(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    q1 = db.question_create(topic_id=t.topic_id, asked_by="a", question_text="q1")
    q2 = db.question_create(topic_id=t.topic_id, asked_by="a", question_text="q2")

    db.question_cancel(topic_id=t.topic_id, question_id=q1.question_id, reason="nvm")

    db.answer_insert_batch(
        topic_id=t.topic_id,
        answered_by="b",
        items=[(q2.question_id, {"answer_markdown": "hi", "suggested_followups": ["x"]})],
    )

    pending_b = db.question_list_answerable(topic_id=t.topic_id, limit=20, agent_name="b")
    assert [q.question_id for q in pending_b] == []

    pending_c = db.question_list_answerable(topic_id=t.topic_id, limit=20, agent_name="c")
    assert [q.question_id for q in pending_c] == [q2.question_id]


def test_answer_insert_batch_inserts_answer(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")
    q = db.question_create(topic_id=t.topic_id, asked_by="a", question_text="q1")

    saved, skipped = db.answer_insert_batch(
        topic_id=t.topic_id,
        answered_by="b",
        items=[(q.question_id, {"answer_markdown": "hi", "suggested_followups": ["x"]})],
    )
    assert (saved, skipped) == (1, 0)

    q2 = db.question_get(question_id=q.question_id)
    assert q2.status == "pending"

    answers = db.answers_list(question_id=q.question_id)
    assert len(answers) == 1
    assert answers[0].answered_by == "b"
    assert answers[0].payload["answer_markdown"] == "hi"


def test_answer_insert_batch_forbids_self_answer(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")
    q1 = db.question_create(topic_id=t.topic_id, asked_by="a", question_text="q1")
    q2 = db.question_create(topic_id=t.topic_id, asked_by="a", question_text="q2")

    try:
        db.answer_insert_batch(
            topic_id=t.topic_id,
            answered_by="a",
            items=[
                (q1.question_id, {"answer_markdown": "nope", "suggested_followups": ["x"]}),
                (q2.question_id, {"answer_markdown": "nope", "suggested_followups": ["x"]}),
            ],
        )
    except SelfAnswerForbiddenError as e:
        assert set(e.question_ids) == {q1.question_id, q2.question_id}
    else:  # pragma: no cover
        raise AssertionError("expected SelfAnswerForbiddenError")


def test_answer_insert_batch_forbids_duplicate_answers_by_same_agent(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")
    q = db.question_create(topic_id=t.topic_id, asked_by="a", question_text="q1")

    db.answer_insert_batch(
        topic_id=t.topic_id,
        answered_by="b",
        items=[(q.question_id, {"answer_markdown": "hi", "suggested_followups": ["x"]})],
    )

    try:
        db.answer_insert_batch(
            topic_id=t.topic_id,
            answered_by="b",
            items=[(q.question_id, {"answer_markdown": "again", "suggested_followups": ["y"]})],
        )
    except AlreadyAnsweredError as e:
        assert e.question_ids == [q.question_id]
    else:  # pragma: no cover
        raise AssertionError("expected AlreadyAnsweredError")


def test_question_mark_answered_idempotent(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")
    q = db.question_create(topic_id=t.topic_id, asked_by="a", question_text="q1")

    q1, already1 = db.question_mark_answered(topic_id=t.topic_id, question_id=q.question_id)
    assert already1 is False
    assert q1.status == "answered"

    q2, already2 = db.question_mark_answered(topic_id=t.topic_id, question_id=q.question_id)
    assert already2 is True
    assert q2.status == "answered"


def test_question_mark_answered_rejects_cancelled(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")
    q = db.question_create(topic_id=t.topic_id, asked_by="a", question_text="q1")

    db.question_cancel(topic_id=t.topic_id, question_id=q.question_id, reason=None)
    try:
        db.question_mark_answered(topic_id=t.topic_id, question_id=q.question_id)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_question_cancel_missing_raises(tmp_path):
    db = AgentBusDB(path=str(tmp_path / "bus.sqlite"))
    t = db.topic_create(name="pink", metadata=None, mode="new")

    try:
        db.question_cancel(topic_id=t.topic_id, question_id="missing", reason=None)
    except QuestionNotFoundError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected QuestionNotFoundError")
