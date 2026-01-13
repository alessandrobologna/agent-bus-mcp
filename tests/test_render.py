from __future__ import annotations

from agent_bus.render import render_answer


def test_render_answer_contains_followup_block():
    text = render_answer(
        topic_id="t1",
        answer_markdown="Answer.",
        suggested_followups=["Q1", "Q2", "Q3"],
    )
    assert "FOLLOW_UP_REQUIRED" in text
    assert 'ask(topic_id="t1", question="<your question>")' in text
    assert "1) Q1" in text
    assert "NO_FOLLOWUP_NEEDED" in text
