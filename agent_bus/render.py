from __future__ import annotations

from typing import Any


def strip_teacher_notes(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    out = dict(payload)
    out.pop("teacher_notes", None)
    return out


def render_student_answer(
    *, topic_id: str, answer_markdown: str, suggested_followups: list[str]
) -> str:
    followups = suggested_followups[:5]

    lines: list[str] = []
    lines.append(answer_markdown.rstrip())
    lines.append("")
    lines.append("---")
    lines.append("FOLLOW_UP_REQUIRED")
    lines.append("Choose ONE follow-up question and call:")
    lines.append(f'ask(topic_id="{topic_id}", question="<your question>")')
    lines.append("")
    lines.append("Suggested follow-ups:")
    for i, q in enumerate(followups, start=1):
        lines.append(f"{i}) {q}")
    lines.append("")
    lines.append("If you fully understand, reply with:")
    lines.append("NO_FOLLOWUP_NEEDED")
    lines.append("and provide a 3-5 bullet summary of what you learned.")
    return "\n".join(lines).rstrip() + "\n"
