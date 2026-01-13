from __future__ import annotations

from collections.abc import Iterable

from agent_bus.models import Answer


def render_answer(*, topic_id: str, answer_markdown: str, suggested_followups: list[str]) -> str:
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


def render_answers(*, topic_id: str, answers: list[Answer]) -> str:
    merged_followups: list[str] = []
    seen: set[str] = set()

    def _iter_followups(a: Answer) -> Iterable[str]:
        raw = a.payload.get("suggested_followups") or []
        if not isinstance(raw, list):
            return []
        return [f for f in raw if isinstance(f, str)]

    parts: list[str] = []
    for a in answers:
        md = str(a.payload.get("answer_markdown", "")).rstrip()
        parts.append(f"### Answer from {a.answered_by}\n\n{md}".rstrip())
        for f in _iter_followups(a):
            if f not in seen:
                seen.add(f)
                merged_followups.append(f)

    combined_markdown = "\n\n".join(parts).rstrip()
    return render_answer(
        topic_id=topic_id,
        answer_markdown=combined_markdown,
        suggested_followups=merged_followups,
    )
