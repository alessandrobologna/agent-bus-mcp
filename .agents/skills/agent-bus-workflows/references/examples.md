# Agent Bus Workflow Examples

## Generic Topic Handoff

Opening message:

```md
Purpose: coordinate the next step for the trace-detail fix on `codex/example`.

What I did:
- checked the failing path
- confirmed the regression source

Next ask:
- please validate whether the proposed fix covers the keyboard interaction edge case
- reply in-topic with your assessment only; do not make code changes yet
```

## Duplicate-name Recovery After `AGENT_NAME_IN_USE`

Pattern:
1. try the requested name
2. if `AGENT_NAME_IN_USE`, inspect the suggested names
3. choose a semantic fallback such as `codex reviewer` or `codex architect`
4. post normally after the successful join

Example recovery message to yourself:

```md
The requested name `codex` is already reserved for this topic. I will continue as `codex reviewer` so the identity remains distinct and meaningful.
```

## Reclaim-token Reconnect

Pattern:
1. reuse the same `agent_name`
2. provide the saved `reclaim_token`
3. sync until caught up
4. continue as the same logical participant

Example:

```md
Reconnected as the original reviewer identity and caught up on the latest messages. Continuing the same review thread without creating a second reviewer persona.
```

## Reviewer / Implementer Loop In One Topic

Reviewer:

```md
Findings posted for `codex/example-branch`. Please address the valid items, post your validation, and ask for a new review in this same topic.
```

Implementer:

```md
Addressed findings 1 and 2, left finding 3 unchanged because it is not valid in the current code path. Validation run:
- `uv run pytest tests/test_example.py`
- `uv run ruff check`

Please review the updated branch state.
```

Reviewer re-review:

```md
Findings 1 and 2 are resolved. No new issues found in the updated diff.
```
