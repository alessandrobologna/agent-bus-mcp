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

## Reviewer Default When Asked To Create A Topic

Pattern:
1. finish the review locally
2. create the topic
3. join as `reviewer`
4. post the findings in the same step
5. ask for fixes or re-review

Example:

```md
Review context:
- branch: `codex/example-branch`
- merge base: `abc1234`

Findings:
1. [P1] ...
2. [P2] ...

Please address the valid findings in this branch, post back what changed, and ask for another review.
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
3. `sync(wait_seconds=0)` until caught up
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

## Join Existing Topic With Pending Review

Pattern:
1. join the requested topic
2. sync with `wait_seconds=0` until caught up
3. if you find pending review items and the user only asked you to inspect or join, summarize them to the user
4. ask whether they want implementation before making changes

Example message to the user:

```md
I joined the requested topic and found pending review findings:
- [P1] ...
- [P2] ...

If you want, I can address those findings in the same topic and post back for another review.
```
