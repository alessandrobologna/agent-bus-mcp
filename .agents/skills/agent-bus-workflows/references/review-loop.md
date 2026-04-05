# Review Loop Reference

Use this reference when the task is specifically a reviewer or implementer workflow in Agent Bus.

## Reviewer Opening Post

Use one message with:
- branch or commit under review
- findings ordered by severity
- concise validation notes
- explicit request for fixes and re-review

Example:

```md
Review context:
- branch: `codex/example-branch`
- merge base: `abc1234`

Findings:
1. [P1] ...
2. [P2] ...

Validation:
- `uv run pytest tests/test_example.py`
- `uv run ruff check`

Please address the valid findings in this branch, post back what changed, and ask for another review.
```

## Implementer Fix Summary

Use one message with:
- which findings were addressed
- which findings were not addressed and why
- what validation was run
- explicit request for another review

Example:

```md
Implemented fixes for findings 1 and 2.

Changes:
- ...
- ...

Not changed:
- finding 3 is not valid because ...

Validation:
- `uv run pytest tests/test_example.py`
- `uv run ruff check`

Please review the latest branch state and confirm whether the findings are resolved.
```

## Reviewer Re-validation Post

Use one message with:
- resolved findings
- remaining findings
- overall verdict

Example:

```md
Re-review result:
- finding 1 resolved
- finding 2 still open because ...

Overall:
- patch is not yet ready

Please address the remaining issue and post back for another pass.
```

## Review Polling Pattern

Default review polling:
- 3 rounds
- `wait_seconds=5`
- `max_items=20`

After the bounded loop:
- if new messages arrived, continue the workflow
- if not, stop and report that no reviewer or implementer response arrived yet
