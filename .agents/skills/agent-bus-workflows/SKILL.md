---
name: agent-bus-workflows
description: Use when an agent needs to collaborate through Agent Bus, coordinate via topic threads, run reviewer or implementer handoffs, recover from AGENT_NAME_IN_USE, or continue a code-review loop in an existing Agent Bus topic.
---

# Agent Bus Workflows

Use this skill when work should happen through Agent Bus instead of only in chat.
This skill is written to be portable across projects: the repository under discussion can be any codebase, as long as the host agent has Agent Bus MCP configured and available.

This skill covers:
- creating or reusing discussion topics
- joining safely with a stable agent identity
- short bounded polling loops
- reviewer and implementer handoffs in one topic
- follow-up validation in the same topic

Read [references/review-loop.md](references/review-loop.md) when the task is specifically a review or fix loop.
Read [references/examples.md](references/examples.md) when you need concrete message patterns or recovery examples.

## Core Rules

- Prefer semantic agent names such as `reviewer`, `implementer`, `codex reviewer`, or `codex architect`.
- Treat the returned `agent_name` and `reclaim_token` as authoritative.
- If `AGENT_NAME_IN_USE` is returned, either:
  - retry with the original `reclaim_token` if this is the same logical agent reconnecting, or
  - choose a meaningful fallback name from the suggestions.
- Prefer joining by `topic_id` when available. Use `name` only when a topic id is not available yet.
- Keep `sync.max_items <= 20`.
- When catching up on topic history, call `sync(wait_seconds=0)` repeatedly until `has_more=false`.
- Keep polling bounded by default:
  - 3 rounds
  - `wait_seconds=5`
  - `max_items=20`
- Persist or remember the last successful `agent_name` plus `reclaim_token` for reconnects.
- If the user asks you to create a topic after you already have review findings, assume the intent includes joining as a reviewer and posting those findings in the new topic.
- If the user asks you to join or check an existing topic and you discover pending review findings, summarize them to the user and ask whether they want them addressed unless they already explicitly asked for implementation.

## Workflow: Start Or Join A Topic

### Start a new topic

Use this when the user did not provide an established topic.

1. Create a topic unless the user clearly asked to reuse an existing one.
2. Join it with a semantic role name. If you are posting existing review findings, join as `reviewer`.
3. If you already have review findings, post them immediately instead of creating an empty placeholder topic.
4. Otherwise post an opening message with enough context for the next participant to act without guessing.
5. Poll briefly after posting.

Opening messages should include:
- the purpose of the topic
- the branch, commit, or work area when relevant
- the specific ask for the next agent
- whether the next participant is expected to reply only or make changes

### Join an existing topic

Use this when the user provides a topic id, topic name, or says a thread already exists.

1. Join the topic first.
2. If only a topic name is available, resolve or join by name and prefer the returned `topic_id` afterward.
3. Call `sync(wait_seconds=0)` repeatedly until `has_more=false`.
4. If the topic contains pending review findings and the user only asked you to join or inspect the topic, summarize those findings to the user and ask whether they want them addressed.
5. Stop there until the user answers. Only post a reply in-topic or start implementing after the user confirms what they want next.

## Workflow: Generic Discussion Or Handoff

Use this for lightweight coordination, brainstorming, or passing work between agents.

1. Create or join the topic.
2. Call `sync(wait_seconds=0)` repeatedly until `has_more=false`.
3. Post a message that makes the next step obvious.
4. Poll up to 3 times for responses.
5. Stop and report back if nothing new arrives.

Good handoff messages include:
- what was done
- what remains
- any validation or uncertainty
- whether the next agent should reply only, investigate, or make changes
- the exact question or next action for the other agent

Do not leave vague messages like "take a look" or "thoughts?" without context.

## Workflow: Code Review Reviewer

Use this when you are reviewing changes and publishing findings through Agent Bus.

If the user asks you to create a topic after you already have findings, assume they want the full reviewer handoff: create the topic, join it as a reviewer, post the findings, and ask for fixes or re-review. Do not stop after topic creation alone.

1. Create a review topic unless one is already established.
2. Join with a reviewer-style name.
3. Post findings with enough context to act on them:
  - branch or commit under review
  - findings ordered by severity
  - why each finding matters
  - what kind of fix or validation is expected
4. Ask explicitly for fixes or for a follow-up validation pass.
5. Poll up to 3 times.
6. If implementer updates arrive, re-review in the same topic instead of starting a new one.

Reviewer messages should separate:
- resolved findings
- remaining findings
- new regressions introduced by the fix

## Workflow: Code Review Implementer

Use this when the user says there is review feedback in an Agent Bus topic and wants you to address it.

If the user only asked you to join or inspect a topic and you find pending review items there, do not assume implementation. First summarize the findings to the user and ask whether they want you to address them.

1. Join the provided topic.
2. Call `sync(wait_seconds=0)` repeatedly until `has_more=false`.
3. Summarize the findings into:
  - valid and will fix
  - unclear and needs user confirmation
  - not valid, with concrete reasoning
4. Implement only the valid findings.
5. Post back in the same topic with:
  - what changed
  - what was validated
  - which findings were intentionally not changed and why
  - a request for another review
6. Poll up to 3 times for reviewer follow-up.

Do not silently dismiss findings. If a finding is not valid, explain why in-topic or to the user.

## Workflow: Re-review

Use this after the implementer posts a fix and asks for validation.

1. Stay in the same topic.
2. Call `sync(wait_seconds=0)` repeatedly until `has_more=false`.
3. Validate the claimed fixes.
4. Post one of:
  - all reviewed findings resolved
  - some findings remain
  - new issue introduced
5. Poll briefly for any final follow-up.

Keep re-review messages short and decisive. Avoid re-triaging the entire history unless the new changes require it.

## Failure Recovery

### `AGENT_NAME_IN_USE`

- If this is the same agent reconnecting, retry with the saved `reclaim_token`.
- Otherwise choose a semantic fallback name from the suggestions.
- Do not keep retrying the same conflicting name without changing inputs.

### `AGENT_NOT_JOINED`

- Join the topic first, then retry `sync()` or cursor actions.

### `INVALID_ARGUMENT` on `sync.max_items`

- Reduce `max_items` to 20 or lower.
- Keep the default bounded polling settings unless the user asked for a different watch loop.

### No visible messages after posting

- Check `sync(wait_seconds=0).sent` first. With the default `include_self=false`, a successful post may still leave `received` empty.
- Use `include_self=true` only when you specifically need a self-echo.
- Reserve `cursor_reset(topic_id=..., last_seq=0)` for real replay or cursor-recovery needs, not routine confirmation of your own post.

### Polling discipline

- Poll only a few times by default.
- If nothing arrives after the bounded loop, stop and report the current state.
- Do not wait indefinitely unless the user explicitly asks for a watch loop.
