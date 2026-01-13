from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _bin(name: str) -> str:
    return str(Path(sys.executable).with_name(name))


@pytest.mark.anyio
async def test_two_process_smoke(tmp_path):
    db_path = str(tmp_path / "bus.sqlite")
    env = {**os.environ, "AGENT_BUS_DB": db_path}

    peer_server = StdioServerParameters(command=_bin("agent-bus"), env=env)

    async with (
        stdio_client(peer_server) as (a_read, a_write),
        ClientSession(a_read, a_write) as agent_a,
    ):
        await agent_a.initialize()
        a_tools = await agent_a.list_tools()
        a_tool_names = {t.name for t in a_tools.tools}
        assert "topic_create" in a_tool_names
        assert "topic_join" in a_tool_names
        assert "ask" in a_tool_names
        assert "pending_list" in a_tool_names
        assert "answer" in a_tool_names
        assert "question_mark_answered" in a_tool_names

        created = await agent_a.call_tool("topic_create", {"name": "pink"})
        assert created.isError is False
        topic_id = created.structuredContent["topic_id"]

        joined_a = await agent_a.call_tool(
            "topic_join",
            {"agent_name": "red-squirrel", "topic_id": topic_id},
        )
        assert joined_a.isError is False

        async with (
            stdio_client(peer_server) as (b_read, b_write),
            ClientSession(b_read, b_write) as agent_b,
        ):
            await agent_b.initialize()

            joined_b = await agent_b.call_tool(
                "topic_join",
                {"agent_name": "crimson-cat", "name": "pink"},
            )
            assert joined_b.isError is False
            assert joined_b.structuredContent["topic_id"] == topic_id

            asked = await agent_a.call_tool(
                "ask",
                {"topic_id": topic_id, "question": "hello", "wait_seconds": 0},
            )
            assert asked.isError is False
            assert asked.structuredContent["status"] == "queued"
            question_id = asked.structuredContent["question_id"]
            assert asked.structuredContent["next_action"]["tool"] == "ask_poll"
            assert asked.structuredContent["next_action"]["args"]["topic_id"] == topic_id
            assert asked.structuredContent["next_action"]["args"]["question_id"] == question_id
            assert asked.structuredContent["client_plan"]["max_attempts"] == 3

            # Agents must not answer their own questions.
            forbidden = await agent_a.call_tool(
                "answer",
                {
                    "topic_id": topic_id,
                    "responses": [
                        {
                            "question_id": question_id,
                            "answer_markdown": "Should fail.",
                            "repo_pointers": [],
                            "suggested_followups": ["Next?"],
                        }
                    ],
                },
            )
            assert forbidden.isError is True
            assert forbidden.structuredContent["error"]["code"] == "FORBIDDEN_SELF_ANSWER"
            assert forbidden.structuredContent["forbidden_question_ids"] == [question_id]

            # The asker should not see their own question in pending_list().
            drained_self = await agent_a.call_tool(
                "pending_list", {"topic_id": topic_id, "limit": 20, "wait_seconds": 0}
            )
            assert drained_self.isError is False
            assert drained_self.structuredContent["pending"] == []

            drained = await agent_b.call_tool(
                "pending_list", {"topic_id": topic_id, "limit": 20, "wait_seconds": 0}
            )
            pending = drained.structuredContent["pending"]
            assert any(q["question_id"] == question_id for q in pending)

            published = await agent_b.call_tool(
                "answer",
                {
                    "topic_id": topic_id,
                    "responses": [
                        {
                            "question_id": question_id,
                            "answer_markdown": "Answer here.",
                            "repo_pointers": ["a.py"],
                            "suggested_followups": ["Next?"],
                        }
                    ],
                },
            )
            assert published.isError is False
            assert published.structuredContent["saved"] == 1
            assert published.structuredContent["next_action"]["tool"] == "pending_list"
            assert published.structuredContent["client_plan"]["max_attempts"] == 3

            # An agent must not answer the same question twice.
            dup = await agent_b.call_tool(
                "answer",
                {
                    "topic_id": topic_id,
                    "responses": [
                        {
                            "question_id": question_id,
                            "answer_markdown": "Duplicate should fail.",
                            "repo_pointers": [],
                            "suggested_followups": ["Next?"],
                        }
                    ],
                },
            )
            assert dup.isError is True
            assert dup.structuredContent["error"]["code"] == "FORBIDDEN_ALREADY_ANSWERED"
            assert dup.structuredContent["already_answered_question_ids"] == [question_id]

            async with (
                stdio_client(peer_server) as (c_read, c_write),
                ClientSession(c_read, c_write) as agent_c,
            ):
                await agent_c.initialize()
                joined_c = await agent_c.call_tool(
                    "topic_join",
                    {"agent_name": "green-gecko", "topic_id": topic_id},
                )
                assert joined_c.isError is False

                drained_c = await agent_c.call_tool(
                    "pending_list", {"topic_id": topic_id, "limit": 20, "wait_seconds": 0}
                )
                pending_c = drained_c.structuredContent["pending"]
                assert any(q["question_id"] == question_id for q in pending_c)

                published_c = await agent_c.call_tool(
                    "answer",
                    {
                        "topic_id": topic_id,
                        "responses": [
                            {
                                "question_id": question_id,
                                "answer_markdown": "Second answer.",
                                "repo_pointers": [],
                                "suggested_followups": ["Another?"],
                            }
                        ],
                    },
                )
                assert published_c.isError is False
                assert published_c.structuredContent["saved"] == 1

            polled = await agent_a.call_tool(
                "ask_poll", {"topic_id": topic_id, "question_id": question_id}
            )
            assert polled.isError is False
            assert polled.structuredContent["status"] == "answered"
            assert "FOLLOW_UP_REQUIRED" in polled.content[0].text
            assert polled.structuredContent["answers_count"] == 2

            closed = await agent_a.call_tool(
                "question_mark_answered", {"topic_id": topic_id, "question_id": question_id}
            )
            assert closed.isError is False
            assert closed.structuredContent["status"] == "answered"

            drained_after_close = await agent_b.call_tool(
                "pending_list", {"topic_id": topic_id, "limit": 20, "wait_seconds": 0}
            )
            pending_after_close = drained_after_close.structuredContent["pending"]
            assert all(q["question_id"] != question_id for q in pending_after_close)

            # Only the asking agent can cancel.
            asked2 = await agent_a.call_tool(
                "ask",
                {"topic_id": topic_id, "question": "second", "wait_seconds": 0},
            )
            q2 = asked2.structuredContent["question_id"]

            bad_cancel = await agent_b.call_tool(
                "ask_cancel",
                {"topic_id": topic_id, "question_id": q2, "reason": "nope"},
            )
            assert bad_cancel.isError is True
            assert bad_cancel.structuredContent["error"]["code"] == "FORBIDDEN_NOT_ASKER"

            ok_cancel = await agent_a.call_tool(
                "ask_cancel",
                {"topic_id": topic_id, "question_id": q2, "reason": "nope"},
            )
            assert ok_cancel.isError is False

            drained2 = await agent_b.call_tool(
                "pending_list", {"topic_id": topic_id, "limit": 50, "wait_seconds": 0}
            )
            pending2 = drained2.structuredContent["pending"]
            assert all(q["question_id"] != q2 for q in pending2)

            # Truncation warnings: repo_pointers > 10 and suggested_followups > 5.
            asked3 = await agent_a.call_tool(
                "ask",
                {"topic_id": topic_id, "question": "truncate", "wait_seconds": 0},
            )
            q3 = asked3.structuredContent["question_id"]

            published2 = await agent_b.call_tool(
                "answer",
                {
                    "topic_id": topic_id,
                    "responses": [
                        {
                            "question_id": q3,
                            "answer_markdown": "Truncation test.",
                            "repo_pointers": [f"file{i}.py" for i in range(15)],
                            "suggested_followups": [f"F{i}?" for i in range(7)],
                        }
                    ],
                },
            )
            assert published2.isError is False
            warning_codes = {
                w["code"] for w in (published2.structuredContent.get("warnings") or [])
            }
            assert "REPO_POINTERS_TRUNCATED" in warning_codes
            assert "FOLLOWUPS_TRUNCATED" in warning_codes

            polled2 = await agent_a.call_tool("ask_poll", {"topic_id": topic_id, "question_id": q3})
            assert polled2.isError is False
            assert polled2.structuredContent["answers_count"] == 1
            payload = polled2.structuredContent["answers"][0]["payload"]
            assert len(payload["repo_pointers"]) == 10
            assert len(payload["suggested_followups"]) == 5
