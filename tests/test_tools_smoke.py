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

    teacher_server = StdioServerParameters(command=_bin("agent-bus-teacher"), env=env)
    student_server = StdioServerParameters(command=_bin("agent-bus-student"), env=env)

    async with (
        stdio_client(teacher_server) as (t_read, t_write),
        ClientSession(t_read, t_write) as teacher,
    ):
        await teacher.initialize()
        teacher_tools = await teacher.list_tools()
        teacher_tool_names = {t.name for t in teacher_tools.tools}
        assert "topic_create" in teacher_tool_names
        assert "teacher_publish" in teacher_tool_names
        assert "ask" not in teacher_tool_names

        created = await teacher.call_tool("topic_create", {"name": "pink"})
        assert created.isError is False
        topic_id = created.structuredContent["topic_id"]

        async with (
            stdio_client(student_server) as (s_read, s_write),
            ClientSession(s_read, s_write) as student,
        ):
            await student.initialize()
            student_tools = await student.list_tools()
            student_tool_names = {t.name for t in student_tools.tools}
            assert "ask" in student_tool_names
            assert "ask_poll" in student_tool_names
            assert "teacher_publish" not in student_tool_names

            resolved = await student.call_tool("topic_resolve", {"name": "pink"})
            assert resolved.isError is False
            assert resolved.structuredContent["topic_id"] == topic_id

            asked = await student.call_tool(
                "ask",
                {"topic_id": topic_id, "question": "hello", "wait_seconds": 0},
            )
            assert asked.isError is False
            assert asked.structuredContent["status"] == "queued"
            question_id = asked.structuredContent["question_id"]

            drained = await teacher.call_tool("teacher_drain", {"topic_id": topic_id, "limit": 20})
            pending = drained.structuredContent["pending"]
            assert any(q["question_id"] == question_id for q in pending)

            published = await teacher.call_tool(
                "teacher_publish",
                {
                    "topic_id": topic_id,
                    "responses": [
                        {
                            "question_id": question_id,
                            "answer_markdown": "Answer here.",
                            "repo_pointers": ["a.py"],
                            "suggested_followups": ["Next?"],
                            "teacher_notes": "internal",
                        }
                    ],
                },
            )
            assert published.isError is False
            assert published.structuredContent["saved"] == 1

            polled = await student.call_tool(
                "ask_poll",
                {"topic_id": topic_id, "question_id": question_id},
            )
            assert polled.isError is False
            assert polled.structuredContent["status"] == "answered"
            assert "FOLLOW_UP_REQUIRED" in polled.content[0].text
            assert "teacher_notes" not in polled.structuredContent["answer_payload"]

            # Cancelled questions should not be drained.
            asked2 = await student.call_tool(
                "ask",
                {"topic_id": topic_id, "question": "second", "wait_seconds": 0},
            )
            q2 = asked2.structuredContent["question_id"]
            await student.call_tool(
                "ask_cancel",
                {"topic_id": topic_id, "question_id": q2, "reason": "nope"},
            )

            drained2 = await teacher.call_tool("teacher_drain", {"topic_id": topic_id, "limit": 50})
            pending2 = drained2.structuredContent["pending"]
            assert all(q["question_id"] != q2 for q in pending2)

            # Truncation warnings: repo_pointers > 10 and suggested_followups > 5.
            asked3 = await student.call_tool(
                "ask",
                {"topic_id": topic_id, "question": "truncate", "wait_seconds": 0},
            )
            q3 = asked3.structuredContent["question_id"]

            published2 = await teacher.call_tool(
                "teacher_publish",
                {
                    "topic_id": topic_id,
                    "responses": [
                        {
                            "question_id": q3,
                            "answer_markdown": "Truncation test.",
                            "repo_pointers": [f"file{i}.py" for i in range(15)],
                            "suggested_followups": [f"F{i}?" for i in range(7)],
                            "teacher_notes": "internal",
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

            polled2 = await student.call_tool(
                "ask_poll",
                {"topic_id": topic_id, "question_id": q3},
            )
            assert polled2.isError is False
            payload = polled2.structuredContent["answer_payload"]
            assert len(payload["repo_pointers"]) == 10
            assert len(payload["suggested_followups"]) == 5
            assert "teacher_notes" not in payload
