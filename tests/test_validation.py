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
async def test_student_ask_max_question_length_validation(tmp_path):
    env = {
        **os.environ,
        "AGENT_BUS_DB": str(tmp_path / "bus.sqlite"),
        "AGENT_BUS_MAX_QUESTION_CHARS": "3",
    }
    student_server = StdioServerParameters(command=_bin("agent-bus-student"), env=env)

    async with (
        stdio_client(student_server) as (s_read, s_write),
        ClientSession(s_read, s_write) as student,
    ):
        await student.initialize()
        res = await student.call_tool(
            "ask",
            {"topic_id": "t", "question": "hello", "wait_seconds": 0},
        )
        assert res.isError is True
        assert res.structuredContent["error"]["code"] == "INVALID_ARGUMENT"
        assert "exceeds max length" in res.structuredContent["error"]["message"]
