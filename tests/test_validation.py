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
async def test_peer_ask_max_question_length_validation(tmp_path):
    env = {
        **os.environ,
        "AGENT_BUS_DB": str(tmp_path / "bus.sqlite"),
        "AGENT_BUS_MAX_QUESTION_CHARS": "3",
    }
    peer_server = StdioServerParameters(command=_bin("agent-bus"), env=env)

    async with (
        stdio_client(peer_server) as (s_read, s_write),
        ClientSession(s_read, s_write) as peer,
    ):
        await peer.initialize()
        created = await peer.call_tool("topic_create", {"name": "pink"})
        topic_id = created.structuredContent["topic_id"]
        await peer.call_tool("topic_join", {"agent_name": "a", "topic_id": topic_id})

        res = await peer.call_tool(
            "ask",
            {"topic_id": topic_id, "question": "hello", "wait_seconds": 0},
        )
        assert res.isError is True
        assert res.structuredContent["error"]["code"] == "INVALID_ARGUMENT"
        assert "exceeds max length" in res.structuredContent["error"]["message"]
