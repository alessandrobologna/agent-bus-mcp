from __future__ import annotations

import json
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
        assert "topic_presence" in a_tool_names
        assert "cursor_reset" in a_tool_names
        assert "sync" in a_tool_names

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

            sent = await agent_a.call_tool(
                "sync",
                {
                    "topic_id": topic_id,
                    "outbox": [{"content_markdown": "hello", "message_type": "message"}],
                    "wait_seconds": 0,
                },
            )
            assert sent.isError is False
            assert sent.structuredContent["received_count"] == 0
            assert sent.structuredContent["sent"][0]["duplicate"] is False
            sent_msg = sent.structuredContent["sent"][0]["message"]
            assert sent_msg["sender"] == "red-squirrel"
            assert sent_msg["content_markdown"] == "hello"

            drained = await agent_b.call_tool(
                "sync",
                {"topic_id": topic_id, "wait_seconds": 0},
            )
            assert drained.isError is False
            assert drained.structuredContent["status"] == "ready"
            assert drained.structuredContent["received_count"] == 1
            msg_hello = drained.structuredContent["received"][0]
            assert msg_hello["sender"] == "red-squirrel"
            assert msg_hello["content_markdown"] == "hello"

            # Compat: some clients accidentally pass outbox as a JSON-encoded string.
            sent_json_string = await agent_a.call_tool(
                "sync",
                {
                    "topic_id": topic_id,
                    "outbox": json.dumps(
                        [{"content_markdown": "hello-json", "message_type": "message"}]
                    ),
                    "wait_seconds": 0,
                },
            )
            assert sent_json_string.isError is False
            assert sent_json_string.structuredContent["sent"][0]["duplicate"] is False
            assert (
                sent_json_string.structuredContent["sent"][0]["message"]["content_markdown"]
                == "hello-json"
            )

            drained_json_string = await agent_b.call_tool(
                "sync",
                {"topic_id": topic_id, "wait_seconds": 0},
            )
            assert drained_json_string.isError is False
            assert drained_json_string.structuredContent["received_count"] == 1
            msg_hello_json = drained_json_string.structuredContent["received"][0]
            assert msg_hello_json["sender"] == "red-squirrel"
            assert msg_hello_json["content_markdown"] == "hello-json"

            invalid_json_outbox = await agent_a.call_tool(
                "sync",
                {"topic_id": topic_id, "outbox": "not json", "wait_seconds": 0},
            )
            assert invalid_json_outbox.isError is True

            presence = await agent_a.call_tool(
                "topic_presence",
                {"topic_id": topic_id, "window_seconds": 3600},
            )
            assert presence.isError is False
            peer_names = {p["agent_name"] for p in presence.structuredContent["peers"]}
            assert "red-squirrel" in peer_names
            assert "crimson-cat" in peer_names

            reply = await agent_b.call_tool(
                "sync",
                {
                    "topic_id": topic_id,
                    "outbox": [
                        {
                            "content_markdown": "world",
                            "message_type": "message",
                            "reply_to": msg_hello["message_id"],
                        }
                    ],
                    "wait_seconds": 0,
                },
            )
            assert reply.isError is False
            assert reply.structuredContent["sent"][0]["message"]["sender"] == "crimson-cat"

            received = await agent_a.call_tool(
                "sync",
                {"topic_id": topic_id, "wait_seconds": 0},
            )
            assert received.isError is False
            assert received.structuredContent["status"] == "ready"
            assert received.structuredContent["received_count"] == 1
            assert received.structuredContent["received"][0]["sender"] == "crimson-cat"
            assert received.structuredContent["received"][0]["content_markdown"] == "world"

            reset = await agent_a.call_tool("cursor_reset", {"topic_id": topic_id})
            assert reset.isError is False
            assert reset.structuredContent["cursor"]["last_seq"] == 0

            replayed = await agent_a.call_tool(
                "sync",
                {"topic_id": topic_id, "wait_seconds": 0},
            )
            assert replayed.isError is False
            assert replayed.structuredContent["status"] == "ready"
            assert replayed.structuredContent["received_count"] == 1
            assert replayed.structuredContent["received"][0]["sender"] == "crimson-cat"
            assert replayed.structuredContent["received"][0]["content_markdown"] == "world"
