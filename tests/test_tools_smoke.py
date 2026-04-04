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
    env = {**os.environ, "AGENT_BUS_DB": db_path, "AGENT_BUS_EMBEDDINGS_AUTOINDEX": "0"}

    peer_server = StdioServerParameters(command=_bin("agent-bus"), env=env)

    async with (
        stdio_client(peer_server) as (a_read, a_write),
        ClientSession(a_read, a_write) as agent_a,
    ):
        await agent_a.initialize()
        ping = await agent_a.call_tool("ping", {})
        assert ping.isError is False
        assert ping.structuredContent["spec_version"] == "v6.3"
        a_tools = await agent_a.list_tools()
        a_tool_names = {t.name for t in a_tools.tools}
        tools_by_name = {t.name: t for t in a_tools.tools}
        assert "topic_create" in a_tool_names
        assert "topic_join" in a_tool_names
        assert "topic_presence" in a_tool_names
        assert "cursor_reset" in a_tool_names
        assert "messages_search" in a_tool_names
        assert "sync" in a_tool_names
        assert tools_by_name["sync"].outputSchema is not None
        assert tools_by_name["messages_search"].outputSchema is not None
        assert (
            tools_by_name["sync"].inputSchema["properties"]["max_items"]["description"]
            == "Maximum number of items to return. Keep this small and no greater than 20; loop until has_more=false."
        )
        assert tools_by_name["sync"].inputSchema["properties"]["max_items"]["maximum"] == 20
        assert "exactly one of topic_id or name" in tools_by_name["topic_join"].description
        assert tools_by_name["topic_create"].inputSchema["properties"]["mode"]["enum"] == [
            "reuse",
            "new",
        ]
        assert "mode='new'" in tools_by_name["topic_create"].description
        assert "returned topic_id" in tools_by_name["topic_create"].description
        assert (
            "Use 'new' for a fresh topic"
            in tools_by_name["topic_create"].inputSchema["properties"]["mode"]["description"]
        )
        assert "topic_create(mode='new')" in tools_by_name["topic_join"].description
        assert "reclaim_token" in tools_by_name["topic_join"].description
        assert (
            "Prefer this after topic_create(mode='new')"
            in tools_by_name["topic_join"].inputSchema["properties"]["topic_id"]["description"]
        )
        assert (
            "reclaim_token"
            in tools_by_name["topic_join"].inputSchema["properties"]["reclaim_token"]["description"]
        )

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
            assert any("hello" in getattr(c, "text", "") for c in drained.content)

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

            # Search (FTS/hybrid) is optional depending on SQLite build; tool should exist either way.
            searched = await agent_a.call_tool(
                "messages_search",
                {
                    "query": "hello",
                    "topic_id": topic_id,
                    "mode": "fts",
                    "limit": 5,
                    "include_content": True,
                },
            )
            if searched.isError:
                assert "FTS5" in searched.structuredContent["error"]["message"]
            else:
                assert searched.structuredContent["count"] >= 1
                found = [
                    r
                    for r in searched.structuredContent["results"]
                    if r["message_id"] == sent_msg["message_id"]
                ]
                assert found
                assert found[0]["content_markdown"] == "hello"
                assert any("hello" in getattr(c, "text", "") for c in searched.content)

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


@pytest.mark.anyio
async def test_sync_schema_respects_configured_max_items(tmp_path):
    db_path = str(tmp_path / "bus.sqlite")
    env = {
        **os.environ,
        "AGENT_BUS_DB": db_path,
        "AGENT_BUS_EMBEDDINGS_AUTOINDEX": "0",
        "AGENT_BUS_MAX_SYNC_ITEMS": "3",
    }

    peer_server = StdioServerParameters(command=_bin("agent-bus"), env=env)

    async with (
        stdio_client(peer_server) as (read, write),
        ClientSession(read, write) as peer,
    ):
        await peer.initialize()
        tools = await peer.list_tools()
        tools_by_name = {t.name: t for t in tools.tools}
        sync_schema = tools_by_name["sync"].inputSchema["properties"]["max_items"]
        assert sync_schema["default"] == 3
        assert sync_schema["maximum"] == 3
        assert (
            sync_schema["description"]
            == "Maximum number of items to return. Keep this small and no greater than 3; loop until has_more=false."
        )

        created = await peer.call_tool("topic_create", {"name": "small-limit"})
        assert created.isError is False
        topic_id = created.structuredContent["topic_id"]

        joined = await peer.call_tool("topic_join", {"agent_name": "peer", "topic_id": topic_id})
        assert joined.isError is False

        synced = await peer.call_tool("sync", {"topic_id": topic_id, "wait_seconds": 0})
        assert synced.isError is False
        assert synced.structuredContent["received_count"] == 0


@pytest.mark.anyio
async def test_sync_schema_keeps_default_20_when_cap_is_higher(tmp_path):
    db_path = str(tmp_path / "bus.sqlite")
    env = {
        **os.environ,
        "AGENT_BUS_DB": db_path,
        "AGENT_BUS_EMBEDDINGS_AUTOINDEX": "0",
        "AGENT_BUS_MAX_SYNC_ITEMS": "50",
    }

    peer_server = StdioServerParameters(command=_bin("agent-bus"), env=env)

    async with (
        stdio_client(peer_server) as (read, write),
        ClientSession(read, write) as peer,
    ):
        await peer.initialize()
        tools = await peer.list_tools()
        tools_by_name = {t.name: t for t in tools.tools}
        sync_schema = tools_by_name["sync"].inputSchema["properties"]["max_items"]
        assert sync_schema["default"] == 20
        assert sync_schema["maximum"] == 50
        assert (
            sync_schema["description"]
            == "Maximum number of items to return. Keep this small and no greater than 50; loop until has_more=false."
        )

        created = await peer.call_tool("topic_create", {"name": "high-limit"})
        assert created.isError is False
        topic_id = created.structuredContent["topic_id"]

        joined = await peer.call_tool("topic_join", {"agent_name": "peer", "topic_id": topic_id})
        assert joined.isError is False

        for i in range(25):
            sent = await peer.call_tool(
                "sync",
                {
                    "topic_id": topic_id,
                    "outbox": [{"content_markdown": f"msg-{i}", "message_type": "message"}],
                    "wait_seconds": 0,
                },
            )
            assert sent.isError is False

        async with (
            stdio_client(peer_server) as (read_b, write_b),
            ClientSession(read_b, write_b) as peer_b,
        ):
            await peer_b.initialize()
            joined_b = await peer_b.call_tool(
                "topic_join",
                {"agent_name": "peer-b", "topic_id": topic_id},
            )
            assert joined_b.isError is False

            replayed = await peer_b.call_tool("sync", {"topic_id": topic_id, "wait_seconds": 0})
            assert replayed.isError is False
            assert replayed.structuredContent["received_count"] == 20
            assert replayed.structuredContent["has_more"] is True


@pytest.mark.anyio
async def test_topic_join_rejects_duplicates_and_supports_reclaim(tmp_path):
    db_path = str(tmp_path / "bus.sqlite")
    env = {**os.environ, "AGENT_BUS_DB": db_path, "AGENT_BUS_EMBEDDINGS_AUTOINDEX": "0"}
    peer_server = StdioServerParameters(command=_bin("agent-bus"), env=env)

    async with (
        stdio_client(peer_server) as (a_read, a_write),
        ClientSession(a_read, a_write) as agent_a,
    ):
        await agent_a.initialize()
        created = await agent_a.call_tool("topic_create", {"name": "dupe-name"})
        assert created.isError is False
        topic_id = created.structuredContent["topic_id"]

        joined_a = await agent_a.call_tool(
            "topic_join",
            {"agent_name": "codex", "topic_id": topic_id},
        )
        assert joined_a.isError is False
        reclaim_token = joined_a.structuredContent["reclaim_token"]
        assert joined_a.structuredContent["agent_name"] == "codex"
        assert reclaim_token
        assert any(
            f"reclaim_token={reclaim_token}" in getattr(content, "text", "")
            for content in joined_a.content
        )

        sent = await agent_a.call_tool(
            "sync",
            {
                "topic_id": topic_id,
                "outbox": [{"content_markdown": "hello from codex", "message_type": "message"}],
                "wait_seconds": 0,
            },
        )
        assert sent.isError is False
        assert sent.structuredContent["received_count"] == 0

    async with (
        stdio_client(peer_server) as (b_read, b_write),
        ClientSession(b_read, b_write) as agent_b,
    ):
        await agent_b.initialize()

        rejected = await agent_b.call_tool(
            "topic_join",
            {"agent_name": "codex", "topic_id": topic_id},
        )
        assert rejected.isError is True
        assert rejected.structuredContent["error"]["code"] == "AGENT_NAME_IN_USE"
        assert rejected.structuredContent["requested_agent_name"] == "codex"
        assert "codex reviewer" in rejected.structuredContent["suggested_agent_names"]

        joined_b = await agent_b.call_tool(
            "topic_join",
            {"agent_name": "codex reviewer", "topic_id": topic_id},
        )
        assert joined_b.isError is False
        assert joined_b.structuredContent["agent_name"] == "codex reviewer"
        assert joined_b.structuredContent["reclaim_token"]

        reply = await agent_b.call_tool(
            "sync",
            {
                "topic_id": topic_id,
                "outbox": [{"content_markdown": "hello from reviewer", "message_type": "message"}],
                "wait_seconds": 0,
            },
        )
        assert reply.isError is False
        assert reply.structuredContent["sent"][0]["message"]["sender"] == "codex reviewer"

    async with (
        stdio_client(peer_server) as (c_read, c_write),
        ClientSession(c_read, c_write) as agent_c,
    ):
        await agent_c.initialize()

        joined_c = await agent_c.call_tool(
            "topic_join",
            {
                "agent_name": "codex",
                "topic_id": topic_id,
                "reclaim_token": reclaim_token,
            },
        )
        assert joined_c.isError is False
        assert joined_c.structuredContent["agent_name"] == "codex"
        assert joined_c.structuredContent["reclaim_token"] == reclaim_token
        assert any(
            f"reclaim_token={reclaim_token}" in getattr(content, "text", "")
            for content in joined_c.content
        )

        received = await agent_c.call_tool(
            "sync",
            {"topic_id": topic_id, "wait_seconds": 0},
        )
        assert received.isError is False
        assert received.structuredContent["received_count"] == 1
        assert received.structuredContent["received"][0]["sender"] == "codex reviewer"
        assert (
            received.structuredContent["received"][0]["content_markdown"] == "hello from reviewer"
        )
