"""FastAPI web server for the Agent Bus SPA and browser APIs."""

from __future__ import annotations

import asyncio
import json
import signal
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)

from agent_bus.db import AgentBusDB, DBBusyError, TopicNotFoundError
from agent_bus.models import Cursor, Message

STATIC_DIR = Path(__file__).parent / "static"
SPA_INDEX = STATIC_DIR / "index.html"
DEFAULT_PAGE_SIZE = 50
TOPICS_STREAM_INTERVAL_SECONDS = 2.0
TOPIC_STREAM_INTERVAL_SECONDS = 2.0
STREAM_HEARTBEAT_SECONDS = 15.0
PRESENCE_WINDOW_SECONDS = 300
TOPICS_SIGNATURE_SCAN_LIMIT = 2_147_483_647
SERVER_SHUTDOWN_GRACE_SECONDS = 2

SearchMode = Literal["fts", "semantic", "hybrid"]
TopicStatusFilter = Literal["open", "closed", "all"]
TopicSort = Literal["last_updated_desc", "created_desc", "created_asc"]

app = FastAPI(title="Agent Bus", docs_url=None, redoc_url=None)

_db: AgentBusDB | None = None


class SSEStreamingResponse(StreamingResponse):
    async def __call__(self, scope, receive, send) -> None:  # type: ignore[override]
        try:
            await super().__call__(scope, receive, send)
        except asyncio.CancelledError:
            return


class ImmediateSigintServer:
    def __init__(self, config) -> None:
        import uvicorn

        class _Server(uvicorn.Server):
            def handle_exit(self, sig: int, frame) -> None:  # type: ignore[override]
                super().handle_exit(sig, frame)
                if sig == signal.SIGINT:
                    self.force_exit = True

        self._server = _Server(config=config)

    def run(self) -> None:
        self._server.run()


def get_db() -> AgentBusDB:
    if _db is None:
        raise RuntimeError("Database not initialized")
    return _db


def init_db(db_path: str | None = None) -> None:
    global _db
    _db = AgentBusDB(path=db_path)


def now() -> float:
    return time.time()


def format_missing_bundle_response() -> Response:
    return HTMLResponse(
        """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <title>Agent Bus Web UI</title>
            <style>
              body {
                margin: 0;
                min-height: 100vh;
                display: grid;
                place-items: center;
                background: #f6f1e9;
                color: #211b16;
                font-family: "Geist Variable", system-ui, sans-serif;
              }
              main {
                max-width: 42rem;
                padding: 2rem;
                border-radius: 1rem;
                background: white;
                box-shadow: 0 20px 60px rgba(33, 27, 22, 0.12);
              }
              code {
                background: #f1e8da;
                border-radius: 0.375rem;
                padding: 0.15rem 0.4rem;
              }
            </style>
          </head>
          <body>
            <main>
              <h1>Frontend bundle not found</h1>
              <p>
                This checkout does not have built web assets yet. Run
                <code>pnpm --dir frontend install</code> and
                <code>pnpm --dir frontend build</code>, then restart
                <code>agent-bus serve</code>.
              </p>
            </main>
          </body>
        </html>
        """,
        status_code=503,
    )


def spa_index_response() -> Response:
    if not SPA_INDEX.exists():
        return format_missing_bundle_response()
    return FileResponse(SPA_INDEX)


def serialize_cursor(cursor: Cursor) -> dict[str, Any]:
    return {
        "topic_id": cursor.topic_id,
        "agent_name": cursor.agent_name,
        "last_seq": cursor.last_seq,
        "updated_at": cursor.updated_at,
    }


def serialize_message(message: Message, sender_by_msg_id: dict[str, str]) -> dict[str, Any]:
    return {
        "message_id": message.message_id,
        "topic_id": message.topic_id,
        "seq": message.seq,
        "sender": message.sender,
        "message_type": message.message_type,
        "reply_to": message.reply_to,
        "reply_to_sender": sender_by_msg_id.get(message.reply_to) if message.reply_to else None,
        "content_markdown": message.content_markdown,
        "metadata": message.metadata,
        "client_message_id": message.client_message_id,
        "created_at": message.created_at,
    }


def normalize_topic_summary(row: dict[str, Any]) -> dict[str, Any]:
    counts = cast(dict[str, int], row["counts"])
    return {
        "topic_id": row["topic_id"],
        "name": row["name"],
        "status": row["status"],
        "created_at": row["created_at"],
        "closed_at": row["closed_at"],
        "close_reason": row["close_reason"],
        "metadata": row["metadata"],
        "message_count": counts["messages"],
        "last_seq": counts["last_seq"],
        "last_message_at": row.get("last_message_at"),
        "last_updated_at": row.get("last_updated_at", row["created_at"]),
    }


def list_topic_summaries(
    db: AgentBusDB,
    *,
    status: TopicStatusFilter,
    sort: TopicSort,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows = db.topic_list_with_counts(status=status, sort=sort, query=query, limit=limit)
    return [normalize_topic_summary(row) for row in rows]


def get_topic_summary(db: AgentBusDB, *, topic_id: str) -> dict[str, Any]:
    return normalize_topic_summary(db.topic_get_with_counts(topic_id=topic_id))


def serialize_topic_messages(
    db: AgentBusDB,
    messages: list[Message],
) -> tuple[list[dict[str, Any]], int | None, int | None]:
    reply_to_ids = [message.reply_to for message in messages if message.reply_to]
    sender_lookup = db.get_senders_by_message_ids(reply_to_ids) if reply_to_ids else {}
    payload = [serialize_message(message, sender_lookup) for message in messages]
    first_seq = messages[0].seq if messages else None
    last_seq = messages[-1].seq if messages else None
    return payload, first_seq, last_seq


def run_search(
    *,
    db: AgentBusDB,
    query: str,
    mode: str,
    limit: int,
    topic_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    from agent_bus.search import DEFAULT_EMBEDDING_MODEL, search_messages

    query = query.strip()
    if not query:
        return [], []

    mode_value = cast(SearchMode, mode.lower())
    results, warnings = search_messages(
        db,
        query=query,
        mode=mode_value,
        topic_id=topic_id,
        limit=max(1, min(limit, 50)),
        model=DEFAULT_EMBEDDING_MODEL,
    )
    return list(results), list(warnings)


def encode_sse(event: str, data: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


type TopicSignature = tuple[str, str, int, float, str, float | None]


def topics_signature(db: AgentBusDB) -> list[TopicSignature]:
    summaries = list_topic_summaries(
        db,
        status="all",
        sort="last_updated_desc",
        query="",
        limit=TOPICS_SIGNATURE_SCAN_LIMIT,
    )
    return [
        (
            item["topic_id"],
            item["name"],
            item["message_count"],
            item["last_updated_at"],
            item["status"],
            item["closed_at"],
        )
        for item in summaries
    ]


def format_export_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z")


def topic_stream_state(db: AgentBusDB, *, topic_id: str) -> dict[str, Any]:
    summary = get_topic_summary(db, topic_id=topic_id)
    presence = db.get_presence(topic_id=topic_id, window_seconds=PRESENCE_WINDOW_SECONDS)
    return {
        "topic_id": topic_id,
        "last_seq": summary["last_seq"],
        "message_count": summary["message_count"],
        "presence": [serialize_cursor(item) for item in presence],
    }


@app.get("/api/topics")
async def api_topics(
    status: TopicStatusFilter = "all",
    sort: TopicSort = "last_updated_desc",
    q: str = "",
    limit: int = Query(200, ge=1, le=1000),
) -> dict[str, Any]:
    db = get_db()
    topics = list_topic_summaries(db, status=status, sort=sort, query=q, limit=limit)
    return {"topics": topics}


@app.get("/api/topics/{topic_id}")
async def api_topic_detail(topic_id: str, focus: str | None = None) -> dict[str, Any]:
    db = get_db()
    summary = get_topic_summary(db, topic_id=topic_id)
    context_mode = False
    focus_message_id: str | None = None

    if focus:
        try:
            focused = db.get_message_by_id(message_id=focus)
        except ValueError:
            raise HTTPException(status_code=404, detail="Message not found") from None
        if focused.topic_id != topic_id:
            raise HTTPException(status_code=404, detail="Message not found") from None

        window = 25
        messages = db.get_messages(
            topic_id=topic_id,
            after_seq=max(0, focused.seq - window - 1),
            before_seq=focused.seq + window + 1,
            limit=(window * 2) + 1,
        )
        context_mode = True
        focus_message_id = focus
    else:
        messages = db.get_latest_messages(topic_id=topic_id, limit=DEFAULT_PAGE_SIZE)

    payload, first_seq, last_seq = serialize_topic_messages(db, messages)
    presence = db.get_presence(topic_id=topic_id, window_seconds=PRESENCE_WINDOW_SECONDS)

    return {
        "topic": summary,
        "messages": payload,
        "message_count": summary["message_count"],
        "first_seq": first_seq,
        "last_seq": last_seq,
        "has_earlier": bool(first_seq and first_seq > 1),
        "context_mode": context_mode,
        "focus_message_id": focus_message_id,
        "presence": [serialize_cursor(item) for item in presence],
    }


@app.get("/api/topics/{topic_id}/messages")
async def api_topic_messages(
    topic_id: str,
    after_seq: int = Query(0, ge=0),
    before_seq: int | None = Query(None, ge=0),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=200),
) -> dict[str, Any]:
    db = get_db()

    try:
        db.get_topic(topic_id=topic_id)
    except TopicNotFoundError:
        raise HTTPException(status_code=404, detail="Topic not found") from None

    messages = db.get_messages(
        topic_id=topic_id,
        after_seq=after_seq,
        before_seq=before_seq,
        limit=limit,
    )
    payload, first_seq, last_seq = serialize_topic_messages(db, messages)
    return {
        "messages": payload,
        "first_seq": first_seq,
        "last_seq": last_seq,
        "has_earlier": bool(first_seq and first_seq > 1),
    }


@app.get("/api/search")
async def api_global_search(
    q: str = "",
    mode: SearchMode = "hybrid",
    limit: int = Query(20, ge=1, le=50),
) -> dict[str, Any]:
    db = get_db()
    results, warnings = run_search(db=db, query=q, mode=mode, limit=limit)
    return {"query": q.strip(), "mode": mode, "warnings": warnings, "results": results}


@app.get("/api/topics/{topic_id}/search")
async def api_topic_search(
    topic_id: str,
    q: str = "",
    mode: SearchMode = "hybrid",
    limit: int = Query(20, ge=1, le=50),
) -> dict[str, Any]:
    db = get_db()
    try:
        db.get_topic(topic_id=topic_id)
    except TopicNotFoundError:
        raise HTTPException(status_code=404, detail="Topic not found") from None

    results, warnings = run_search(db=db, query=q, mode=mode, limit=limit, topic_id=topic_id)
    return {
        "topic_id": topic_id,
        "query": q.strip(),
        "mode": mode,
        "warnings": warnings,
        "results": results,
    }


@app.get("/api/topics/{topic_id}/export", response_class=PlainTextResponse)
async def api_topic_export(topic_id: str) -> PlainTextResponse:
    db = get_db()
    try:
        summary = get_topic_summary(db, topic_id=topic_id)
    except TopicNotFoundError:
        raise HTTPException(status_code=404, detail="Topic not found") from None

    messages = db.get_messages(topic_id=topic_id, after_seq=0, limit=10_000)
    lines = [
        f"# {summary['name']}",
        "",
        f"**Topic ID:** {summary['topic_id']}",
        f"**Status:** {summary['status']}",
        f"**Messages:** {summary['message_count']}",
        "",
        "---",
        "",
    ]
    for message in messages:
        lines.append(f"### [{message.seq}] {message.sender}")
        lines.append(f"*{format_export_timestamp(message.created_at)}*")
        if message.reply_to:
            lines.append(f"*Reply to: {message.reply_to}*")
        lines.append("")
        lines.append(message.content_markdown)
        lines.append("")
        lines.append("---")
        lines.append("")

    safe_name = "".join(
        char if char.isalnum() or char in ("-", "_") else "-" for char in summary["name"]
    )
    return PlainTextResponse(
        content="\n".join(lines),
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{safe_name or "topic"}.md"'},
    )


@app.delete("/api/topics/{topic_id}")
async def api_delete_topic(topic_id: str) -> dict[str, Any]:
    db = get_db()
    deleted = db.delete_topic(topic_id=topic_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Topic not found") from None
    return {"status": "ok", "topic_id": topic_id, "deleted": True}


@app.delete("/api/topics/{topic_id}/messages")
async def api_delete_messages(
    topic_id: str,
    message_ids: Annotated[list[str], Body(embed=True)],
) -> dict[str, Any]:
    db = get_db()
    try:
        db.get_topic(topic_id=topic_id)
    except TopicNotFoundError:
        raise HTTPException(status_code=404, detail="Topic not found") from None

    deleted_message_ids = db.delete_messages_batch(topic_id=topic_id, message_ids=message_ids)
    return {
        "status": "ok",
        "deleted_count": len(deleted_message_ids),
        "deleted_message_ids": deleted_message_ids,
    }


@app.get("/api/stream/topics")
async def api_topics_stream(request: Request) -> StreamingResponse:
    db = get_db()

    async def event_stream():
        previous_signature: list[TopicSignature] | None = None
        last_heartbeat = 0.0
        try:
            while True:
                if await request.is_disconnected():
                    return

                try:
                    signature = topics_signature(db)
                except DBBusyError:
                    if now() - last_heartbeat >= STREAM_HEARTBEAT_SECONDS:
                        last_heartbeat = now()
                        yield encode_sse("heartbeat", {"timestamp": last_heartbeat})
                else:
                    if signature != previous_signature:
                        previous_signature = signature
                        last_heartbeat = now()
                        yield encode_sse("topics.invalidate", {"timestamp": last_heartbeat})
                    elif now() - last_heartbeat >= STREAM_HEARTBEAT_SECONDS:
                        last_heartbeat = now()
                        yield encode_sse("heartbeat", {"timestamp": last_heartbeat})

                await asyncio.sleep(TOPICS_STREAM_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            return

    return SSEStreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/stream/topics/{topic_id}")
async def api_topic_stream(topic_id: str, request: Request) -> StreamingResponse:
    db = get_db()

    async def event_stream():
        previous_state: dict[str, Any] | None = None
        last_heartbeat = 0.0
        try:
            while True:
                if await request.is_disconnected():
                    return

                try:
                    state = topic_stream_state(db, topic_id=topic_id)
                except TopicNotFoundError:
                    yield encode_sse("topic.deleted", {"topic_id": topic_id})
                    return
                except DBBusyError:
                    if now() - last_heartbeat >= STREAM_HEARTBEAT_SECONDS:
                        last_heartbeat = now()
                        yield encode_sse("heartbeat", {"timestamp": last_heartbeat})
                else:
                    if state != previous_state:
                        previous_state = state
                        last_heartbeat = now()
                        yield encode_sse("topic.update", state)
                    elif now() - last_heartbeat >= STREAM_HEARTBEAT_SECONDS:
                        last_heartbeat = now()
                        yield encode_sse("heartbeat", {"timestamp": last_heartbeat})

                await asyncio.sleep(TOPIC_STREAM_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            return

    return SSEStreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
async def spa_root() -> Response:
    return spa_index_response()


@app.get("/topics/{topic_id}", response_class=HTMLResponse)
async def spa_topic_page(topic_id: str) -> Response:
    _ = topic_id
    return spa_index_response()


@app.get("/{path:path}")
async def spa_assets(path: str) -> Response:
    if not path:
        return spa_index_response()

    candidate = (STATIC_DIR / path).resolve()
    try:
        candidate.relative_to(STATIC_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found") from None

    if candidate.is_file():
        return FileResponse(candidate)

    if "." in Path(path).name:
        raise HTTPException(status_code=404, detail="Not found") from None

    return spa_index_response()


def run_server(host: str = "127.0.0.1", port: int = 8080, db_path: str | None = None) -> None:
    import uvicorn

    init_db(db_path)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        lifespan="off",
        timeout_graceful_shutdown=SERVER_SHUTDOWN_GRACE_SECONDS,
    )
    with suppress(KeyboardInterrupt):
        ImmediateSigintServer(config).run()
