"""FastAPI web server for Agent Bus UI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

from agent_bus.db import AgentBusDB, TopicNotFoundError

# Template directory
TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="Agent Bus", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Global DB instance (initialized on startup)
_db: AgentBusDB | None = None


def get_db() -> AgentBusDB:
    """Get the database instance."""
    if _db is None:
        raise RuntimeError("Database not initialized")
    return _db


def init_db(db_path: str | None = None) -> None:
    """Initialize the database."""
    global _db
    _db = AgentBusDB(path=db_path)


# Sender colors for consistent coloring
SENDER_COLORS = ["cyan", "magenta", "amber", "emerald", "blue", "rose"]
_sender_color_map: dict[str, str] = {}


def get_sender_color(sender: str) -> str:
    """Get a consistent Tailwind color for a sender."""
    if sender not in _sender_color_map:
        _sender_color_map[sender] = SENDER_COLORS[len(_sender_color_map) % len(SENDER_COLORS)]
    return _sender_color_map[sender]


def format_age(timestamp: float) -> str:
    """Format a timestamp as a human-readable age."""
    age = time.time() - timestamp
    if age < 60:
        return f"{int(age)}s ago"
    if age < 3600:
        return f"{int(age / 60)}m ago"
    if age < 86400:
        return f"{int(age / 3600)}h ago"
    return f"{int(age / 86400)}d ago"


def sidebar_topics(db: AgentBusDB) -> list[dict[str, Any]]:
    topics = db.topic_list_with_counts(status="all", limit=100)
    for topic in topics:
        if topic.get("created_at"):
            topic["age"] = format_age(topic["created_at"])
    return topics


# Register template filters
@app.on_event("startup")
async def setup_template_filters() -> None:
    """Register custom Jinja2 filters."""
    templates.env.filters["sender_color"] = get_sender_color
    templates.env.filters["format_age"] = format_age


@app.get("/", response_class=HTMLResponse)
async def topics_list(request: Request) -> Any:
    """Show list of topics."""
    db = get_db()
    topics = sidebar_topics(db)

    return templates.TemplateResponse(
        "topics/list.html",
        {
            "request": request,
            "topics": topics,
            "active_topic_id": None,
            "now": time.time(),
        },
    )


DEFAULT_PAGE_SIZE = 50


@app.get("/topics/{topic_id}", response_class=HTMLResponse)
async def topic_detail(request: Request, topic_id: str, focus: str | None = None) -> Any:
    """Show messages in a topic."""
    db = get_db()

    try:
        topic = db.get_topic(topic_id=topic_id)
    except TopicNotFoundError:
        raise HTTPException(status_code=404, detail="Topic not found") from None

    topics = sidebar_topics(db)
    context_mode = False
    focus_message_id: str | None = None

    if focus:
        # Load a window around a specific message (for deep links from search results).
        try:
            msg = db.get_message_by_id(message_id=focus)
        except ValueError:
            raise HTTPException(status_code=404, detail="Message not found") from None
        if msg.topic_id != topic_id:
            raise HTTPException(status_code=404, detail="Message not found") from None

        window = 25
        after_seq = max(0, msg.seq - window - 1)
        before_seq = msg.seq + window + 1
        messages = db.get_messages(
            topic_id=topic_id,
            after_seq=after_seq,
            before_seq=before_seq,
            limit=(window * 2) + 1,
        )
        context_mode = True
        focus_message_id = focus
    else:
        # Load only the latest N messages initially for faster rendering.
        messages = db.get_latest_messages(topic_id=topic_id, limit=DEFAULT_PAGE_SIZE)
    presence = db.get_presence(topic_id=topic_id, window_seconds=300)

    # Build sender lookup for reply_to references
    sender_by_msg_id: dict[str, str] = {msg.message_id: msg.sender for msg in messages}

    # Flat message list with metadata
    all_messages = [
        {
            "message": msg,
            "color": get_sender_color(msg.sender),
            "age": format_age(msg.created_at),
            "reply_to_sender": sender_by_msg_id.get(msg.reply_to) if msg.reply_to else None,
        }
        for msg in messages
    ]

    first_seq = messages[0].seq if messages else 0
    last_seq = messages[-1].seq if messages else 0
    # There are earlier messages if first message isn't seq 1
    has_earlier = first_seq > 1
    # Prefer the full topic message count (not just the loaded page size).
    message_count = next(
        (t["counts"]["messages"] for t in topics if t["topic_id"] == topic_id),
        len(messages),
    )

    return templates.TemplateResponse(
        "topics/detail.html",
        {
            "request": request,
            "topics": topics,
            "active_topic_id": topic_id,
            "topic": topic,
            "all_messages": all_messages,
            "message_count": message_count,
            "first_seq": first_seq,
            "last_seq": last_seq,
            "has_earlier": has_earlier,
            "presence": presence,
            "context_mode": context_mode,
            "focus_message_id": focus_message_id,
            "now": time.time(),
        },
    )


@app.get("/topics/{topic_id}/messages", response_class=HTMLResponse)
async def topic_messages_partial(
    request: Request,
    topic_id: str,
    after_seq: int = 0,
    before_seq: int | None = None,
    limit: int = 50,
) -> Any:
    """HTMX partial: get messages with pagination support.

    Args:
        after_seq: Get messages after this sequence (for polling new messages).
        before_seq: Get messages before this sequence (for "load earlier").
        limit: Maximum number of messages to return (default 50).
    """
    db = get_db()

    try:
        db.get_topic(topic_id=topic_id)
    except TopicNotFoundError:
        raise HTTPException(status_code=404, detail="Topic not found") from None

    after_seq = max(0, after_seq)
    if before_seq is not None:
        before_seq = max(0, before_seq)
    limit = max(1, min(limit, 200))

    messages = db.get_messages(
        topic_id=topic_id, after_seq=after_seq, before_seq=before_seq, limit=limit
    )

    # Build sender lookup only for reply_to IDs we need (not all messages)
    reply_to_ids = [msg.reply_to for msg in messages if msg.reply_to]
    sender_by_msg_id: dict[str, str] = {}
    if reply_to_ids:
        sender_by_msg_id = db.get_senders_by_message_ids(reply_to_ids)

    first_seq = messages[0].seq if messages else None
    last_seq = messages[-1].seq if messages else None

    return templates.TemplateResponse(
        "components/messages.html",
        {
            "request": request,
            "messages": [
                {
                    "message": msg,
                    "color": get_sender_color(msg.sender),
                    "age": format_age(msg.created_at),
                    "reply_to_sender": sender_by_msg_id.get(msg.reply_to) if msg.reply_to else None,
                }
                for msg in messages
            ],
            "first_seq": first_seq,
            "last_seq": last_seq,
            "topic_id": topic_id,
            "update_last_seq": before_seq is None,
            "update_load_earlier": before_seq is not None and after_seq == 0,
        },
    )


@app.get("/topics/{topic_id}/presence", response_class=HTMLResponse)
async def topic_presence_partial(request: Request, topic_id: str) -> Any:
    """HTMX partial: get presence indicators."""
    db = get_db()

    try:
        db.get_topic(topic_id=topic_id)
    except TopicNotFoundError:
        raise HTTPException(status_code=404, detail="Topic not found") from None

    presence = db.get_presence(topic_id=topic_id, window_seconds=300)

    return templates.TemplateResponse(
        "components/presence.html",
        {
            "request": request,
            "presence": presence,
            "now": time.time(),
        },
    )


@app.get("/topics/{topic_id}/export", response_class=PlainTextResponse)
async def topic_export(topic_id: str) -> Any:
    """Export topic messages as Markdown."""
    db = get_db()

    try:
        topic = db.get_topic(topic_id=topic_id)
    except TopicNotFoundError:
        raise HTTPException(status_code=404, detail="Topic not found") from None

    messages = db.get_messages(topic_id=topic_id, after_seq=0, limit=10000)

    lines = [
        f"# {topic.name}",
        "",
        f"**Topic ID:** {topic.topic_id}",
        f"**Status:** {topic.status}",
        f"**Messages:** {len(messages)}",
        "",
        "---",
        "",
    ]

    for msg in messages:
        lines.append(f"### [{msg.seq}] {msg.sender}")
        lines.append(f"*{format_age(msg.created_at)}*")
        if msg.reply_to:
            lines.append(f"*Reply to: {msg.reply_to}*")
        lines.append("")
        lines.append(msg.content_markdown)
        lines.append("")
        lines.append("---")
        lines.append("")

    return PlainTextResponse(
        content="\n".join(lines),
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{topic.name}.md"'},
    )


@app.get("/topics/{topic_id}/search", response_class=HTMLResponse)
async def topic_search(
    request: Request,
    topic_id: str,
    q: str | None = None,
    mode: str = "hybrid",
    limit: int = 20,
) -> Any:
    """HTMX partial: search within a topic."""
    db = get_db()

    try:
        db.get_topic(topic_id=topic_id)
    except TopicNotFoundError:
        raise HTTPException(status_code=404, detail="Topic not found") from None

    query = (q or "").strip()
    if not query:
        return templates.TemplateResponse(
            "components/search_results.html",
            {"request": request, "topic_id": topic_id, "query": "", "results": [], "warnings": []},
        )

    try:
        from agent_bus.search import DEFAULT_EMBEDDING_MODEL, search_messages

        results, warnings = search_messages(
            db,
            query=query,
            mode=mode.lower(),
            topic_id=topic_id,
            limit=max(1, min(limit, 50)),
            model=DEFAULT_EMBEDDING_MODEL,
        )
    except Exception as e:
        return templates.TemplateResponse(
            "components/search_results.html",
            {
                "request": request,
                "topic_id": topic_id,
                "query": query,
                "results": [],
                "warnings": [str(e)],
            },
        )

    return templates.TemplateResponse(
        "components/search_results.html",
        {
            "request": request,
            "topic_id": topic_id,
            "query": query,
            "results": results,
            "warnings": warnings,
        },
    )


@app.delete("/topics/{topic_id}")
async def delete_topic(request: Request, topic_id: str) -> Any:
    """Delete a topic and all its messages."""
    db = get_db()
    deleted = db.delete_topic(topic_id=topic_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Topic not found") from None

    if request.headers.get("hx-request") == "true":
        return HTMLResponse("")

    return {"status": "ok", "topic_id": topic_id, "deleted": True}


@app.delete("/topics/{topic_id}/messages")
async def delete_messages(topic_id: str, message_ids: list[str]) -> dict[str, Any]:
    """Delete selected messages and their reply chains."""
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


def run_server(host: str = "127.0.0.1", port: int = 8080, db_path: str | None = None) -> None:
    """Run the web server."""
    import uvicorn

    init_db(db_path)
    uvicorn.run(app, host=host, port=port)
