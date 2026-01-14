from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from agent_bus.db import AgentBusDB


@click.group()
@click.option(
    "--db-path",
    default=None,
    help="SQLite DB path (defaults to $AGENT_BUS_DB or ~/.agent_bus/agent_bus.sqlite).",
)
@click.pass_context
def cli(ctx: click.Context, db_path: str | None) -> None:
    """Administrative CLI for Agent Bus."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path


def _db(ctx: click.Context) -> AgentBusDB:
    db_path = None
    if ctx.obj:
        db_path = ctx.obj.get("db_path")
    return AgentBusDB(path=db_path)


@cli.group("db")
def db_group() -> None:
    """Database operations."""


@db_group.command("wipe")
@click.option("--yes", is_flag=True, help="Do not prompt for confirmation.")
@click.pass_context
def db_wipe(ctx: click.Context, *, yes: bool) -> None:
    """Delete the local Agent Bus SQLite database file (and WAL/SHM sidecars)."""
    db = _db(ctx)
    db_path = db.path
    if db_path == ":memory:":
        raise click.ClickException("Cannot wipe an in-memory DB.")

    main = Path(db_path)
    wal = Path(f"{db_path}-wal")
    shm = Path(f"{db_path}-shm")
    candidates = [main, wal, shm]

    click.echo(f"DB path: {main}")
    existing = [p for p in candidates if p.exists()]
    if not existing:
        click.echo("Nothing to delete (DB file not found).")
        return

    click.echo("Will delete:")
    for p in existing:
        click.echo(f"- {p}")

    if not yes and not click.confirm("Delete these files?", default=False):
        raise click.ClickException("Canceled.")

    removed = 0
    for p in existing:
        try:
            p.unlink()
        except FileNotFoundError:  # pragma: no cover
            continue
        removed += 1

    click.echo(f"Deleted {removed} file(s).")


@cli.group("topics")
def topics_group() -> None:
    """Topic operations."""


@topics_group.command("list")
@click.option(
    "--status",
    type=click.Choice(["open", "closed", "all"], case_sensitive=False),
    default="open",
    show_default=True,
)
@click.option("--limit", type=int, default=200, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of a table.")
@click.pass_context
def topics_list(ctx: click.Context, *, status: str, limit: int, as_json: bool) -> None:
    """List topics with message counts."""
    if limit <= 0:
        raise click.ClickException("limit must be > 0")

    db = _db(ctx)
    rows = db.topic_list_with_counts(status=status.lower(), limit=limit)

    if as_json:
        click.echo(json.dumps({"topics": rows}, ensure_ascii=True, sort_keys=True, indent=2))
        return

    click.echo(f"DB path: {db.path}")
    click.echo(f"Topics ({status}): {len(rows)}")
    if not rows:
        return

    headers = ["topic_id", "name", "status", "messages", "last_seq"]
    cols = {h: len(h) for h in headers}
    for r in rows:
        cols["topic_id"] = max(cols["topic_id"], len(str(r["topic_id"])))
        cols["name"] = max(cols["name"], len(str(r["name"])))
        cols["status"] = max(cols["status"], len(str(r["status"])))
        cols["messages"] = max(cols["messages"], len(str(r["counts"]["messages"])))
        cols["last_seq"] = max(cols["last_seq"], len(str(r["counts"]["last_seq"])))

    def _cell(key: str, val: Any) -> str:
        s = str(val)
        return s.rjust(cols[key]) if key in {"messages", "last_seq"} else s.ljust(cols[key])

    click.echo(
        " ".join(
            [
                _cell("topic_id", "topic_id"),
                _cell("name", "name"),
                _cell("status", "status"),
                _cell("messages", "messages"),
                _cell("last_seq", "last_seq"),
            ]
        )
    )
    for r in rows:
        click.echo(
            " ".join(
                [
                    _cell("topic_id", r["topic_id"]),
                    _cell("name", r["name"]),
                    _cell("status", r["status"]),
                    _cell("messages", r["counts"]["messages"]),
                    _cell("last_seq", r["counts"]["last_seq"]),
                ]
            )
        )


@topics_group.command("presence")
@click.argument("topic_id")
@click.option(
    "--window",
    "window_seconds",
    type=int,
    default=300,
    show_default=True,
    help="Consider peers active if seen within this many seconds.",
)
@click.option("--limit", type=int, default=200, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of text.")
@click.pass_context
def topics_presence(
    ctx: click.Context,
    topic_id: str,
    *,
    window_seconds: int,
    limit: int,
    as_json: bool,
) -> None:
    """Show peers recently active on a topic."""
    from agent_bus.db import TopicNotFoundError

    if window_seconds <= 0:
        raise click.ClickException("window must be > 0")
    if limit <= 0:
        raise click.ClickException("limit must be > 0")

    db = _db(ctx)
    try:
        topic = db.get_topic(topic_id=topic_id)
        cursors = db.get_presence(topic_id=topic_id, window_seconds=window_seconds, limit=limit)
    except TopicNotFoundError:
        raise click.ClickException(f"Topic not found: {topic_id}") from None
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    now_ts = time.time()
    peers = [
        {
            "agent_name": c.agent_name,
            "last_seq": c.last_seq,
            "updated_at": c.updated_at,
            "age_seconds": max(0.0, now_ts - c.updated_at),
        }
        for c in cursors
    ]

    if as_json:
        click.echo(
            json.dumps(
                {
                    "topic_id": topic_id,
                    "topic_name": topic.name,
                    "status": topic.status,
                    "window_seconds": window_seconds,
                    "peers": peers,
                },
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
            )
        )
        return

    click.echo(click.style(f"Topic: {topic.name} ({topic_id})", fg="green", bold=True))
    click.echo(click.style(f"Status: {topic.status}", dim=True))
    click.echo(f"Active peers in last {window_seconds}s: {len(peers)}")
    if not peers:
        return

    for p in peers:
        click.echo(f"- {p['agent_name']} last_seq={p['last_seq']} age={p['age_seconds']:.1f}s")


@topics_group.command("rename")
@click.argument("topic_id")
@click.argument("new_name")
@click.pass_context
def topics_rename(ctx: click.Context, topic_id: str, new_name: str) -> None:
    """Rename a topic."""
    from agent_bus.db import TopicNotFoundError

    db = _db(ctx)
    try:
        topic, unchanged = db.topic_rename(topic_id=topic_id, new_name=new_name)
    except TopicNotFoundError:
        raise click.ClickException(f"Topic not found: {topic_id}") from None
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    if unchanged:
        click.echo(f'No-op: topic "{topic.topic_id}" already named "{topic.name}".')
        return

    click.echo(f'Renamed topic "{topic.topic_id}" to "{topic.name}".')


@topics_group.command("delete")
@click.argument("topic_id")
@click.option("--yes", is_flag=True, help="Do not prompt for confirmation.")
@click.pass_context
def topics_delete(ctx: click.Context, topic_id: str, *, yes: bool) -> None:
    """Delete a topic and all related data (messages, cursors, sequences)."""
    from agent_bus.db import TopicNotFoundError

    db = _db(ctx)
    try:
        topic = db.get_topic(topic_id=topic_id)
    except TopicNotFoundError:
        raise click.ClickException(f"Topic not found: {topic_id}") from None

    click.echo(click.style(f"Topic: {topic.name} ({topic_id})", fg="green", bold=True))
    click.echo(click.style("This will delete the topic and all related data.", fg="red"))

    if not yes and not click.confirm("Delete this topic?", default=False):
        raise click.ClickException("Canceled.")

    deleted = db.delete_topic(topic_id=topic_id)
    if not deleted:  # pragma: no cover
        raise click.ClickException(f"Topic not found: {topic_id}")

    click.echo(f"Deleted topic {topic_id}.")


# Colors for different senders (cycles through these)
_SENDER_COLORS = ["cyan", "magenta", "yellow", "green", "blue", "red"]
_sender_color_map: dict[str, str] = {}


def _get_sender_color(sender: str) -> str:
    """Get a consistent color for a sender."""
    if sender not in _sender_color_map:
        _sender_color_map[sender] = _SENDER_COLORS[len(_sender_color_map) % len(_SENDER_COLORS)]
    return _sender_color_map[sender]


def _format_message(msg: Any, *, show_time: bool = True) -> str:
    """Format a message for display."""
    color = _get_sender_color(msg.sender)
    sender_styled = click.style(msg.sender, fg=color, bold=True)
    seq_styled = click.style(f"[{msg.seq}]", fg="white", dim=True)

    parts = [seq_styled, sender_styled]

    if show_time:
        ts = datetime.fromtimestamp(msg.created_at).strftime("%H:%M:%S")
        time_styled = click.style(ts, fg="white", dim=True)
        parts.append(time_styled)

    # Get first line of content for preview, or full content if short
    content = msg.content_markdown
    lines = content.split("\n")
    preview = lines[0][:80] + " ..." if len(lines) > 1 else content[:100]

    return f"{' '.join(parts)}: {preview}"


@topics_group.command("watch")
@click.argument("topic_id")
@click.option(
    "--follow",
    "-f",
    "--tail",
    is_flag=True,
    help="Wait for new messages (like tail -f). Alias: --tail.",
)
@click.option(
    "--last",
    "-n",
    type=int,
    default=10,
    show_default=True,
    help="Show last N messages initially.",
)
@click.option("--full", is_flag=True, help="Show full message content instead of preview.")
@click.pass_context
def topics_watch(
    ctx: click.Context,
    topic_id: str,
    *,
    follow: bool,
    last: int,
    full: bool,
) -> None:
    """Watch messages on a topic in real-time.

    Examples:

        agent-bus cli topics watch <topic_id>          # Show recent messages
        agent-bus cli topics watch <topic_id> -f       # Follow new messages
        agent-bus cli topics watch <topic_id> -f -n 0  # Follow, skip history
    """
    from agent_bus.db import TopicNotFoundError

    db = _db(ctx)

    # Verify topic exists
    try:
        topic = db.get_topic(topic_id=topic_id)
    except TopicNotFoundError:
        raise click.ClickException(f"Topic not found: {topic_id}") from None

    click.echo(click.style(f"Watching topic: {topic.name} ({topic_id})", fg="green", bold=True))
    click.echo(click.style(f"Status: {topic.status}", dim=True))
    click.echo()

    # Get initial messages
    if last > 0:
        initial_msgs = db.get_latest_messages(topic_id=topic_id, limit=last)
        for msg in initial_msgs:
            if full:
                click.echo(_format_message(msg))
                # Print full content indented
                for line in msg.content_markdown.split("\n"):
                    click.echo(click.style(f"    {line}", dim=True))
            else:
                click.echo(_format_message(msg))

        last_seq = initial_msgs[-1].seq if initial_msgs else 0
    else:
        # If last=0, we still need the latest seq to start following from
        # Use a small limit just to get the last message
        initial_msgs = db.get_latest_messages(topic_id=topic_id, limit=1)
        last_seq = initial_msgs[-1].seq if initial_msgs else 0

    if not follow:
        return

    click.echo()
    click.echo(click.style("--- Waiting for new messages (Ctrl+C to exit) ---", dim=True))
    click.echo()

    try:
        while True:
            new_msgs = db.get_messages(topic_id=topic_id, after_seq=last_seq, limit=100)

            for msg in new_msgs:
                if full:
                    click.echo(_format_message(msg))
                    for line in msg.content_markdown.split("\n"):
                        click.echo(click.style(f"    {line}", dim=True))
                else:
                    click.echo(_format_message(msg))
                last_seq = msg.seq

            time.sleep(1.0)
    except KeyboardInterrupt:
        click.echo()
        click.echo(click.style("Stopped watching.", dim=True))


def _format_export_markdown(
    messages: list[Any],
    *,
    include_metadata: bool,
    topic_name: str,
    topic_id: str,
) -> str:
    """Format messages as markdown for export."""
    lines: list[str] = []
    lines.append(f"# {topic_name}")
    lines.append("")
    lines.append(f"**Topic ID:** {topic_id}")
    lines.append(f"**Messages:** {len(messages)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Build reply_to lookup for threading context
    msg_by_id: dict[str, Any] = {m.message_id: m for m in messages}

    for msg in messages:
        # Header with sender and seq (consistent with Web UI)
        lines.append(f"### [{msg.seq}] {msg.sender}")

        if include_metadata:
            ts = datetime.fromtimestamp(msg.created_at).strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"*{ts}*")

        # Reply context
        if msg.reply_to and msg.reply_to in msg_by_id:
            parent = msg_by_id[msg.reply_to]
            lines.append(f"*↩︎ reply to {parent.sender} (#{parent.seq})*")

        lines.append("")
        lines.append(msg.content_markdown)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _format_export_json(messages: list[Any], *, topic_name: str, topic_id: str) -> str:
    """Format messages as JSON for export."""
    data = {
        "topic_id": topic_id,
        "topic_name": topic_name,
        "message_count": len(messages),
        "messages": [
            {
                "message_id": m.message_id,
                "seq": m.seq,
                "sender": m.sender,
                "message_type": m.message_type,
                "reply_to": m.reply_to,
                "content_markdown": m.content_markdown,
                "metadata": m.metadata,
                "created_at": m.created_at,
            }
            for m in messages
        ],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


@topics_group.command("export")
@click.argument("topic_id")
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["markdown", "json"], case_sensitive=False),
    default="markdown",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file path (default: stdout).",
)
@click.option(
    "--include-metadata",
    is_flag=True,
    help="Include timestamps in markdown output.",
)
@click.option(
    "--after-seq",
    type=int,
    default=0,
    help="Export only messages after this sequence number (for delta export).",
)
@click.pass_context
def topics_export(
    ctx: click.Context,
    topic_id: str,
    *,
    fmt: str,
    output: str | None,
    include_metadata: bool,
    after_seq: int,
) -> None:
    """Export all messages from a topic.

    Examples:

        agent-bus cli topics export <topic_id>                    # Markdown to stdout
        agent-bus cli topics export <topic_id> -f json            # JSON to stdout
        agent-bus cli topics export <topic_id> -o chat.md         # Save to file
        agent-bus cli topics export <topic_id> --include-metadata # With timestamps
        agent-bus cli topics export <topic_id> --after-seq 50     # Delta export
    """
    from agent_bus.db import TopicNotFoundError

    db = _db(ctx)

    # Verify topic exists
    try:
        topic = db.get_topic(topic_id=topic_id)
    except TopicNotFoundError:
        raise click.ClickException(f"Topic not found: {topic_id}") from None

    # Fetch messages (use large limit)
    messages = db.get_messages(topic_id=topic_id, after_seq=after_seq, limit=100000)

    if not messages:
        click.echo("No messages to export.", err=True)
        return

    # Format output
    if fmt.lower() == "json":
        content = _format_export_json(messages, topic_name=topic.name, topic_id=topic_id)
    else:
        content = _format_export_markdown(
            messages,
            include_metadata=include_metadata,
            topic_name=topic.name,
            topic_id=topic_id,
        )

    # Write output
    if output:
        Path(output).write_text(content, encoding="utf-8")
        click.echo(f"Exported {len(messages)} messages to {output}", err=True)
    else:
        click.echo(content)
