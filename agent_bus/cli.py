from __future__ import annotations

import json
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
    """List topics with question counts."""
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

    headers = ["topic_id", "name", "status", "total", "pending", "answered", "cancelled"]
    cols = {h: len(h) for h in headers}
    for r in rows:
        cols["topic_id"] = max(cols["topic_id"], len(str(r["topic_id"])))
        cols["name"] = max(cols["name"], len(str(r["name"])))
        cols["status"] = max(cols["status"], len(str(r["status"])))

    def _cell(key: str, val: Any) -> str:
        s = str(val)
        return (
            s.rjust(cols[key])
            if key in {"total", "pending", "answered", "cancelled"}
            else s.ljust(cols[key])
        )

    click.echo(
        " ".join(
            [
                _cell("topic_id", "topic_id"),
                _cell("name", "name"),
                _cell("status", "status"),
                _cell("total", "total"),
                _cell("pending", "pending"),
                _cell("answered", "answered"),
                _cell("cancelled", "cancelled"),
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
                    _cell("total", r["counts"]["total"]),
                    _cell("pending", r["counts"]["pending"]),
                    _cell("answered", r["counts"]["answered"]),
                    _cell("cancelled", r["counts"]["cancelled"]),
                ]
            )
        )
