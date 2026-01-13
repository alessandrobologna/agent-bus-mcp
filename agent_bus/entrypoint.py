from __future__ import annotations

import click

from agent_bus.cli import cli as cli_group


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """Agent Bus peer MCP server and administrative CLI."""
    if ctx.invoked_subcommand is None:
        from agent_bus.peer_server import main as server_main

        server_main()


@main.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind to.")
@click.option("--port", "-p", default=8080, show_default=True, help="Port to bind to.")
@click.option(
    "--db-path",
    default=None,
    help="SQLite DB path (defaults to $AGENT_BUS_DB or ~/.agent_bus/agent_bus.sqlite).",
)
def serve_command(host: str, port: int, db_path: str | None) -> None:
    """Start the Agent Bus web UI server."""
    try:
        from agent_bus.web.server import run_server
    except ImportError:
        raise click.ClickException(
            "Web UI dependencies not installed. Install with: uv sync --extra web"
        ) from None

    click.echo(f"Starting Agent Bus Web UI at http://{host}:{port}")
    click.echo("Press Ctrl+C to stop.")
    run_server(host=host, port=port, db_path=db_path)


main.add_command(cli_group, name="cli")
