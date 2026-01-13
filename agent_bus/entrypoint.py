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


main.add_command(cli_group, name="cli")
