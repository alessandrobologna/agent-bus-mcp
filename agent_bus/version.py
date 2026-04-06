from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

PACKAGE_NAME = "agent-bus-mcp"


def get_package_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:  # pragma: no cover
        return "0.0.0+unknown"


__version__ = get_package_version()
