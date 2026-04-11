from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def replace_version(path: Path, current: str, updated: str) -> None:
    text = path.read_text(encoding="utf-8")
    new_text = text.replace(f'version = "{current}"', f'version = "{updated}"', 1)
    if new_text == text:
        raise SystemExit(f"Failed to update version in {path}")
    path.write_text(new_text, encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: apply_version_overrides.py <python-version> <cargo-version>")

    py_version = sys.argv[1].strip()
    cargo_version = sys.argv[2].strip()
    if not py_version or not cargo_version:
        raise SystemExit("expected both python and cargo version overrides")

    pyproject_path = Path("pyproject.toml")
    cargo_path = Path("Cargo.toml")

    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    cargo = tomllib.loads(cargo_path.read_text(encoding="utf-8"))

    current_py_version = pyproject["project"]["version"]
    current_cargo_version = cargo["package"]["version"]

    replace_version(pyproject_path, current_py_version, py_version)
    replace_version(cargo_path, current_cargo_version, cargo_version)


if __name__ == "__main__":
    main()
