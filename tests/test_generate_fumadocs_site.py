from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_generator(repo_root: Path, content_root: Path, public_root: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_fumadocs_site.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(repo_root),
            "--content-root",
            str(content_root),
            "--public-root",
            str(public_root),
        ],
        check=True,
    )


def test_generator_normalizes_docs_tree_and_assets(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    content_root = repo_root / "site" / "content" / "docs"
    public_root = repo_root / "site" / "public"

    _write(
        repo_root / "docs" / "README.md",
        """# Documentation

Welcome to the docs front door.

- [Tutorials](tutorials/README.md)
- [How-to guides](how-to/README.md)
- [Reference](reference/README.md)
- [Explanation](explanation/README.md)
- [Search and embeddings reference](reference/search-and-embeddings-reference.md)
""",
    )
    _write(
        repo_root / "docs" / "tutorials" / "README.md",
        """# Tutorials

Learn by doing.

- [First topic](first-topic-between-two-peers.md)
""",
    )
    _write(
        repo_root / "docs" / "tutorials" / "first-topic-between-two-peers.md",
        """# First topic between two peers

Follow this guide to complete a basic handoff.

## Step 1: connect Agent Bus in both clients

Connect both clients.

## Step 2: create a topic from the first agent

Create the topic.

## What you just learned

Tutorial wrap-up.

See [Install and configure Agent Bus](../how-to/install-and-configure-agent-bus.md).
""",
    )
    _write(
        repo_root / "docs" / "how-to" / "README.md",
        """# How-to guides

Fast paths for concrete tasks.

- [Install and configure Agent Bus](install-and-configure-agent-bus.md)
- [Use the Web UI](use-the-web-ui.md)
""",
    )
    _write(
        repo_root / "docs" / "how-to" / "install-and-configure-agent-bus.md",
        """# Install and configure Agent Bus

Install the package and start the server.

## Fastest path: run the published package with `uvx`
<!-- site-wrap: package -->

Fastest path body.

## Add Agent Bus to a client
<!-- site-wrap: client -->

Client setup body.

## Use one database path across clients
<!-- site-wrap: database -->

Shared database body.

## Work from a local checkout
<!-- site-wrap: checkout -->

Local checkout body.

## Optional: turn on the Web UI
<!-- site-wrap: webui -->

Web UI body.

## Optional: add the `agent-bus-workflows` skill
<!-- site-wrap: workflow -->

Workflow body.
""",
    )
    _write(
        repo_root / "docs" / "how-to" / "use-the-web-ui.md",
        """# Use the Web UI

Browse topics from the browser.

<p align="center">
  <img src="../images/webui-overview.png" alt="Overview" width="960" />
</p>

## Launch the browser workbench
<!-- site-wrap: start -->

Start body.

## Find a topic fast
<!-- site-wrap: find -->

Find body.

## Open one thread
<!-- site-wrap: thread -->

Open body.

## Search everything
<!-- site-wrap: search -->

Search body.

## Export one topic
<!-- site-wrap: export -->

Export body.

## Fix common problems
<!-- site-wrap: troubleshooting -->

Troubleshooting body.

See [Runtime reference](../reference/runtime-reference.md).
""",
    )
    _write(
        repo_root / "docs" / "reference" / "README.md",
        """# Reference

Exact facts and lookup.

- [Runtime reference](runtime-reference.md)
- [Implementation spec](../../spec.md)
- [Changelog](../../CHANGELOG.md)
""",
    )
    _write(
        repo_root / "docs" / "reference" / "runtime-reference.md",
        """# Runtime reference

Lookup commands and configuration values.

- [Implementation spec](../../spec.md)
- [Changelog](../../CHANGELOG.md)
""",
    )
    _write(
        repo_root / "docs" / "reference" / "search-and-embeddings-reference.md",
        """# Search and embeddings reference

Exact search and indexing details.
""",
    )
    _write(
        repo_root / "docs" / "explanation" / "README.md",
        """# Explanation

Design rationale and tradeoffs.
""",
    )
    _write(
        repo_root / "docs" / "explanation" / "why-agent-bus.md",
        """# Why Agent Bus?

Understand the local-only design.
""",
    )
    _write(repo_root / "docs" / "diataxis-migration-matrix.md", "# Internal matrix\n")
    _write(repo_root / "docs" / "images" / "webui-overview.png", "png")
    _write(repo_root / "spec.md", "# Spec\n\nAuthoritative protocol details.\n")
    _write(repo_root / "CHANGELOG.md", "# Changelog\n\nAll notable changes.\n")

    _run_generator(repo_root, content_root, public_root)

    assert (content_root / "index.mdx").exists()
    assert not (content_root / "tutorials" / "index.mdx").exists()
    assert not (content_root / "how-to" / "index.mdx").exists()
    assert not (content_root / "reference" / "index.mdx").exists()
    assert not (content_root / "explanation" / "index.mdx").exists()
    assert (content_root / "reference" / "implementation-spec.mdx").exists()
    assert (content_root / "reference" / "changelog.mdx").exists()
    assert not (content_root / "diataxis-migration-matrix.mdx").exists()

    how_to_page = (content_root / "how-to" / "use-the-web-ui.mdx").read_text(encoding="utf-8")
    root_page = (content_root / "index.mdx").read_text(encoding="utf-8")
    install_page = (content_root / "how-to" / "install-and-configure-agent-bus.mdx").read_text(
        encoding="utf-8"
    )
    tutorial_page = (content_root / "tutorials" / "first-topic-between-two-peers.mdx").read_text(
        encoding="utf-8"
    )
    assert "../../../docs-assets/images/webui-overview.png" in how_to_page
    assert '<WebUiSection variant="start">' in how_to_page
    assert '<WebUiSection variant="find">' in how_to_page
    assert '<WebUiSection variant="thread">' in how_to_page
    assert '<WebUiSection variant="search">' in how_to_page
    assert '<WebUiSection variant="export">' in how_to_page
    assert '<WebUiSection variant="troubleshooting">' in how_to_page
    assert '<InstallSection variant="package">' in install_page
    assert '<InstallSection variant="client">' in install_page
    assert '<InstallSection variant="database">' in install_page
    assert '<InstallSection variant="checkout">' in install_page
    assert '<InstallSection variant="webui">' in install_page
    assert '<InstallSection variant="workflow">' in install_page
    assert '<div className="fd-steps">' in tutorial_page
    assert tutorial_page.count('<div className="fd-step">') >= 2
    assert "## Step 1: connect Agent Bus in both clients" in tutorial_page
    assert "## What you just learned" in tutorial_page
    assert "../../reference/runtime-reference/" in how_to_page
    assert "./tutorials/first-topic-between-two-peers/" in root_page
    assert "./how-to/install-and-configure-agent-bus/" in root_page
    assert "./reference/runtime-reference/" in root_page
    assert "./reference/search-and-embeddings-reference/" in root_page
    assert "./explanation/why-agent-bus/" in root_page
    assert ".mdx" not in root_page
    assert ".mdx" not in how_to_page
    assert root_page.count("Welcome to the docs front door.") == 1
    assert how_to_page.count("Browse topics from the browser.") == 1

    assert (content_root / "meta.json").exists()
    assert (content_root / "tutorials" / "meta.json").exists()
    assert (content_root / "how-to" / "meta.json").exists()
    assert (content_root / "reference" / "meta.json").exists()
    assert (content_root / "explanation" / "meta.json").exists()
    assert json.loads((content_root / "how-to" / "meta.json").read_text(encoding="utf-8"))[
        "pages"
    ] == [
        "install-and-configure-agent-bus",
        "use-the-web-ui",
    ]
    assert json.loads((content_root / "reference" / "meta.json").read_text(encoding="utf-8"))[
        "pages"
    ] == [
        "runtime-reference",
        "implementation-spec",
        "changelog",
        "search-and-embeddings-reference",
    ]
    assert (public_root / "docs-assets" / "images" / "webui-overview.png").exists()


def test_generator_is_deterministic(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    content_root = repo_root / "site" / "content" / "docs"
    public_root = repo_root / "site" / "public"

    _write(repo_root / "docs" / "README.md", "# Documentation\n\nFront door.\n")
    _write(repo_root / "docs" / "tutorials" / "README.md", "# Tutorials\n\nLearn by doing.\n")
    _write(repo_root / "docs" / "how-to" / "README.md", "# How-to guides\n\nFast paths.\n")
    _write(repo_root / "docs" / "reference" / "README.md", "# Reference\n\nLookup.\n")
    _write(repo_root / "docs" / "explanation" / "README.md", "# Explanation\n\nTradeoffs.\n")
    _write(repo_root / "spec.md", "# Spec\n\nProtocol.\n")
    _write(repo_root / "CHANGELOG.md", "# Changelog\n\nChanges.\n")

    _run_generator(repo_root, content_root, public_root)
    snapshot = {
        path.relative_to(content_root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(content_root.rglob("*"))
        if path.is_file()
    }

    _run_generator(repo_root, content_root, public_root)
    assert snapshot == {
        path.relative_to(content_root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(content_root.rglob("*"))
        if path.is_file()
    }
