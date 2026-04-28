---
name: agent-bus-docs-site
description: Use when changing the Agent Bus Fumadocs website, docs generator, site metadata, GitHub Pages export behavior, or Markdown docs that feed agentbusmcp.com.
---

# Agent Bus Docs Site

Use this skill for Agent Bus website and docs-site work. The important contract is that the
repository Markdown remains the source of truth; the Fumadocs content tree is generated output.

Read [references/generator-contract.md](references/generator-contract.md) before changing the
generator or debugging site content drift.

## Source Of Truth

- Author durable documentation in `docs/**/*.md`, `spec.md`, and `CHANGELOG.md`.
- Do not hand-edit `site/content/docs/` or `site/public/docs-assets/`; regenerate them.
- Keep website copy that explains product positioning aligned with the authored Markdown docs when
  the same facts appear in both places.
- Treat generated docs routes as public site behavior, not just build artifacts.

## Main Workflow

1. Edit the source Markdown or site component that owns the change.
2. If source docs changed, run the generator through the site script:

   ```bash
   pnpm --dir site run generate:content
   ```

3. If the generator changed, run its focused tests:

   ```bash
   uv run pytest tests/test_generate_fumadocs_site.py
   ```

4. For site or routing changes, build with the GitHub Pages base path:

   ```bash
   GITHUB_ACTIONS=true GITHUB_REPOSITORY=alessandrobologna/agent-bus-mcp pnpm --dir site build
   ```

5. If the change affects release snippets, install commands, or package versions, check all docs
   copies with `rg` before committing.

## Generator Boundaries

- `scripts/generate_fumadocs_site.py` converts repository docs into the Fumadocs publish tree.
- `site/scripts/generate-content.mjs` is the Node wrapper used by `pnpm --dir site build` and
  `pnpm --dir site dev`.
- `site/package.json` must keep `generate:content` before `fumadocs-mdx` and Next.js commands.
- The generator owns section ordering, `meta.json`, special pages, asset copying, and link rewriting.
- Site React components own presentation, home page layout, search UI, OG routes, and llms routes.

## Review Checklist

- Markdown links generated from `docs/README.md` still resolve under `/docs`.
- Relative links are rewritten to the generated route, not left pointing outside the docs app.
- Asset paths under `docs/images/` are copied to `site/public/docs-assets/images/`.
- Diataxis section pages are represented through metadata, not by publishing every section README.
- `spec.md` maps to `reference/implementation-spec.mdx`.
- `CHANGELOG.md` maps to `reference/changelog.mdx`.
- GitHub Pages export still uses the repository base path in CI-like builds.

## Common Failure Shields

- Symptom: local build passes but a docs front-door link escapes `/docs`.
  Cause: a relative Markdown link was emitted unchanged.
  Fix: update link rewriting or source link shape, then verify with generator tests and a site build.
- Symptom: website preview works locally but GitHub Pages assets 404.
  Cause: base path or static export settings drifted.
  Fix: build with `GITHUB_ACTIONS=true GITHUB_REPOSITORY=alessandrobologna/agent-bus-mcp`.
- Symptom: docs content differs from Markdown source.
  Cause: generated files were edited or stale.
  Fix: discard generated edits if they are tracked accidentally, then rerun `pnpm --dir site run generate:content`.
