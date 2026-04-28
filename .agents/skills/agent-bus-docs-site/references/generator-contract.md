# Generator Contract

This repo uses a custom Markdown-to-Fumadocs pipeline. The goal is to keep docs authored in normal
repo Markdown while publishing a richer Fumadocs site.

## Key Files

- `scripts/generate_fumadocs_site.py`: canonical content generator.
- `tests/test_generate_fumadocs_site.py`: generator regression tests.
- `site/scripts/generate-content.mjs`: Node entrypoint that finds Python and runs the generator.
- `site/source.config.ts`: Fumadocs source config for `site/content/docs`.
- `site/next.config.mjs`: static export and base-path integration.
- `site/base-path.shared.js`: shared repo-aware base-path helper.
- `site/src/lib/shared.ts`: site helpers, including base-path URL handling.

## Generated Tree

The generator writes:

- `site/content/docs/**/*.mdx`
- `site/content/docs/**/meta.json`
- `site/public/docs-assets/images/**`

Do not treat these as authoring surfaces. They should be reproducible from:

- `docs/**/*.md`
- `spec.md`
- `CHANGELOG.md`
- `docs/images/**`

## Mapping Rules

- `docs/README.md` becomes `site/content/docs/index.mdx`.
- `docs/tutorials/*.md` becomes `site/content/docs/tutorials/*.mdx`.
- `docs/how-to/*.md` becomes `site/content/docs/how-to/*.mdx`.
- `docs/reference/*.md` becomes `site/content/docs/reference/*.mdx`.
- `docs/explanation/*.md` becomes `site/content/docs/explanation/*.mdx`.
- `spec.md` becomes `site/content/docs/reference/implementation-spec.mdx`.
- `CHANGELOG.md` becomes `site/content/docs/reference/changelog.mdx`.
- `docs/diataxis-migration-matrix.md` is intentionally excluded.

Section `README.md` files are used for titles and metadata but are not published as normal pages.

## Content Rewrites

The generator is responsible for:

- stripping duplicate leading H1s where the Fumadocs page shell supplies the title
- extracting descriptions for metadata
- rewriting Markdown links to generated routes
- rewriting Markdown and HTML image sources into `docs-assets`
- wrapping selected tutorial and install-guide sections with site-specific presentation classes
- writing deterministic `meta.json` files for root and section ordering

Do not add one-off content patches in generated MDX when the source Markdown or generator rule should
own the behavior.

## Validation

Run focused generator tests after changing mapping, link rewriting, assets, or metadata:

```bash
uv run pytest tests/test_generate_fumadocs_site.py
```

Run the CI-like site build after changing routes, base-path behavior, Fumadocs config, or home/docs
layout:

```bash
GITHUB_ACTIONS=true GITHUB_REPOSITORY=alessandrobologna/agent-bus-mcp pnpm --dir site build
```

If `site/node_modules` is missing in a fresh worktree, install site dependencies first:

```bash
pnpm --dir site install --frozen-lockfile
```
