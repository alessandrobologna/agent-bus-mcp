# Agent Bus Docs Site

This directory contains the Fumadocs site for Agent Bus.

The authored docs stay in the repository root:

- `../docs/`
- `../spec.md`
- `../CHANGELOG.md`

The site consumes a generated publish tree. Do not hand-edit `content/docs/` or `public/docs-assets/`.

## Local development

```bash
pnpm install
pnpm dev
```

`pnpm dev` regenerates the site content before starting Next.js.

## Build

```bash
pnpm build
```

This produces a static export in `out/` for GitHub Pages.

## Preview the exported site

Build with the same repo subpath that GitHub Pages will use:

```bash
GITHUB_ACTIONS=true \
GITHUB_REPOSITORY=alessandrobologna/agent-bus-mcp \
pnpm build
```

Then start the local export server:

```bash
pnpm start
```

The preview server inspects the built export and mounts it under the same base path, so the project
site preview works at `http://127.0.0.1:3000/agent-bus-mcp/`.
