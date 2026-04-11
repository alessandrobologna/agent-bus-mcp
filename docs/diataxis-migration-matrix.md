# Diataxis Migration Matrix

Scope for this pass:

- In scope: `README.md`, `spec.md`, `CHANGELOG.md`
- Excluded: `AGENTS.md` and `CLAUDE.md` because they are repository/agent instruction files rather
  than end-user product documentation

Follow-up additions in this pass:

- `docs/how-to/use-the-web-ui.md` adds a dedicated task-oriented guide for the local browser
  workbench.
- the README keeps a lightweight Web UI teaser, while the procedural browser workflow now lives in
  `how-to/`.

| source_path | target_mode | action | notes |
| --- | --- | --- | --- |
| `README.md` | `explanation` + `how-to` + `reference` + `tutorials` | `split` | Rewrite the README as the front door, then move mixed content into Diataxis pages under `docs/` |
| `spec.md` | `reference` | `rewrite` | Keep `spec.md` in place as the authoritative protocol/reference spec and link to it from the new reference landing page |
| `CHANGELOG.md` | `reference` | `rewrite` | Keep in place and link from the reference landing page as release/upgrade reference |
| `docs/tutorials/` | `tutorials` | `move` | New landing page and first hands-on walkthrough |
| `docs/how-to/` | `how-to` | `move` | New setup-oriented docs for install, client config, and optional web/skill setup |
| `docs/how-to/use-the-web-ui.md` | `how-to` | `add` | Dedicated goal-driven guide for launching, browsing, searching, and exporting in the Web UI |
| `docs/reference/` | `reference` | `move` | New lookup-oriented runtime reference plus links to the raw spec and changelog |
| `docs/explanation/` | `explanation` | `move` | New product rationale page explaining where Agent Bus fits and why its local architecture matters |

Risky moves in this pass:

- The README is heavily rewritten to become a front door instead of a combined install guide and
  reference page.
- Existing details are preserved additively in new docs pages rather than removed from the repo.
