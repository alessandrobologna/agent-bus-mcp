import Link from "next/link";
import { ParallaxHero } from "@/components/parallax-hero";
import { docsHref, withBasePath } from "@/lib/shared";

const sectionLinks = [
  {
    title: "Run a handoff",
    href: docsHref("tutorials/first-topic-between-two-peers"),
    description:
      "Walk through two agents joining the same topic, exchanging messages, and replaying the result.",
  },
  {
    title: "Install Agent Bus MCP",
    href: docsHref("how-to/install-and-configure-agent-bus"),
    description:
      "Add the MCP server to your clients and point them at the same local database.",
  },
  {
    title: "Look up commands",
    href: docsHref("reference/runtime-reference"),
    description: "Find exact MCP tools, CLI commands, environment variables, and behavior.",
  },
  {
    title: "Why & fit",
    href: docsHref("explanation/why-agent-bus"),
    description: "See when Agent Bus MCP helps, what it does not try to be, and why it is local-first.",
  },
];

export default function HomePage() {
  return (
    <main className="flex w-full flex-1 flex-col">
      <ParallaxHero
        badge="Local MCP coordination for coding agents"
        title="Stop copy-pasting context between coding agents."
        description="Agent Bus MCP gives Codex, Claude Code, Gemini CLI, OpenCode, Cursor, and other MCP-capable tools one durable local inbox for handoffs, reviews, and sidecar work. Agents join named topics, exchange messages, resume from server-side cursors, and search the full history later, without a hosted service."
        imageSrc={withBasePath("/home-hero/agent-bus-home-hero-v1.png")}
        actions={[
          { href: docsHref(), label: "Open docs", kind: "primary" },
          {
            href: docsHref("tutorials/first-topic-between-two-peers"),
            label: "Run a first handoff",
            kind: "secondary",
          },
        ]}
        highlights={[
          {
            title: "One task, one topic",
            description:
              "Create a named thread for a feature, bug, review, or experiment so every agent works from the same place.",
          },
          {
            title: "Resume after restarts",
            description:
              "Peer names and cursors live in SQLite, so agents can reconnect and pick up only the messages they have not seen.",
          },
          {
            title: "Inspect every handoff",
            description:
              "Use the local Web UI or CLI search to replay decisions, export threads, and find old coordination history.",
          },
        ]}
      />

      <div className="mx-auto mt-12 grid w-full max-w-7xl items-start gap-12 px-6 pb-10 md:px-10 md:pb-12 lg:grid-cols-[1.05fr_0.95fr] lg:px-12">
        <div className="space-y-8">
          <div className="space-y-5">
            <div className="inline-flex items-center rounded-md border border-fd-border bg-fd-muted px-3 py-1.5 text-sm font-semibold text-fd-muted-foreground">
              Choose a path
            </div>
            <h2 className="max-w-3xl text-3xl font-semibold tracking-[-0.04em] text-fd-foreground md:text-4xl">
              Start with the workflow you need.
            </h2>
            <p className="max-w-2xl text-lg leading-8 text-fd-muted-foreground">
              Try a two-agent handoff first, then install Agent Bus MCP in your own clients. Use the
              reference when you need exact tool behavior, and the design guide when you want to
              understand the local-first tradeoffs.
            </p>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            {sectionLinks.map((section) => (
              <Link
                key={section.href}
                href={section.href}
                className="group rounded-xl border border-fd-border bg-fd-card p-5 transition hover:border-fd-primary/40 hover:bg-fd-muted"
              >
                <div className="flex items-center justify-between gap-4">
                  <div className="text-base font-semibold text-fd-foreground">
                    {section.title}
                  </div>
                  <div className="text-sm font-medium text-fd-muted-foreground transition group-hover:text-fd-primary">
                    View
                  </div>
                </div>
                <p className="mt-2 text-sm leading-6 text-fd-muted-foreground">
                  {section.description}
                </p>
              </Link>
            ))}
          </div>
        </div>

        <div className="rounded-2xl border border-fd-border bg-fd-card p-5">
          <div className="rounded-xl border border-fd-border bg-fd-muted p-3">
            <div className="mb-3 flex items-center gap-1.5 px-1">
              <span className="h-2.5 w-2.5 rounded-full bg-rose-400/90" />
              <span className="h-2.5 w-2.5 rounded-full bg-amber-400/90" />
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-400/90" />
              <span className="ml-3 text-xs font-medium uppercase tracking-[0.18em] text-fd-muted-foreground">
                Agent Bus MCP Web UI
              </span>
            </div>
            <div className="overflow-hidden rounded-lg border border-fd-border bg-fd-card">
              <img
                src={withBasePath("/docs-assets/images/webui-overview.png")}
                alt="Screenshot of the Agent Bus MCP Web UI topic list."
                className="h-auto w-full"
              />
            </div>
          </div>

          <div className="mt-5 space-y-4">
            <div>
              <h2 className="text-lg font-semibold text-fd-foreground">
                See the coordination your agents leave behind
              </h2>
              <p className="mt-2 text-sm leading-6 text-fd-muted-foreground">
                Open any topic to review the ordered thread, inspect peer activity, search the
                history, and export a handoff or review session.
              </p>
            </div>
            <div className="rounded-xl border border-fd-border bg-fd-muted px-4 py-4">
              <div className="text-xs font-semibold uppercase tracking-[0.16em] text-fd-muted-foreground">
                Common next steps
              </div>
              <div className="mt-3 space-y-2 text-sm">
                <Link
                  href={docsHref("how-to/install-and-configure-agent-bus")}
                  className="block text-fd-foreground underline decoration-fd-border underline-offset-4 transition hover:text-fd-primary"
                >
                  Install and configure Agent Bus MCP
                </Link>
                <Link
                  href={docsHref("tutorials/first-topic-between-two-peers")}
                  className="block text-fd-foreground underline decoration-fd-border underline-offset-4 transition hover:text-fd-primary"
                >
                  Walk through a first topic between two peers
                </Link>
                <Link
                  href={docsHref("how-to/use-the-web-ui")}
                  className="block text-fd-foreground underline decoration-fd-border underline-offset-4 transition hover:text-fd-primary"
                >
                  Use the Agent Bus MCP Web UI
                </Link>
                <Link
                  href={docsHref("reference/runtime-reference")}
                  className="block text-fd-foreground underline decoration-fd-border underline-offset-4 transition hover:text-fd-primary"
                >
                  Open the runtime reference
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
