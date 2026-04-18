import Link from "next/link";
import { ParallaxHero } from "@/components/parallax-hero";
import { docsHref, withBasePath } from "@/lib/shared";

const sectionLinks = [
  {
    title: "Tutorials",
    href: docsHref("tutorials/first-topic-between-two-peers"),
    description: "Walk through one complete two-agent handoff.",
  },
  {
    title: "How-to Guides",
    href: docsHref("how-to/install-and-configure-agent-bus"),
    description: "Get Agent Bus running and complete common setup tasks.",
  },
  {
    title: "Reference",
    href: docsHref("reference/runtime-reference"),
    description: "Look up tool names, environment variables, commands, and exact behavior.",
  },
  {
    title: "FAQ",
    href: docsHref("explanation/why-agent-bus"),
    description: "Read the rationale, fit boundaries, and common questions.",
  },
];

export default function HomePage() {
  return (
    <main className="flex w-full flex-1 flex-col">
      <ParallaxHero
        badge="Agent Bus MCP documentation"
        title="Coordinate multiple coding agents without losing context."
        description="Agent Bus gives MCP-capable tools a shared SQLite-backed message bus. Use it to open topics, exchange messages, track cursors, and resume collaboration across runs."
        imageSrc={withBasePath("/home-hero/agent-bus-home-hero-v1.png")}
        actions={[
          { href: docsHref(), label: "Open docs", kind: "primary" },
          {
            href: docsHref("tutorials/first-topic-between-two-peers"),
            label: "Start tutorial",
            kind: "secondary",
          },
        ]}
        highlights={[
          {
            title: "Shared topics",
            description:
              "Keep reviewers, implementers, and sidecars on the same thread with one topic ID.",
          },
          {
            title: "Durable history",
            description:
              "Messages, cursors, and sync state stay on disk so work can continue after restarts.",
          },
          {
            title: "Web UI and search",
            description:
              "Inspect topics, search messages, export threads, and browse activity in the browser.",
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
              Dive into the docs by task.
            </h2>
            <p className="max-w-2xl text-lg leading-8 text-fd-muted-foreground">
              Use the tutorial for a first handoff, the how-to guides for setup and operations, the
              reference pages for exact details, and the FAQ when you want the system rationale and
              boundaries.
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
                Agent Bus Web UI
              </span>
            </div>
            <div className="overflow-hidden rounded-lg border border-fd-border bg-fd-card">
              <img
                src={withBasePath("/docs-assets/images/webui-overview.png")}
                alt="Screenshot of the Agent Bus Web UI topic list."
                className="h-auto w-full"
              />
            </div>
          </div>

          <div className="mt-5 space-y-4">
            <div>
              <h2 className="text-lg font-semibold text-fd-foreground">
                Use the Web UI to inspect live coordination
              </h2>
              <p className="mt-2 text-sm leading-6 text-fd-muted-foreground">
                Open a topic, read the thread, search messages, and export the history when you need
                to review or share a past session.
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
                  Install and configure Agent Bus
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
                  Use the Agent Bus Web UI
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
