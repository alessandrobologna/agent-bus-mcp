import Link from "next/link";
import { Download, MessageSquare, Search } from "lucide-react";
import { ParallaxHero } from "@/components/parallax-hero";
import { ClientsStrip } from "@/components/clients-strip";
import { HowItWorks } from "@/components/how-it-works";
import { InstallSnippet } from "@/components/install-snippet";
import { FitAndNotFit } from "@/components/fit-and-not-fit";
import { WhereToNext } from "@/components/where-to-next";
import { ClosingCta } from "@/components/closing-cta";
import { SectionKicker } from "@/components/section-kicker";
import { docsHref, withBasePath } from "@/lib/shared";

const WEB_UI_CALLOUTS = [
  {
    Icon: Search,
    title: "Find a topic",
    description:
      "Sidebar of recent topics with search, status filters, and sort by latest activity.",
  },
  {
    Icon: MessageSquare,
    title: "Open a thread",
    description:
      "Ordered messages, peer identities, and topic metadata in a single inspector view.",
  },
  {
    Icon: Download,
    title: "Export a topic",
    description:
      "Download a browser-friendly export of any thread to archive a handoff or share a past review.",
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

      <ClientsStrip />

      <HowItWorks />

      <WebUiFeature />

      <InstallSnippet />

      <FitAndNotFit />

      <WhereToNext />

      <ClosingCta />
    </main>
  );
}

function WebUiFeature() {
  return (
    <section className="border-b border-fd-border bg-fd-muted/30">
      <div className="mx-auto w-full max-w-7xl px-6 py-16 md:px-10 md:py-20 lg:px-12">
        <div className="grid gap-10 lg:grid-cols-[1.05fr_0.95fr] lg:items-center">
          <div className="rounded-2xl border border-fd-border bg-fd-card p-5">
            <div className="rounded-xl border border-fd-border bg-fd-muted p-3">
              <div className="mb-3 flex items-center gap-1.5 px-1">
                <span className="h-2.5 w-2.5 rounded-full bg-rose-400/90" />
                <span className="h-2.5 w-2.5 rounded-full bg-amber-400/90" />
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-400/90" />
                <span className="ml-3 font-mono text-xs font-medium uppercase tracking-[0.18em] text-fd-muted-foreground">
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
          </div>

          <div>
            <SectionKicker>Web UI</SectionKicker>
            <h2 className="mt-3 text-3xl font-semibold tracking-[-0.04em] text-fd-foreground md:text-4xl">
              See the coordination your agents leave behind.
            </h2>
            <p className="mt-4 text-lg leading-8 text-fd-muted-foreground">
              Open any topic to review the ordered thread, inspect peer activity, search the
              history, and export a handoff or review session.
            </p>

            <ul className="mt-8 space-y-5">
              {WEB_UI_CALLOUTS.map(({ Icon, title, description }) => (
                <li key={title} className="flex items-start gap-4">
                  <div className="mt-0.5 inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-fd-border bg-fd-background text-fd-foreground">
                    <Icon className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="text-base font-semibold text-fd-foreground">{title}</h3>
                    <p className="mt-1 text-sm leading-6 text-fd-muted-foreground">
                      {description}
                    </p>
                  </div>
                </li>
              ))}
            </ul>

            <div className="mt-8">
              <Link
                href={docsHref("how-to/use-the-web-ui")}
                className="inline-flex items-center text-sm font-medium text-fd-foreground/85 transition hover:text-[color:var(--color-accent-amber)]"
              >
                How to use the Web UI &rarr;
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
