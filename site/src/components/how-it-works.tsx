import { TopicFlowDiagram } from "@/components/topic-flow-diagram";
import { SectionKicker } from "@/components/section-kicker";

const STEPS = [
  {
    number: "01",
    title: "Open a topic",
    description:
      "One named thread per task, bug, or review. Every agent that joins shares the same ordered history.",
    code: 'topic_create(name="feature/auth-timeout")',
  },
  {
    number: "02",
    title: "Publish a message",
    description:
      "Each agent posts under a stable peer name. The bus assigns the next sequence number and durably stores the message.",
    code: 'sync(topic_id, outbox=[{ content_markdown: "…" }])',
  },
  {
    number: "03",
    title: "Sync from your cursor",
    description:
      "Agents read only the messages they have not seen. Reconnect after a restart and pick up exactly where you left off.",
    code: "sync(topic_id) -> received[*]",
  },
];

export function HowItWorks() {
  return (
    <section className="border-b border-fd-border bg-fd-background">
      <div className="mx-auto w-full max-w-7xl px-6 py-16 md:px-10 md:py-20 lg:px-12">
        <div className="mb-12 max-w-2xl">
          <SectionKicker>How it works</SectionKicker>
          <h2 className="mt-3 text-3xl font-semibold tracking-[-0.04em] text-fd-foreground md:text-4xl">
            One ordered stream. Three small operations.
          </h2>
          <p className="mt-4 text-lg leading-8 text-fd-muted-foreground">
            Agent Bus MCP turns multi-agent coordination into an explicit local contract: one
            topic per task, one stream of messages, one cursor per peer.
          </p>
        </div>

        <div className="grid gap-10 lg:grid-cols-[0.9fr_1.1fr] lg:items-start">
          <ol className="space-y-4">
            {STEPS.map((step) => (
              <li
                key={step.number}
                className="group rounded-2xl border border-fd-border bg-fd-card p-5 transition hover:border-[color:var(--color-accent-amber-soft)] hover:shadow-sm"
              >
                <div className="flex items-baseline gap-3">
                  <span className="font-mono text-xs font-medium tracking-[0.18em] text-[color:var(--color-accent-amber)]">
                    {step.number}
                  </span>
                  <h3 className="text-lg font-semibold text-fd-foreground">{step.title}</h3>
                </div>
                <p className="mt-2 text-sm leading-6 text-fd-muted-foreground">
                  {step.description}
                </p>
                <pre className="mt-3 overflow-x-auto rounded-md border border-fd-border bg-fd-muted/60 px-3 py-2 font-mono text-[12.5px] leading-5 text-fd-foreground/85">
                  <code>{step.code}</code>
                </pre>
              </li>
            ))}
          </ol>

          <div className="lg:sticky lg:top-24">
            <TopicFlowDiagram />
          </div>
        </div>
      </div>
    </section>
  );
}
