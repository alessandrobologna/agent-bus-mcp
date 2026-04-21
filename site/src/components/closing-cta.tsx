import Link from "next/link";
import { docsHref } from "@/lib/shared";

export function ClosingCta() {
  return (
    <section className="bg-fd-background">
      <div className="mx-auto w-full max-w-7xl px-6 py-16 md:px-10 md:py-20 lg:px-12">
        <div className="relative overflow-hidden rounded-3xl border border-[color:var(--color-accent-amber-soft)] bg-fd-card p-10 md:p-14">
          <div
            aria-hidden="true"
            className="grid-paper pointer-events-none absolute inset-0 opacity-60"
          />
          <div
            aria-hidden="true"
            className="pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-[color:var(--color-accent-amber-faint)] blur-3xl"
          />
          <div className="relative flex flex-col items-start gap-6 md:flex-row md:items-center md:justify-between">
            <div className="max-w-xl">
              <p className="font-mono text-xs font-medium uppercase tracking-[0.22em] text-[color:var(--color-accent-amber)]">
                Get started
              </p>
              <h2 className="mt-3 text-3xl font-semibold tracking-[-0.04em] text-fd-foreground md:text-4xl">
                Run your first handoff in two minutes.
              </h2>
              <p className="mt-3 text-base leading-7 text-fd-muted-foreground md:text-lg">
                Two agents, one topic, one ordered thread. Local-first, no hosted service.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <Link
                href={docsHref("tutorials/first-topic-between-two-peers")}
                className="inline-flex items-center rounded-md bg-fd-foreground px-5 py-2.5 text-sm font-semibold text-fd-background transition hover:bg-fd-foreground/90"
              >
                Run a first handoff
              </Link>
              <Link
                href={docsHref("how-to/install-and-configure-agent-bus")}
                className="inline-flex items-center rounded-md border border-fd-border bg-fd-background px-5 py-2.5 text-sm font-semibold text-fd-foreground transition hover:border-fd-foreground/30"
              >
                Install Agent Bus MCP
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
