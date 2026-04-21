import Link from "next/link";
import { ArrowRight, Check, X } from "lucide-react";
import { docsHref } from "@/lib/shared";
import { SectionKicker } from "@/components/section-kicker";

const BEST_FIT = [
  "Reviewer / implementer / re-review loops on one workstation",
  "Multi-agent coding sessions across Codex, Claude Code, Gemini CLI, and OpenCode",
  "Durable local audit trails for agent collaboration",
  "Searchable topic history with optional semantic indexing",
  "Reconnecting after restarts without replaying everything manually",
];

const NOT_A_FIT = [
  "Multi-machine coordination over a network",
  "Auth- or tenancy-heavy environments",
  "Workflows that need a hosted service and external participants",
  "A single agent session with no durable handoffs",
];

export function FitAndNotFit() {
  return (
    <section className="border-b border-fd-border bg-fd-muted/30">
      <div className="mx-auto w-full max-w-7xl px-6 py-16 md:px-10 md:py-20 lg:px-12">
        <div className="mb-10 max-w-2xl">
          <SectionKicker>Fit</SectionKicker>
          <h2 className="mt-3 text-3xl font-semibold tracking-[-0.04em] text-fd-foreground md:text-4xl">
            When Agent Bus MCP helps, and when it doesn&rsquo;t.
          </h2>
          <p className="mt-4 text-lg leading-8 text-fd-muted-foreground">
            It is a local-first coordination layer for agents on one workstation. Outside that
            scope, simpler or different tools are usually better.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          <FitColumn
            tone="positive"
            title="Best fit"
            description="Multi-agent coding work that needs durable structure."
            items={BEST_FIT}
          />
          <FitColumn
            tone="negative"
            title="Not a fit"
            description="Things this project does not try to be."
            items={NOT_A_FIT}
          />
        </div>

        <div className="mt-8">
          <Link
            href={docsHref("explanation/why-agent-bus")}
            className="inline-flex items-center gap-1.5 text-sm font-medium text-fd-muted-foreground transition hover:text-fd-foreground"
          >
            Read the full design rationale
            <ArrowRight aria-hidden="true" className="h-3.5 w-3.5" />
          </Link>
        </div>
      </div>
    </section>
  );
}

function FitColumn({
  tone,
  title,
  description,
  items,
}: {
  tone: "positive" | "negative";
  title: string;
  description: string;
  items: string[];
}) {
  const Icon = tone === "positive" ? Check : X;
  const iconClasses =
    tone === "positive"
      ? "text-[color:var(--color-accent-amber)]"
      : "text-fd-muted-foreground/80";

  return (
    <div className="rounded-2xl border border-fd-border bg-fd-card p-6">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-fd-foreground">{title}</h3>
        <p className="mt-1 text-sm text-fd-muted-foreground">{description}</p>
      </div>
      <ul className="space-y-3">
        {items.map((item) => (
          <li key={item} className="flex items-start gap-3 text-sm leading-6 text-fd-foreground/90">
            <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${iconClasses}`} aria-hidden="true" />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
