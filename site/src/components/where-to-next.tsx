import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { docsHref } from "@/lib/shared";
import { SectionKicker } from "@/components/section-kicker";

const CARDS = [
  {
    title: "See it in action",
    description: "Walk through two agents joining the same topic and exchanging messages.",
    href: docsHref("tutorials/first-topic-between-two-peers"),
  },
  {
    title: "Add it to your client",
    description: "Install the package and point Codex, Claude Code, or OpenCode at it.",
    href: docsHref("how-to/install-and-configure-agent-bus"),
  },
  {
    title: "Look up commands",
    description: "Find exact MCP tools, CLI commands, environment variables, and behavior.",
    href: docsHref("reference/runtime-reference"),
  },
];

export function WhereToNext() {
  return (
    <section className="border-b border-fd-border bg-fd-background">
      <div className="mx-auto w-full max-w-7xl px-6 py-16 md:px-10 md:py-20 lg:px-12">
        <div className="mb-8 max-w-2xl">
          <SectionKicker>Where to next</SectionKicker>
          <h2 className="mt-3 text-3xl font-semibold tracking-[-0.04em] text-fd-foreground md:text-4xl">
            Pick the next thread.
          </h2>
        </div>

        <div className="grid gap-4 md:grid-cols-3">
          {CARDS.map((card) => (
            <Link
              key={card.href}
              href={card.href}
              className="group flex flex-col rounded-2xl border border-fd-border bg-fd-card p-6 transition hover:-translate-y-0.5 hover:border-[color:var(--color-accent-amber-soft)] hover:shadow-sm"
            >
              <h3 className="text-lg font-semibold text-fd-foreground">{card.title}</h3>
              <p className="mt-2 flex-1 text-sm leading-6 text-fd-muted-foreground">
                {card.description}
              </p>
              <span className="mt-4 inline-flex items-center text-sm font-medium text-fd-foreground/85 transition group-hover:text-[color:var(--color-accent-amber)]">
                <ArrowRight
                  aria-hidden="true"
                  className="h-4 w-4 transition-transform group-hover:translate-x-0.5"
                />
              </span>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}
