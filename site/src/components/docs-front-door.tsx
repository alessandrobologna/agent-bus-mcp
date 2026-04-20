import Link from "next/link";
import type { LucideIcon } from "lucide-react";
import {
  BookOpenText,
  FileSearch,
  Lightbulb,
  ListChecks,
  ArrowRight,
} from "lucide-react";
import { docsHref } from "@/lib/shared";

type DocsSectionKey = "tutorials" | "how-to" | "reference" | "explanation";

type DocsSectionDef = {
  key: DocsSectionKey;
  order: string;
  title: string;
  subtitle: string;
  blurb: string;
  href: string;
  Icon: LucideIcon;
};

const DOCS_SECTIONS: Record<DocsSectionKey, DocsSectionDef> = {
  tutorials: {
    key: "tutorials",
    order: "01",
    title: "Tutorial",
    subtitle: "walkthrough",
    blurb: "Start with one complete handoff so the coordination model clicks before setup details.",
    href: docsHref("tutorials/first-topic-between-two-peers"),
    Icon: BookOpenText,
  },
  "how-to": {
    key: "how-to",
    order: "02",
    title: "How-to",
    subtitle: "tasks",
    blurb: "Install Agent Bus MCP, share one database across clients, and inspect work in the browser.",
    href: docsHref("how-to/install-and-configure-agent-bus"),
    Icon: ListChecks,
  },
  reference: {
    key: "reference",
    order: "03",
    title: "Reference",
    subtitle: "facts",
    blurb: "Look up exact commands, tool names, config values, and behavior.",
    href: docsHref("reference/runtime-reference"),
    Icon: FileSearch,
  },
  explanation: {
    key: "explanation",
    order: "04",
    title: "Why & fit",
    subtitle: "design and boundaries",
    blurb: "See why Agent Bus MCP exists, when it helps, and when simpler coordination is enough.",
    href: docsHref("explanation/why-agent-bus"),
    Icon: Lightbulb,
  },
};

const SECTION_ORDER: DocsSectionKey[] = ["tutorials", "how-to", "reference", "explanation"];

export function DocsFrontDoor() {
  return (
    <section className="mb-8 mt-7">
      <div className="mb-4 border-b border-fd-border pb-3">
        <p className="text-xs font-medium uppercase tracking-[0.24em] text-fd-muted-foreground">
          Pick A Route
        </p>
        <p className="mt-2 max-w-2xl text-sm text-fd-muted-foreground">
          Start with a handoff, then choose the section that matches the question you have now.
          Each card opens the best first page for that route.
        </p>
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        {SECTION_ORDER.map((key) => (
          <DocsFrontDoorCard key={key} section={DOCS_SECTIONS[key]} />
        ))}
      </div>
    </section>
  );
}

function DocsFrontDoorCard({ section }: { section: DocsSectionDef }) {
  const { Icon } = section;

  return (
    <Link
      href={section.href}
      className="group relative overflow-hidden rounded-2xl border border-fd-border bg-fd-card/65 p-5 transition-[transform,border-color,background-color,box-shadow] duration-150 hover:-translate-y-0.5 hover:border-fd-primary/35 hover:bg-fd-card hover:shadow-sm"
    >
      <div className="absolute right-4 top-4 text-sm font-medium tracking-[0.18em] text-fd-muted-foreground">
        {section.order}
      </div>
      <div className="mb-5 inline-flex h-12 w-12 items-center justify-center rounded-xl border border-fd-border bg-fd-background/70 text-fd-foreground">
        <Icon className="h-6 w-6" />
      </div>
      <div className="space-y-2">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight text-fd-foreground">
            {section.title}
          </h2>
          <p className="mt-1 text-sm text-fd-muted-foreground">{section.subtitle}</p>
        </div>
        <div className="h-px w-full bg-fd-border" />
        <p className="max-w-[28rem] text-sm leading-6 text-fd-muted-foreground">{section.blurb}</p>
      </div>
      <div className="mt-5 inline-flex items-center gap-2 text-sm font-medium text-fd-foreground/85 transition-colors group-hover:text-fd-primary">
        Open section
        <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
      </div>
    </Link>
  );
}

export function DocsSectionIntro({ section }: { section: DocsSectionKey }) {
  const meta = DOCS_SECTIONS[section];
  const { Icon } = meta;

  return (
    <section className="mb-7 mt-7 overflow-hidden rounded-2xl border border-fd-border bg-fd-card/55">
      <div className="flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-4">
          <div className="inline-flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border border-fd-border bg-fd-background/70 text-fd-foreground">
            <Icon className="h-6 w-6" />
          </div>
          <div>
            <div className="text-xs font-medium uppercase tracking-[0.22em] text-fd-muted-foreground">
              {meta.order} {meta.title}
            </div>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-fd-muted-foreground">
              {meta.blurb}
            </p>
          </div>
        </div>
        <Link
          href={docsHref()}
          className="inline-flex items-center gap-2 text-sm font-medium text-fd-muted-foreground transition-colors hover:text-fd-primary"
        >
          Back to docs front door
          <ArrowRight className="h-4 w-4" />
        </Link>
      </div>
    </section>
  );
}

export function getDocsSection(path: string): DocsSectionKey | null {
  const [firstSegment] = path.split("/");
  if (firstSegment === "tutorials" || firstSegment === "how-to" || firstSegment === "reference" || firstSegment === "explanation") {
    return firstSegment;
  }
  return null;
}
