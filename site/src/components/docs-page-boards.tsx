import Link from "next/link";
import type { LucideIcon } from "lucide-react";
import {
  ArrowRight,
  Clock3,
  HardDrive,
  MessageCircle,
  MessageSquare,
  Route,
  Users,
} from "lucide-react";

type BoardCard = {
  order: string;
  title: string;
  subtitle: string;
  href: string;
  Icon: LucideIcon;
};

const WHY_AGENT_BUS_CARDS: BoardCard[] = [
  {
    order: "01",
    title: "Topics",
    subtitle: "replace implicit threads",
    href: "#topics-instead-of-implicit-threads",
    Icon: MessageSquare,
  },
  {
    order: "02",
    title: "Stable peers",
    subtitle: "keep names meaningful",
    href: "#stable-peer-identity",
    Icon: Users,
  },
  {
    order: "03",
    title: "Server cursors",
    subtitle: "resume without client checkpoints",
    href: "#server-side-cursors",
    Icon: Clock3,
  },
  {
    order: "04",
    title: "Local surface",
    subtitle: "stdio, SQLite, optional UI",
    href: "#one-local-dependency-surface",
    Icon: HardDrive,
  },
  {
    order: "05",
    title: "Where it fits",
    subtitle: "strong for local coordination",
    href: "#where-agent-bus-fits",
    Icon: Route,
  },
  {
    order: "06",
    title: "Why not chat",
    subtitle: "conversation is not enough",
    href: "#why-not-just-use-chat",
    Icon: MessageCircle,
  },
];

export function WhyAgentBusBoard() {
  return (
    <DocsActionBoard
      eyebrow="Design Map"
      description="The core idea is simple: coordination works better when topics, peers, cursors, and fit boundaries are explicit."
      cards={WHY_AGENT_BUS_CARDS}
    />
  );
}

function DocsActionBoard({
  eyebrow,
  description,
  cards,
}: {
  eyebrow: string;
  description: string;
  cards: BoardCard[];
}) {
  return (
    <section className="mb-8 mt-7">
      <div className="mb-4 border-b border-fd-border pb-3">
        <p className="text-xs font-medium uppercase tracking-[0.24em] text-fd-muted-foreground">
          {eyebrow}
        </p>
        <p className="mt-2 max-w-2xl text-sm text-fd-muted-foreground">{description}</p>
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        {cards.map((card) => (
          <DocsActionCard key={card.href} card={card} />
        ))}
      </div>
    </section>
  );
}

function DocsActionCard({ card }: { card: BoardCard }) {
  const { Icon } = card;

  return (
    <Link
      href={card.href}
      className="group relative overflow-hidden rounded-2xl border border-fd-border bg-fd-card/55 p-5 transition-[transform,border-color,background-color,box-shadow] duration-150 hover:-translate-y-0.5 hover:border-fd-primary/35 hover:bg-fd-card hover:shadow-sm"
    >
      <div className="absolute right-4 top-4 text-sm font-medium tracking-[0.18em] text-fd-muted-foreground">
        {card.order}
      </div>
      <div className="mb-5 inline-flex h-11 w-11 items-center justify-center rounded-xl border border-fd-border bg-fd-background/70 text-fd-foreground">
        <Icon className="h-[22px] w-[22px]" />
      </div>
      <div className="space-y-2">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-fd-foreground">{card.title}</h2>
          <p className="mt-1 text-sm text-fd-muted-foreground">{card.subtitle}</p>
        </div>
        <div className="h-px w-full bg-fd-border" />
      </div>
      <div className="mt-4 inline-flex items-center gap-2 text-sm font-medium text-fd-foreground/80 transition-colors group-hover:text-fd-primary">
        Jump to section
        <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
      </div>
    </Link>
  );
}
