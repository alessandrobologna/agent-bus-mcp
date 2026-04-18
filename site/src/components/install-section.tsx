import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import {
  Boxes,
  Database,
  Download,
  LayoutPanelTop,
  MessageSquare,
  Play,
  Search,
  Settings2,
  Sparkles,
  SquareTerminal,
  TriangleAlert,
} from "lucide-react";

type InstallSectionVariant =
  | "package"
  | "client"
  | "database"
  | "checkout"
  | "webui"
  | "workflow";

type WebUiSectionVariant =
  | "start"
  | "find"
  | "thread"
  | "search"
  | "export"
  | "troubleshooting";

type InstallSectionMeta = {
  kicker: string;
  Icon: LucideIcon;
};

const SECTION_META: Record<InstallSectionVariant | WebUiSectionVariant, InstallSectionMeta> = {
  package: {
    kicker: "Published package",
    Icon: Boxes,
  },
  client: {
    kicker: "Client setup",
    Icon: Settings2,
  },
  database: {
    kicker: "Shared database",
    Icon: Database,
  },
  checkout: {
    kicker: "Local checkout",
    Icon: SquareTerminal,
  },
  webui: {
    kicker: "Web UI",
    Icon: LayoutPanelTop,
  },
  workflow: {
    kicker: "Workflow skill",
    Icon: Sparkles,
  },
  start: {
    kicker: "Start the Web UI",
    Icon: Play,
  },
  find: {
    kicker: "Find a topic",
    Icon: Search,
  },
  thread: {
    kicker: "Thread view",
    Icon: MessageSquare,
  },
  search: {
    kicker: "Search",
    Icon: Search,
  },
  export: {
    kicker: "Export",
    Icon: Download,
  },
  troubleshooting: {
    kicker: "Troubleshooting",
    Icon: TriangleAlert,
  },
};

export function InstallSection({
  variant,
  children,
}: {
  variant: InstallSectionVariant;
  children: ReactNode;
}) {
  return <GuideSection variant={variant}>{children}</GuideSection>;
}

export function WebUiSection({
  variant,
  children,
}: {
  variant: WebUiSectionVariant;
  children: ReactNode;
}) {
  return <GuideSection variant={variant}>{children}</GuideSection>;
}

function GuideSection({
  variant,
  children,
}: {
  variant: InstallSectionVariant | WebUiSectionVariant;
  children: ReactNode;
}) {
  const { Icon, kicker } = SECTION_META[variant];
  return (
    <section className="my-8 rounded-2xl border border-fd-border bg-fd-card/45 px-5 py-5 sm:px-6">
      <div className="flex items-start gap-4">
        <div className="mt-0.5 inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-fd-border bg-fd-background/75 text-fd-foreground">
          <Icon className="h-[22px] w-[22px]" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="mb-2 text-xs font-medium uppercase tracking-[0.22em] text-fd-muted-foreground">
            {kicker}
          </p>
          <div className="[&>h2]:mt-0 [&>h2]:border-none [&>h2]:pb-0 [&>h2]:text-3xl [&>h2]:font-semibold [&>h2]:tracking-tight [&>p:first-of-type]:mt-2">
            {children}
          </div>
        </div>
      </div>
    </section>
  );
}
