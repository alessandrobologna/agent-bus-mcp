import { SectionKicker } from "@/components/section-kicker";

const CLIENTS = [
  "Codex",
  "Claude Code",
  "Gemini CLI",
  "OpenCode",
  "Cursor",
];

export function ClientsStrip() {
  return (
    <section className="border-b border-fd-border bg-fd-background">
      <div className="mx-auto flex w-full max-w-7xl flex-col items-start gap-4 px-6 py-7 md:flex-row md:items-center md:gap-6 md:px-10 md:py-8 lg:px-12">
        <SectionKicker>Works inside</SectionKicker>
        <ul className="flex flex-wrap items-center gap-2.5">
          {CLIENTS.map((name) => (
            <li
              key={name}
              className="rounded-full border border-fd-border bg-fd-card px-3.5 py-1.5 font-mono text-xs font-medium text-fd-foreground/85"
            >
              {name}
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
