import type { ReactNode } from "react";

export function SectionKicker({ children }: { children: ReactNode }) {
  return (
    <p className="flex items-center gap-2.5 text-xs font-medium uppercase tracking-[0.22em] text-fd-muted-foreground">
      <span
        aria-hidden="true"
        className="inline-block h-1.5 w-1.5 rounded-full bg-[color:var(--color-accent-amber)]"
      />
      {children}
    </p>
  );
}
