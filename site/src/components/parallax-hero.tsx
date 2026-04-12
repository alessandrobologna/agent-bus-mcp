"use client";

import Link from "next/link";
import {
  startTransition,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from "react";

type Action = {
  href: string;
  label: string;
  kind: "primary" | "secondary";
};

type Highlight = {
  title: string;
  description: string;
};

type ParallaxHeroProps = {
  badge: string;
  title: string;
  description: string;
  imageSrc: string;
  actions: Action[];
  highlights: Highlight[];
};

const MAX_OFFSET = 54;

export function ParallaxHero({
  badge,
  title,
  description,
  imageSrc,
  actions,
  highlights,
}: ParallaxHeroProps) {
  const [offset, setOffset] = useState(0);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);
  const offsetRef = useRef(0);

  const syncOffset = useEffectEvent(() => {
    if (prefersReducedMotion || typeof window === "undefined") {
      offsetRef.current = 0;
      startTransition(() => setOffset(0));
      return;
    }

    const nextOffset = Math.min(MAX_OFFSET, window.scrollY * 0.12);
    if (Math.abs(nextOffset - offsetRef.current) < 0.5) return;
    offsetRef.current = nextOffset;
    startTransition(() => setOffset(nextOffset));
  });

  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const applyPreference = () => setPrefersReducedMotion(media.matches);
    applyPreference();
    media.addEventListener("change", applyPreference);
    return () => media.removeEventListener("change", applyPreference);
  }, []);

  useEffect(() => {
    syncOffset();
    if (prefersReducedMotion) return;

    let frame = 0;
    const scheduleSync = () => {
      if (frame) return;
      frame = window.requestAnimationFrame(() => {
        frame = 0;
        syncOffset();
      });
    };

    window.addEventListener("scroll", scheduleSync, { passive: true });
    window.addEventListener("resize", scheduleSync);

    return () => {
      if (frame) window.cancelAnimationFrame(frame);
      window.removeEventListener("scroll", scheduleSync);
      window.removeEventListener("resize", scheduleSync);
    };
  }, [prefersReducedMotion, syncOffset]);

  return (
    <section className="relative overflow-hidden bg-neutral-950">
      <div className="absolute inset-0">
        <img
          src={imageSrc}
          alt=""
          aria-hidden="true"
          className="absolute inset-y-0 right-0 hidden h-full w-[56%] max-w-none object-cover object-right opacity-90 md:block"
          style={{
            filter: "saturate(0.35) brightness(0.86) contrast(1.08)",
            transform: prefersReducedMotion
              ? "scale(1.03)"
              : `translateY(${offset}px) scale(1.06)`,
          }}
        />
        <div className="absolute inset-0 bg-[linear-gradient(90deg,rgba(10,10,10,0.97)_0%,rgba(10,10,10,0.88)_34%,rgba(10,10,10,0.60)_62%,rgba(10,10,10,0.22)_100%)]" />
        <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(23,23,23,0.06)_0%,rgba(23,23,23,0.28)_100%)]" />
      </div>

      <div className="relative z-10 mx-auto max-w-7xl px-6 py-12 md:px-10 md:py-16 lg:px-12 lg:py-20">
        <div className="max-w-3xl space-y-6">
          <div className="inline-flex items-center rounded-md border border-white/10 bg-white/6 px-3 py-1.5 text-sm font-semibold text-slate-200 backdrop-blur-sm">
            {badge}
          </div>

          <div className="space-y-4">
            <h1 className="max-w-3xl text-4xl font-semibold tracking-[-0.05em] text-white md:text-6xl">
              {title}
            </h1>
            <p className="max-w-2xl text-lg leading-8 text-slate-200/86">
              {description}
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            {actions.map((action) => (
              <Link
                key={action.href}
                href={action.href}
                className={
                  action.kind === "primary"
                    ? "inline-flex items-center rounded-md bg-white px-5 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-slate-100"
                    : "inline-flex items-center rounded-md border border-white/12 bg-white/6 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-white/10"
                }
              >
                {action.label}
              </Link>
            ))}
          </div>

          <div className="grid gap-3 pt-2 sm:grid-cols-3">
            {highlights.map((highlight) => (
              <div
                key={highlight.title}
                className="rounded-xl border border-white/10 bg-black/18 px-4 py-4 backdrop-blur-sm"
              >
                <div className="text-sm font-semibold text-white">{highlight.title}</div>
                <p className="mt-1 text-sm leading-6 text-slate-200/76">
                  {highlight.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
