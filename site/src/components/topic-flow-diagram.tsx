"use client";

export function TopicFlowDiagram() {
  return (
    <figure className="topic-flow not-prose my-8 overflow-hidden rounded-2xl border border-fd-border bg-fd-card p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between gap-3">
        <figcaption className="text-sm font-semibold text-fd-foreground">
          Topic Flow Schematic
        </figcaption>
        <div className="rounded-full border border-fd-border bg-fd-muted px-3 py-1 text-xs font-medium text-fd-muted-foreground">
          publish -&gt; sequence -&gt; sync
        </div>
      </div>

      <svg
        viewBox="0 0 960 384"
        role="img"
        aria-label="Technical drawing showing a reviewer publishing into an Agent Bus MCP topic, the bus assigning the next sequence number, and an implementer syncing from its own cursor."
        className="h-auto w-full"
      >
        <defs>
          <pattern id="topic-flow-grid" width="48" height="48" patternUnits="userSpaceOnUse">
            <path d="M 48 0 L 0 0 0 48" className="grid-line" />
          </pattern>
          <marker id="topic-flow-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10" className="arrow-head" />
          </marker>
        </defs>

        <rect className="grid-fill" x="0" y="0" width="960" height="384" />

        <g>
          <rect className="frame frame-peer" x="78" y="118" width="178" height="150" rx="18" />
          <rect className="frame frame-bus" x="306" y="86" width="348" height="214" rx="22" />
          <rect className="frame frame-peer" x="704" y="118" width="178" height="150" rx="18" />
        </g>

        <g className="construction">
          <line x1="167" y1="118" x2="167" y2="98" />
          <line x1="480" y1="86" x2="480" y2="60" />
          <line x1="793" y1="118" x2="793" y2="98" />
          <line x1="167" y1="268" x2="167" y2="302" />
          <line x1="793" y1="268" x2="793" y2="302" />
        </g>

        <g>
          <text className="small-label" x="167" y="104" textAnchor="middle">
            PEER A
          </text>
          <text className="small-label" x="480" y="66" textAnchor="middle">
            AGENT BUS
          </text>
          <text className="small-label" x="793" y="104" textAnchor="middle">
            PEER B
          </text>
        </g>

        <g>
          <text className="peer-name" x="167" y="164" textAnchor="middle">
            reviewer
          </text>
          <text className="peer-annotation" x="120" y="198">
            op
          </text>
          <line className="guide-line" x1="144" y1="193" x2="236" y2="193" />
          <text className="peer-value" x="236" y="198" textAnchor="end">
            publish
          </text>
          <text className="peer-annotation" x="120" y="234">
            cursor
          </text>
          <line className="guide-line" x1="160" y1="229" x2="236" y2="229" />
          <text className="peer-value" x="236" y="234" textAnchor="end">
            22
          </text>
        </g>

        <g>
          <text className="topic-kicker" x="480" y="132" textAnchor="middle">
            TOPIC
          </text>
          <text className="topic-name" x="480" y="170" textAnchor="middle">
            release-docs
          </text>
          <text className="topic-caption" x="480" y="198" textAnchor="middle">
            ordered topic stream
          </text>
        </g>

        <g className="sequence-ruler">
          <line className="sequence-line" x1="374" y1="248" x2="586" y2="248" markerEnd="url(#topic-flow-arrow)" />
          <line className="sequence-tick" x1="414" y1="238" x2="414" y2="258" />
          <line className="sequence-tick" x1="480" y1="238" x2="480" y2="258" />
          <line className="sequence-tick sequence-tick-active" x1="546" y1="232" x2="546" y2="264" />
          <text className="tick-label" x="414" y="282" textAnchor="middle">
            21
          </text>
          <text className="tick-label" x="480" y="282" textAnchor="middle">
            22
          </text>
          <text className="tick-label tick-label-active" x="546" y="282" textAnchor="middle">
            23
          </text>
          <text className="callout-text" x="546" y="222" textAnchor="middle">
            next seq
          </text>
        </g>

        <line className="main-flow main-flow-soft" x1="256" y1="214" x2="704" y2="214" />
        <line className="main-flow" x1="256" y1="214" x2="704" y2="214" markerEnd="url(#topic-flow-arrow)" />

        <circle className="port" cx="256" cy="214" r="6" />
        <circle className="port port-active" cx="704" cy="214" r="6" />
        <circle className="seq-node" cx="546" cy="214" r="18" />
        <circle className="seq-core" cx="546" cy="214" r="6" />

        <g className="message-token" aria-hidden="true">
          <rect className="token" x="248" y="206" width="16" height="16" rx="2" />
        </g>

        <g className="arrival-ring-wrap" aria-hidden="true">
          <circle className="arrival-ring" cx="704" cy="214" r="18" />
        </g>

        <g>
          <text className="peer-name" x="793" y="164" textAnchor="middle">
            implementer
          </text>
          <text className="peer-annotation" x="744" y="198">
            op
          </text>
          <line className="guide-line" x1="768" y1="193" x2="862" y2="193" />
          <text className="peer-value" x="862" y="198" textAnchor="end">
            sync
          </text>
          <text className="peer-annotation" x="744" y="234">
            cursor
          </text>
          <line className="guide-line" x1="784" y1="229" x2="862" y2="229" />
          <text className="peer-value peer-value-active" x="862" y="234" textAnchor="end">
            23
          </text>
        </g>

        <g className="dimension-text">
          <text x="167" y="322" textAnchor="middle">
            publish peer
          </text>
          <text x="480" y="322" textAnchor="middle">
            sequence + durable history
          </text>
          <text x="793" y="322" textAnchor="middle">
            sync peer
          </text>
        </g>
      </svg>

      <p className="mt-3 text-sm leading-6 text-fd-muted-foreground">
        Each publish enters one ordered topic stream, receives the next sequence number, and becomes visible to peers resuming from their own cursor.
      </p>

      <style jsx>{`
        .topic-flow {
          --line: color-mix(in srgb, var(--color-fd-foreground) 28%, transparent);
          --line-strong: color-mix(in srgb, var(--color-fd-foreground) 42%, transparent);
          --line-faint: color-mix(in srgb, var(--color-fd-foreground) 12%, transparent);
          --text: var(--color-fd-foreground);
          --muted: color-mix(in srgb, var(--color-fd-foreground) 72%, transparent);
          --faint: color-mix(in srgb, var(--color-fd-foreground) 54%, transparent);
          --accent: var(--color-accent-amber, color-mix(in srgb, var(--color-fd-foreground) 92%, transparent));
          --accent-soft: color-mix(in srgb, var(--color-accent-amber, var(--color-fd-foreground)) 32%, transparent);
          --mono: var(--font-mono), "SFMono-Regular", ui-monospace, Menlo, monospace;
          --sans: var(--font-sans), ui-sans-serif, system-ui, sans-serif;
        }

        .grid-fill {
          fill: url(#topic-flow-grid);
          opacity: 0.35;
        }

        .grid-line,
        .construction line,
        .guide-line {
          fill: none;
          stroke: var(--line-faint);
          stroke-width: 1;
        }

        .frame {
          fill: none;
          stroke: var(--line);
          stroke-width: 1.6;
        }

        .frame-bus {
          stroke: var(--line-strong);
        }

        .small-label,
        .topic-kicker,
        .dimension-text text,
        .callout-text,
        .tick-label,
        .peer-annotation {
          fill: var(--muted);
          font-family: var(--mono);
          font-size: 13px;
          letter-spacing: 0.08em;
        }

        .small-label,
        .topic-kicker {
          font-size: 12px;
          letter-spacing: 0.22em;
        }

        .peer-name,
        .topic-name {
          fill: var(--text);
          font-family: var(--sans);
          font-size: 25px;
          font-weight: 650;
        }

        .topic-caption,
        .peer-value {
          fill: var(--text);
          font-family: var(--mono);
          font-size: 15px;
        }

        .peer-value-active,
        .tick-label-active,
        .callout-text {
          fill: var(--accent);
        }

        .main-flow,
        .main-flow-soft,
        .sequence-line,
        .sequence-tick,
        .sequence-tick-active {
          fill: none;
          stroke-linecap: round;
          stroke-linejoin: round;
        }

        .main-flow {
          stroke: var(--line-strong);
          stroke-width: 2.6;
        }

        .main-flow-soft {
          stroke: var(--line-faint);
          stroke-width: 10;
        }

        .sequence-line {
          stroke: var(--line);
          stroke-width: 1.8;
        }

        .sequence-tick {
          stroke: var(--line);
          stroke-width: 1.4;
        }

        .sequence-tick-active {
          stroke: var(--accent);
          stroke-width: 1.8;
        }

        .arrow-head {
          fill: none;
          stroke: var(--line-strong);
          stroke-width: 1.6;
        }

        .port,
        .seq-core {
          fill: var(--text);
        }

        .port-active {
          animation: port-pulse 3.8s ease-in-out infinite;
          transform-origin: 704px 214px;
        }

        .seq-node {
          fill: none;
          stroke: var(--line);
          stroke-width: 1.6;
          transform-origin: 546px 214px;
          animation: seq-pulse 3.8s ease-in-out infinite;
        }

        .token {
          fill: var(--accent);
          stroke: var(--accent-soft);
          stroke-width: 1.2;
          filter: drop-shadow(0 0 8px rgba(255, 255, 255, 0.18));
        }

        .message-token {
          animation: token-traverse 3.8s cubic-bezier(0.2, 0.7, 0.2, 1) infinite;
          transform-box: fill-box;
          transform-origin: center;
        }

        .arrival-ring {
          fill: none;
          stroke: var(--accent);
          stroke-width: 1.6;
          opacity: 0;
          transform-origin: 704px 214px;
          animation: arrival-ring 3.8s ease-in-out infinite;
        }

        @keyframes token-traverse {
          0%,
          12% {
            opacity: 0;
            transform: translateX(0) scale(0.92);
          }
          18% {
            opacity: 1;
          }
          38% {
            opacity: 1;
            transform: translateX(286px) scale(1);
          }
          46%,
          56% {
            opacity: 1;
            transform: translateX(290px) scale(1);
          }
          78%,
          88% {
            opacity: 1;
            transform: translateX(448px) scale(1);
          }
          94%,
          100% {
            opacity: 0;
            transform: translateX(448px) scale(0.96);
          }
        }

        @keyframes seq-pulse {
          0%,
          40%,
          100% {
            transform: scale(1);
            stroke: var(--line);
          }
          48% {
            transform: scale(1.12);
            stroke: var(--accent);
          }
          56% {
            transform: scale(1);
            stroke: var(--line);
          }
        }

        @keyframes port-pulse {
          0%,
          78%,
          100% {
            transform: scale(1);
          }
          86% {
            transform: scale(1.2);
          }
          94% {
            transform: scale(1);
          }
        }

        @keyframes arrival-ring {
          0%,
          78%,
          100% {
            transform: scale(0.7);
            opacity: 0;
          }
          86% {
            transform: scale(1);
            opacity: 0.9;
          }
          94% {
            transform: scale(1.5);
            opacity: 0;
          }
        }

        @media (prefers-reduced-motion: reduce) {
          .message-token,
          .seq-node,
          .arrival-ring,
          .port-active {
            animation: none;
          }

          .message-token {
            opacity: 1;
            transform: translateX(448px);
          }
        }
      `}</style>
    </figure>
  );
}
