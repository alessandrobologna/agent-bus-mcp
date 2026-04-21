"use client";

import { useEffect, useRef, useState } from "react";
import { Check, Copy } from "lucide-react";
import { agentBusPackage, agentBusVersion } from "@/lib/shared";
import { SectionKicker } from "@/components/section-kicker";

type Tab = {
  id: string;
  label: string;
  language: string;
  filename: string;
  body: string;
};

function buildTabs(version: string, pkg: string): Tab[] {
  return [
    {
      id: "claude-code",
      label: "Claude Code",
      language: "json",
      filename: ".mcp.json",
      body: `{
  "mcpServers": {
    "agent-bus": {
      "command": "uvx",
      "args": ["--from", "${pkg}==${version}", "agent-bus"],
      "env": {}
    }
  }
}`,
    },
    {
      id: "codex",
      label: "Codex",
      language: "toml",
      filename: "~/.codex/config.toml",
      body: `[mcp_servers.agent-bus]
command = "uvx"
args = ["--from", "${pkg}==${version}", "agent-bus"]`,
    },
    {
      id: "opencode",
      label: "OpenCode",
      language: "json",
      filename: "opencode.json",
      body: `{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "agent-bus": {
      "type": "local",
      "command": ["uvx", "--from", "${pkg}==${version}", "agent-bus"],
      "enabled": true
    }
  }
}`,
    },
  ];
}

const TERMINAL_BODY = `export AGENT_BUS_VERSION="${agentBusVersion}"
uvx --from "${agentBusPackage}==$AGENT_BUS_VERSION" agent-bus --help`;

export function InstallSnippet() {
  const tabs = buildTabs(agentBusVersion, agentBusPackage);
  const [activeId, setActiveId] = useState(tabs[0].id);
  const active = tabs.find((tab) => tab.id === activeId) ?? tabs[0];

  return (
    <section className="border-b border-fd-border bg-fd-background">
      <div className="mx-auto w-full max-w-7xl px-6 py-16 md:px-10 md:py-20 lg:px-12">
        <div className="mb-10 max-w-2xl">
          <SectionKicker>Install</SectionKicker>
          <h2 className="mt-3 text-3xl font-semibold tracking-[-0.04em] text-fd-foreground md:text-4xl">
            Two commands. One config block.
          </h2>
          <p className="mt-4 text-lg leading-8 text-fd-muted-foreground">
            Run the published package with <code className="font-mono text-[0.95em]">uvx</code>,
            then point your MCP client at it. Pin the version so updates are explicit.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          <CodeCard
            kicker="1. Verify the package"
            filename="terminal"
            language="bash"
            body={TERMINAL_BODY}
            copyValue={TERMINAL_BODY}
            copyLabel="commands"
          />

          <div className="overflow-hidden rounded-2xl border border-fd-border bg-fd-card">
            <div className="flex items-center gap-2 border-b border-fd-border bg-fd-muted/40 px-4 py-3">
              <p className="mr-2 text-xs font-medium uppercase tracking-[0.18em] text-fd-muted-foreground">
                2. Add to your client
              </p>
              <div className="flex flex-wrap gap-1">
                {tabs.map((tab) => {
                  const isActive = tab.id === activeId;
                  return (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => setActiveId(tab.id)}
                      className={
                        isActive
                          ? "relative rounded-md bg-fd-background px-2.5 py-1 font-mono text-xs font-medium text-fd-foreground shadow-sm ring-1 ring-[color:var(--color-accent-amber-soft)]"
                          : "rounded-md px-2.5 py-1 font-mono text-xs font-medium text-fd-muted-foreground transition hover:text-fd-foreground"
                      }
                    >
                      {tab.label}
                    </button>
                  );
                })}
              </div>
            </div>
            <CodeBody
              filename={active.filename}
              language={active.language}
              body={active.body}
              copyValue={active.body}
              copyLabel="config"
            />
          </div>
        </div>
      </div>
    </section>
  );
}

function CodeCard({
  kicker,
  filename,
  language,
  body,
  copyValue,
  copyLabel,
}: {
  kicker: string;
  filename: string;
  language: string;
  body: string;
  copyValue: string;
  copyLabel: string;
}) {
  return (
    <div className="overflow-hidden rounded-2xl border border-fd-border bg-fd-card">
      <div className="flex items-center justify-between gap-3 border-b border-fd-border bg-fd-muted/40 px-4 py-3">
        <p className="text-xs font-medium uppercase tracking-[0.18em] text-fd-muted-foreground">
          {kicker}
        </p>
      </div>
      <CodeBody
        filename={filename}
        language={language}
        body={body}
        copyValue={copyValue}
        copyLabel={copyLabel}
      />
    </div>
  );
}

function CodeBody({
  filename,
  language,
  body,
  copyValue,
  copyLabel,
}: {
  filename: string;
  language: string;
  body: string;
  copyValue: string;
  copyLabel: string;
}) {
  return (
    <div className="relative">
      <div className="flex items-center justify-between border-b border-fd-border/60 px-4 py-2 text-xs text-fd-muted-foreground">
        <span className="font-mono">{filename}</span>
        <CopyButton value={copyValue} label={copyLabel} />
      </div>
      <pre className="overflow-x-auto px-4 py-4 font-mono text-[12.5px] leading-6 text-fd-foreground/90">
        <code data-language={language}>{body}</code>
      </pre>
    </div>
  );
}

function CopyButton({ value, label }: { value: string; label: string }) {
  const [copied, setCopied] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  const markCopied = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setCopied(true);
    timeoutRef.current = setTimeout(() => {
      setCopied(false);
      timeoutRef.current = null;
    }, 1500);
  };

  const fallbackCopy = () => {
    if (typeof document === "undefined") return false;

    const textarea = document.createElement("textarea");
    textarea.value = value;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.top = "-9999px";
    textarea.style.left = "-9999px";
    textarea.style.opacity = "0";

    const selection = document.getSelection();
    const previousRange =
      selection && selection.rangeCount > 0 ? selection.getRangeAt(0).cloneRange() : null;
    const activeElement =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;

    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    textarea.setSelectionRange(0, textarea.value.length);

    let didCopy = false;
    try {
      didCopy = document.execCommand("copy");
    } catch {
      didCopy = false;
    }

    textarea.remove();

    if (selection && previousRange) {
      selection.removeAllRanges();
      selection.addRange(previousRange);
    }
    activeElement?.focus();

    return didCopy;
  };

  const handleClick = async () => {
    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(value);
        markCopied();
        return;
      } catch {
        // Fall through to the legacy path for webviews or browsers that deny clipboard access.
      }
    }

    if (fallbackCopy()) {
      markCopied();
    }
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-md border border-fd-border bg-fd-background px-2 py-1 text-xs text-fd-muted-foreground transition hover:text-fd-foreground"
      aria-label={`Copy ${label} to clipboard`}
      title={`Copy ${label}`}
    >
      {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}
