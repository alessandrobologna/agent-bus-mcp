import { startTransition, useEffect, useEffectEvent, useMemo, useRef, useState, type CSSProperties } from "react"
import ReactMarkdown, { type Components } from "react-markdown"
import { useLocation, useNavigate } from "react-router-dom"
import remarkGfm from "remark-gfm"
import { toast } from "sonner"
import {
  AlertCircleIcon,
  ArrowDownToLineIcon,
  ArrowDownWideNarrowIcon,
  ArrowUpDownIcon,
  CheckIcon,
  FolderSearch2Icon,
  ListFilterIcon,
  MessageSquareMoreIcon,
  Trash2Icon,
  XIcon,
} from "lucide-react"

import { fetchSearchResults, fetchTopicDetail, fetchTopicMessages, fetchTopics } from "@/lib/api"
import { deleteMessages, deleteTopic, exportTopicUrl } from "@/lib/api"
import { formatAbsoluteTime, formatRelativeTime, initialsFor } from "@/lib/format"
import type {
  CursorPresence,
  SearchMode,
  SearchResult,
  TopicDetailResponse,
  TopicMessage,
  TopicSort,
  TopicStatusFilter,
  TopicStreamUpdate,
  TopicSummary,
  WorkbenchState,
} from "@/lib/types"
import {
  loadWorkbenchState,
  saveWorkbenchState,
} from "@/lib/workbench-state"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInput,
  SidebarProvider,
  SidebarSeparator,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

type SearchState = {
  query: string
  mode: SearchMode
  results: SearchResult[]
  warnings: string[]
  loading: boolean
  error: string | null
}

type FindState = {
  open: boolean
  query: string
  activeIndex: number
}

type TextRange = {
  start: number
  end: number
}

type MessageHighlightSpec = {
  terms: string[]
}

type ThreadMapMeasuredMarker = {
  messageId: string
  seq: number
  sender: string
  top: number
  height: number
}

type ThreadMapMarker = ThreadMapMeasuredMarker & {
  focused: boolean
  localMatched: boolean
  localActive: boolean
}

type ThreadMapViewport = {
  top: number
  height: number
}

type ThreadMapSenderTone = {
  key: string
  borderColor: string
  backgroundColor: string
  lineColor: string
}

const SEARCH_HIGHLIGHT_START = "\uE000"
const SEARCH_HIGHLIGHT_END = "\uE001"
const THREAD_MAP_DESKTOP_BREAKPOINT = 1280
const THREAD_MAP_AUTO_HIDE_MS = 1800
const EMPTY_TOPIC_MESSAGES: TopicMessage[] = []
const THREAD_MAP_LINE_PATTERNS = [
  [92, 76, 84],
  [84, 93, 68],
  [89, 72, 80],
] as const

const SIDEBAR_LAYOUT_STYLE = {
  "--sidebar-width": "22rem",
  "--sidebar-width-mobile": "22rem",
} as CSSProperties

const TOPIC_SORT_OPTIONS: Array<{ value: TopicSort; label: string }> = [
  { value: "last_updated_desc", label: "Last updated" },
  { value: "created_desc", label: "Newest first" },
  { value: "created_asc", label: "Oldest first" },
]

const TOPIC_STATUS_OPTIONS: Array<{ value: TopicStatusFilter; label: string }> = [
  { value: "all", label: "All topics" },
  { value: "open", label: "Open" },
  { value: "closed", label: "Closed" },
]

const TOPIC_SORT_LABELS = Object.fromEntries(
  TOPIC_SORT_OPTIONS.map((option) => [option.value, option.label])
) as Record<TopicSort, string>

const TOPIC_STATUS_LABELS = Object.fromEntries(
  TOPIC_STATUS_OPTIONS.map((option) => [option.value, option.label])
) as Record<TopicStatusFilter, string>

function mergeMessages(existing: TopicMessage[], incoming: TopicMessage[]): TopicMessage[] {
  const merged = new Map(existing.map((message) => [message.message_id, message]))
  for (const message of incoming) {
    merged.set(message.message_id, message)
  }
  return Array.from(merged.values()).sort((left, right) => left.seq - right.seq)
}

function statusVariant(status: TopicSummary["status"]): "default" | "secondary" | "outline" {
  return status === "open" ? "default" : "secondary"
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function hashString(value: string): number {
  let hash = 0
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0
  }
  return hash
}

function getThreadMapSenderTone(sender: string): ThreadMapSenderTone {
  const seed = hashString(sender)
  const hue = 198 + (seed % 48)
  const saturation = 18 + (Math.floor(seed / 48) % 4) * 10
  return {
    key: `${hue}-${saturation}`,
    borderColor: `hsla(${hue}, ${Math.min(saturation + 8, 58)}%, 76%, 0.36)`,
    backgroundColor: `hsla(${hue}, ${saturation}%, 62%, 0.14)`,
    lineColor: `hsla(${hue}, ${Math.min(saturation + 14, 64)}%, 92%, 0.62)`,
  }
}

function snippetLabel(result: SearchResult): string {
  if (
    result.semantic_score !== undefined &&
    result.semantic_score !== null &&
    result.fts_rank !== undefined &&
    result.fts_rank !== null
  ) {
    return `hybrid ${result.semantic_score.toFixed(4)}`
  }
  if (result.semantic_score !== undefined && result.semantic_score !== null) {
    return `semantic ${result.semantic_score.toFixed(4)}`
  }
  if (
    (result.rank !== undefined && result.rank !== null) ||
    (result.fts_rank !== undefined && result.fts_rank !== null)
  ) {
    return "fts"
  }
  return result.message_type
}

function renderSnippet(snippet: string) {
  const parts = splitSearchSnippet(snippet)
  return parts.map((part, index) => {
    if (part.highlight) {
      return (
        <mark
          key={`snippet-${index}`}
          className="rounded-sm bg-primary/15 px-0.5 text-zinc-200"
        >
          {part.text}
        </mark>
      )
    }
    return <span key={`snippet-${index}`}>{part.text}</span>
  })
}

function uniqueStrings(values: string[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const value of values) {
    const trimmed = value.trim()
    if (!trimmed) {
      continue
    }
    const key = trimmed.toLocaleLowerCase()
    if (seen.has(key)) {
      continue
    }
    seen.add(key)
    out.push(trimmed)
  }
  return out
}

function normalizeSearchText(value: string): string {
  return value.toLocaleLowerCase().normalize("NFKD").replace(/\p{M}+/gu, "")
}

function splitSearchSnippet(snippet: string): Array<{ text: string; highlight: boolean }> {
  const parts: Array<{ text: string; highlight: boolean }> = []
  let cursor = 0

  while (cursor < snippet.length) {
    const start = snippet.indexOf(SEARCH_HIGHLIGHT_START, cursor)
    if (start === -1) {
      const text = snippet.slice(cursor)
      if (text) {
        parts.push({ text, highlight: false })
      }
      break
    }

    if (start > cursor) {
      parts.push({ text: snippet.slice(cursor, start), highlight: false })
    }

    const highlightStart = start + SEARCH_HIGHLIGHT_START.length
    const end = snippet.indexOf(SEARCH_HIGHLIGHT_END, highlightStart)
    if (end === -1) {
      parts.push({ text: snippet.slice(start), highlight: false })
      break
    }

    const text = snippet.slice(highlightStart, end)
    if (text) {
      parts.push({ text, highlight: true })
    }
    cursor = end + SEARCH_HIGHLIGHT_END.length
  }

  return parts
}

function extractSnippetTerms(snippet: string): string[] {
  return uniqueStrings(
    splitSearchSnippet(snippet)
      .filter((part) => part.highlight)
      .map((part) => part.text)
  )
}

function tokenizeSearchQuery(query: string): string[] {
  return uniqueStrings(
    query
      .split(/\s+/u)
      .map((part) => part.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, ""))
      .filter((part) => part.length >= 2)
  )
}

function mergeRanges(ranges: TextRange[]): TextRange[] {
  if (ranges.length === 0) {
    return []
  }
  const merged: TextRange[] = []
  const sorted = [...ranges].sort((left, right) => left.start - right.start || left.end - right.end)
  for (const range of sorted) {
    if (range.end <= range.start) {
      continue
    }
    const last = merged.at(-1)
    if (!last || range.start > last.end) {
      merged.push({ ...range })
      continue
    }
    last.end = Math.max(last.end, range.end)
  }
  return merged
}

function findTermRanges(text: string, terms: string[]): TextRange[] {
  if (!text || terms.length === 0) {
    return []
  }

  const haystack = normalizeSearchText(text)
  const normalizedIndexToOriginalRange: TextRange[] = []
  let originalIndex = 0
  for (const char of text) {
    const start = originalIndex
    const end = start + char.length
    const normalizedChar = normalizeSearchText(char)
    for (let normalizedIndex = 0; normalizedIndex < normalizedChar.length; normalizedIndex += 1) {
      normalizedIndexToOriginalRange.push({ start, end })
    }
    originalIndex = end
  }

  const ranges: TextRange[] = []

  for (const term of uniqueStrings(terms)) {
    const needle = normalizeSearchText(term)
    if (!needle) {
      continue
    }
    let searchStart = 0
    while (searchStart < haystack.length) {
      const normalizedMatchIndex = haystack.indexOf(needle, searchStart)
      if (normalizedMatchIndex === -1) {
        break
      }
      const normalizedMatchEnd = normalizedMatchIndex + needle.length - 1
      const start = normalizedIndexToOriginalRange[normalizedMatchIndex]?.start
      const end = normalizedIndexToOriginalRange[normalizedMatchEnd]?.end
      if (start !== undefined && end !== undefined && end > start) {
        ranges.push({ start, end })
      }
      searchStart = normalizedMatchIndex + Math.max(needle.length, 1)
    }
  }

  return mergeRanges(ranges)
}

function clearMessageHighlights(root: HTMLElement): void {
  root.querySelectorAll("mark[data-ab-highlight='true']").forEach((mark) => {
    const parent = mark.parentNode
    if (!parent) {
      return
    }
    while (mark.firstChild) {
      parent.insertBefore(mark.firstChild, mark)
    }
    parent.removeChild(mark)
    parent.normalize()
  })
}

function applyMessageHighlights(root: HTMLElement, spec: MessageHighlightSpec | null): void {
  clearMessageHighlights(root)

  if (!spec) {
    return
  }

  const nodeFilter = root.ownerDocument.defaultView?.NodeFilter ?? NodeFilter
  const walker = root.ownerDocument.createTreeWalker(root, nodeFilter.SHOW_TEXT)
  const textNodes: Array<{ node: Text; start: number; end: number }> = []

  let currentNode = walker.nextNode()
  let cursor = 0
  while (currentNode) {
    const textNode = currentNode as Text
    const value = textNode.data
    if (value) {
      textNodes.push({
        node: textNode,
        start: cursor,
        end: cursor + value.length,
      })
      cursor += value.length
    }
    currentNode = walker.nextNode()
  }

  if (textNodes.length === 0) {
    return
  }

  const fullText = textNodes.map((entry) => entry.node.data).join("")
  const ranges = findTermRanges(fullText, spec.terms)

  if (ranges.length === 0) {
    return
  }

  for (const entry of [...textNodes].reverse()) {
    const localRanges = ranges
      .filter((range) => range.start < entry.end && range.end > entry.start)
      .map((range) => ({
        start: Math.max(range.start, entry.start) - entry.start,
        end: Math.min(range.end, entry.end) - entry.start,
      }))

    if (localRanges.length === 0) {
      continue
    }

    const fragment = root.ownerDocument.createDocumentFragment()
    let localCursor = 0

    for (const range of localRanges) {
      if (range.start > localCursor) {
        fragment.append(root.ownerDocument.createTextNode(entry.node.data.slice(localCursor, range.start)))
      }
      const mark = root.ownerDocument.createElement("mark")
      mark.dataset.abHighlight = "true"
      mark.className =
        "rounded-sm bg-amber-300/45 px-0.5 text-amber-50 ring-1 ring-inset ring-amber-200/45"
      mark.textContent = entry.node.data.slice(range.start, range.end)
      fragment.append(mark)
      localCursor = range.end
    }

    if (localCursor < entry.node.data.length) {
      fragment.append(root.ownerDocument.createTextNode(entry.node.data.slice(localCursor)))
    }

    entry.node.parentNode?.replaceChild(fragment, entry.node)
  }
}

function fallbackTopic(topicId: string): TopicSummary {
  const timestamp = Date.now() / 1000
  return {
    topic_id: topicId,
    name: `Topic ${topicId.slice(0, 8)}`,
    status: "open",
    created_at: timestamp,
    closed_at: null,
    close_reason: null,
    metadata: null,
    message_count: 0,
    last_seq: 0,
    last_message_at: null,
    last_updated_at: timestamp,
  }
}

function messageContainsText(message: TopicMessage, query: string): boolean {
  const haystack = [message.sender, message.message_type, message.content_markdown].join("\n").toLowerCase()
  return haystack.includes(query.toLowerCase())
}

const markdownComponents: Components = {
  p: ({ children }) => (
    <p className="mb-4 whitespace-pre-wrap text-[15px] leading-7 text-zinc-100 [overflow-wrap:anywhere] last:mb-0">
      {children}
    </p>
  ),
  ul: ({ children }) => (
    <ul className="mb-4 list-disc pl-5 text-[15px] leading-7 text-zinc-100 marker:text-zinc-500 [&>li]:mb-1 last:mb-0">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="mb-4 list-decimal pl-5 text-[15px] leading-7 text-zinc-100 marker:text-zinc-500 [&>li]:mb-1 last:mb-0">
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="whitespace-pre-wrap [overflow-wrap:anywhere]">{children}</li>,
  blockquote: ({ children }) => (
    <blockquote className="mb-4 border-l-2 border-zinc-700 pl-4 text-zinc-300 italic last:mb-0">
      {children}
    </blockquote>
  ),
  strong: ({ children }) => <strong className="font-semibold text-zinc-50">{children}</strong>,
  em: ({ children }) => <em className="italic text-zinc-200">{children}</em>,
  code: ({ children, className, ...props }) => {
    const inline = !className
    if (inline) {
      return (
        <code
          className="rounded bg-zinc-800/90 px-1.5 py-0.5 font-mono text-[0.92em] text-sky-100 [overflow-wrap:anywhere]"
          {...props}
        >
          {children}
        </code>
      )
    }
    return (
      <code className="font-mono text-[13px] leading-6 text-zinc-100" {...props}>
        {children}
      </code>
    )
  },
  pre: ({ children }) => (
    <pre className="mb-4 overflow-x-auto rounded-md border border-zinc-800 bg-[#17191f] p-3 font-mono text-[13px] leading-6 text-zinc-100 last:mb-0">
      {children}
    </pre>
  ),
  a: ({ children, ...props }) => (
    <a className="text-sky-300 underline underline-offset-4 hover:text-sky-200" {...props}>
      {children}
    </a>
  ),
}

function MessageMarkdown(props: { content: string; highlight: MessageHighlightSpec | null }) {
  const { content, highlight } = props
  const containerRef = useRef<HTMLDivElement | null>(null)
  const highlightTermsKey = highlight?.terms.join("\u0000") ?? ""

  useEffect(() => {
    const container = containerRef.current
    if (!container) {
      return
    }
    applyMessageHighlights(container, highlight)
    return () => {
      clearMessageHighlights(container)
    }
  }, [content, highlightTermsKey])

  return (
    <div ref={containerRef} className="min-w-0 text-sm">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {content}
      </ReactMarkdown>
    </div>
  )
}

function MessageCard(props: {
  message: TopicMessage
  highlight: MessageHighlightSpec | null
  focused: boolean
  localMatched: boolean
  localActive: boolean
  selectable: boolean
  selected: boolean
  onSelectedChange: (next: boolean) => void
}) {
  const { message, highlight, focused, localMatched, localActive, selectable, selected, onSelectedChange } = props

  return (
    <article
      id={`msg-${message.message_id}`}
      data-ab-message-id={message.message_id}
      className={[
        "w-full min-w-0 border px-5 py-5 transition-colors",
        localActive
          ? "border-sky-500 bg-sky-500/10"
          : localMatched
            ? "border-cyan-500/50 bg-cyan-500/8"
            : focused
              ? "border-primary bg-accent/60"
              : "border-border bg-card",
      ].join(" ")}
    >
      <div className="flex items-start gap-3">
        {selectable ? (
          <Checkbox
            checked={selected}
            onCheckedChange={(checked) => onSelectedChange(checked === true)}
            aria-label={`Select message ${message.seq}`}
          />
        ) : null}
        <div className="flex size-8 shrink-0 items-center justify-center rounded-sm bg-muted text-xs font-semibold text-muted-foreground">
          {initialsFor(message.sender)}
        </div>
        <div className="flex min-w-0 flex-1 flex-col gap-4">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[15px] font-semibold text-foreground">{message.sender}</span>
            <span className="text-xs text-muted-foreground">#{message.seq}</span>
            <span className="text-xs text-muted-foreground">
              {formatRelativeTime(message.created_at)}
            </span>
            {message.message_type !== "message" ? (
              <Badge variant="outline">{message.message_type}</Badge>
            ) : null}
            {message.reply_to_sender ? (
              <Badge variant="secondary">reply to {message.reply_to_sender}</Badge>
            ) : null}
          </div>
          <MessageMarkdown content={message.content_markdown} highlight={highlight} />
          <div className="text-xs text-muted-foreground">{formatAbsoluteTime(message.created_at)}</div>
        </div>
      </div>
    </article>
  )
}

function TopicInspectorPanel(props: { topicDetail: TopicDetailResponse }) {
  const { topicDetail } = props

  return (
    <div className="flex flex-1 flex-col gap-4 p-4 text-sm">
      <div className="grid grid-cols-2 gap-2">
        <div className="border border-border bg-background px-3 py-3">
          <div className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Status</div>
          <div className="mt-2 font-medium">{topicDetail.topic.status}</div>
        </div>
        <div className="border border-border bg-background px-3 py-3">
          <div className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Messages</div>
          <div className="mt-2 font-medium">{topicDetail.message_count}</div>
        </div>
        <div className="border border-border bg-background px-3 py-3">
          <div className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Last seq</div>
          <div className="mt-2 font-medium">{topicDetail.last_seq ?? 0}</div>
        </div>
        <div className="border border-border bg-background px-3 py-3">
          <div className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Updated</div>
          <div className="mt-2 font-medium">{formatRelativeTime(topicDetail.topic.last_updated_at)}</div>
        </div>
      </div>
      <div className="border border-border bg-background px-3 py-3">
        <div className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Created</div>
        <div className="mt-2 font-medium">{formatAbsoluteTime(topicDetail.topic.created_at)}</div>
      </div>
      <Separator />
      <div className="flex flex-col gap-3">
        <div className="font-medium">Presence</div>
        {topicDetail.presence.length === 0 ? (
          <p className="text-muted-foreground">No peers have touched this topic recently.</p>
        ) : (
          topicDetail.presence.map((presence: CursorPresence) => (
            <div key={presence.agent_name} className="border border-border bg-background px-3 py-3">
              <div className="font-medium">{presence.agent_name}</div>
              <div className="text-xs text-muted-foreground">
                last seq {presence.last_seq} · {formatRelativeTime(presence.updated_at)}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

function ThreadMap(props: {
  markers: ThreadMapMarker[]
  viewport: ThreadMapViewport | null
  visible: boolean
  onJumpToMessage: (messageId: string) => void
  onViewportDrag: (top: number) => void
}) {
  const { markers, viewport, visible, onJumpToMessage, onViewportDrag } = props
  const mapRef = useRef<HTMLDivElement | null>(null)
  const dragStateRef = useRef<{ grabOffset: number; viewportHeight: number } | null>(null)
  const senderTones = useMemo(() => {
    const tones = new Map<string, ThreadMapSenderTone>()
    for (const marker of markers) {
      if (!tones.has(marker.sender)) {
        tones.set(marker.sender, getThreadMapSenderTone(marker.sender))
      }
    }
    return tones
  }, [markers])

  const moveViewportCursor = (clientY: number) => {
    const map = mapRef.current
    const dragState = dragStateRef.current
    if (!map || !dragState) {
      return
    }

    const rect = map.getBoundingClientRect()
    if (rect.height <= 0) {
      return
    }

    const pointerTop = (clientY - rect.top) / rect.height
    const maxTop = Math.max(1 - dragState.viewportHeight, 0)
    onViewportDrag(clamp(pointerTop - dragState.grabOffset, 0, maxTop))
  }

  return (
    <div
      data-ab-thread-map="true"
      data-visible={visible ? "true" : "false"}
      aria-hidden={visible ? undefined : true}
      className={`absolute inset-y-3 right-3 z-20 w-[7rem] overflow-hidden border border-border/60 bg-[#17191e]/84 shadow-[0_8px_20px_rgba(0,0,0,0.24)] backdrop-blur-[2px] transition-all duration-200 ${
        visible ? "opacity-100" : "pointer-events-none translate-x-2 opacity-0"
      }`}
    >
      <div ref={mapRef} data-ab-thread-map-frame="true" className="relative h-full w-full">
        <div className="absolute inset-1 bg-muted/8" />
        {viewport ? (
          <div
            data-ab-thread-map-viewport="true"
            className="absolute inset-x-1 z-10 cursor-grab border-y border-primary/35 bg-primary/10 active:cursor-grabbing"
            style={{
              top: `${clamp(viewport.top * 100, 0, 100)}%`,
              height: `${clamp(viewport.height * 100, 8, 100)}%`,
            }}
            onClick={(event) => event.stopPropagation()}
            onPointerDown={(event) => {
              if (event.button !== 0) {
                return
              }

              const map = mapRef.current
              if (!map) {
                return
              }

              const rect = map.getBoundingClientRect()
              if (rect.height <= 0) {
                return
              }

              const viewportHeight = clamp(viewport.height, 0, 1)
              dragStateRef.current = {
                grabOffset: clamp((event.clientY - rect.top) / rect.height - viewport.top, 0, viewportHeight),
                viewportHeight,
              }
              event.currentTarget.setPointerCapture(event.pointerId)
              event.preventDefault()
              event.stopPropagation()
              moveViewportCursor(event.clientY)
            }}
            onPointerMove={(event) => {
              if (!dragStateRef.current) {
                return
              }

              event.preventDefault()
              event.stopPropagation()
              moveViewportCursor(event.clientY)
            }}
            onPointerUp={(event) => {
              if (!dragStateRef.current) {
                return
              }

              dragStateRef.current = null
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId)
              }
              event.preventDefault()
              event.stopPropagation()
            }}
            onPointerCancel={(event) => {
              dragStateRef.current = null
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId)
              }
            }}
          />
        ) : null}
        {markers.map((marker, index) => {
          const center = clamp((marker.top + marker.height / 2) * 100, 0, 100)
          const previousMarker = markers[index - 1]
          const nextMarker = markers[index + 1]
          const previousCenter = previousMarker
            ? clamp((previousMarker.top + previousMarker.height / 2) * 100, 0, 100)
            : 0
          const nextCenter = nextMarker
            ? clamp((nextMarker.top + nextMarker.height / 2) * 100, 0, 100)
            : 100
          const slotTop = index === 0 ? 0 : (previousCenter + center) / 2
          const slotBottom = index === markers.length - 1 ? 100 : (center + nextCenter) / 2
          const slotHeight = Math.max(slotBottom - slotTop, 0.25)
          const glyphTop = clamp(((center - slotTop) / slotHeight) * 100, 0, 100)
          const visibleHeight = marker.localActive ? 11 : marker.focused ? 10 : marker.localMatched ? 9 : 8
          const senderTone = senderTones.get(marker.sender) ?? getThreadMapSenderTone(marker.sender)
          const tone = marker.localActive
            ? {
                accentColor: "rgba(125, 211, 252, 0.95)",
                backgroundColor: "rgba(56, 189, 248, 0.12)",
                lineColor: "rgba(224, 242, 254, 0.84)",
              }
            : marker.localMatched
              ? {
                  accentColor: "rgba(103, 232, 249, 0.82)",
                  backgroundColor: "rgba(34, 211, 238, 0.09)",
                  lineColor: "rgba(236, 254, 255, 0.76)",
                }
              : marker.focused
                ? {
                    accentColor: "rgba(113, 113, 122, 0.82)",
                    backgroundColor: "rgba(244, 244, 245, 0.08)",
                    lineColor: "rgba(250, 250, 250, 0.74)",
                  }
                : {
                    accentColor: senderTone.borderColor,
                    backgroundColor: "transparent",
                    lineColor: senderTone.lineColor,
                  }
          const linePattern = THREAD_MAP_LINE_PATTERNS[marker.seq % THREAD_MAP_LINE_PATTERNS.length]!
          const stateLabel = marker.localActive
            ? "active find match"
            : marker.localMatched
              ? "find match"
              : marker.focused
                ? "focused message"
                : "message"

          return (
            <button
              key={marker.messageId}
              type="button"
              tabIndex={-1}
              data-ab-thread-map-marker={marker.messageId}
              data-focused={marker.focused ? "true" : "false"}
              data-local-matched={marker.localMatched ? "true" : "false"}
              data-local-active={marker.localActive ? "true" : "false"}
              data-sender-tone={senderTone.key}
              aria-current={marker.localActive ? "true" : undefined}
              aria-label={`Jump to message #${marker.seq} by ${marker.sender} (${stateLabel})`}
              className="absolute left-0 right-0 transition-colors focus-visible:ring-2 focus-visible:ring-primary/70 focus-visible:outline-none"
              style={{
                top: `${slotTop}%`,
                height: `${slotHeight}%`,
              }}
              onClick={() => onJumpToMessage(marker.messageId)}
            >
              <span
                className="absolute left-2 right-2 block px-2"
                style={{
                  top: `${glyphTop}%`,
                  height: `${visibleHeight}px`,
                  transform: "translateY(-50%)",
                  backgroundColor: tone.backgroundColor,
                }}
              >
                {(marker.localActive || marker.localMatched || marker.focused) ? (
                  <span
                    className="absolute inset-y-0 left-0 w-[2px]"
                    style={{ backgroundColor: tone.accentColor }}
                  />
                ) : null}
                <span
                  className="flex h-full w-full flex-col justify-center gap-[2px]"
                  style={{
                    paddingLeft: marker.localActive || marker.localMatched || marker.focused ? "6px" : "0px",
                  }}
                >
                  {linePattern.map((width, index) => (
                    <span
                      key={index}
                      className="block h-px"
                      style={{
                        width: `${width}%`,
                        backgroundColor: tone.lineColor,
                      }}
                    />
                  ))}
                </span>
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function TopicView(props: {
  topicDetail: TopicDetailResponse
  activeSearchResult: SearchResult | null
  sidebarSearchQuery: string
  findState: FindState
  findMatches: TopicMessage[]
  selectionMode: boolean
  selectedMessageIds: Set<string>
  findInputRef: React.RefObject<HTMLInputElement | null>
  onCloseFind: () => void
  onFindQueryChange: (next: string) => void
  onFindPrevious: () => void
  onFindNext: () => void
  onToggleSelectionMode: () => void
  onDeleteSelected: () => void
  onMessageSelectionChange: (messageId: string, next: boolean) => void
  onLoadEarlier: () => void
  onDeleteTopic: () => void
}) {
  const {
    topicDetail,
    activeSearchResult,
    sidebarSearchQuery,
    findState,
    findMatches,
    selectionMode,
    selectedMessageIds,
    findInputRef,
    onCloseFind,
    onFindQueryChange,
    onFindPrevious,
    onFindNext,
    onToggleSelectionMode,
    onDeleteSelected,
    onMessageSelectionChange,
    onLoadEarlier,
    onDeleteTopic,
  } = props

  const trimmedFindQuery = findState.query.trim()
  const hasTrimmedFindQuery = Boolean(trimmedFindQuery)
  const activeFindMessageId =
    hasTrimmedFindQuery && findMatches.length > 0
      ? findMatches[Math.min(findState.activeIndex, findMatches.length - 1)]?.message_id ?? null
      : null
  const activeSearchTerms = activeSearchResult ? extractSnippetTerms(activeSearchResult.snippet) : []
  const localMatchedMessageIds = useMemo(
    () => new Set(findMatches.map((message) => message.message_id)),
    [findMatches]
  )
  const threadViewportRef = useRef<HTMLDivElement | null>(null)
  const threadListRef = useRef<HTMLDivElement | null>(null)
  const threadMapHideTimerRef = useRef<number | null>(null)
  const threadMapHideAtRef = useRef(0)
  const threadMapHideTimerDueAtRef = useRef(0)
  const [isDesktopThreadMap, setIsDesktopThreadMap] = useState(false)
  const [threadMapMeasuredMarkers, setThreadMapMeasuredMarkers] = useState<
    ThreadMapMeasuredMarker[]
  >([])
  const [threadMapViewport, setThreadMapViewport] = useState<ThreadMapViewport | null>(null)
  const [threadMapQualifies, setThreadMapQualifies] = useState(false)
  const [threadMapVisible, setThreadMapVisible] = useState(false)
  const threadMapMarkers = useMemo(
    () =>
      threadMapMeasuredMarkers.map((marker) => ({
        ...marker,
        focused: topicDetail.focus_message_id === marker.messageId,
        localMatched: localMatchedMessageIds.has(marker.messageId),
        localActive: activeFindMessageId === marker.messageId,
      })),
    [
      activeFindMessageId,
      localMatchedMessageIds,
      threadMapMeasuredMarkers,
      topicDetail.focus_message_id,
    ]
  )

  useEffect(() => {
    if (typeof window === "undefined") {
      return
    }

    const mediaQuery = window.matchMedia(`(min-width: ${THREAD_MAP_DESKTOP_BREAKPOINT}px)`)
    const updateDesktopState = () => setIsDesktopThreadMap(mediaQuery.matches)

    updateDesktopState()
    mediaQuery.addEventListener("change", updateDesktopState)
    return () => mediaQuery.removeEventListener("change", updateDesktopState)
  }, [])

  const clearThreadMapHideTimer = useEffectEvent(() => {
    if (threadMapHideTimerRef.current !== null) {
      window.clearTimeout(threadMapHideTimerRef.current)
      threadMapHideTimerRef.current = null
    }
    threadMapHideAtRef.current = 0
    threadMapHideTimerDueAtRef.current = 0
  })

  const scheduleThreadMapHide = useEffectEvent((delay = THREAD_MAP_AUTO_HIDE_MS) => {
    if (typeof window === "undefined") {
      return
    }

    const armTimer = (timeout: number) => {
      threadMapHideTimerDueAtRef.current = Date.now() + timeout
      threadMapHideTimerRef.current = window.setTimeout(() => {
        const remaining = threadMapHideAtRef.current - Date.now()
        if (remaining > 0) {
          armTimer(remaining)
          return
        }

        setThreadMapVisible(false)
        threadMapHideTimerRef.current = null
        threadMapHideAtRef.current = 0
        threadMapHideTimerDueAtRef.current = 0
      }, timeout)
    }

    const nextHideAt = Date.now() + delay
    threadMapHideAtRef.current = nextHideAt

    if (threadMapHideTimerRef.current !== null) {
      if (nextHideAt < threadMapHideTimerDueAtRef.current - 16) {
        window.clearTimeout(threadMapHideTimerRef.current)
        armTimer(delay)
      }
      return
    }

    armTimer(delay)
  })

  const revealThreadMap = useEffectEvent((delay = THREAD_MAP_AUTO_HIDE_MS) => {
    if (typeof window === "undefined" || !isDesktopThreadMap || !threadMapQualifies) {
      return
    }

    setThreadMapVisible(true)
    scheduleThreadMapHide(delay)
  })

  useEffect(() => {
    if (!isDesktopThreadMap || !threadMapQualifies) {
      setThreadMapVisible(false)
      clearThreadMapHideTimer()
    }

    return () => clearThreadMapHideTimer()
  }, [clearThreadMapHideTimer, isDesktopThreadMap, threadMapQualifies])

  useEffect(() => {
    if (findState.open || activeFindMessageId || topicDetail.focus_message_id) {
      revealThreadMap()
    }
  }, [activeFindMessageId, findState.open, isDesktopThreadMap, revealThreadMap, threadMapQualifies, topicDetail.focus_message_id])

  useEffect(() => {
    if (!isDesktopThreadMap) {
      return
    }

    const viewport = threadViewportRef.current
    const threadList = threadListRef.current
    if (!viewport || !threadList) {
      return
    }

    const messageMap = new Map(topicDetail.messages.map((message) => [message.message_id, message]))

    const updateViewport = () => {
      const nextQualifies = viewport.scrollHeight > viewport.clientHeight + 1
      setThreadMapQualifies(nextQualifies)

      if (!nextQualifies || viewport.scrollHeight <= 0) {
        setThreadMapViewport(null)
        return
      }

      const nextViewport = {
        top: clamp(viewport.scrollTop / viewport.scrollHeight, 0, 1),
        height: clamp(viewport.clientHeight / viewport.scrollHeight, 0, 1),
      }

      setThreadMapViewport((current) => {
        if (
          current &&
          Math.abs(current.top - nextViewport.top) < 0.001 &&
          Math.abs(current.height - nextViewport.height) < 0.001
        ) {
          return current
        }
        return nextViewport
      })
    }

    const measureMarkers = () => {
      const nextQualifies = viewport.scrollHeight > viewport.clientHeight + 1
      setThreadMapQualifies(nextQualifies)

      if (!nextQualifies || viewport.scrollHeight <= 0) {
        setThreadMapMeasuredMarkers([])
        setThreadMapViewport(null)
        return
      }

      const nextMarkers = Array.from(
        threadList.querySelectorAll<HTMLElement>("[data-ab-message-id]")
      ).flatMap((node) => {
        const messageId = node.dataset.abMessageId
        if (!messageId) {
          return []
        }

        const entry = messageMap.get(messageId)
        if (!entry) {
          return []
        }

        return [
          {
            messageId,
            seq: entry.seq,
            sender: entry.sender,
            top: clamp(node.offsetTop / viewport.scrollHeight, 0, 1),
            height: clamp(node.offsetHeight / viewport.scrollHeight, 0.0025, 1),
          } satisfies ThreadMapMeasuredMarker,
        ]
      })

      setThreadMapMeasuredMarkers(nextMarkers)
      updateViewport()
    }

    const onScroll = () => {
      updateViewport()
      revealThreadMap()
    }
    viewport.addEventListener("scroll", onScroll, { passive: true })

    const resizeObserver =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => measureMarkers())
    if (resizeObserver) {
      resizeObserver.observe(viewport)
      resizeObserver.observe(threadList)
    } else {
      window.addEventListener("resize", measureMarkers)
    }

    measureMarkers()

    return () => {
      viewport.removeEventListener("scroll", onScroll)
      resizeObserver?.disconnect()
      if (!resizeObserver) {
        window.removeEventListener("resize", measureMarkers)
      }
    }
  }, [isDesktopThreadMap, topicDetail.messages])

  function jumpToMessage(messageId: string) {
    revealThreadMap()
    const element = document.getElementById(`msg-${messageId}`)
    if (!element) {
      return
    }
    element.scrollIntoView({ behavior: "smooth", block: "center" })
  }

  function dragThreadMapViewport(top: number) {
    const viewport = threadViewportRef.current
    if (!viewport) {
      return
    }

    revealThreadMap()
    viewport.scrollTop = clamp(top, 0, 1) * viewport.scrollHeight
    viewport.dispatchEvent(new Event("scroll"))
  }

  const showThreadMapOverlay = isDesktopThreadMap && threadMapQualifies

  return (
    <div className="grid h-full min-h-0 flex-1 grid-cols-1 gap-0 border border-border bg-card xl:grid-cols-[minmax(0,1fr)_12rem]">
      <section className="flex min-h-0 flex-col overflow-hidden">
        <CardHeader className="border-b border-border bg-card px-4 py-3">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="flex min-w-0 flex-col gap-2">
              <div className="flex flex-wrap items-center gap-2">
                <CardTitle className="truncate text-lg font-medium">{topicDetail.topic.name}</CardTitle>
                <Badge variant={statusVariant(topicDetail.topic.status)}>
                  {topicDetail.topic.status}
                </Badge>
                <Badge variant="outline">{topicDetail.message_count} messages</Badge>
              </div>
              <CardDescription className="flex flex-wrap items-center gap-2">
                <span>created {formatAbsoluteTime(topicDetail.topic.created_at)}</span>
                <Separator orientation="vertical" className="h-4" />
                <span>updated {formatRelativeTime(topicDetail.topic.last_updated_at)}</span>
                <Separator orientation="vertical" className="h-4" />
                <span>{topicDetail.topic.topic_id}</span>
              </CardDescription>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="outline" size="sm" onClick={onToggleSelectionMode}>
                <CheckIcon data-icon="inline-start" />
                {selectionMode ? "Cancel selection" : "Select messages"}
              </Button>
              <Button asChild variant="outline" size="sm">
                <a href={exportTopicUrl(topicDetail.topic.topic_id)} download>
                  <ArrowDownToLineIcon data-icon="inline-start" />
                  Export
                </a>
              </Button>
              <Button variant="outline" size="sm" onClick={onDeleteTopic}>
                <Trash2Icon data-icon="inline-start" />
                Delete topic
              </Button>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            {topicDetail.presence.length === 0 ? (
              <span className="text-muted-foreground">No active peers</span>
            ) : (
              topicDetail.presence.map((presence) => (
                <Badge key={presence.agent_name} variant="secondary">
                  {presence.agent_name}
                </Badge>
              ))
            )}
          </div>
        </CardHeader>
        <CardContent className="flex min-h-0 flex-1 flex-col gap-0 p-0">
          {findState.open ? (
            <div className="flex flex-wrap items-center gap-2 border-b border-border bg-[#1d2026] px-4 py-3">
              <Input
                ref={findInputRef}
                value={findState.query}
                onChange={(event) => onFindQueryChange(event.target.value)}
                placeholder="Find in this thread"
                className="h-9 min-w-[16rem] flex-1 border-zinc-700 bg-[#111318] text-zinc-100 placeholder:text-zinc-500 focus-visible:border-sky-500 focus-visible:ring-sky-500/25"
              />
              <div className="min-w-20 text-right text-sm text-muted-foreground">
                {findState.query.trim()
                  ? `${findMatches.length === 0 ? 0 : Math.min(findState.activeIndex + 1, findMatches.length)}/${findMatches.length}`
                  : "0/0"}
              </div>
              <Button variant="outline" size="sm" onClick={onFindPrevious} disabled={findMatches.length === 0}>
                Prev
              </Button>
              <Button variant="outline" size="sm" onClick={onFindNext} disabled={findMatches.length === 0}>
                Next
              </Button>
              <Button variant="ghost" size="sm" onClick={onCloseFind}>
                <XIcon data-icon="inline-start" />
                Close
              </Button>
            </div>
          ) : null}
          {selectionMode ? (
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border bg-accent/40 px-4 py-3">
              <div className="text-sm text-muted-foreground">
                {selectedMessageIds.size} selected
              </div>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={onDeleteSelected}
                  disabled={selectedMessageIds.size === 0}
                >
                  <Trash2Icon data-icon="inline-start" />
                  Delete selected
                </Button>
              </div>
            </div>
          ) : null}
          <div className="flex min-h-0 flex-1 flex-col">
            {topicDetail.has_earlier ? (
              <div className="flex justify-center border-b border-border px-4 py-3">
                <Button variant="outline" size="sm" onClick={onLoadEarlier}>
                  <ArrowUpDownIcon data-icon="inline-start" />
                  Load earlier
                </Button>
              </div>
            ) : null}
            <div className="flex items-center justify-between border-b border-border px-4 py-2 text-xs uppercase tracking-[0.12em] text-muted-foreground">
              <span>Thread</span>
              <span>{topicDetail.message_count} message{topicDetail.message_count === 1 ? "" : "s"}</span>
            </div>
            <div
              data-ab-thread-map-region="true"
              className="relative flex min-h-0 flex-1 flex-col"
              onPointerMove={() => {
                if (threadMapVisible) {
                  revealThreadMap()
                }
              }}
              onPointerLeave={() => {
                if (threadMapVisible) {
                  scheduleThreadMapHide(500)
                }
              }}
            >
              {showThreadMapOverlay ? (
                <>
                  <div
                    data-ab-thread-map-hotspot="true"
                    className="absolute inset-y-0 right-0 z-10 w-10"
                    onPointerEnter={() => revealThreadMap()}
                    onPointerMove={() => revealThreadMap()}
                  />
                  <ThreadMap
                    visible={threadMapVisible}
                    markers={threadMapMarkers}
                    viewport={threadMapViewport}
                    onJumpToMessage={jumpToMessage}
                    onViewportDrag={dragThreadMapViewport}
                  />
                </>
              ) : null}
              <ScrollArea
                className="min-h-0 flex-1 bg-[#1a1d22]"
                data-ab-topic-thread-scroll-area="true"
                viewportRef={threadViewportRef}
              >
                <div ref={threadListRef} className="flex min-h-full w-full min-w-0 flex-col gap-3 p-4">
                  {topicDetail.messages.length === 0 ? (
                    <div className="flex min-h-64 flex-col items-center justify-center gap-3 border border-dashed border-border bg-muted/20 text-center">
                      <MessageSquareMoreIcon className="size-8 text-muted-foreground" />
                      <div className="flex flex-col gap-1">
                        <div className="font-medium">No messages yet</div>
                        <div className="text-sm text-muted-foreground">
                          This topic exists, but it does not have any content yet.
                        </div>
                      </div>
                    </div>
                  ) : (
                    topicDetail.messages.map((message) => {
                      const localMatched =
                        hasTrimmedFindQuery && localMatchedMessageIds.has(message.message_id)
                      const localFindQuery = localMatched ? trimmedFindQuery : null
                      const focusedSearchMessage =
                        activeSearchResult?.message_id === message.message_id ? activeSearchResult : null
                      const highlightTerms = uniqueStrings([
                        ...(localFindQuery ? [localFindQuery] : []),
                        ...(focusedSearchMessage
                          ? activeSearchTerms.length > 0
                            ? activeSearchTerms
                            : sidebarSearchQuery.trim()
                              ? tokenizeSearchQuery(sidebarSearchQuery)
                              : []
                          : []),
                      ])
                      const highlight =
                        highlightTerms.length > 0
                          ? {
                              terms: highlightTerms,
                            }
                          : null

                      return (
                        <MessageCard
                          key={message.message_id}
                          message={message}
                          highlight={highlight}
                          focused={topicDetail.focus_message_id === message.message_id}
                          localMatched={localMatched}
                          localActive={activeFindMessageId === message.message_id}
                          selectable={selectionMode}
                          selected={selectedMessageIds.has(message.message_id)}
                          onSelectedChange={(next) => onMessageSelectionChange(message.message_id, next)}
                        />
                      )
                    })
                  )}
                </div>
              </ScrollArea>
            </div>
          </div>
        </CardContent>
      </section>
      <aside className="hidden min-h-0 flex-col overflow-hidden border-l border-border bg-[#202227] xl:flex">
        <>
          <CardHeader className="border-b border-border px-4 py-3">
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Inspector</div>
            <CardTitle className="text-base">Topic metadata</CardTitle>
          </CardHeader>
          <TopicInspectorPanel topicDetail={topicDetail} />
        </>
      </aside>
    </div>
  )
}

function WorkbenchKeyboardShortcuts(props: {
  routeTopicId: string | null
  onFocusSidebarSearch: () => void
  onOpenTopicFind: () => void
  onCloseTopicFind: () => void
}) {
  const { routeTopicId, onFocusSidebarSearch, onOpenTopicFind, onCloseTopicFind } = props
  const { isMobile, setOpen, setOpenMobile } = useSidebar()

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const isPrimaryModifier = event.metaKey || event.ctrlKey
      if (isPrimaryModifier && event.key.toLowerCase() === "k") {
        event.preventDefault()
        if (isMobile) {
          setOpenMobile(true)
        } else {
          setOpen(true)
        }
        window.setTimeout(onFocusSidebarSearch, 0)
      }
      if (isPrimaryModifier && event.key.toLowerCase() === "f" && routeTopicId) {
        event.preventDefault()
        onOpenTopicFind()
      }
      if (event.key === "Escape") {
        onCloseTopicFind()
      }
    }

    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [isMobile, onCloseTopicFind, onFocusSidebarSearch, onOpenTopicFind, routeTopicId, setOpen, setOpenMobile])

  return null
}

function AppSidebar(props: {
  routeTopicId: string | null
  focusMessageId: string | null
  workbenchState: WorkbenchState
  topics: TopicSummary[]
  topicsLoading: boolean
  topicsError: string | null
  railSearchState: SearchState
  sidebarSearchInputRef: React.RefObject<HTMLInputElement | null>
  setRailSearchState: React.Dispatch<React.SetStateAction<SearchState>>
  setWorkbenchState: React.Dispatch<React.SetStateAction<WorkbenchState>>
  openTopic: (topicId: string, options?: { focusMessageId?: string | null; replace?: boolean }) => void
}) {
  const {
    routeTopicId,
    focusMessageId,
    workbenchState,
    topics,
    topicsLoading,
    topicsError,
    railSearchState,
    sidebarSearchInputRef,
    setRailSearchState,
    setWorkbenchState,
    openTopic,
  } = props
  const { isMobile, setOpenMobile } = useSidebar()

  return (
    <Sidebar collapsible="offcanvas" className="border-r border-sidebar-border">
      <div className="flex h-full min-h-0 flex-col bg-sidebar text-sidebar-foreground">
        <SidebarHeader className="gap-3 border-b border-sidebar-border px-3 py-3">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              <span className="text-[10px] uppercase tracking-[0.24em] text-muted-foreground">
                Agent Bus MCP
              </span>
              <div className="truncate text-sm font-medium">Workbench</div>
            </div>
            {isMobile ? (
              <SidebarTrigger
                variant="ghost"
                size="icon-sm"
                className="shrink-0 rounded-md border border-sidebar-border/70 bg-sidebar hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
              />
            ) : null}
          </div>
          <SidebarInput
            ref={sidebarSearchInputRef}
            className="rounded-sm border-sidebar-border bg-background"
            value={railSearchState.query}
            onChange={(event) => {
              const query = event.target.value
              setRailSearchState((current) => ({
                ...current,
                query,
              }))
              setWorkbenchState((current) => ({
                ...current,
                sidebarQuery: query,
              }))
            }}
            placeholder="Search"
          />
        </SidebarHeader>
        <SidebarContent className="min-h-0">
          <SidebarGroup className="gap-3 border-b border-sidebar-border px-3 py-3">
            <SidebarGroupLabel className="h-auto px-1 text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
              Browse
            </SidebarGroupLabel>
            <SidebarGroupContent className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <DropdownMenu>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="outline"
                          size="icon-sm"
                          className="rounded-sm border-sidebar-border bg-background"
                          aria-label="Topic status filter"
                        >
                          <ListFilterIcon />
                        </Button>
                      </DropdownMenuTrigger>
                    </TooltipTrigger>
                    <TooltipContent side="bottom">Filter topics</TooltipContent>
                  </Tooltip>
                  <DropdownMenuContent align="start" className="w-44 min-w-44">
                    <DropdownMenuLabel>Status</DropdownMenuLabel>
                    <DropdownMenuRadioGroup
                      value={workbenchState.sidebarStatus}
                      onValueChange={(value) =>
                        setWorkbenchState((current) => ({
                          ...current,
                          sidebarStatus: value as TopicStatusFilter,
                        }))
                      }
                    >
                      {TOPIC_STATUS_OPTIONS.map((option) => (
                        <DropdownMenuRadioItem key={option.value} value={option.value}>
                          {option.label}
                        </DropdownMenuRadioItem>
                      ))}
                    </DropdownMenuRadioGroup>
                  </DropdownMenuContent>
                </DropdownMenu>

                <DropdownMenu>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="outline"
                          size="icon-sm"
                          className="rounded-sm border-sidebar-border bg-background"
                          aria-label="Topic sort order"
                        >
                          <ArrowDownWideNarrowIcon />
                        </Button>
                      </DropdownMenuTrigger>
                    </TooltipTrigger>
                    <TooltipContent side="bottom">Sort topics</TooltipContent>
                  </Tooltip>
                  <DropdownMenuContent align="start" className="w-44 min-w-44">
                    <DropdownMenuLabel>Sort</DropdownMenuLabel>
                    <DropdownMenuRadioGroup
                      value={workbenchState.sidebarSort}
                      onValueChange={(value) =>
                        setWorkbenchState((current) => ({
                          ...current,
                          sidebarSort: value as TopicSort,
                        }))
                      }
                    >
                      {TOPIC_SORT_OPTIONS.map((option) => (
                        <DropdownMenuRadioItem key={option.value} value={option.value}>
                          {option.label}
                        </DropdownMenuRadioItem>
                      ))}
                    </DropdownMenuRadioGroup>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
              <div className="px-1 text-[11px] uppercase tracking-[0.08em] text-muted-foreground">
                {TOPIC_STATUS_LABELS[workbenchState.sidebarStatus]} ·{" "}
                {TOPIC_SORT_LABELS[workbenchState.sidebarSort]}
              </div>
            </SidebarGroupContent>
          </SidebarGroup>
          <SidebarSeparator className="mx-0" />
          <SidebarGroup className="min-h-0 flex-1 gap-0 p-0">
            <SidebarGroupLabel className="px-4 py-3 text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
              {railSearchState.query.trim() ? "Search results" : "Topics"}
            </SidebarGroupLabel>
            <SidebarGroupContent className="min-h-0 flex-1">
              <ScrollArea className="min-h-0 flex-1 border-t border-sidebar-border">
                <div className="px-2 py-2">
                  {railSearchState.query.trim() ? (
                    railSearchState.loading ? (
                      <div className="px-2 py-3 text-sm text-muted-foreground">Searching messages…</div>
                    ) : railSearchState.error ? (
                      <div className="border border-destructive/30 bg-destructive/10 px-3 py-3 text-sm text-destructive">
                        {railSearchState.error}
                      </div>
                    ) : (
                      <div className="flex flex-col gap-2">
                        {railSearchState.warnings.length > 0 ? (
                          <div className="border border-amber-500/30 bg-amber-500/10 px-3 py-3 text-sm text-amber-100">
                            {railSearchState.warnings.join(" ")}
                          </div>
                        ) : null}
                        {railSearchState.results.length === 0 ? (
                          <div className="border border-dashed border-sidebar-border px-3 py-4 text-sm text-muted-foreground">
                            No messages match this search yet.
                          </div>
                        ) : (
                          <div className="flex flex-col gap-1">
                            {railSearchState.results.map((result) => (
                              <button
                                key={result.message_id}
                                type="button"
                                onClick={() => {
                                  openTopic(result.topic_id, {
                                    focusMessageId: result.message_id,
                                  })
                                  if (isMobile) {
                                    setOpenMobile(false)
                                  }
                                }}
                                className={[
                                  "flex w-full min-w-0 flex-col gap-2 border-l-2 px-3 py-3 text-left transition",
                                  routeTopicId === result.topic_id && focusMessageId === result.message_id
                                    ? "border-primary bg-sidebar-accent"
                                    : "border-transparent hover:bg-sidebar-accent/80",
                                ].join(" ")}
                              >
                                <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-start gap-2">
                                  <span className="break-words text-[13px] font-medium leading-5 text-sidebar-foreground">
                                    {result.topic_name}
                                  </span>
                                  <Badge
                                    variant="outline"
                                    className="mt-0.5 h-5 w-fit rounded-full px-1.5 text-[10px]"
                                  >
                                    #{result.seq}
                                  </Badge>
                                </div>
                                <div className="flex min-w-0 flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                                  <span className="break-all leading-5">{result.sender}</span>
                                  <Badge
                                    variant="outline"
                                    className="h-5 max-w-full rounded-full px-1.5 text-[10px] leading-none"
                                  >
                                    {snippetLabel(result)}
                                  </Badge>
                                </div>
                                <div className="line-clamp-3 break-words text-[12px] leading-5 text-zinc-400">
                                  {renderSnippet(result.snippet)}
                                </div>
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    )
                  ) : topicsLoading ? (
                    <div className="px-2 py-3 text-sm text-muted-foreground">Loading topics…</div>
                  ) : topicsError ? (
                    <div className="border border-destructive/30 bg-destructive/10 px-3 py-3 text-sm text-destructive">
                      {topicsError}
                    </div>
                  ) : topics.length === 0 ? (
                    <div className="border border-dashed border-sidebar-border px-3 py-4 text-sm text-muted-foreground">
                      No topics match this filter yet.
                    </div>
                  ) : (
                    <div className="flex flex-col gap-1">
                      {topics.map((topic) => (
                        <button
                          key={topic.topic_id}
                          type="button"
                          onClick={() => {
                            openTopic(topic.topic_id)
                            if (isMobile) {
                              setOpenMobile(false)
                            }
                          }}
                          className={[
                            "grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2 border-l-2 px-2 py-2 text-left transition",
                            routeTopicId === topic.topic_id
                              ? "border-primary bg-sidebar-accent"
                              : "border-transparent hover:bg-sidebar-accent/80",
                          ].join(" ")}
                        >
                          <div className="min-w-0">
                            <div className="truncate text-[13px] font-medium leading-5">{topic.name}</div>
                            <div className="mt-0.5 text-[11px] text-muted-foreground">
                              {formatRelativeTime(topic.last_updated_at)}
                            </div>
                          </div>
                          <Badge variant="outline" className="h-5 min-w-5 rounded-full px-1.5 text-[10px]">
                            {topic.message_count}
                          </Badge>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </ScrollArea>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
        <SidebarFooter className="shrink-0 border-t border-sidebar-border px-4 py-3 pb-4 text-[11px] text-muted-foreground">
          <>
            Search the bus with <kbd className="rounded-sm bg-sidebar-accent px-1.5 py-0.5">⌘K</kbd>
          </>
        </SidebarFooter>
      </div>
    </Sidebar>
  )
}

export default function App() {
  const navigate = useNavigate()
  const location = useLocation()
  const [workbenchState, setWorkbenchState] = useState<WorkbenchState>(() => loadWorkbenchState())
  const [topics, setTopics] = useState<TopicSummary[]>([])
  const [topicsLoading, setTopicsLoading] = useState(true)
  const [topicsError, setTopicsError] = useState<string | null>(null)
  const [topicDetail, setTopicDetail] = useState<TopicDetailResponse | null>(null)
  const [topicLoading, setTopicLoading] = useState(false)
  const [topicError, setTopicError] = useState<string | null>(null)
  const [railSearchState, setRailSearchState] = useState<SearchState>({
    query: workbenchState.sidebarQuery,
    mode: "hybrid",
    results: [],
    warnings: [],
    loading: false,
    error: null,
  })
  const [topicFindState, setTopicFindState] = useState<FindState>({ open: false, query: "", activeIndex: 0 })
  const [selectionMode, setSelectionMode] = useState(false)
  const [selectedMessageIds, setSelectedMessageIds] = useState<Set<string>>(new Set())
  const topicDetailRef = useRef<TopicDetailResponse | null>(null)
  const topicStreamUpdateTokenRef = useRef(0)
  const restoredInitialRoute = useRef(false)
  const sidebarSearchInputRef = useRef<HTMLInputElement | null>(null)
  const topicFindInputRef = useRef<HTMLInputElement | null>(null)
  const tabStripRef = useRef<HTMLDivElement | null>(null)
  const activeTabRef = useRef<HTMLButtonElement | null>(null)
  const suppressNextFindScrollRef = useRef(false)

  const pathnameParts = location.pathname.split("/").filter(Boolean)
  const routeTopicId = pathnameParts[0] === "topics" ? decodeURIComponent(pathnameParts[1] ?? "") : null
  const focusMessageId = new URLSearchParams(location.search).get("focus")

  useEffect(() => {
    saveWorkbenchState(workbenchState)
  }, [workbenchState])

  useEffect(() => {
    topicDetailRef.current = topicDetail
  }, [topicDetail])

  useEffect(() => {
    topicStreamUpdateTokenRef.current += 1
  }, [focusMessageId, routeTopicId])

  useEffect(() => {
    if (restoredInitialRoute.current) {
      return
    }
    restoredInitialRoute.current = true
    if (!routeTopicId && workbenchState.activeTopicId) {
      startTransition(() => navigate(`/topics/${workbenchState.activeTopicId}`, { replace: true }))
    }
  }, [navigate, routeTopicId, workbenchState.activeTopicId])

  useEffect(() => {
    if (!routeTopicId) {
      return
    }

    setWorkbenchState((current) => {
      const openTopicIds = current.openTopicIds.includes(routeTopicId)
        ? current.openTopicIds
        : [...current.openTopicIds, routeTopicId]
      return {
        ...current,
        openTopicIds,
        activeTopicId: routeTopicId,
      }
    })
  }, [routeTopicId])

  useEffect(() => {
    let cancelled = false

    async function loadTopics() {
      setTopicsLoading(true)
      setTopicsError(null)
      try {
        const nextTopics = await fetchTopics({
          status: workbenchState.sidebarStatus,
          sort: workbenchState.sidebarSort,
          query: "",
        })
        if (!cancelled) {
          setTopics(nextTopics)
        }
      } catch (error) {
        if (!cancelled) {
          setTopicsError(error instanceof Error ? error.message : "Failed to load topics")
        }
      } finally {
        if (!cancelled) {
          setTopicsLoading(false)
        }
      }
    }

    void loadTopics()
    return () => {
      cancelled = true
    }
  }, [
    workbenchState.sidebarSort,
    workbenchState.sidebarStatus,
  ])

  useEffect(() => {
    if (!routeTopicId) {
      setTopicDetail(null)
      setTopicError(null)
      setTopicLoading(false)
      return
    }

    const topicId = routeTopicId
    let cancelled = false

    async function loadTopic() {
      setTopicLoading(true)
      setTopicError(null)
      try {
        const nextTopic = await fetchTopicDetail(topicId, focusMessageId)
        if (!cancelled) {
          setTopicDetail(nextTopic)
        }
      } catch (error) {
        if (!cancelled) {
          setTopicError(error instanceof Error ? error.message : "Failed to load topic")
          setTopicDetail(null)
        }
      } finally {
        if (!cancelled) {
          setTopicLoading(false)
        }
      }
    }

    void loadTopic()
    return () => {
      cancelled = true
    }
  }, [focusMessageId, routeTopicId])

  useEffect(() => {
    setSelectionMode(false)
    setSelectedMessageIds(new Set())
    setTopicFindState({ open: false, query: "", activeIndex: 0 })
  }, [routeTopicId])

  useEffect(() => {
    if (!railSearchState.query.trim()) {
      setRailSearchState((current) => ({
        ...current,
        results: [],
        warnings: [],
        loading: false,
        error: null,
      }))
      return
    }

    const timer = window.setTimeout(async () => {
      setRailSearchState((current) => ({ ...current, loading: true, error: null }))
      try {
        const response = await fetchSearchResults({
          query: railSearchState.query,
          mode: "hybrid",
          limit: 50,
        })
        setRailSearchState((current) => ({
          ...current,
          results: response.results,
          warnings: response.warnings,
          loading: false,
          error: null,
        }))
      } catch (error) {
        setRailSearchState((current) => ({
          ...current,
          loading: false,
          error: error instanceof Error ? error.message : "Search failed",
        }))
      }
    }, 250)

    return () => window.clearTimeout(timer)
  }, [railSearchState.query])

  useEffect(() => {
    const stream = new EventSource("/api/stream/topics")
    const invalidate = () => {
      void fetchTopics({
        status: workbenchState.sidebarStatus,
        sort: workbenchState.sidebarSort,
        query: "",
      })
        .then((nextTopics) => setTopics(nextTopics))
        .catch(() => {
          // Keep the current sidebar state on transient stream refresh errors.
        })
    }

    stream.addEventListener("topics.invalidate", invalidate)
    return () => {
      stream.removeEventListener("topics.invalidate", invalidate)
      stream.close()
    }
  }, [
    workbenchState.sidebarSort,
    workbenchState.sidebarStatus,
  ])

  useEffect(() => {
    if (!routeTopicId) {
      return
    }

    const topicId = routeTopicId
    const stream = new EventSource(`/api/stream/topics/${topicId}`)

    async function handleUpdate(event: Event) {
      const payload = JSON.parse((event as MessageEvent<string>).data) as TopicStreamUpdate

      const currentDetail = topicDetailRef.current
      if (!currentDetail || currentDetail.topic.topic_id !== topicId) {
        return
      }

      const updateToken = ++topicStreamUpdateTokenRef.current

      setTopicDetail((current) => {
        if (!current || current.topic.topic_id !== topicId) {
          return current
        }
        return {
          ...current,
          presence: payload.presence,
        }
      })

      const currentLastSeq = currentDetail.last_seq ?? 0
      const currentFocusMessageId = currentDetail.focus_message_id
      const appendOnly =
        !currentDetail.context_mode &&
        payload.last_seq > currentLastSeq &&
        payload.message_count > currentDetail.message_count

      async function refreshTopicDetail() {
        const refreshedDetail = await fetchTopicDetail(topicId, currentFocusMessageId)
        setTopicDetail((current) => {
          if (!current || current.topic.topic_id !== topicId) {
            return current
          }
          if (updateToken !== topicStreamUpdateTokenRef.current) {
            return current
          }
          return {
            ...refreshedDetail,
            presence: payload.presence,
          }
        })
      }

      if (appendOnly) {
        try {
          const pageLimit = 200
          const maxAppendPages = 5
          let afterSeq = currentLastSeq
          let appendedMessages: TopicMessage[] = []
          let pagesFetched = 0

          while (afterSeq < payload.last_seq && pagesFetched < maxAppendPages) {
            const nextMessages = await fetchTopicMessages(topicId, {
              afterSeq,
              limit: pageLimit,
            })
            const batch = nextMessages.messages
            pagesFetched += 1

            if (batch.length === 0) {
              break
            }

            appendedMessages = mergeMessages(appendedMessages, batch)

            const batchLastSeq = batch[batch.length - 1]?.seq ?? afterSeq
            if (batchLastSeq <= afterSeq) {
              break
            }
            afterSeq = batchLastSeq

            if (batch.length < pageLimit) {
              break
            }
          }

          if (afterSeq < payload.last_seq) {
            await refreshTopicDetail()
            return
          }

          setTopicDetail((current) => {
            if (!current || current.topic.topic_id !== topicId) {
              return current
            }
            if (updateToken !== topicStreamUpdateTokenRef.current) {
              return current
            }
            const messages = mergeMessages(current.messages, appendedMessages)
            return {
              ...current,
              messages,
              first_seq: messages[0]?.seq ?? current.first_seq,
              last_seq: messages[messages.length - 1]?.seq ?? current.last_seq,
              message_count: payload.message_count,
              topic: {
                ...current.topic,
                message_count: payload.message_count,
                last_seq: payload.last_seq,
                last_updated_at: Date.now() / 1000,
              },
              presence: payload.presence,
            }
          })
        } catch {
          // Ignore transient stream refresh failures; the next event or manual refresh will catch up.
        }
        return
      }

      if (
        payload.last_seq === currentLastSeq &&
        payload.message_count === currentDetail.message_count
      ) {
        return
      }

      try {
        await refreshTopicDetail()
      } catch {
        // Ignore transient stream refresh failures; the next event or manual refresh will catch up.
      }
    }

    function handleDeleted() {
      toast.error("This topic was deleted.")
      closeTopic(topicId)
    }

    stream.addEventListener("topic.update", handleUpdate)
    stream.addEventListener("topic.deleted", handleDeleted)
    return () => {
      topicStreamUpdateTokenRef.current += 1
      stream.removeEventListener("topic.update", handleUpdate)
      stream.removeEventListener("topic.deleted", handleDeleted)
      stream.close()
    }
  }, [routeTopicId])

  const topicMessages = topicDetail?.messages ?? EMPTY_TOPIC_MESSAGES

  const topicFindMatches = useMemo(() => {
    if (!topicFindState.query.trim()) {
      return []
    }

    return topicMessages.filter((message) =>
      messageContainsText(message, topicFindState.query.trim())
    )
  }, [topicMessages, topicFindState.query])
  const activeRailSearchResult =
    routeTopicId && focusMessageId
      ? railSearchState.results.find(
          (result) => result.topic_id === routeTopicId && result.message_id === focusMessageId
        ) ?? null
      : null

  useEffect(() => {
    if (!topicDetail?.focus_message_id) {
      return
    }
    suppressNextFindScrollRef.current = topicFindState.open && topicFindMatches.length > 0
    const element = document.getElementById(`msg-${topicDetail.focus_message_id}`)
    if (!element) {
      return
    }
    element.scrollIntoView({ behavior: "smooth", block: "center" })
  }, [topicDetail, topicFindMatches.length, topicFindState.open])

  useEffect(() => {
    if (!topicFindState.open || !topicFindState.query.trim() || topicFindMatches.length === 0) {
      suppressNextFindScrollRef.current = false
    }
  }, [topicFindMatches.length, topicFindState.open, topicFindState.query])

  useEffect(() => {
    activeTabRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
      inline: "nearest",
    })
  }, [routeTopicId, workbenchState.openTopicIds])

  useEffect(() => {
    setTopicFindState((current) => {
      if (!current.query.trim()) {
        return current.activeIndex === 0 ? current : { ...current, activeIndex: 0 }
      }
      const maxIndex = Math.max(topicFindMatches.length - 1, 0)
      const nextIndex = Math.min(current.activeIndex, maxIndex)
      return nextIndex === current.activeIndex ? current : { ...current, activeIndex: nextIndex }
    })
  }, [topicFindMatches.length, topicFindState.query])

  useEffect(() => {
    if (!topicFindState.open || !topicFindState.query.trim() || topicFindMatches.length === 0) {
      return
    }
    if (suppressNextFindScrollRef.current) {
      suppressNextFindScrollRef.current = false
      return
    }
    const activeMatch = topicFindMatches[Math.min(topicFindState.activeIndex, topicFindMatches.length - 1)]
    if (!activeMatch) {
      return
    }
    const element = document.getElementById(`msg-${activeMatch.message_id}`)
    if (!element) {
      return
    }
    element.scrollIntoView({ behavior: "smooth", block: "center" })
  }, [topicFindMatches, topicFindState.activeIndex, topicFindState.open, topicFindState.query])

  function openTopic(topicId: string, options?: { focusMessageId?: string | null; replace?: boolean }) {
    setWorkbenchState((current) => ({
      ...current,
      openTopicIds: current.openTopicIds.includes(topicId)
        ? current.openTopicIds
        : [...current.openTopicIds, topicId],
      activeTopicId: topicId,
    }))

    const search = options?.focusMessageId
      ? `?focus=${encodeURIComponent(options.focusMessageId)}`
      : ""
    startTransition(() =>
      navigate(`/topics/${encodeURIComponent(topicId)}${search}`, {
        replace: options?.replace ?? false,
      })
    )
  }

  function closeTopic(topicId: string) {
    const remaining = workbenchState.openTopicIds.filter((value) => value !== topicId)
    const nextActive =
      workbenchState.activeTopicId === topicId
        ? (remaining.at(-1) ?? null)
        : workbenchState.activeTopicId

    setWorkbenchState((current) => ({
      ...current,
      openTopicIds: remaining,
      activeTopicId: nextActive,
    }))

    if (workbenchState.activeTopicId === topicId) {
      startTransition(() => navigate(nextActive ? `/topics/${encodeURIComponent(nextActive)}` : "/", { replace: true }))
    }
  }

  async function loadEarlierMessages() {
    if (!routeTopicId || !topicDetail?.first_seq) {
      return
    }

    try {
      const payload = await fetchTopicMessages(routeTopicId, {
        beforeSeq: topicDetail.first_seq,
        limit: 50,
      })
      setTopicDetail((current) => {
        if (!current || current.topic.topic_id !== routeTopicId) {
          return current
        }
        const messages = mergeMessages(payload.messages, current.messages)
        return {
          ...current,
          messages,
          first_seq: messages[0]?.seq ?? current.first_seq,
          last_seq: messages[messages.length - 1]?.seq ?? current.last_seq,
          has_earlier: payload.has_earlier,
        }
      })
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to load earlier messages")
    }
  }

  async function handleDeleteTopic() {
    if (!topicDetail) {
      return
    }
    if (!window.confirm(`Delete topic "${topicDetail.topic.name}" and all of its messages?`)) {
      return
    }

    try {
      await deleteTopic(topicDetail.topic.topic_id)
      toast.success(`Deleted ${topicDetail.topic.name}`)
      closeTopic(topicDetail.topic.topic_id)
      const nextTopics = await fetchTopics({
        status: workbenchState.sidebarStatus,
        sort: workbenchState.sidebarSort,
        query: "",
      })
      setTopics(nextTopics)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to delete topic")
    }
  }

  async function handleDeleteSelected() {
    if (!routeTopicId || selectedMessageIds.size === 0) {
      return
    }
    if (!window.confirm(`Delete ${selectedMessageIds.size} selected messages?`)) {
      return
    }

    try {
      await deleteMessages(routeTopicId, Array.from(selectedMessageIds))
      const fresh = await fetchTopicDetail(routeTopicId)
      setTopicDetail(fresh)
      setSelectedMessageIds(new Set())
      setSelectionMode(false)
      toast.success("Messages deleted")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to delete messages")
    }
  }

  const topicMap = new Map(topics.map((topic) => [topic.topic_id, topic]))
  const openTabs = workbenchState.openTopicIds.map((topicId) => {
    if (topicDetail?.topic.topic_id === topicId) {
      return topicDetail.topic
    }
    return topicMap.get(topicId) ?? fallbackTopic(topicId)
  })
  return (
    <SidebarProvider defaultOpen className="h-svh overflow-hidden bg-background text-foreground" style={SIDEBAR_LAYOUT_STYLE}>
      <WorkbenchKeyboardShortcuts
        routeTopicId={routeTopicId}
        onFocusSidebarSearch={() => sidebarSearchInputRef.current?.focus()}
        onOpenTopicFind={() => {
          setTopicFindState((current) => ({ ...current, open: true }))
          window.setTimeout(() => topicFindInputRef.current?.focus(), 0)
        }}
        onCloseTopicFind={() =>
          setTopicFindState((current) => (current.open ? { ...current, open: false } : current))
        }
      />
      <div className="flex h-full w-full overflow-hidden bg-[#1b1d23] text-foreground">
        <AppSidebar
          routeTopicId={routeTopicId}
          focusMessageId={focusMessageId}
          workbenchState={workbenchState}
          topics={topics}
          topicsLoading={topicsLoading}
          topicsError={topicsError}
          railSearchState={railSearchState}
          sidebarSearchInputRef={sidebarSearchInputRef}
          setRailSearchState={setRailSearchState}
          setWorkbenchState={setWorkbenchState}
          openTopic={openTopic}
        />
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
              <header className="sticky top-0 z-20 border-b border-border bg-[#181a1f]">
                <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
                  <div className="flex items-center gap-2">
                    <SidebarTrigger
                      variant="ghost"
                      size="icon-sm"
                      className="rounded-md border border-border/70 bg-card hover:bg-accent hover:text-foreground"
                    />
                  </div>
                  <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.12em] text-muted-foreground">
                    <FolderSearch2Icon className="size-4" />
                    <span>Activity-sorted workbench</span>
                  </div>
                </div>
                <div
                  ref={tabStripRef}
                  className="flex shrink-0 gap-0 overflow-x-auto overflow-y-hidden border-t border-border bg-[#1f2229] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
                  onWheel={(event) => {
                    const container = tabStripRef.current
                    if (!container || container.scrollWidth <= container.clientWidth) {
                      return
                    }
                    if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) {
                      return
                    }
                    event.preventDefault()
                    container.scrollLeft += event.deltaY
                  }}
                >
                  {openTabs.length === 0 ? (
                    <div className="flex-1 px-4 py-3 text-sm text-muted-foreground">
                      Open a topic from the sidebar search.
                    </div>
                  ) : (
                    openTabs.map((topic) => (
                      <div
                        key={topic.topic_id}
                        className={[
                          "flex shrink-0 items-center gap-2 border-r border-border px-3 py-2.5",
                          workbenchState.activeTopicId === topic.topic_id
                            ? "border-t-2 border-t-primary bg-card text-foreground"
                            : "border-t-2 border-t-transparent bg-[#1f2229] text-muted-foreground",
                        ].join(" ")}
                      >
                        <button
                          type="button"
                          ref={workbenchState.activeTopicId === topic.topic_id ? activeTabRef : null}
                          className="flex min-w-0 items-center gap-2 text-left"
                          onClick={() => openTopic(topic.topic_id)}
                        >
                          <span className="max-w-44 truncate text-[13px]">{topic.name}</span>
                          <Badge variant="outline" className="h-5 min-w-5 rounded-full px-1.5 text-[10px]">
                            {topic.message_count}
                          </Badge>
                        </button>
                        <button
                          type="button"
                          className="rounded-sm p-1 text-muted-foreground transition hover:bg-accent hover:text-foreground"
                          onClick={() => closeTopic(topic.topic_id)}
                          aria-label={`Close ${topic.name}`}
                        >
                          <XIcon className="size-3.5" />
                        </button>
                      </div>
                    ))
                  )}
                </div>
              </header>
              <main className="flex min-h-0 flex-1 flex-col gap-0 overflow-hidden bg-[#1b1d23] p-2">
                {topicLoading ? (
                  <Card className="flex flex-1 rounded-none border-border bg-card shadow-none">
                    <CardContent className="flex min-h-[24rem] items-center justify-center">
                      <div className="text-sm text-muted-foreground">Loading topic…</div>
                    </CardContent>
                  </Card>
                ) : topicError ? (
                  <Card className="flex flex-1 rounded-none border-destructive/30 bg-destructive/10 shadow-none">
                    <CardContent className="flex min-h-[24rem] items-center justify-center gap-3 p-8 text-destructive">
                      <AlertCircleIcon className="size-5" />
                      <span>{topicError}</span>
                    </CardContent>
                  </Card>
                ) : topicDetail ? (
        <TopicView
          key={topicDetail.topic.topic_id}
          topicDetail={topicDetail}
          activeSearchResult={activeRailSearchResult}
          sidebarSearchQuery={railSearchState.query}
          findState={topicFindState}
          findMatches={topicFindMatches}
                    selectionMode={selectionMode}
                    selectedMessageIds={selectedMessageIds}
                    findInputRef={topicFindInputRef}
                    onCloseFind={() => setTopicFindState((current) => ({ ...current, open: false }))}
                    onFindQueryChange={(query) =>
                      setTopicFindState((current) => ({ ...current, query, activeIndex: 0 }))
                    }
                    onFindPrevious={() =>
                      setTopicFindState((current) => ({
                        ...current,
                        activeIndex:
                          topicFindMatches.length === 0
                            ? 0
                            : (current.activeIndex - 1 + topicFindMatches.length) % topicFindMatches.length,
                      }))
                    }
                    onFindNext={() =>
                      setTopicFindState((current) => ({
                        ...current,
                        activeIndex:
                          topicFindMatches.length === 0
                            ? 0
                            : (current.activeIndex + 1) % topicFindMatches.length,
                      }))
                    }
                    onToggleSelectionMode={() => {
                      setSelectionMode((current) => !current)
                      setSelectedMessageIds(new Set())
                    }}
                    onDeleteSelected={() => void handleDeleteSelected()}
                    onMessageSelectionChange={(messageId, next) =>
                      setSelectedMessageIds((current) => {
                        const nextSet = new Set(current)
                        if (next) {
                          nextSet.add(messageId)
                        } else {
                          nextSet.delete(messageId)
                        }
                        return nextSet
                      })
                    }
                    onLoadEarlier={() => void loadEarlierMessages()}
                    onDeleteTopic={() => void handleDeleteTopic()}
                  />
                ) : (
                  <Card className="flex flex-1 rounded-none border-border bg-card shadow-none">
                    <CardContent className="flex min-h-[28rem] flex-col items-center justify-center gap-7 px-6 text-center">
                      <img
                        src="/app-mark.svg"
                        alt=""
                        className="h-auto w-full max-w-[18.5rem] opacity-95"
                        aria-hidden="true"
                      />
                      <div className="flex max-w-lg flex-col gap-2">
                        <h1 className="text-2xl font-medium tracking-tight">Agent Bus MCP Workbench</h1>
                        <p className="text-muted-foreground">
                          Browse topics by creation or latest activity, keep several threads open at
                          once, and search across the whole bus from the sidebar without leaving the
                          main shell.
                        </p>
                      </div>
                      <Badge variant="secondary">Use ⌘K to jump to search</Badge>
                    </CardContent>
                  </Card>
                )}
              </main>
        </div>
      </div>
    </SidebarProvider>
  )
}
