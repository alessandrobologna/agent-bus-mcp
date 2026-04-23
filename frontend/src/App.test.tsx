import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter, useNavigate } from "react-router-dom"
import { describe, expect, test, vi } from "vitest"

import App from "@/App"
import { TooltipProvider } from "@/components/ui/tooltip"
import type { TopicDetailResponse, TopicSummary } from "@/lib/types"
import { DEFAULT_WORKBENCH_STATE, loadWorkbenchState } from "@/lib/workbench-state"

const SEARCH_HIGHLIGHT_START = "\uE000"
const SEARCH_HIGHLIGHT_END = "\uE001"

const topicsPayload: { topics: TopicSummary[] } = {
  topics: [
    {
      topic_id: "t-1",
      name: "Alpha review",
      status: "open",
      created_at: 1_700_000_000,
      closed_at: null,
      close_reason: null,
      metadata: null,
      message_count: 1,
      last_seq: 1,
      last_message_at: 1_700_000_100,
      last_updated_at: 1_700_000_100,
    },
    {
      topic_id: "t-2",
      name: "Beta thread",
      status: "open",
      created_at: 1_700_000_010,
      closed_at: null,
      close_reason: null,
      metadata: null,
      message_count: 1,
      last_seq: 1,
      last_message_at: 1_700_000_090,
      last_updated_at: 1_700_000_090,
    },
  ],
}

function topicDetail(
  topicId: string,
  content: string,
  options?: { focus?: string | null }
): TopicDetailResponse {
  return {
    topic: topicsPayload.topics.find((topic) => topic.topic_id === topicId)!,
    messages: [
      {
        message_id: options?.focus ?? `${topicId}-m-1`,
        topic_id: topicId,
        seq: 1,
        sender: "reviewer",
        message_type: "message",
        reply_to: null,
        reply_to_sender: null,
        content_markdown: content,
        metadata: null,
        client_message_id: null,
        created_at: 1_700_000_100,
      },
    ],
    message_count: 1,
    first_seq: 1,
    last_seq: 1,
    has_earlier: false,
    context_mode: Boolean(options?.focus),
    focus_message_id: options?.focus ?? null,
    presence: [
      {
        topic_id: topicId,
        agent_name: "codex reviewer",
        last_seq: 1,
        updated_at: 1_700_000_110,
      },
    ],
  }
}

function topicDetailWithMessages(
  topicId: string,
  contents: string[],
  options?: {
    focusMessageId?: string | null
    hasEarlier?: boolean
    startSeq?: number
    messageIds?: string[]
  }
): TopicDetailResponse {
  const startSeq = options?.startSeq ?? 1
  const messages = contents.map((content, index) => {
    const seq = startSeq + index
    return {
      message_id: options?.messageIds?.[index] ?? `${topicId}-m-${seq}`,
      topic_id: topicId,
      seq,
      sender: index % 2 === 0 ? "reviewer" : "architect",
      message_type: "message" as const,
      reply_to: null,
      reply_to_sender: null,
      content_markdown: content,
      metadata: null,
      client_message_id: null,
      created_at: 1_700_000_100 + index,
    }
  })

  return {
    topic: {
      ...topicsPayload.topics.find((topic) => topic.topic_id === topicId)!,
      message_count: messages.length,
      last_seq: messages.at(-1)?.seq ?? 0,
      last_message_at: messages.at(-1)?.created_at ?? null,
      last_updated_at: messages.at(-1)?.created_at ?? 1_700_000_100,
    },
    messages,
    message_count: messages.length,
    first_seq: messages[0]?.seq ?? null,
    last_seq: messages.at(-1)?.seq ?? null,
    has_earlier: options?.hasEarlier ?? false,
    context_mode: Boolean(options?.focusMessageId),
    focus_message_id: options?.focusMessageId ?? null,
    presence: [
      {
        topic_id: topicId,
        agent_name: "codex reviewer",
        last_seq: messages.at(-1)?.seq ?? 0,
        updated_at: 1_700_000_110,
      },
    ],
  }
}

function jsonResponse(payload: unknown) {
  return Promise.resolve(
    new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  )
}

function installFetchMock() {
  return vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
    const url = new URL(String(input), "http://localhost")

    if (url.pathname === "/api/topics") {
      return jsonResponse(topicsPayload)
    }
    if (url.pathname === "/api/topics/t-2" && url.searchParams.get("focus") === "focused-message") {
      return jsonResponse(
        topicDetail("t-2", "beta context", { focus: "focused-message" })
      )
    }
    if (url.pathname === "/api/topics/t-1") {
      return jsonResponse(topicDetail("t-1", "hello from alpha"))
    }
    if (url.pathname === "/api/topics/t-2") {
      return jsonResponse(topicDetail("t-2", "beta context"))
    }
    if (url.pathname === "/api/search") {
      return jsonResponse({
        query: url.searchParams.get("q") ?? "",
        mode: "hybrid",
        warnings: [],
        results: [
          {
            topic_id: "t-2",
            topic_name: "Beta thread",
            message_id: "focused-message",
            seq: 7,
            sender: "architect",
            message_type: "message",
            snippet: `<img src=x onerror=alert(1)> ${SEARCH_HIGHLIGHT_START}handoff${SEARCH_HIGHLIGHT_END} summary`,
            semantic_score: 0.81,
          },
        ],
      })
    }
    throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
  })
}

function renderApp(initialEntries: string[]) {
  return render(
    <TooltipProvider>
      <MemoryRouter initialEntries={initialEntries}>
        <App />
      </MemoryRouter>
    </TooltipProvider>
  )
}

function renderAppWithControls(initialEntries: string[]) {
  function NavigationControls() {
    const navigate = useNavigate()

    return (
      <button type="button" onClick={() => navigate("/topics/t-1?focus=focused-message")}>
        Focus same topic
      </button>
    )
  }

  return render(
    <TooltipProvider>
      <MemoryRouter initialEntries={initialEntries}>
        <NavigationControls />
        <App />
      </MemoryRouter>
    </TooltipProvider>
  )
}

function setDesktopWidth(width = 1440) {
  window.innerWidth = width
  window.dispatchEvent(new Event("resize"))
}

function triggerResizeObservers() {
  const ResizeObserverCtor = globalThis.ResizeObserver as unknown as {
    instances: Array<{ trigger: () => void }>
  }

  for (const observer of ResizeObserverCtor.instances) {
    observer.trigger()
  }
}

function setTopicThreadLayout(props: {
  scrollHeight: number
  clientHeight: number
  scrollTop?: number
  messageHeights?: number[]
  messageGap?: number
}) {
  const {
    scrollHeight,
    clientHeight,
    scrollTop = 0,
    messageHeights = [],
    messageGap = 12,
  } = props

  const scrollAreaRoot = document.querySelector<HTMLElement>("[data-ab-topic-thread-scroll-area='true']")
  expect(scrollAreaRoot).toBeTruthy()

  const viewport = scrollAreaRoot!.querySelector<HTMLElement>("[data-slot='scroll-area-viewport']")
  expect(viewport).toBeTruthy()

  Object.defineProperty(viewport!, "scrollHeight", {
    configurable: true,
    value: scrollHeight,
  })
  Object.defineProperty(viewport!, "clientHeight", {
    configurable: true,
    value: clientHeight,
  })
  Object.defineProperty(viewport!, "scrollTop", {
    configurable: true,
    writable: true,
    value: scrollTop,
  })

  let offset = 0
  const messageNodes = Array.from(document.querySelectorAll<HTMLElement>("[data-ab-message-id]"))
  for (const [index, node] of messageNodes.entries()) {
    const height = messageHeights[index] ?? 180
    Object.defineProperty(node, "offsetTop", {
      configurable: true,
      value: offset,
    })
    Object.defineProperty(node, "offsetHeight", {
      configurable: true,
      value: height,
    })
    offset += height + messageGap
  }

  triggerResizeObservers()
  viewport!.dispatchEvent(new Event("scroll"))
}

function revealThreadMapOverlay() {
  const hotspot = document.querySelector<HTMLElement>("[data-ab-thread-map-hotspot='true']")
  expect(hotspot).toBeTruthy()
  fireEvent.pointerEnter(hotspot!)
  fireEvent.pointerMove(hotspot!)
}

function topicStream(topicId: string) {
  const EventSourceCtor = globalThis.EventSource as unknown as {
    instances: Array<{ url: string; emit: (type: string, payload?: unknown) => void }>
  }

  const stream = EventSourceCtor.instances.find(
    (instance) => instance.url === `/api/stream/topics/${topicId}`
  )
  expect(stream).toBeDefined()
  return stream!
}

describe("App", () => {
  test("restores the last active topic from localStorage", async () => {
    window.localStorage.setItem(
      "agent-bus.workbench.v1",
      JSON.stringify({
        openTopicIds: ["t-1"],
        activeTopicId: "t-1",
        sidebarQuery: "",
        sidebarStatus: "all",
        sidebarSort: "last_updated_desc",
      })
    )
    installFetchMock()

    renderApp(["/"])

    expect(await screen.findByText("hello from alpha")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Close Alpha review/i })).toBeInTheDocument()
  })

  test("opens a topic from the sidebar and closes its tab", async () => {
    installFetchMock()

    renderApp(["/"])

    const topicButton = await screen.findByRole("button", { name: /Alpha review/i })
    fireEvent.click(topicButton)

    expect(await screen.findByText("hello from alpha")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /Close Alpha review/i }))

    expect(await screen.findByText(/Agent Bus MCP Workbench/i)).toBeInTheDocument()
  })

  test("uses the sidebar search to navigate to a focused result", async () => {
    installFetchMock()

    renderApp(["/"])

    fireEvent.change(await screen.findByPlaceholderText(/^Search$/i), {
      target: { value: "handoff" },
    })

    const result = await screen.findByRole("button", { name: /Beta thread/i })
    fireEvent.click(result)

    await waitFor(() => expect(screen.getByText("beta context")).toBeInTheDocument())
  })

  test("highlights lexical sidebar matches inside the focused message", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-2" && url.searchParams.get("focus") === "focused-message") {
        return jsonResponse(
          topicDetail("t-2", "beta handoff summary context", { focus: "focused-message" })
        )
      }
      if (url.pathname === "/api/search") {
        return jsonResponse({
          query: url.searchParams.get("q") ?? "",
          mode: "hybrid",
          warnings: [],
          results: [
            {
              topic_id: "t-2",
              topic_name: "Beta thread",
              message_id: "focused-message",
              seq: 7,
              sender: "architect",
              message_type: "message",
              snippet: `beta ${SEARCH_HIGHLIGHT_START}handoff${SEARCH_HIGHLIGHT_END} summary`,
              rank: 0.2,
            },
          ],
        })
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    const { container } = renderApp(["/"])

    fireEvent.change(await screen.findByPlaceholderText(/^Search$/i), {
      target: { value: "handoff" },
    })

    fireEvent.click(await screen.findByRole("button", { name: /Beta thread/i }))

    await waitFor(() =>
      expect(
        Array.from(container.querySelectorAll("mark[data-ab-highlight='true']")).some(
          (element) => element.textContent === "handoff"
        )
      ).toBe(true)
    )
  })

  test("highlights overlapping query terms for semantic results", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-2" && url.searchParams.get("focus") === "focused-message") {
        return jsonResponse(
          topicDetail("t-2", "beta handoff\nsummary context", { focus: "focused-message" })
        )
      }
      if (url.pathname === "/api/search") {
        return jsonResponse({
          query: url.searchParams.get("q") ?? "",
          mode: "hybrid",
          warnings: [],
          results: [
            {
              topic_id: "t-2",
              topic_name: "Beta thread",
              message_id: "focused-message",
              seq: 7,
              sender: "architect",
              message_type: "message",
              snippet: "handoff summary",
              semantic_score: 0.81,
            },
          ],
        })
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    const { container } = renderApp(["/"])

    fireEvent.change(await screen.findByPlaceholderText(/^Search$/i), {
      target: { value: "semantic handoff" },
    })

    fireEvent.click(await screen.findByRole("button", { name: /Beta thread/i }))

    await waitFor(() => {
      const highlightedText = Array.from(container.querySelectorAll("mark[data-ab-highlight='true']")).map(
        (element) => element.textContent ?? ""
      )
      expect(highlightedText).toContain("handoff")
    })
  })

  test("highlights Unicode case-insensitive matches without misaligned ranges", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-2" && url.searchParams.get("focus") === "focused-message") {
        return jsonResponse(
          topicDetail("t-2", "İstanbul handoff context", { focus: "focused-message" })
        )
      }
      if (url.pathname === "/api/search") {
        return jsonResponse({
          query: url.searchParams.get("q") ?? "",
          mode: "hybrid",
          warnings: [],
          results: [
            {
              topic_id: "t-2",
              topic_name: "Beta thread",
              message_id: "focused-message",
              seq: 7,
              sender: "architect",
              message_type: "message",
              snippet: "İstanbul handoff",
              semantic_score: 0.81,
            },
          ],
        })
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    const { container } = renderApp(["/"])

    fireEvent.change(await screen.findByPlaceholderText(/^Search$/i), {
      target: { value: "istanbul" },
    })

    fireEvent.click(await screen.findByRole("button", { name: /Beta thread/i }))

    await waitFor(() => {
      const highlightedText = Array.from(container.querySelectorAll("mark[data-ab-highlight='true']")).map(
        (element) => element.textContent ?? ""
      )
      expect(highlightedText).toContain("İstanbul")
    })
  })

  test("labels fts-only sidebar results using fts_rank", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/search") {
        return jsonResponse({
          query: url.searchParams.get("q") ?? "",
          mode: "hybrid",
          warnings: [],
          results: [
            {
              topic_id: "t-2",
              topic_name: "Beta thread",
              message_id: "focused-message",
              seq: 7,
              sender: "architect",
              message_type: "message",
              snippet: `beta ${SEARCH_HIGHLIGHT_START}handoff${SEARCH_HIGHLIGHT_END} summary`,
              fts_rank: 0.2,
            },
          ],
        })
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderApp(["/"])

    fireEvent.change(await screen.findByPlaceholderText(/^Search$/i), {
      target: { value: "handoff" },
    })

    expect(await screen.findByText(/^fts$/i)).toBeInTheDocument()
  })

  test("renders search snippets as text instead of attacker-controlled HTML", async () => {
    installFetchMock()

    const { container } = renderApp(["/"])

    fireEvent.change(await screen.findByPlaceholderText(/^Search$/i), {
      target: { value: "handoff" },
    })

    expect(await screen.findByText("handoff")).toBeInTheDocument()
    expect(container.querySelector("img[src='x']")).toBeNull()
    expect(screen.getByText(/<img src=x onerror=alert\(1\)>/i)).toBeInTheDocument()
  })

  test("opens local find with the keyboard shortcut inside the active topic", async () => {
    installFetchMock()

    const { container } = renderApp(["/topics/t-1"])

    expect(await screen.findByText("hello from alpha")).toBeInTheDocument()
    fireEvent.keyDown(window, { key: "f", metaKey: true })
    fireEvent.change(await screen.findByPlaceholderText(/Find in this thread/i), {
      target: { value: "alpha" },
    })

    expect(await screen.findByText("1/1")).toBeInTheDocument()
    await waitFor(() =>
      expect(
        Array.from(container.querySelectorAll("mark[data-ab-highlight='true']")).some(
          (element) => element.textContent === "alpha"
        )
      ).toBe(true)
    )
  })

  test("keeps the inspector in the rail and reveals the thread-map overlay from the hotspot", async () => {
    setDesktopWidth()

    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1") {
        return jsonResponse(
          topicDetailWithMessages("t-1", [
            "alpha section one",
            "alpha section two",
            "alpha section three",
            "alpha section four",
          ])
        )
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderApp(["/topics/t-1"])

    expect(await screen.findByText("alpha section one")).toBeInTheDocument()

    setTopicThreadLayout({
      scrollHeight: 1400,
      clientHeight: 420,
      messageHeights: [220, 260, 180, 240],
    })

    await waitFor(() => expect(screen.getByText("Topic metadata")).toBeInTheDocument())
    expect(screen.queryByRole("tab", { name: "Map" })).not.toBeInTheDocument()
    expect(document.querySelector("[data-ab-thread-map='true']")).toHaveAttribute("data-visible", "false")

    revealThreadMapOverlay()

    await waitFor(() =>
      expect(document.querySelector("[data-ab-thread-map='true']")).toHaveAttribute("data-visible", "true")
    )
  })

  test("auto-hides the thread-map overlay after pointer inactivity", async () => {
    setDesktopWidth()

    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1") {
        return jsonResponse(
          topicDetailWithMessages("t-1", [
            "alpha section one",
            "alpha section two",
            "alpha section three",
            "alpha section four",
          ])
        )
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderApp(["/topics/t-1"])

    expect(await screen.findByText("alpha section one")).toBeInTheDocument()

    setTopicThreadLayout({
      scrollHeight: 1400,
      clientHeight: 420,
      messageHeights: [220, 260, 180, 240],
    })

    await waitFor(() =>
      expect(document.querySelector("[data-ab-thread-map-hotspot='true']")).toBeTruthy()
    )

    vi.useFakeTimers()

    try {
      revealThreadMapOverlay()

      expect(document.querySelector("[data-ab-thread-map='true']")).toHaveAttribute("data-visible", "true")

      await act(async () => {
        vi.advanceTimersByTime(1900)
      })

      expect(document.querySelector("[data-ab-thread-map='true']")).toHaveAttribute("data-visible", "false")
    } finally {
      vi.useRealTimers()
    }
  })

  test("uses stable sender tones for default thread-map markers", async () => {
    setDesktopWidth()

    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1") {
        return jsonResponse(
          topicDetailWithMessages("t-1", [
            "alpha section one",
            "alpha section two",
            "alpha section three",
            "alpha section four",
          ])
        )
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderApp(["/topics/t-1"])

    expect(await screen.findByText("alpha section one")).toBeInTheDocument()

    setTopicThreadLayout({
      scrollHeight: 1400,
      clientHeight: 420,
      messageHeights: [220, 260, 180, 240],
    })

    await waitFor(() => {
      const firstMarker = document.querySelector<HTMLElement>("[data-ab-thread-map-marker='t-1-m-1']")
      const secondMarker = document.querySelector<HTMLElement>("[data-ab-thread-map-marker='t-1-m-2']")
      expect(firstMarker).toHaveAttribute("data-sender-tone")
      expect(secondMarker).toHaveAttribute("data-sender-tone")
      expect(firstMarker?.getAttribute("data-sender-tone")).not.toBe(secondMarker?.getAttribute("data-sender-tone"))
    })
  })

  test("keeps thread-map markers out of the normal tab order", async () => {
    setDesktopWidth()

    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1") {
        return jsonResponse(
          topicDetailWithMessages("t-1", [
            "alpha section one",
            "alpha section two",
            "alpha section three",
          ])
        )
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderApp(["/topics/t-1"])

    expect(await screen.findByText("alpha section one")).toBeInTheDocument()

    setTopicThreadLayout({
      scrollHeight: 1300,
      clientHeight: 360,
      messageHeights: [220, 220, 220],
    })

    await waitFor(() =>
      expect(
        document.querySelector<HTMLElement>("[data-ab-thread-map-marker='t-1-m-1']")
      ).toHaveAttribute("tabindex", "-1")
    )
  })

  test("keeps the inspector-only rail for desktop topics without overflow", async () => {
    setDesktopWidth()

    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1") {
        return jsonResponse(topicDetailWithMessages("t-1", ["short one", "short two"]))
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderApp(["/topics/t-1"])

    expect(await screen.findByText("short one")).toBeInTheDocument()

    setTopicThreadLayout({
      scrollHeight: 400,
      clientHeight: 400,
      messageHeights: [140, 140],
    })

    await waitFor(() => expect(document.querySelector("[data-ab-thread-map='true']")).toBeNull())
    expect(screen.getByText("Topic metadata")).toBeInTheDocument()
  })

  test("clicking a thread-map marker band jumps to its message", async () => {
    setDesktopWidth()

    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1") {
        return jsonResponse(
          topicDetailWithMessages("t-1", ["first jump target", "second jump target", "third jump target"])
        )
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderApp(["/topics/t-1"])

    expect(await screen.findByText("first jump target")).toBeInTheDocument()

    setTopicThreadLayout({
      scrollHeight: 1300,
      clientHeight: 380,
      messageHeights: [200, 220, 240],
    })

    await waitFor(() => expect(document.querySelector("[data-ab-thread-map='true']")).toBeInTheDocument())
    revealThreadMapOverlay()

    const target = document.getElementById("msg-t-1-m-2")
    expect(target).toBeTruthy()
    const scrollSpy = vi.spyOn(target!, "scrollIntoView").mockImplementation(() => {})
    const markerBand = screen.getByRole("button", { name: /Jump to message #2 by architect/i })

    expect(Number.parseFloat(markerBand.style.height)).toBeGreaterThan(10)
    expect(markerBand.style.height).toContain("%")

    fireEvent.click(markerBand)

    expect(scrollSpy).toHaveBeenCalled()
  })

  test("dragging the thread-map viewport scrolls the thread", async () => {
    setDesktopWidth()

    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1") {
        return jsonResponse(
          topicDetailWithMessages("t-1", ["first drag target", "second drag target", "third drag target"])
        )
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderApp(["/topics/t-1"])

    expect(await screen.findByText("first drag target")).toBeInTheDocument()

    setTopicThreadLayout({
      scrollHeight: 1200,
      clientHeight: 300,
      messageHeights: [240, 260, 280],
    })

    await waitFor(() => expect(document.querySelector("[data-ab-thread-map='true']")).toBeInTheDocument())
    revealThreadMapOverlay()

    const mapFrame = document.querySelector<HTMLElement>("[data-ab-thread-map-frame='true']")
    expect(mapFrame).toBeTruthy()
    vi.spyOn(mapFrame!, "getBoundingClientRect").mockReturnValue({
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      bottom: 300,
      right: 100,
      width: 100,
      height: 300,
      toJSON: () => {},
    } as DOMRect)

    const viewport = document.querySelector<HTMLElement>(
      "[data-ab-topic-thread-scroll-area='true'] [data-slot='scroll-area-viewport']"
    )
    expect(viewport).toBeTruthy()
    expect(viewport!.scrollTop).toBe(0)

    const cursor = document.querySelector<HTMLElement>("[data-ab-thread-map-viewport='true']")
    expect(cursor).toBeTruthy()
    cursor!.setPointerCapture = vi.fn()
    cursor!.releasePointerCapture = vi.fn()

    fireEvent.pointerDown(cursor!, { button: 0, pointerId: 1, clientY: 15 })
    fireEvent.pointerMove(cursor!, { pointerId: 1, clientY: 165 })
    fireEvent.pointerUp(cursor!, { pointerId: 1, clientY: 165 })

    expect(viewport!.scrollTop).toBeGreaterThan(0)
  })

  test("updates thread-map marker states from local thread find", async () => {
    setDesktopWidth()

    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1") {
        return jsonResponse(
          topicDetailWithMessages("t-1", [
            "alpha one",
            "alpha two",
            "plain context",
          ])
        )
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderApp(["/topics/t-1"])

    expect(await screen.findByText("alpha one")).toBeInTheDocument()

    setTopicThreadLayout({
      scrollHeight: 1200,
      clientHeight: 360,
      messageHeights: [190, 190, 190],
    })

    await waitFor(() => expect(document.querySelector("[data-ab-thread-map='true']")).toBeInTheDocument())

    fireEvent.keyDown(window, { key: "f", metaKey: true })
    fireEvent.change(await screen.findByPlaceholderText(/Find in this thread/i), {
      target: { value: "alpha" },
    })

    setTopicThreadLayout({
      scrollHeight: 1200,
      clientHeight: 360,
      messageHeights: [190, 190, 190],
    })

    const firstMarker = document.querySelector<HTMLElement>("[data-ab-thread-map-marker='t-1-m-1']")
    const secondMarker = document.querySelector<HTMLElement>("[data-ab-thread-map-marker='t-1-m-2']")

    await waitFor(() => {
      expect(firstMarker).toHaveAttribute("data-local-matched", "true")
      expect(firstMarker).toHaveAttribute("data-local-active", "true")
      expect(secondMarker).toHaveAttribute("data-local-matched", "true")
      expect(secondMarker).toHaveAttribute("data-local-active", "false")
    })

    fireEvent.click(screen.getByRole("button", { name: "Next" }))

    await waitFor(() => {
      expect(firstMarker).toHaveAttribute("data-local-active", "false")
      expect(secondMarker).toHaveAttribute("data-local-active", "true")
    })
  })

  test("shows the focused-message marker for route-driven focus navigation", async () => {
    setDesktopWidth()

    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1" && url.searchParams.get("focus") === "focused-message") {
        return jsonResponse(
          topicDetailWithMessages(
            "t-1",
            ["focused detail", "background context"],
            {
              focusMessageId: "focused-message",
              messageIds: ["focused-message", "t-1-m-2"],
            }
          )
        )
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderApp(["/topics/t-1?focus=focused-message"])

    expect(await screen.findByText("focused detail")).toBeInTheDocument()

    setTopicThreadLayout({
      scrollHeight: 1100,
      clientHeight: 320,
      messageHeights: [210, 210],
    })

    await waitFor(() =>
      expect(
        document.querySelector<HTMLElement>("[data-ab-thread-map-marker='focused-message']")
      ).toHaveAttribute("data-focused", "true")
    )
  })

  test("recomputes the thread map after loading earlier messages without losing the selected rail", async () => {
    setDesktopWidth()

    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1") {
        return jsonResponse(
          topicDetailWithMessages("t-1", ["later one", "later two"], {
            hasEarlier: true,
            startSeq: 101,
          })
        )
      }
      if (url.pathname === "/api/topics/t-1/messages" && url.searchParams.get("before_seq") === "101") {
        const earlier = topicDetailWithMessages("t-1", ["earlier one", "earlier two"], {
          startSeq: 1,
        }).messages

        return jsonResponse({
          messages: earlier,
          first_seq: earlier[0]?.seq ?? null,
          last_seq: earlier[earlier.length - 1]?.seq ?? null,
          has_earlier: false,
        })
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderApp(["/topics/t-1"])

    expect(await screen.findByText("later one")).toBeInTheDocument()

    setTopicThreadLayout({
      scrollHeight: 1200,
      clientHeight: 360,
      messageHeights: [260, 260],
    })

    await waitFor(() => expect(document.querySelector("[data-ab-thread-map='true']")).toBeInTheDocument())
    expect(screen.getByText("Topic metadata")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /Load earlier/i }))
    expect(await screen.findByText("earlier one")).toBeInTheDocument()

    setTopicThreadLayout({
      scrollHeight: 1800,
      clientHeight: 360,
      messageHeights: [200, 200, 260, 260],
    })

    await waitFor(() =>
      expect(document.querySelectorAll("[data-ab-thread-map-marker]")).toHaveLength(4)
    )
  })

  test("does not recreate thread-map observers on presence-only stream updates", async () => {
    setDesktopWidth()

    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1") {
        return jsonResponse(
          topicDetailWithMessages("t-1", [
            "alpha section one",
            "alpha section two",
            "alpha section three",
          ])
        )
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderApp(["/topics/t-1"])

    expect(await screen.findByText("alpha section one")).toBeInTheDocument()

    setTopicThreadLayout({
      scrollHeight: 1300,
      clientHeight: 360,
      messageHeights: [220, 220, 220],
    })

    const ResizeObserverCtor = globalThis.ResizeObserver as unknown as {
      instances: Array<unknown>
    }

    await waitFor(() => expect(document.querySelector("[data-ab-thread-map='true']")).toBeInTheDocument())
    const initialObserverCount = ResizeObserverCtor.instances.length

    topicStream("t-1").emit("topic.update", {
      topic_id: "t-1",
      last_seq: 3,
      message_count: 3,
      presence: [
        {
          topic_id: "t-1",
          agent_name: "presence probe",
          last_seq: 3,
          updated_at: 1_700_000_330,
        },
      ],
    })

    await waitFor(() => expect(screen.getAllByText("presence probe").length).toBeGreaterThan(0))

    await waitFor(() =>
      expect(ResizeObserverCtor.instances.length).toBe(initialObserverCount)
    )
  })

  test("focused-message navigation does not suppress the first later local-find scroll", async () => {
    const scrollIntoViewSpy = vi.spyOn(Element.prototype, "scrollIntoView").mockImplementation(() => {})
    let detailRequestCount = 0
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1" && url.searchParams.get("focus") === "focused-message") {
        return jsonResponse(topicDetail("t-1", "focused detail", { focus: "focused-message" }))
      }
      if (url.pathname === "/api/topics/t-1") {
        detailRequestCount += 1
        if (detailRequestCount === 1) {
          return jsonResponse(topicDetail("t-1", "hello from alpha"))
        }
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderAppWithControls(["/topics/t-1"])

    expect(await screen.findByText("hello from alpha")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Focus same topic" }))
    expect(await screen.findByText("focused detail")).toBeInTheDocument()

    scrollIntoViewSpy.mockClear()

    fireEvent.keyDown(window, { key: "f", metaKey: true })
    fireEvent.change(await screen.findByPlaceholderText(/Find in this thread/i), {
      target: { value: "focused" },
    })

    expect(await screen.findByText("1/1")).toBeInTheDocument()
    await waitFor(() => expect(scrollIntoViewSpy).toHaveBeenCalled())
  })

  test("focuses the sidebar search with the keyboard shortcut", async () => {
    installFetchMock()

    renderApp(["/"])

    const searchInput = await screen.findByPlaceholderText(/^Search$/i)
    fireEvent.keyDown(window, { key: "k", metaKey: true })

    await waitFor(() => expect(searchInput).toHaveFocus())
  })

  test("sanitizes invalid persisted workbench state fields", () => {
    window.localStorage.setItem(
      "agent-bus.workbench.v1",
      JSON.stringify({
        openTopicIds: ["t-1", 7, null],
        activeTopicId: 42,
        sidebarQuery: ["bad"],
        sidebarStatus: "bogus",
        sidebarSort: "bogus",
      })
    )

    expect(loadWorkbenchState()).toEqual({
      ...DEFAULT_WORKBENCH_STATE,
      openTopicIds: ["t-1"],
    })
  })

  test("refetches full topic detail when a stream update moves backward", async () => {
    let currentDetail: TopicDetailResponse = topicDetail("t-1", "hello from alpha")
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1") {
        return jsonResponse(currentDetail)
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderApp(["/topics/t-1"])

    expect(await screen.findByText("hello from alpha")).toBeInTheDocument()

    currentDetail = {
      ...currentDetail,
      messages: [],
      message_count: 0,
      first_seq: null,
      last_seq: null,
      topic: {
        ...currentDetail.topic,
        message_count: 0,
        last_seq: 0,
      },
      presence: [
        {
          topic_id: "t-1",
          agent_name: "remote reviewer",
          last_seq: 0,
          updated_at: 1_700_000_220,
        },
      ],
    }

    topicStream("t-1").emit("topic.update", {
      topic_id: "t-1",
      last_seq: 0,
      message_count: 0,
      presence: currentDetail.presence,
    })

    await waitFor(() => expect(screen.queryByText("hello from alpha")).not.toBeInTheDocument())
    await waitFor(() => expect(screen.getAllByText("remote reviewer").length).toBeGreaterThan(0))
    expect(fetchSpy).toHaveBeenCalledWith("/api/topics/t-1", expect.any(Object))
  })

  test("keeps presence in sync even when append refresh fails", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1") {
        return jsonResponse(topicDetail("t-1", "hello from alpha"))
      }
      if (url.pathname === "/api/topics/t-1/messages") {
        return Promise.reject(new Error("stream append fetch failed"))
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderApp(["/topics/t-1"])

    expect(await screen.findByText("hello from alpha")).toBeInTheDocument()

    topicStream("t-1").emit("topic.update", {
      topic_id: "t-1",
      last_seq: 2,
      message_count: 2,
      presence: [
        {
          topic_id: "t-1",
          agent_name: "append probe",
          last_seq: 2,
          updated_at: 1_700_000_330,
        },
      ],
    })

    await waitFor(() => expect(screen.getAllByText("append probe").length).toBeGreaterThan(0))
    expect(screen.getByText("hello from alpha")).toBeInTheDocument()
  })

  test("drains multiple append pages for one stream update", async () => {
    const batchOne = Array.from({ length: 200 }, (_, index) => ({
      message_id: `t-1-m-${index + 2}`,
      topic_id: "t-1",
      seq: index + 2,
      sender: "reviewer",
      message_type: "message",
      reply_to: null,
      reply_to_sender: null,
      content_markdown: `message ${index + 2}`,
      metadata: null,
      client_message_id: null,
      created_at: 1_700_000_200 + index,
    }))
    const batchTwo = Array.from({ length: 4 }, (_, index) => ({
      message_id: `t-1-m-${index + 202}`,
      topic_id: "t-1",
      seq: index + 202,
      sender: "reviewer",
      message_type: "message",
      reply_to: null,
      reply_to_sender: null,
      content_markdown: `message ${index + 202}`,
      metadata: null,
      client_message_id: null,
      created_at: 1_700_000_500 + index,
    }))

    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1") {
        return jsonResponse(topicDetail("t-1", "hello from alpha"))
      }
      if (url.pathname === "/api/topics/t-1/messages" && url.searchParams.get("after_seq") === "1") {
        return jsonResponse({
          messages: batchOne,
          first_seq: 2,
          last_seq: 201,
          has_earlier: false,
        })
      }
      if (url.pathname === "/api/topics/t-1/messages" && url.searchParams.get("after_seq") === "201") {
        return jsonResponse({
          messages: batchTwo,
          first_seq: 202,
          last_seq: 205,
          has_earlier: false,
        })
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderApp(["/topics/t-1"])

    expect(await screen.findByText("hello from alpha")).toBeInTheDocument()

    topicStream("t-1").emit("topic.update", {
      topic_id: "t-1",
      last_seq: 205,
      message_count: 205,
      presence: [
        {
          topic_id: "t-1",
          agent_name: "bulk sender",
          last_seq: 205,
          updated_at: 1_700_000_440,
        },
      ],
    })

    expect(await screen.findByText("message 205")).toBeInTheDocument()
    expect(
      fetchSpy.mock.calls.filter(([input]) =>
        String(input).includes("/api/topics/t-1/messages")
      )
    ).toHaveLength(2)
  })

  test("falls back to full topic detail when append catch-up exceeds the page cap", async () => {
    const appendBatches = new Map(
      [1, 201, 401, 601, 801].map((afterSeq) => [
        String(afterSeq),
        Array.from({ length: 200 }, (_, index) => {
          const seq = afterSeq + index + 1
          return {
            message_id: `t-1-m-${seq}`,
            topic_id: "t-1",
            seq,
            sender: "reviewer",
            message_type: "message",
            reply_to: null,
            reply_to_sender: null,
            content_markdown: `message ${seq}`,
            metadata: null,
            client_message_id: null,
            created_at: 1_700_001_000 + seq,
          }
        }),
      ])
    )

    let currentDetail: TopicDetailResponse = topicDetail("t-1", "hello from alpha")
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1") {
        return jsonResponse(currentDetail)
      }
      if (url.pathname === "/api/topics/t-1/messages") {
        const afterSeq = url.searchParams.get("after_seq")
        const messages = afterSeq ? appendBatches.get(afterSeq) : null

        if (messages) {
          return jsonResponse({
            messages,
            first_seq: messages[0]?.seq ?? null,
            last_seq: messages[messages.length - 1]?.seq ?? null,
            has_earlier: false,
          })
        }
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderApp(["/topics/t-1"])

    expect(await screen.findByText("hello from alpha")).toBeInTheDocument()

    currentDetail = {
      ...currentDetail,
      messages: [
        {
          message_id: "t-1-m-1205",
          topic_id: "t-1",
          seq: 1205,
          sender: "reviewer",
          message_type: "message",
          reply_to: null,
          reply_to_sender: null,
          content_markdown: "message 1205",
          metadata: null,
          client_message_id: null,
          created_at: 1_700_002_205,
        },
      ],
      message_count: 1205,
      first_seq: 1205,
      last_seq: 1205,
      topic: {
        ...currentDetail.topic,
        message_count: 1205,
        last_seq: 1205,
        last_message_at: 1_700_002_205,
        last_updated_at: 1_700_002_205,
      },
    }

    topicStream("t-1").emit("topic.update", {
      topic_id: "t-1",
      last_seq: 1205,
      message_count: 1205,
      presence: [
        {
          topic_id: "t-1",
          agent_name: "bulk sender",
          last_seq: 1205,
          updated_at: 1_700_002_210,
        },
      ],
    })

    expect(await screen.findByText("message 1205")).toBeInTheDocument()
    expect(
      fetchSpy.mock.calls.filter(([input]) =>
        String(input).includes("/api/topics/t-1/messages")
      )
    ).toHaveLength(5)
    expect(
      fetchSpy.mock.calls.filter(([input]) => String(input) === "/api/topics/t-1")
    ).toHaveLength(2)
  })

  test("refetches full topic detail when last_seq advances without message_count growth", async () => {
    let currentDetail: TopicDetailResponse = topicDetail("t-1", "hello from alpha")
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1") {
        return jsonResponse(currentDetail)
      }
      if (url.pathname === "/api/topics/t-1/messages") {
        return jsonResponse({
          messages: [],
          first_seq: null,
          last_seq: null,
          has_earlier: false,
        })
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderApp(["/topics/t-1"])

    expect(await screen.findByText("hello from alpha")).toBeInTheDocument()

    currentDetail = {
      ...currentDetail,
      messages: [
        {
          message_id: "t-1-m-2",
          topic_id: "t-1",
          seq: 2,
          sender: "reviewer",
          message_type: "message",
          reply_to: null,
          reply_to_sender: null,
          content_markdown: "replacement message",
          metadata: null,
          client_message_id: null,
          created_at: 1_700_000_200,
        },
      ],
      first_seq: 2,
      last_seq: 2,
      message_count: 1,
      topic: {
        ...currentDetail.topic,
        last_seq: 2,
        message_count: 1,
        last_message_at: 1_700_000_200,
        last_updated_at: 1_700_000_200,
      },
    }

    topicStream("t-1").emit("topic.update", {
      topic_id: "t-1",
      last_seq: 2,
      message_count: 1,
      presence: [
        {
          topic_id: "t-1",
          agent_name: "replacement reviewer",
          last_seq: 2,
          updated_at: 1_700_000_210,
        },
      ],
    })

    expect(await screen.findByText("replacement message")).toBeInTheDocument()
    await waitFor(() => expect(screen.queryByText("hello from alpha")).not.toBeInTheDocument())
    expect(
      fetchSpy.mock.calls.filter(([input]) =>
        String(input).includes("/api/topics/t-1/messages")
      )
    ).toHaveLength(0)
  })

  test("ignores stale refresh results after a newer topic update lands", async () => {
    let detailRequestCount = 0
    let resolveStaleRefresh!: (response: Response) => void
    const staleRefresh = new Promise<Response>((resolve) => {
      resolveStaleRefresh = resolve
    })

    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1") {
        detailRequestCount += 1
        if (detailRequestCount === 1) {
          return jsonResponse(topicDetail("t-1", "hello from alpha"))
        }
        if (detailRequestCount === 2) {
          return staleRefresh
        }
      }
      if (url.pathname === "/api/topics/t-1/messages" && url.searchParams.get("after_seq") === "1") {
        return jsonResponse({
          messages: [
            {
              message_id: "t-1-m-2",
              topic_id: "t-1",
              seq: 2,
              sender: "reviewer",
              message_type: "message",
              reply_to: null,
              reply_to_sender: null,
              content_markdown: "message 2",
              metadata: null,
              client_message_id: null,
              created_at: 1_700_000_220,
            },
          ],
          first_seq: 2,
          last_seq: 2,
          has_earlier: false,
        })
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderApp(["/topics/t-1"])

    expect(await screen.findByText("hello from alpha")).toBeInTheDocument()

    topicStream("t-1").emit("topic.update", {
      topic_id: "t-1",
      last_seq: 0,
      message_count: 0,
      presence: [
        {
          topic_id: "t-1",
          agent_name: "stale reviewer",
          last_seq: 0,
          updated_at: 1_700_000_230,
        },
      ],
    })

    topicStream("t-1").emit("topic.update", {
      topic_id: "t-1",
      last_seq: 2,
      message_count: 2,
      presence: [
        {
          topic_id: "t-1",
          agent_name: "fresh reviewer",
          last_seq: 2,
          updated_at: 1_700_000_240,
        },
      ],
    })

    expect(await screen.findByText("message 2")).toBeInTheDocument()

    resolveStaleRefresh(
      new Response(JSON.stringify(topicDetail("t-1", "stale detail")), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    )

    await waitFor(() => expect(screen.queryByText("stale detail")).not.toBeInTheDocument())
    expect(screen.getByText("message 2")).toBeInTheDocument()
    expect(
      fetchSpy.mock.calls.filter(([input]) =>
        String(input).includes("/api/topics/t-1/messages")
      )
    ).toHaveLength(1)
  })

  test("ignores stale stream refreshes after the topic focus changes", async () => {
    let detailRequestCount = 0
    let resolveStaleRefresh!: (response: Response) => void
    const staleRefresh = new Promise<Response>((resolve) => {
      resolveStaleRefresh = resolve
    })

    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = new URL(String(input), "http://localhost")

      if (url.pathname === "/api/topics") {
        return jsonResponse(topicsPayload)
      }
      if (url.pathname === "/api/topics/t-1" && url.searchParams.get("focus") === "focused-message") {
        return jsonResponse(topicDetail("t-1", "focused detail", { focus: "focused-message" }))
      }
      if (url.pathname === "/api/topics/t-1") {
        detailRequestCount += 1
        if (detailRequestCount === 1) {
          return jsonResponse(topicDetail("t-1", "hello from alpha"))
        }
        if (detailRequestCount === 2) {
          return staleRefresh
        }
      }

      throw new Error(`Unhandled fetch ${url.pathname}${url.search}`)
    })

    renderAppWithControls(["/topics/t-1"])

    expect(await screen.findByText("hello from alpha")).toBeInTheDocument()

    topicStream("t-1").emit("topic.update", {
      topic_id: "t-1",
      last_seq: 0,
      message_count: 0,
      presence: [
        {
          topic_id: "t-1",
          agent_name: "stale reviewer",
          last_seq: 0,
          updated_at: 1_700_000_250,
        },
      ],
    })

    fireEvent.click(screen.getByRole("button", { name: "Focus same topic" }))

    expect(await screen.findByText("focused detail")).toBeInTheDocument()

    resolveStaleRefresh(
      new Response(JSON.stringify(topicDetail("t-1", "stale detail")), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    )

    await waitFor(() => expect(screen.queryByText("stale detail")).not.toBeInTheDocument())
    expect(screen.getByText("focused detail")).toBeInTheDocument()
  })
})
