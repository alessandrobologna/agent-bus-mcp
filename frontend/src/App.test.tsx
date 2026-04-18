import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, test, vi } from "vitest"

import App from "@/App"
import { TooltipProvider } from "@/components/ui/tooltip"
import type { TopicDetailResponse, TopicSummary } from "@/lib/types"
import { DEFAULT_WORKBENCH_STATE, loadWorkbenchState } from "@/lib/workbench-state"

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
      message_count: 2,
      last_seq: 2,
      last_message_at: 1_700_000_120,
      last_updated_at: 1_700_000_120,
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
            snippet: '<img src=x onerror=alert(1)> [handoff] summary',
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

    expect(await screen.findByText(/Agent Bus Workbench/i)).toBeInTheDocument()
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

    renderApp(["/topics/t-1"])

    expect(await screen.findByText("hello from alpha")).toBeInTheDocument()
    fireEvent.keyDown(window, { key: "f", metaKey: true })
    fireEvent.change(await screen.findByPlaceholderText(/Find in this thread/i), {
      target: { value: "alpha" },
    })

    expect(await screen.findByText("1/1")).toBeInTheDocument()
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
})
