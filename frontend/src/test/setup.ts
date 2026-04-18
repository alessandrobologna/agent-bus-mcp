import "@testing-library/jest-dom/vitest"
import { afterEach, beforeEach, vi } from "vitest"
import { cleanup } from "@testing-library/react"

type EventListenerMap = Map<string, Set<EventListenerOrEventListenerObject>>

class MockEventSource {
  static instances: MockEventSource[] = []
  listeners: EventListenerMap = new Map()
  onmessage: ((event: MessageEvent<string>) => void) | null = null
  url: string

  constructor(url: string) {
    this.url = url
    MockEventSource.instances.push(this)
  }

  addEventListener(type: string, listener: EventListenerOrEventListenerObject) {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, new Set())
    }
    this.listeners.get(type)?.add(listener)
  }

  removeEventListener(type: string, listener: EventListenerOrEventListenerObject) {
    this.listeners.get(type)?.delete(listener)
  }

  close() {}

  emit(type: string, payload?: unknown) {
    const event = payload === undefined
      ? ({} as Event)
      : ({ data: JSON.stringify(payload) } as MessageEvent<string>)

    for (const listener of this.listeners.get(type) ?? []) {
      if (typeof listener === "function") {
        listener(event)
      } else {
        listener.handleEvent(event)
      }
    }

    if (type === "message" && payload !== undefined) {
      this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent<string>)
    }
  }
}

Object.defineProperty(globalThis, "EventSource", {
  writable: true,
  value: MockEventSource,
})

const storage = new Map<string, string>()

Object.defineProperty(window, "localStorage", {
  writable: true,
  value: {
    getItem(key: string) {
      return storage.has(key) ? storage.get(key)! : null
    },
    setItem(key: string, value: string) {
      storage.set(key, value)
    },
    removeItem(key: string) {
      storage.delete(key)
    },
    clear() {
      storage.clear()
    },
  },
})

beforeEach(() => {
  window.localStorage.clear()
  MockEventSource.instances = []
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})
