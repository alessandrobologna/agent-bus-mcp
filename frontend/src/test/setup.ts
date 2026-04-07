import "@testing-library/jest-dom/vitest"
import { afterEach, beforeEach, vi } from "vitest"
import { cleanup } from "@testing-library/react"

type EventListenerMap = Map<string, Set<EventListenerOrEventListenerObject>>

class MockEventSource {
  listeners: EventListenerMap = new Map()
  onmessage: ((event: MessageEvent<string>) => void) | null = null
  url: string

  constructor(url: string) {
    this.url = url
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
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})
