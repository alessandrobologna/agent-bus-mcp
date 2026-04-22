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

class MockResizeObserver {
  static instances: MockResizeObserver[] = []
  callback: ResizeObserverCallback

  constructor(callback: ResizeObserverCallback) {
    this.callback = callback
    MockResizeObserver.instances.push(this)
  }

  observe() {}

  unobserve() {}

  disconnect() {}

  trigger() {
    this.callback([], this as unknown as ResizeObserver)
  }
}

type MockMediaQueryList = {
  media: string
  matches: boolean
  onchange: ((event: MediaQueryListEvent) => void) | null
  addEventListener: (type: string, listener: EventListenerOrEventListenerObject) => void
  removeEventListener: (type: string, listener: EventListenerOrEventListenerObject) => void
  addListener: (listener: EventListenerOrEventListenerObject) => void
  removeListener: (listener: EventListenerOrEventListenerObject) => void
  dispatchEvent: (event: Event) => boolean
  _listeners: Set<EventListenerOrEventListenerObject>
}

function matchesMediaQuery(query: string): boolean {
  const min = query.match(/min-width:\s*(\d+)px/)
  if (min && window.innerWidth < Number(min[1])) {
    return false
  }

  const max = query.match(/max-width:\s*(\d+)px/)
  if (max && window.innerWidth > Number(max[1])) {
    return false
  }

  return true
}

const mediaQueries = new Set<MockMediaQueryList>()

Object.defineProperty(globalThis, "EventSource", {
  writable: true,
  value: MockEventSource,
})

Object.defineProperty(globalThis, "ResizeObserver", {
  writable: true,
  value: MockResizeObserver,
})

Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string): MediaQueryList => {
    const mql: MockMediaQueryList = {
      media: query,
      matches: matchesMediaQuery(query),
      onchange: null,
      _listeners: new Set(),
      addEventListener(type, listener) {
        if (type === "change") {
          this._listeners.add(listener)
        }
      },
      removeEventListener(type, listener) {
        if (type === "change") {
          this._listeners.delete(listener)
        }
      },
      addListener(listener) {
        this._listeners.add(listener)
      },
      removeListener(listener) {
        this._listeners.delete(listener)
      },
      dispatchEvent(event) {
        for (const listener of this._listeners) {
          if (typeof listener === "function") {
            listener(event)
          } else {
            listener.handleEvent(event)
          }
        }
        return true
      },
    }

    mediaQueries.add(mql)
    return mql as unknown as MediaQueryList
  },
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
  MockResizeObserver.instances = []
  mediaQueries.clear()
  window.innerWidth = 1024
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

window.addEventListener("resize", () => {
  for (const mediaQuery of mediaQueries) {
    const nextMatches = matchesMediaQuery(mediaQuery.media)
    mediaQuery.matches = nextMatches
    const event = { matches: nextMatches, media: mediaQuery.media } as MediaQueryListEvent
    mediaQuery.onchange?.(event)
    mediaQuery.dispatchEvent(event as unknown as Event)
  }
})
