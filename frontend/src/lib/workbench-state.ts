import type { WorkbenchState } from "@/lib/types"

const STORAGE_KEY = "agent-bus.workbench.v1"

export const DEFAULT_WORKBENCH_STATE: WorkbenchState = {
  openTopicIds: [],
  activeTopicId: null,
  sidebarQuery: "",
  sidebarStatus: "all",
  sidebarSort: "last_updated_desc",
}

export function loadWorkbenchState(): WorkbenchState {
  if (typeof window === "undefined") {
    return DEFAULT_WORKBENCH_STATE
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) {
      return DEFAULT_WORKBENCH_STATE
    }

    const parsed = JSON.parse(raw) as Partial<WorkbenchState>
    return {
      ...DEFAULT_WORKBENCH_STATE,
      ...parsed,
      openTopicIds: Array.isArray(parsed.openTopicIds)
        ? parsed.openTopicIds.filter((value): value is string => typeof value === "string")
        : [],
    }
  } catch {
    return DEFAULT_WORKBENCH_STATE
  }
}

export function saveWorkbenchState(state: WorkbenchState): void {
  if (typeof window === "undefined") {
    return
  }

  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
}
