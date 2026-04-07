import type { WorkbenchState } from "@/lib/types"

const STORAGE_KEY = "agent-bus.workbench.v1"
const SIDEBAR_STATUS_VALUES = new Set<WorkbenchState["sidebarStatus"]>(["all", "open", "closed"])
const SIDEBAR_SORT_VALUES = new Set<WorkbenchState["sidebarSort"]>([
  "last_updated_desc",
  "created_desc",
  "created_asc",
])

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
      activeTopicId: typeof parsed.activeTopicId === "string" ? parsed.activeTopicId : null,
      sidebarQuery: typeof parsed.sidebarQuery === "string" ? parsed.sidebarQuery : "",
      sidebarStatus: SIDEBAR_STATUS_VALUES.has(parsed.sidebarStatus ?? "all")
        ? parsed.sidebarStatus!
        : DEFAULT_WORKBENCH_STATE.sidebarStatus,
      sidebarSort: SIDEBAR_SORT_VALUES.has(parsed.sidebarSort ?? "last_updated_desc")
        ? parsed.sidebarSort!
        : DEFAULT_WORKBENCH_STATE.sidebarSort,
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
