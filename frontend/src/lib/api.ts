import type {
  SearchMode,
  SearchResponse,
  TopicDetailResponse,
  TopicMessagesResponse,
  TopicSort,
  TopicStatusFilter,
  TopicSummary,
} from "@/lib/types"

const API_BASE = "/api"

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string }
    if (body.detail) {
      return body.detail
    }
  } catch {
    // Fall through to text.
  }

  const text = await response.text()
  return text || `${response.status} ${response.statusText}`
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  })

  if (!response.ok) {
    throw new Error(await readError(response))
  }

  return (await response.json()) as T
}

export async function fetchTopics(params: {
  status: TopicStatusFilter
  sort: TopicSort
  query: string
  limit?: number
}): Promise<TopicSummary[]> {
  const search = new URLSearchParams({
    status: params.status,
    sort: params.sort,
    q: params.query,
    limit: String(params.limit ?? 200),
  })
  const payload = await fetchJson<{ topics: TopicSummary[] }>(`/topics?${search.toString()}`)
  return payload.topics
}

export async function fetchTopicDetail(
  topicId: string,
  focusMessageId?: string | null
): Promise<TopicDetailResponse> {
  const search = new URLSearchParams()
  if (focusMessageId) {
    search.set("focus", focusMessageId)
  }
  const suffix = search.size > 0 ? `?${search.toString()}` : ""
  return fetchJson<TopicDetailResponse>(`/topics/${topicId}${suffix}`)
}

export async function fetchTopicMessages(
  topicId: string,
  params: {
    afterSeq?: number
    beforeSeq?: number
    limit?: number
  }
): Promise<TopicMessagesResponse> {
  const search = new URLSearchParams()
  if (params.afterSeq !== undefined) {
    search.set("after_seq", String(params.afterSeq))
  }
  if (params.beforeSeq !== undefined) {
    search.set("before_seq", String(params.beforeSeq))
  }
  if (params.limit !== undefined) {
    search.set("limit", String(params.limit))
  }
  return fetchJson<TopicMessagesResponse>(`/topics/${topicId}/messages?${search.toString()}`)
}

export async function fetchSearchResults(params: {
  query: string
  mode: SearchMode
  limit?: number
  topicId?: string
}): Promise<SearchResponse> {
  const search = new URLSearchParams({
    q: params.query,
    mode: params.mode,
    limit: String(params.limit ?? 20),
  })
  const path = params.topicId
    ? `/topics/${params.topicId}/search?${search.toString()}`
    : `/search?${search.toString()}`
  return fetchJson<SearchResponse>(path)
}

export async function deleteTopic(topicId: string): Promise<void> {
  await fetchJson(`/topics/${topicId}`, { method: "DELETE" })
}

export async function deleteMessages(topicId: string, messageIds: string[]): Promise<void> {
  await fetchJson(`/topics/${topicId}/messages`, {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message_ids: messageIds }),
  })
}

export function exportTopicUrl(topicId: string): string {
  return `${API_BASE}/topics/${topicId}/export`
}
