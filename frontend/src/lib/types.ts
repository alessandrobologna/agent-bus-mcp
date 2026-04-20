export type TopicStatus = "open" | "closed"
export type TopicStatusFilter = TopicStatus | "all"
export type TopicSort = "last_updated_desc" | "created_desc" | "created_asc"
export type SearchMode = "fts" | "semantic" | "hybrid"

export type JsonMap = Record<string, unknown>

export interface TopicSummary {
  topic_id: string
  name: string
  status: TopicStatus
  created_at: number
  closed_at: number | null
  close_reason: string | null
  metadata: JsonMap | null
  message_count: number
  last_seq: number
  last_message_at: number | null
  last_updated_at: number
}

export interface CursorPresence {
  topic_id: string
  agent_name: string
  last_seq: number
  updated_at: number
}

export interface TopicMessage {
  message_id: string
  topic_id: string
  seq: number
  sender: string
  message_type: string
  reply_to: string | null
  reply_to_sender: string | null
  content_markdown: string
  metadata: JsonMap | null
  client_message_id: string | null
  created_at: number
}

export interface TopicDetailResponse {
  topic: TopicSummary
  messages: TopicMessage[]
  message_count: number
  first_seq: number | null
  last_seq: number | null
  has_earlier: boolean
  context_mode: boolean
  focus_message_id: string | null
  presence: CursorPresence[]
}

export interface TopicMessagesResponse {
  messages: TopicMessage[]
  first_seq: number | null
  last_seq: number | null
  has_earlier: boolean
}

export interface SearchResult {
  topic_id: string
  topic_name: string
  message_id: string
  seq: number
  sender: string
  message_type: string
  created_at?: number
  snippet: string
  rank?: number | null
  fts_rank?: number | null
  semantic_score?: number | null
  content_markdown?: string | null
}

export interface SearchResponse {
  query: string
  mode: SearchMode
  warnings: string[]
  results: SearchResult[]
  topic_id?: string
}

export interface TopicStreamUpdate {
  topic_id: string
  last_seq: number
  message_count: number
  presence: CursorPresence[]
}

export interface WorkbenchState {
  openTopicIds: string[]
  activeTopicId: string | null
  sidebarQuery: string
  sidebarStatus: TopicStatusFilter
  sidebarSort: TopicSort
}
