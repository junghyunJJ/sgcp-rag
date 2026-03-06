export interface SearchResult {
  id: string
  page_content: string
  metadata: {
    source?: string
    file_id?: string
    [key: string]: any
  }
  score: number
}

export interface AgenticSearchParams {
  question: string
  search_type?: 'semantic' | 'keyword' | 'hybrid'
  search_limit?: number
  max_rewrites?: number
  filter?: Record<string, any>
}

export interface AgenticSearchResult {
  generation: string
  relevant_documents: Array<{
    page_content: string
    metadata: Record<string, any>
  }>
  steps: string[]
  query_rewrites: string[]
  rewrite_count: number
  error: string | null
}
