export type LLMWikiSection = 'sources' | 'concepts'

export interface LLMWikiContributingSource {
  id: string
  title: string
  path: string
  source?: string | null
}

export interface LLMWikiManifestItem {
  type: 'source' | 'concept'
  title: string
  path: string
  slug: string
  id?: string | null
  file_id?: string | null
  source?: string | null
  chunk_count?: number | null
  reference_count?: number | null
  contributing_sources?: LLMWikiContributingSource[] | null
}

export interface LLMWikiIndexResponse {
  collection_id: string
  status: 'available'
  generated_at?: string | null
  index_markdown: string
  sources: LLMWikiManifestItem[]
  concepts: LLMWikiManifestItem[]
}

export interface LLMWikiPageResponse {
  collection_id: string
  section: LLMWikiSection
  slug: string
  title: string
  path: string
  markdown: string
}
