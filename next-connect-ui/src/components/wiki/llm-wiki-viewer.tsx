'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import {
  ArrowLeft,
  BookOpen,
  FileText,
  Lightbulb,
  Loader2,
  RefreshCw,
} from 'lucide-react'
import { toast } from 'sonner'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useTranslation } from '@/hooks/use-translation'
import type {
  LLMWikiIndexResponse,
  LLMWikiManifestItem,
  LLMWikiPageResponse,
  LLMWikiSection,
} from '@/types/llm-wiki'

interface LLMWikiViewerProps {
  collectionId: string
}

type ActivePage =
  | { kind: 'index' }
  | { kind: 'page'; section: LLMWikiSection; slug: string }

type WikiStatus = 'loading' | 'available' | 'missing' | 'error'

interface CollectionPayload {
  uuid: string
  name?: string
}

interface StructuredPageMarkdown {
  title: string
  notice: string
  summary: string
  keywords: string[]
  references: string[]
  contributingSources: ContributingSource[]
}

interface StructuredIndexItem {
  summary: string
  keywords: string[]
  detail: string
}

interface StructuredIndexMarkdown {
  title: string
  generatedAt: string
  notice: string
  itemDetails: Map<string, StructuredIndexItem>
}

interface SourceReference {
  fileId: string
  chunkId: string
  label: string
}

interface ContributingSource {
  title: string
  path: string
  detail: string
  section: LLMWikiSection
  slug: string
}

interface ChunkDetail {
  id: string
  content?: string | null
  page_content?: string | null
  metadata?: Record<string, unknown> | null
  collection_id?: string | null
}

async function readPayload(response: Response) {
  try {
    return await response.json()
  } catch {
    return null
  }
}

function wikiLinkTarget(href?: string): { section: LLMWikiSection; slug: string } | null {
  if (!href) return null
  const clean = href.split('#')[0].split('?')[0].replace(/^\.\//, '')
  const match = clean.match(/^(sources|concepts)\/([a-z0-9][a-z0-9-]*)\.md$/)
  if (!match) return null
  return { section: match[1] as LLMWikiSection, slug: match[2] }
}

function parseSourceReference(reference: string): SourceReference | null {
  const clean = reference.trim()
  const separatorIndex = clean.indexOf(':')
  if (separatorIndex <= 0 || separatorIndex === clean.length - 1) {
    return null
  }
  return {
    fileId: clean.slice(0, separatorIndex),
    chunkId: clean.slice(separatorIndex + 1),
    label: clean,
  }
}

function parseContributingSource(reference: string): ContributingSource | null {
  const clean = reference.replace(/^[-*]\s*/, '').trim()
  const match = clean.match(/^\[([^\]]+)\]\(([^)]+)\)(?:\s+-\s+(.+))?$/)
  if (!match) {
    return null
  }
  const target = wikiLinkTarget(match[2])
  if (!target) {
    return null
  }
  return {
    title: match[1].trim(),
    path: match[2].trim(),
    detail: match[3]?.trim() || '',
    section: target.section,
    slug: target.slug,
  }
}

function readMarkdownSection(markdown: string, heading: string) {
  const escapedHeading = heading.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const match = markdown.match(
    new RegExp(`(?:^|\\n)##\\s+${escapedHeading}\\s*\\n([\\s\\S]*?)(?=\\n##\\s+|$)`)
  )
  return match?.[1]?.trim() || ''
}

function parseGeneratedPageMarkdown(markdown: string): StructuredPageMarkdown | null {
  const normalized = markdown.replace(/\r\n/g, '\n')
  const title = normalized.match(/^#\s+(.+)$/m)?.[1]?.trim()
  const summary = readMarkdownSection(normalized, 'Summary')
  const keywordText = readMarkdownSection(normalized, 'Keywords')
  const referenceText = readMarkdownSection(normalized, 'Navigation Source References')
  const contributingSourceText = readMarkdownSection(normalized, 'Contributing Sources')

  if (!title || !summary || !keywordText || (!referenceText && !contributingSourceText)) {
    return null
  }

  const notice = normalized
    .split('\n')
    .filter((line) => line.trim().startsWith('>'))
    .map((line) => line.replace(/^>\s?/, '').trim())
    .join(' ')

  const keywords = keywordText
    .split(/[,\n]/)
    .map((keyword) => keyword.replace(/^[-*]\s*/, '').replace(/`/g, '').trim())
    .filter(Boolean)
    .filter((keyword) => keyword.toLowerCase() !== 'none')

  const references = referenceText
    .split('\n')
    .map((reference) => reference.replace(/^[-*]\s*/, '').replace(/`/g, '').trim())
    .filter(Boolean)

  const contributingSources = contributingSourceText
    .split('\n')
    .map(parseContributingSource)
    .filter((source): source is ContributingSource => source !== null)

  return { title, notice, summary, keywords, references, contributingSources }
}

function parseIndexMarkdown(
  markdown: string,
  generatedAtFallback?: string | null
): StructuredIndexMarkdown | null {
  const normalized = markdown.replace(/\r\n/g, '\n')
  const title = normalized.match(/^#\s+(.+)$/m)?.[1]?.trim()
  if (!title) return null

  const generatedAt =
    normalized.match(/Generated at\s+`([^`]+)`/)?.[1]?.trim() ||
    generatedAtFallback ||
    ''
  const notice =
    normalized.match(/Generated files are replaceable\.[^\n]+/)?.[0]?.trim() || ''
  const itemDetails = new Map<string, StructuredIndexItem>()

  for (const line of normalized.split('\n')) {
    const match = line.match(
      /^-\s+\[([^\]]+)\]\(((?:sources|concepts)\/[a-z0-9][a-z0-9-]*\.md)\)\s+-\s+([\s\S]*?)\s+\(([^)]*)\)\s*$/
    )
    if (!match) continue

    const path = match[2]
    const summary = match[3].trim()
    const details = match[4]
    const keywords = details.match(/keywords:\s*([^;]+)/)?.[1]
      ?.split(',')
      .map((keyword) => keyword.trim())
      .filter(Boolean) || []
    const detail =
      details
        .split(';')
        .map((part) => part.trim())
        .find((part) => !part.startsWith('keywords:')) || ''

    itemDetails.set(path, { summary, keywords, detail })
  }

  return { title, generatedAt, notice, itemDetails }
}

export function LLMWikiViewer({ collectionId }: LLMWikiViewerProps) {
  const { t } = useTranslation()
  const [collectionName, setCollectionName] = useState(collectionId)
  const [status, setStatus] = useState<WikiStatus>('loading')
  const [index, setIndex] = useState<LLMWikiIndexResponse | null>(null)
  const [markdown, setMarkdown] = useState('')
  const [activePage, setActivePage] = useState<ActivePage>({ kind: 'index' })
  const [errorMessage, setErrorMessage] = useState('')
  const [pageLoading, setPageLoading] = useState(false)
  const [rebuilding, setRebuilding] = useState(false)
  const [chunkDialogOpen, setChunkDialogOpen] = useState(false)
  const [selectedReference, setSelectedReference] = useState<SourceReference | null>(null)
  const [selectedChunk, setSelectedChunk] = useState<ChunkDetail | null>(null)
  const [chunkLoading, setChunkLoading] = useState(false)
  const [chunkError, setChunkError] = useState('')

  const loadCollection = useCallback(async () => {
    try {
      const response = await fetch(`/api/collections/${collectionId}`)
      const payload = await readPayload(response)
      if (response.ok && payload?.success && payload.data) {
        const collection = payload.data as CollectionPayload
        setCollectionName(collection.name || collection.uuid || collectionId)
      }
    } catch {
      setCollectionName(collectionId)
    }
  }, [collectionId])

  const loadIndex = useCallback(async () => {
    setStatus('loading')
    setErrorMessage('')
    const response = await fetch(`/api/collections/${collectionId}/llm-wiki`)
    const payload = await readPayload(response)

    if (!response.ok || !payload?.success) {
      if (payload?.code === 'wiki_not_generated') {
        setIndex(null)
        setMarkdown('')
        setActivePage({ kind: 'index' })
        setStatus('missing')
        return
      }
      setErrorMessage(payload?.message || t('wiki.loadError'))
      setStatus('error')
      return
    }

    const data = payload.data as LLMWikiIndexResponse
    setIndex(data)
    setMarkdown(data.index_markdown)
    setActivePage({ kind: 'index' })
    setStatus('available')
  }, [collectionId, t])

  useEffect(() => {
    void loadCollection()
    void loadIndex()
  }, [loadCollection, loadIndex])

  const selectIndex = useCallback(() => {
    if (!index) return
    setMarkdown(index.index_markdown)
    setActivePage({ kind: 'index' })
  }, [index])

  const loadPage = useCallback(
    async (section: LLMWikiSection, slug: string) => {
      setPageLoading(true)
      setErrorMessage('')
      try {
        const response = await fetch(
          `/api/collections/${collectionId}/llm-wiki/pages/${section}/${slug}`
        )
        const payload = await readPayload(response)
        if (!response.ok || !payload?.success) {
          throw new Error(payload?.message || t('wiki.pageLoadError'))
        }
        const data = payload.data as LLMWikiPageResponse
        setMarkdown(data.markdown)
        setActivePage({ kind: 'page', section, slug })
      } catch (error: any) {
        setErrorMessage(error.message || t('wiki.pageLoadError'))
        setStatus('error')
      } finally {
        setPageLoading(false)
      }
    },
    [collectionId, t]
  )

  const rebuildWiki = useCallback(async () => {
    setRebuilding(true)
    setErrorMessage('')
    try {
      const response = await fetch(`/api/collections/${collectionId}/llm-wiki`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
      })
      const payload = await readPayload(response)
      if (!response.ok || !payload?.success) {
        throw new Error(payload?.message || t('wiki.rebuildError'))
      }
      toast.success(t('wiki.rebuildSuccess'))
      await loadIndex()
    } catch (error: any) {
      setErrorMessage(error.message || t('wiki.rebuildError'))
      setStatus('error')
    } finally {
      setRebuilding(false)
    }
  }, [collectionId, loadIndex, t])

  const openChunkReference = useCallback(
    async (reference: string) => {
      const parsed = parseSourceReference(reference)
      if (!parsed) {
        toast.error('Invalid source reference')
        return
      }

      setSelectedReference(parsed)
      setSelectedChunk(null)
      setChunkError('')
      setChunkLoading(true)
      setChunkDialogOpen(true)

      try {
        const response = await fetch(
          `/api/collections/${collectionId}/documents/${encodeURIComponent(parsed.chunkId)}?file_id=${encodeURIComponent(parsed.fileId)}`
        )
        const payload = await readPayload(response)
        if (!response.ok || !payload?.success) {
          throw new Error(payload?.message || 'Failed to load chunk')
        }
        setSelectedChunk(payload.data as ChunkDetail)
      } catch (error: any) {
        setChunkError(error.message || 'Failed to load chunk')
      } finally {
        setChunkLoading(false)
      }
    },
    [collectionId]
  )

  const markdownComponents = useMemo<Components>(
    () => ({
      a: ({ href, children }) => {
        const target = wikiLinkTarget(href)
        if (target) {
          return (
            <button
              type="button"
              aria-label={`Open SNI page ${String(children)} from markdown`}
              className="text-blue-600 underline underline-offset-2 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
              onClick={(event) => {
                event.preventDefault()
                void loadPage(target.section, target.slug)
              }}
            >
              {children}
            </button>
          )
        }
        return (
          <a
            href={href}
            target="_blank"
            rel="noreferrer"
            className="text-blue-600 underline underline-offset-2 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
          >
            {children}
          </a>
        )
      },
    }),
    [loadPage]
  )

  const structuredPage = useMemo(() => {
    if (activePage.kind !== 'page') {
      return null
    }
    return parseGeneratedPageMarkdown(markdown)
  }, [activePage, markdown])

  const structuredIndex = useMemo(() => {
    if (activePage.kind !== 'index' || !index) {
      return null
    }
    return parseIndexMarkdown(index.index_markdown, index.generated_at)
  }, [activePage, index])

  const selectedChunkContent =
    selectedChunk?.content || selectedChunk?.page_content || ''
  const selectedChunkMetadata = selectedChunk?.metadata || {}

  const renderNavItem = (
    section: LLMWikiSection,
    item: LLMWikiManifestItem
  ) => {
    const active =
      activePage.kind === 'page' &&
      activePage.section === section &&
      activePage.slug === item.slug
    return (
      <Button
        key={item.path}
        type="button"
        variant={active ? 'secondary' : 'ghost'}
        className="h-auto w-full justify-start px-3 py-2 text-left"
        onClick={() => void loadPage(section, item.slug)}
      >
        <span className="min-w-0 truncate">{item.title}</span>
      </Button>
    )
  }

  const renderIndexCards = (
    title: string,
    section: LLMWikiSection,
    items: LLMWikiManifestItem[]
  ) => (
    <section className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-lg font-semibold text-gray-950 dark:text-gray-50">
          {title}
        </h2>
        <Badge variant="secondary">{items.length}</Badge>
      </div>
      {items.length > 0 ? (
        <div className="grid gap-3 md:grid-cols-2">
          {items.map((item) => {
            const details = structuredIndex?.itemDetails.get(item.path)
            return (
              <button
                key={item.path}
                type="button"
                aria-label={`Open ${section === 'concepts' ? 'concept' : 'source'} page ${item.title}`}
                className="min-h-36 rounded-md border bg-background p-4 text-left transition-colors hover:border-blue-300 hover:bg-blue-50/50 dark:hover:border-blue-800 dark:hover:bg-blue-950/30"
                onClick={() => void loadPage(section, item.slug)}
              >
                <div className="flex h-full flex-col gap-3">
                  <div className="flex items-start justify-between gap-3">
                    <h3 className="text-base font-semibold leading-6 text-gray-950 dark:text-gray-50">
                      {item.title}
                    </h3>
                    {details?.detail && (
                      <Badge variant="outline" className="shrink-0">
                        {details.detail}
                      </Badge>
                    )}
                  </div>
                  {details?.summary && (
                    <p className="line-clamp-4 text-sm leading-6 text-gray-700 dark:text-gray-200">
                      {details.summary}
                    </p>
                  )}
                  {details?.keywords && details.keywords.length > 0 && (
                    <div className="mt-auto flex flex-wrap gap-1.5">
                      {details.keywords.slice(0, 4).map((keyword) => (
                        <Badge key={keyword} variant="secondary">
                          {keyword}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
              </button>
            )
          })}
        </div>
      ) : (
        <p className="rounded-md border border-dashed p-4 text-sm text-gray-500">
          {section === 'concepts' ? t('wiki.noConcepts') : t('wiki.noSources')}
        </p>
      )}
    </section>
  )

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-7xl space-y-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="space-y-2">
            <Button asChild variant="ghost" size="sm" className="w-fit px-0">
              <Link href="/collections">
                <ArrowLeft className="h-4 w-4" />
                {t('wiki.backToCollections')}
              </Link>
            </Button>
            <div>
              <h1 className="flex items-center gap-3 text-3xl font-bold text-gray-900 dark:text-gray-100">
                <BookOpen className="h-8 w-8 text-blue-500" />
                {t('wiki.title')}
              </h1>
              <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
                {collectionName}
              </p>
            </div>
          </div>
          {status !== 'missing' && (
            <Button
              type="button"
              variant="outline"
              onClick={() => void rebuildWiki()}
              disabled={rebuilding}
            >
              {rebuilding ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              {rebuilding ? t('wiki.rebuilding') : t('wiki.rebuild')}
            </Button>
          )}
        </div>

        {status === 'loading' && (
          <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
            <Skeleton className="h-96 rounded-lg" />
            <Skeleton className="h-96 rounded-lg" />
          </div>
        )}

        {status === 'missing' && (
          <div className="rounded-lg border border-dashed border-gray-300 bg-gray-50 p-10 text-center dark:border-gray-700 dark:bg-gray-900/40">
            <BookOpen className="mx-auto mb-4 h-10 w-10 text-blue-500" />
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
              {t('wiki.notGeneratedTitle')}
            </h2>
            <p className="mx-auto mt-2 max-w-xl text-sm text-gray-600 dark:text-gray-300">
              {t('wiki.notGeneratedDescription')}
            </p>
            <Button
              type="button"
              className="mt-6"
              onClick={() => void rebuildWiki()}
              disabled={rebuilding}
            >
              {rebuilding ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              {rebuilding ? t('wiki.rebuilding') : t('wiki.rebuild')}
            </Button>
          </div>
        )}

        {status === 'error' && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
            {errorMessage || t('wiki.loadError')}
          </div>
        )}

        {status === 'available' && index && (
          <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
            <aside className="rounded-lg border bg-background p-4">
              <div className="mb-4 rounded-md bg-blue-50 p-3 text-xs text-blue-800 dark:bg-blue-950 dark:text-blue-200">
                {t('wiki.generatedNavigationNotice')}
              </div>
              <div className="space-y-4">
                <div>
                  <Button
                    type="button"
                    variant={activePage.kind === 'index' ? 'secondary' : 'ghost'}
                    className="w-full justify-start"
                    onClick={selectIndex}
                  >
                    <BookOpen className="h-4 w-4" />
                    {t('wiki.index')}
                  </Button>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between px-2">
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase text-gray-500 dark:text-gray-400">
                      <Lightbulb className="h-3.5 w-3.5" />
                      {t('wiki.concepts')}
                    </div>
                    <Badge variant="secondary">{index.concepts.length}</Badge>
                  </div>
                  <div className="space-y-1">
                    {index.concepts.length > 0 ? (
                      index.concepts.map((item) => renderNavItem('concepts', item))
                    ) : (
                      <p className="px-2 text-sm text-gray-500">{t('wiki.noConcepts')}</p>
                    )}
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between px-2">
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase text-gray-500 dark:text-gray-400">
                      <FileText className="h-3.5 w-3.5" />
                      {t('wiki.sources')}
                    </div>
                    <Badge variant="secondary">{index.sources.length}</Badge>
                  </div>
                  <div className="space-y-1">
                    {index.sources.length > 0 ? (
                      index.sources.map((item) => renderNavItem('sources', item))
                    ) : (
                      <p className="px-2 text-sm text-gray-500">{t('wiki.noSources')}</p>
                    )}
                  </div>
                </div>
              </div>
            </aside>

            <section className="min-w-0 rounded-lg border bg-background p-6">
              {pageLoading ? (
                <div className="space-y-3">
                  <Skeleton className="h-8 w-1/2" />
                  <Skeleton className="h-5 w-full" />
                  <Skeleton className="h-5 w-5/6" />
                </div>
              ) : structuredIndex ? (
                <article className="space-y-6">
                  <div className="space-y-4">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                      <div className="space-y-2">
                        <h1 className="text-3xl font-semibold tracking-normal text-gray-950 dark:text-gray-50">
                          {structuredIndex.title}
                        </h1>
                        {structuredIndex.generatedAt && (
                          <dl className="flex flex-wrap items-center gap-x-2 text-sm text-gray-600 dark:text-gray-300">
                            <dt className="font-medium">Generated at</dt>
                            <dd className="font-mono text-xs">{structuredIndex.generatedAt}</dd>
                          </dl>
                        )}
                      </div>
                      <div className="flex gap-2">
                        <Badge variant="secondary">{index.concepts.length} concepts</Badge>
                        <Badge variant="secondary">{index.sources.length} sources</Badge>
                      </div>
                    </div>
                    {structuredIndex.notice && (
                      <blockquote className="border-l-4 border-blue-500 bg-blue-50 px-4 py-3 text-sm text-blue-900 dark:bg-blue-950/50 dark:text-blue-100">
                        {structuredIndex.notice}
                      </blockquote>
                    )}
                  </div>

                  {renderIndexCards(t('wiki.concepts'), 'concepts', index.concepts)}
                  {renderIndexCards(t('wiki.sources'), 'sources', index.sources)}
                </article>
              ) : structuredPage ? (
                <article className="space-y-6">
                  <div className="space-y-3">
                    <h1 className="text-3xl font-semibold tracking-normal text-gray-950 dark:text-gray-50">
                      {structuredPage.title}
                    </h1>
                    {structuredPage.notice && (
                      <blockquote className="border-l-4 border-blue-500 bg-blue-50 px-4 py-3 text-sm text-blue-900 dark:bg-blue-950/50 dark:text-blue-100">
                        {structuredPage.notice}
                      </blockquote>
                    )}
                  </div>

                  <section className="space-y-2">
                    <h2 className="text-sm font-semibold uppercase text-gray-500 dark:text-gray-400">
                      Summary
                    </h2>
                    <p className="whitespace-pre-line rounded-md border bg-gray-50 p-4 text-sm leading-6 text-gray-800 dark:bg-gray-900/50 dark:text-gray-100">
                      {structuredPage.summary}
                    </p>
                  </section>

                  {structuredPage.keywords.length > 0 && (
                    <section className="space-y-2">
                      <h2 className="text-sm font-semibold uppercase text-gray-500 dark:text-gray-400">
                        Keywords
                      </h2>
                      <div className="flex flex-wrap gap-2">
                        {structuredPage.keywords.map((keyword) => (
                          <Badge key={keyword} variant="secondary">
                            {keyword}
                          </Badge>
                        ))}
                      </div>
                    </section>
                  )}

                  {activePage.kind === 'page' &&
                  activePage.section === 'concepts' &&
                  structuredPage.contributingSources.length > 0 && (
                    <section className="space-y-2">
                      <h2 className="text-sm font-semibold uppercase text-gray-500 dark:text-gray-400">
                        Contributing Sources
                      </h2>
                      <ul className="grid gap-2 sm:grid-cols-2">
                        {structuredPage.contributingSources.map((source) => (
                          <li key={source.path}>
                            <Button
                              type="button"
                              variant="outline"
                              aria-label={`Open contributing source ${source.title}`}
                              className="h-auto min-h-12 w-full justify-start whitespace-normal px-3 py-2 text-left text-sm text-gray-700 dark:text-gray-200"
                              onClick={() => void loadPage(source.section, source.slug)}
                            >
                              <span className="min-w-0">
                                <span className="block font-medium leading-5">
                                  {source.title}
                                </span>
                                {source.detail && (
                                  <span className="block truncate text-xs text-gray-500 dark:text-gray-400">
                                    {source.detail}
                                  </span>
                                )}
                              </span>
                            </Button>
                          </li>
                        ))}
                      </ul>
                    </section>
                  )}

                  {structuredPage.references.length > 0 && (
                    <section className="space-y-2">
                      <h2 className="text-sm font-semibold uppercase text-gray-500 dark:text-gray-400">
                        Navigation Source References
                      </h2>
                      <ul className="grid gap-2 sm:grid-cols-2">
                        {structuredPage.references.map((reference) => (
                          <li key={reference}>
                            <Button
                              type="button"
                              variant="outline"
                              aria-label={`Open source reference ${reference}`}
                              className="h-auto min-h-12 w-full justify-start whitespace-normal break-all px-3 py-2 text-left font-mono text-xs text-gray-700 dark:text-gray-200"
                              onClick={() => void openChunkReference(reference)}
                            >
                              {reference}
                            </Button>
                          </li>
                        ))}
                      </ul>
                    </section>
                  )}
                </article>
              ) : (
                <div className="prose prose-sm max-w-none break-words dark:prose-invert">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkBreaks]}
                    components={markdownComponents}
                  >
                    {markdown}
                  </ReactMarkdown>
                </div>
              )}
            </section>
          </div>
        )}
      </div>

      <Dialog open={chunkDialogOpen} onOpenChange={setChunkDialogOpen}>
        <DialogContent className="max-h-[90vh] overflow-hidden p-0 sm:max-w-[760px]">
          <DialogHeader className="border-b px-5 py-4">
            <div className="flex items-start justify-between gap-4 pr-8">
              <div className="min-w-0 space-y-1">
                <DialogTitle>Chunk Details</DialogTitle>
                <DialogDescription className="break-all font-mono text-xs">
                  {selectedReference?.label || 'Source reference'}
                </DialogDescription>
              </div>
              {selectedChunkContent && (
                <Badge variant="secondary" className="shrink-0">
                  {selectedChunkContent.length} chars
                </Badge>
              )}
            </div>
          </DialogHeader>

          <div className="px-5 pb-5">
            {chunkLoading ? (
              <div className="space-y-3 py-5">
                <Skeleton className="h-9 w-full" />
                <Skeleton className="h-52 w-full" />
              </div>
            ) : chunkError ? (
              <div className="mt-5 rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
                {chunkError}
              </div>
            ) : selectedChunk ? (
              <Tabs defaultValue="content" className="mt-4 w-full">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="content">Content</TabsTrigger>
                  <TabsTrigger value="metadata">Metadata</TabsTrigger>
                </TabsList>

                <TabsContent
                  value="content"
                  forceMount
                  className="mt-4 space-y-3 data-[state=inactive]:hidden"
                >
                  <ScrollArea className="h-[360px] w-full rounded-md border bg-gray-50 p-4 dark:bg-gray-900/50">
                    <div className="whitespace-pre-wrap pr-4 text-sm leading-6 text-gray-800 dark:text-gray-100">
                      {selectedChunkContent}
                    </div>
                  </ScrollArea>

                  <div className="rounded-md border bg-gray-50 p-3 dark:bg-gray-900/50">
                    <div className="space-y-2 text-xs text-gray-600 dark:text-gray-300">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-medium">Document ID:</span>
                        <code className="rounded border bg-white px-2 py-1 font-mono text-gray-700 dark:bg-gray-800 dark:text-gray-200">
                          {selectedChunk.id}
                        </code>
                      </div>
                      {Boolean(selectedChunkMetadata.file_id) && (
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-medium">File ID:</span>
                          <code className="rounded border bg-white px-2 py-1 font-mono text-gray-700 dark:bg-gray-800 dark:text-gray-200">
                            {String(selectedChunkMetadata.file_id)}
                          </code>
                        </div>
                      )}
                      {Boolean(selectedChunkMetadata.source) && (
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-medium">Source:</span>
                          <span className="text-gray-700 dark:text-gray-200">
                            {String(selectedChunkMetadata.source)}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </TabsContent>

                <TabsContent
                  value="metadata"
                  forceMount
                  className="mt-4 data-[state=inactive]:hidden"
                >
                  <ScrollArea className="h-[420px] w-full">
                    <div className="rounded-md border bg-gray-50 p-4 dark:bg-gray-900/50">
                      <div className="grid grid-cols-[120px_1fr] gap-3">
                        <span className="text-sm font-medium text-gray-600 dark:text-gray-300">
                          Document ID:
                        </span>
                        <code className="break-all rounded border bg-white px-2 py-1 font-mono text-xs text-gray-700 dark:bg-gray-800 dark:text-gray-200">
                          {selectedChunk.id}
                        </code>
                        {Object.entries(selectedChunkMetadata).map(([key, value]) => (
                          <div key={key} className="contents">
                            <span className="text-sm font-medium text-gray-600 dark:text-gray-300">
                              {key}:
                            </span>
                            <div className="min-w-0 text-sm text-gray-700 dark:text-gray-200">
                              {typeof value === 'object' && value !== null ? (
                                <pre className="overflow-x-auto rounded border bg-white p-2 text-xs dark:bg-gray-800">
                                  {JSON.stringify(value, null, 2)}
                                </pre>
                              ) : (
                                <span className="break-all">{String(value)}</span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </ScrollArea>
                </TabsContent>
              </Tabs>
            ) : null}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
