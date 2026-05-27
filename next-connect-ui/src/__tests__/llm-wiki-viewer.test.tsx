import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'

import CollectionsPage from '@/app/(protected)/collections/page'
import { LLMWikiViewer } from '@/components/wiki/llm-wiki-viewer'
import { LanguageProvider } from '@/providers/language-provider'

const COLLECTION_ID = '00000000-0000-0000-0000-000000000001'

function jsonResponse(body: unknown, init: ResponseInit = {}) {
  const status = init.status ?? 200
  return Promise.resolve(
    {
      ok: status >= 200 && status < 300,
      status,
      json: async () => body,
    } as Response
  )
}

function renderWithLanguage(ui: React.ReactElement) {
  return render(<LanguageProvider>{ui}</LanguageProvider>)
}

describe('LLM Wiki viewer', () => {
  const originalFetch = global.fetch

  beforeEach(() => {
    global.fetch = jest.fn()
  })

  afterEach(() => {
    global.fetch = originalFetch
    jest.restoreAllMocks()
  })

  it('renders a Wiki link for each collection row', async () => {
    ;(global.fetch as jest.Mock).mockResolvedValue(
      await jsonResponse({
        success: true,
        data: [
          {
            uuid: COLLECTION_ID,
            name: 'Alpha Collection',
            document_count: 2,
            chunk_count: 12,
            metadata: {},
          },
        ],
      })
    )

    renderWithLanguage(<CollectionsPage />)

    const link = await screen.findByRole('link', {
      name: 'Open wiki for Alpha Collection',
    })
    expect(link).toHaveAttribute('href', `/collections/${COLLECTION_ID}/wiki`)
  })

  it('shows the missing wiki state and rebuilds through the collection wiki API', async () => {
    let wikiGetCount = 0
    ;(global.fetch as jest.Mock).mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === `/api/collections/${COLLECTION_ID}`) {
        return jsonResponse({
          success: true,
          data: { uuid: COLLECTION_ID, name: 'Alpha Collection', metadata: {} },
        })
      }
      if (url === `/api/collections/${COLLECTION_ID}/llm-wiki` && init?.method === 'POST') {
        return jsonResponse({ success: true, data: { status: 'rebuilt' } })
      }
      if (url === `/api/collections/${COLLECTION_ID}/llm-wiki`) {
        wikiGetCount += 1
        if (wikiGetCount === 1) {
          return jsonResponse(
            {
              success: false,
              code: 'wiki_not_generated',
              message: 'Wiki not generated yet',
            },
            { status: 404 }
          )
        }
        return jsonResponse({
          success: true,
          data: {
            collection_id: COLLECTION_ID,
            status: 'available',
            generated_at: '2026-05-19T00:00:00+00:00',
            index_markdown: '# Generated Index',
            sources: [],
            concepts: [],
          },
        })
      }
      throw new Error(`Unexpected fetch ${url}`)
    })

    renderWithLanguage(<LLMWikiViewer collectionId={COLLECTION_ID} />)

    expect(await screen.findByText('Wiki not generated yet')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Rebuild Wiki' }))

    await waitFor(() => {
      expect(screen.getByText('Generated Index')).toBeInTheDocument()
    })
    expect(global.fetch).toHaveBeenCalledWith(
      `/api/collections/${COLLECTION_ID}/llm-wiki`,
      expect.objectContaining({ method: 'POST' })
    )
  })

  it('renders index markdown and fetches source pages from generated navigation', async () => {
    ;(global.fetch as jest.Mock).mockImplementation((input: RequestInfo | URL) => {
      const url = String(input)
      if (url === `/api/collections/${COLLECTION_ID}`) {
        return jsonResponse({
          success: true,
          data: { uuid: COLLECTION_ID, name: 'Alpha Collection', metadata: {} },
        })
      }
      if (url === `/api/collections/${COLLECTION_ID}/llm-wiki`) {
        return jsonResponse({
          success: true,
          data: {
            collection_id: COLLECTION_ID,
            status: 'available',
            generated_at: '2026-05-19T00:00:00+00:00',
            index_markdown: '# Generated Index\n\n[Source One](sources/source-one.md)',
            sources: [
              {
                type: 'source',
                title: 'Source One',
                path: 'sources/source-one.md',
                slug: 'source-one',
                id: 'source-source-one',
                chunk_count: 2,
                reference_count: 2,
              },
            ],
            concepts: [
              {
                type: 'concept',
                title: 'Concept One',
                path: 'concepts/concept-one.md',
                slug: 'concept-one',
                id: 'concept-concept-one',
                reference_count: 3,
              },
            ],
          },
        })
      }
      if (url === `/api/collections/${COLLECTION_ID}/llm-wiki/pages/sources/source-one`) {
        return jsonResponse({
          success: true,
          data: {
            collection_id: COLLECTION_ID,
            section: 'sources',
            slug: 'source-one',
            title: 'Source One',
            path: 'sources/source-one.md',
            markdown: '# Source One\n\nSource body.',
          },
        })
      }
      throw new Error(`Unexpected fetch ${url}`)
    })

    renderWithLanguage(<LLMWikiViewer collectionId={COLLECTION_ID} />)

    expect(await screen.findByText('Generated Index')).toBeInTheDocument()
    expect(screen.getByText('Generated wiki navigation. Use original retrieved chunks as evidence.')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Source One' }))

    expect(await screen.findByText('Source body.')).toBeInTheDocument()
    expect(global.fetch).toHaveBeenCalledWith(
      `/api/collections/${COLLECTION_ID}/llm-wiki/pages/sources/source-one`
    )
  })

  it('orders sidebar navigation as index, concepts, then sources', async () => {
    ;(global.fetch as jest.Mock).mockImplementation((input: RequestInfo | URL) => {
      const url = String(input)
      if (url === `/api/collections/${COLLECTION_ID}`) {
        return jsonResponse({
          success: true,
          data: { uuid: COLLECTION_ID, name: 'Alpha Collection', metadata: {} },
        })
      }
      if (url === `/api/collections/${COLLECTION_ID}/llm-wiki`) {
        return jsonResponse({
          success: true,
          data: {
            collection_id: COLLECTION_ID,
            status: 'available',
            generated_at: '2026-05-19T00:00:00+00:00',
            index_markdown: '# Generated Index',
            sources: [
              {
                type: 'source',
                title: 'Source One',
                path: 'sources/source-one.md',
                slug: 'source-one',
              },
            ],
            concepts: [
              {
                type: 'concept',
                title: 'Concept One',
                path: 'concepts/concept-one.md',
                slug: 'concept-one',
              },
            ],
          },
        })
      }
      throw new Error(`Unexpected fetch ${url}`)
    })

    renderWithLanguage(<LLMWikiViewer collectionId={COLLECTION_ID} />)

    expect(await screen.findByText('Generated Index')).toBeInTheDocument()
    const sidebar = document.querySelector('aside')
    expect(sidebar).not.toBeNull()
    const navButtons = within(sidebar as HTMLElement)
      .getAllByRole('button')
      .map((button) => button.textContent?.trim())

    expect(navButtons).toEqual(['Index', 'Concept One', 'Source One'])
  })

  it('renders the index as generated metadata with concept and source cards', async () => {
    ;(global.fetch as jest.Mock).mockImplementation((input: RequestInfo | URL) => {
      const url = String(input)
      if (url === `/api/collections/${COLLECTION_ID}`) {
        return jsonResponse({
          success: true,
          data: { uuid: COLLECTION_ID, name: 'Alpha Collection', metadata: {} },
        })
      }
      if (url === `/api/collections/${COLLECTION_ID}/llm-wiki`) {
        return jsonResponse({
          success: true,
          data: {
            collection_id: COLLECTION_ID,
            status: 'available',
            generated_at: '2026-05-19T00:00:00+00:00',
            index_markdown: [
              '# Generated Index',
              '',
              'Generated at `2026-05-19T00:00:00+00:00`.',
              '',
              'Generated files are replaceable. Use raw retrieved chunks as evidence.',
              '',
              '## Concepts',
              '',
              '- [Concept One](concepts/concept-one.md) - Concept summary text. (keywords: alpha, beta; sources: 3)',
              '',
              '## Sources',
              '',
              '- [Source One](sources/source-one.md) - Source summary text. (keywords: gamma, delta; chunks: 2)',
            ].join('\n'),
            sources: [
              {
                type: 'source',
                title: 'Source One',
                path: 'sources/source-one.md',
                slug: 'source-one',
                chunk_count: 2,
              },
            ],
            concepts: [
              {
                type: 'concept',
                title: 'Concept One',
                path: 'concepts/concept-one.md',
                slug: 'concept-one',
                reference_count: 3,
              },
            ],
          },
        })
      }
      throw new Error(`Unexpected fetch ${url}`)
    })

    renderWithLanguage(<LLMWikiViewer collectionId={COLLECTION_ID} />)

    expect(await screen.findByRole('heading', { name: 'Generated Index' })).toBeInTheDocument()
    expect(screen.getByText('Generated at')).toBeInTheDocument()
    expect(screen.getByText('2026-05-19T00:00:00+00:00')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Concepts' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Sources' })).toBeInTheDocument()
    expect(screen.getByText('Concept summary text.')).toBeInTheDocument()
    expect(screen.getByText('Source summary text.')).toBeInTheDocument()
    expect(screen.getByText('sources: 3')).toBeInTheDocument()
    expect(screen.getByText('chunks: 2')).toBeInTheDocument()
  })

  it('renders concept markdown with contributing sources and promoted chunk references', async () => {
    ;(global.fetch as jest.Mock).mockImplementation((input: RequestInfo | URL) => {
      const url = String(input)
      if (url === `/api/collections/${COLLECTION_ID}`) {
        return jsonResponse({
          success: true,
          data: { uuid: COLLECTION_ID, name: 'Alpha Collection', metadata: {} },
        })
      }
      if (url === `/api/collections/${COLLECTION_ID}/llm-wiki`) {
        return jsonResponse({
          success: true,
          data: {
            collection_id: COLLECTION_ID,
            status: 'available',
            generated_at: '2026-05-19T00:00:00+00:00',
            index_markdown: '# Generated Index',
            sources: [],
            concepts: [
              {
                type: 'concept',
                title: 'Concept One',
                path: 'concepts/concept-one.md',
                slug: 'concept-one',
              },
            ],
          },
        })
      }
      if (url === `/api/collections/${COLLECTION_ID}/llm-wiki/pages/concepts/concept-one`) {
        return jsonResponse({
          success: true,
          data: {
            collection_id: COLLECTION_ID,
            section: 'concepts',
            slug: 'concept-one',
            title: 'Concept One',
            path: 'concepts/concept-one.md',
            markdown: [
              '# Concept One',
              '',
              '> Generated LLM Wiki navigation memory. This page is replaceable on full rebuild and is not authoritative evidence.',
              '',
              '## Summary',
              'Concept summary text.',
              '',
              '## Keywords',
              'alpha, beta',
              '',
              '## Contributing Sources',
              '- [Source One](sources/source-one.md) - source-one.pdf',
              '- [Source Two](sources/source-two.md) - source-two.pdf',
              '',
              '## Navigation Source References',
              '- `file-1:chunk-1`',
              '- `file-2:chunk-2`',
            ].join('\n'),
          },
        })
      }
      if (url === `/api/collections/${COLLECTION_ID}/documents/chunk-1?file_id=file-1`) {
        return jsonResponse({
          success: true,
          data: {
            id: 'chunk-1',
            content: 'Promoted concept chunk body.',
            metadata: {
              file_id: 'file-1',
              source: 'concept-paper.pdf',
              chunk_index: 1,
            },
            collection_id: COLLECTION_ID,
          },
        })
      }
      if (url === `/api/collections/${COLLECTION_ID}/llm-wiki/pages/sources/source-one`) {
        return jsonResponse({
          success: true,
          data: {
            collection_id: COLLECTION_ID,
            section: 'sources',
            slug: 'source-one',
            title: 'Source One',
            path: 'sources/source-one.md',
            markdown: [
              '# Source One',
              '',
              '> Generated LLM Wiki navigation memory. This page is replaceable on full rebuild and is not authoritative evidence.',
              '',
              '## Summary',
              'Source summary text.',
              '',
              '## Keywords',
              'gamma',
              '',
              '## Navigation Source References',
              '- `file-3:chunk-3`',
            ].join('\n'),
          },
        })
      }
      throw new Error(`Unexpected fetch ${url}`)
    })

    renderWithLanguage(<LLMWikiViewer collectionId={COLLECTION_ID} />)

    fireEvent.click(await screen.findByRole('button', { name: 'Concept One' }))

    expect(await screen.findByRole('heading', { name: 'Concept One' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Summary' })).toBeInTheDocument()
    expect(screen.getByText('Concept summary text.')).toBeInTheDocument()
    expect(screen.getByText('alpha')).toBeInTheDocument()
    expect(screen.getByText('beta')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Contributing Sources' })).toBeInTheDocument()
    expect(screen.getByText('Source One')).toBeInTheDocument()
    expect(screen.getByText('source-one.pdf')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Navigation Source References' })).toBeInTheDocument()
    expect(screen.getByText('file-1:chunk-1')).toBeInTheDocument()

    fireEvent.click(
      screen.getByRole('button', {
        name: 'Open source reference file-1:chunk-1',
      })
    )

    expect(await screen.findByText('Promoted concept chunk body.')).toBeInTheDocument()
    expect(global.fetch).toHaveBeenCalledWith(
      `/api/collections/${COLLECTION_ID}/documents/chunk-1?file_id=file-1`
    )
    fireEvent.click(screen.getByRole('button', { name: 'Close' }))

    fireEvent.click(
      screen.getByRole('button', { name: 'Open contributing source Source One' })
    )

    expect(await screen.findByRole('heading', { name: 'Source One' })).toBeInTheDocument()
    expect(global.fetch).toHaveBeenCalledWith(
      `/api/collections/${COLLECTION_ID}/llm-wiki/pages/sources/source-one`
    )
  })

  it('renders source markdown as structured summary, keyword chips, and source references', async () => {
    ;(global.fetch as jest.Mock).mockImplementation((input: RequestInfo | URL) => {
      const url = String(input)
      if (url === `/api/collections/${COLLECTION_ID}`) {
        return jsonResponse({
          success: true,
          data: { uuid: COLLECTION_ID, name: 'Alpha Collection', metadata: {} },
        })
      }
      if (url === `/api/collections/${COLLECTION_ID}/llm-wiki`) {
        return jsonResponse({
          success: true,
          data: {
            collection_id: COLLECTION_ID,
            status: 'available',
            generated_at: '2026-05-19T00:00:00+00:00',
            index_markdown: '# Generated Index',
            sources: [
              {
                type: 'source',
                title: 'Source One',
                path: 'sources/source-one.md',
                slug: 'source-one',
              },
            ],
            concepts: [],
          },
        })
      }
      if (url === `/api/collections/${COLLECTION_ID}/llm-wiki/pages/sources/source-one`) {
        return jsonResponse({
          success: true,
          data: {
            collection_id: COLLECTION_ID,
            section: 'sources',
            slug: 'source-one',
            title: 'Source One',
            path: 'sources/source-one.md',
            markdown: [
              '# Source One',
              '',
              '> Generated LLM Wiki navigation memory. This page is replaceable on full rebuild and is not authoritative evidence.',
              '',
              '## Summary',
              'Source summary text.',
              '',
              '## Keywords',
              'gamma, delta',
              '',
              '## Navigation Source References',
              '- `file-3:chunk-3`',
              '- `file-4:chunk-4`',
            ].join('\n'),
          },
        })
      }
      throw new Error(`Unexpected fetch ${url}`)
    })

    renderWithLanguage(<LLMWikiViewer collectionId={COLLECTION_ID} />)

    fireEvent.click(await screen.findByRole('button', { name: 'Source One' }))

    expect(await screen.findByRole('heading', { name: 'Source One' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Summary' })).toBeInTheDocument()
    expect(screen.getByText('Source summary text.')).toBeInTheDocument()
    expect(screen.getByText('gamma')).toBeInTheDocument()
    expect(screen.getByText('delta')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Navigation Source References' })).toBeInTheDocument()
    expect(screen.getByText('file-3:chunk-3')).toBeInTheDocument()
    expect(screen.getByText('file-4:chunk-4')).toBeInTheDocument()
  })

  it('opens raw chunk content and metadata from a navigation source reference', async () => {
    ;(global.fetch as jest.Mock).mockImplementation((input: RequestInfo | URL) => {
      const url = String(input)
      if (url === `/api/collections/${COLLECTION_ID}`) {
        return jsonResponse({
          success: true,
          data: { uuid: COLLECTION_ID, name: 'Alpha Collection', metadata: {} },
        })
      }
      if (url === `/api/collections/${COLLECTION_ID}/llm-wiki`) {
        return jsonResponse({
          success: true,
          data: {
            collection_id: COLLECTION_ID,
            status: 'available',
            generated_at: '2026-05-19T00:00:00+00:00',
            index_markdown: '# Generated Index',
            sources: [
              {
                type: 'source',
                title: 'Source One',
                path: 'sources/source-one.md',
                slug: 'source-one',
              },
            ],
            concepts: [],
          },
        })
      }
      if (url === `/api/collections/${COLLECTION_ID}/llm-wiki/pages/sources/source-one`) {
        return jsonResponse({
          success: true,
          data: {
            collection_id: COLLECTION_ID,
            section: 'sources',
            slug: 'source-one',
            title: 'Source One',
            path: 'sources/source-one.md',
            markdown: [
              '# Source One',
              '',
              '> Generated LLM Wiki navigation memory. This page is replaceable on full rebuild and is not authoritative evidence.',
              '',
              '## Summary',
              'Source summary text.',
              '',
              '## Keywords',
              'gamma',
              '',
              '## Navigation Source References',
              '- `file-3:chunk-3`',
            ].join('\n'),
          },
        })
      }
      if (url === `/api/collections/${COLLECTION_ID}/documents/chunk-3?file_id=file-3`) {
        return jsonResponse({
          success: true,
          data: {
            id: 'chunk-3',
            content: 'Raw chunk body text.',
            metadata: {
              file_id: 'file-3',
              source: 'paper.pdf',
              chunk_index: 3,
            },
            collection_id: COLLECTION_ID,
          },
        })
      }
      throw new Error(`Unexpected fetch ${url}`)
    })

    renderWithLanguage(<LLMWikiViewer collectionId={COLLECTION_ID} />)

    fireEvent.click(await screen.findByRole('button', { name: 'Source One' }))
    fireEvent.click(
      await screen.findByRole('button', {
        name: 'Open source reference file-3:chunk-3',
      })
    )

    expect(await screen.findByRole('dialog')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Chunk Details' })).toBeInTheDocument()
    expect(await screen.findByText('Raw chunk body text.')).toBeInTheDocument()
    expect(screen.getAllByText('chunk-3').length).toBeGreaterThan(0)
    expect(screen.getAllByText('paper.pdf').length).toBeGreaterThan(0)

    fireEvent.click(screen.getByRole('tab', { name: 'Metadata' }))

    expect(await screen.findByText('chunk_index:')).toBeInTheDocument()
    expect(screen.getByText('3')).toBeInTheDocument()
    expect(global.fetch).toHaveBeenCalledWith(
      `/api/collections/${COLLECTION_ID}/documents/chunk-3?file_id=file-3`
    )
  })
})
