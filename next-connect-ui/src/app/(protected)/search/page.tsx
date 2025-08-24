'use client'

import { useState, useEffect, useCallback } from 'react'
import { Search, Loader2, FileText, ChevronDown, ChevronUp } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Badge } from '@/components/ui/badge'
import { Collection } from '@/types/collection'
import { SearchResult } from '@/types/search'
import { useTranslation } from '@/hooks/use-translation'

export default function SearchPage() {
  const { t } = useTranslation()
  const [collections, setCollections] = useState<Collection[]>([])
  const [selectedCollection, setSelectedCollection] = useState<string>('')
  const [query, setQuery] = useState('')
  const [limit, setLimit] = useState(5)
  const [searchType, setSearchType] = useState('semantic')
  const [filterJson, setFilterJson] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [loadingSources, setLoadingSources] = useState(false)
  const [sources, setSources] = useState<string[]>([])
  const [showSources, setShowSources] = useState(false)
  const [expandedResults, setExpandedResults] = useState<Set<number>>(new Set())

  const fetchCollections = useCallback(async () => {
    try {
      const response = await fetch('/api/collections')
      const res = await response.json()
      if (!res.success) {
        toast.error(t('common.error'), {
          description: t('collections.messages.fetchError')
        })
        return
      }
      
      const collectionsData: Collection[] = res.data
      setCollections(collectionsData)
      
      if (collectionsData.length > 0) {
        setSelectedCollection(collectionsData[0].uuid)
      }
    } catch (error) {
      console.error('Failed to fetch collections:', error)
      toast.error(t('collections.messages.fetchError'))
    }
  }, [t])

  useEffect(() => {
    fetchCollections()
  }, [fetchCollections])

  const loadSources = async () => {
    if (!selectedCollection) return
    
    setLoadingSources(true)
    try {
      const response = await fetch(`/api/collections/${selectedCollection}/documents?limit=100`)
      const res = await response.json()
      
      if (res.success && res.data) {
        const uniqueSources = new Set<string>()
        res.data.forEach((doc: any) => {
          const source = doc.metadata?.source
          if (source) {
            uniqueSources.add(source)
          }
        })
        setSources(Array.from(uniqueSources).sort())
      } else {
        toast.error(t('common.error'))
      }
    } catch (error) {
      console.error('Failed to load sources:', error)
      toast.error(t('common.error'))
    } finally {
      setLoadingSources(false)
    }
  }

  const handleSearch = async () => {
    if (!query.trim()) {
      toast.error(t('common.error'), {
        description: t('search.enterSearchQuery')
      })
      return
    }
    
    if (!selectedCollection) {
      toast.error(t('common.error'), {
        description: t('search.selectCollectionFirst')
      })
      return
    }

    setLoading(true)
    try {
      const searchData: any = {
        query,
        limit,
        search_type: searchType
      }

      if (filterJson.trim()) {
        try {
          searchData.filter = JSON.parse(filterJson)
        } catch (error) {
          toast.error(t('common.error'), {
            description: t('search.invalidJson')
          })
          setLoading(false)
          return
        }
      }

      const response = await fetch(`/api/collections/${selectedCollection}/documents/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(searchData),
      })

      const res = await response.json()
      
      if (res.success) {
        setResults(res.data || [])
        if (res.data && res.data.length > 0) {
          toast.success(t('search.foundResults', { count: res.data.length }))
        } else {
          toast.info(t('search.noResults'))
        }
      } else {
        toast.error(t('common.error'), {
          description: res.message
        })
      }
    } catch (error) {
      console.error('Search failed:', error)
      toast.error(t('common.error'))
    } finally {
      setLoading(false)
    }
  }

  const toggleResultExpansion = (index: number) => {
    setExpandedResults(prev => {
      const newSet = new Set(prev)
      if (newSet.has(index)) {
        newSet.delete(index)
      } else {
        newSet.add(index)
      }
      return newSet
    })
  }

  


  return (
    <div className="min-h-screen p-6 bg-background dark:bg-background">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-600 dark:from-gray-100 dark:to-gray-400 bg-clip-text text-transparent flex items-center gap-3">
              <Search className="h-8 w-8 text-blue-500" />
              {t('search.title')}
            </h1>
            <p className="text-gray-600 dark:text-gray-300 mt-1">{t('search.description')}</p>
          </div>
        </div>

        <Card className="shadow-none">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Search className="h-5 w-5 text-blue-500" />
              {t('search.title')}
            </CardTitle>
            <CardDescription className="text-gray-500 dark:text-gray-300">
              {t('search.description')}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="collection">{t('search.selectCollection')}</Label>
              <Select value={selectedCollection} onValueChange={setSelectedCollection}>
                <SelectTrigger>
                  <SelectValue placeholder={t('search.selectCollection')} />
                </SelectTrigger>
                <SelectContent>
                  {collections.map((collection) => (
                    <SelectItem key={collection.uuid} value={collection.uuid}>
                      {collection.name} ({collection.uuid.slice(0, 8)}...)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {selectedCollection && (
              <Collapsible open={showSources} onOpenChange={setShowSources}>
                <CollapsibleTrigger asChild>
                  <Button variant="outline" className="w-full justify-between">
                    View Available Sources
                    {showSources ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent className="space-y-4 mt-4">
                  <div className="flex items-center gap-2">
                    <Button 
                      onClick={loadSources} 
                      disabled={loadingSources}
                      size="sm"
                      variant="secondary"
                    >
                      {loadingSources ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin mr-2" />
                          {t('common.loading')}
                        </>
                      ) : (
                        'Load Sources'
                      )}
                    </Button>
                  </div>
                  {sources.length > 0 && (
                    <div className="space-y-2">
                      <p className="font-medium text-sm dark:text-gray-200">Available Sources:</p>
                      <div className="grid grid-cols-1 gap-2">
                        {sources.map((source, index) => (
                          <code key={index} className="text-xs bg-gray-100 dark:bg-gray-800 p-2 rounded block dark:text-gray-200">
                            {`{"source": "${source}"}`}
                          </code>
                        ))}
                      </div>
                    </div>
                  )}
                </CollapsibleContent>
              </Collapsible>
            )}

            <div className="space-y-2">
              <Label htmlFor="query">{t('common.search')}</Label>
              <Input
                id="query"
                placeholder={t('search.searchPlaceholder')}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label htmlFor="limit">Result Limit</Label>
                <Select value={limit.toString()} onValueChange={(value) => setLimit(Number(value))}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {[1, 5, 10, 20, 30, 50].map((num) => (
                      <SelectItem key={num} value={num.toString()}>
                        {num}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="searchType">{t('search.searchType')}</Label>
                <Select value={searchType} onValueChange={setSearchType}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="semantic">{t('search.semanticSearch')}</SelectItem>
                    <SelectItem value="keyword">{t('search.keywordSearch')}</SelectItem>
                    <SelectItem value="hybrid">{t('search.hybridSearch')}</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="filter">{t('collections.table.metadata')} Filter</Label>
                <Textarea
                  id="filter"
                  placeholder={`{"source": "sample.pdf"}

# Other examples
{"file_id": "abc123"}
{"source": "document.pdf", "type": "report"}`}
                  value={filterJson}
                  onChange={(e) => setFilterJson(e.target.value)}
                  rows={2}
                  className="text-sm font-mono"
                />
              </div>
            </div>

            <Button 
              onClick={handleSearch} 
              disabled={loading || !query.trim()}
              className="w-full bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin mr-2" />
                  {t('search.searching')}
                </>
              ) : (
                <>
                  <Search className="w-4 h-4 mr-2" />
                  {t('search.searchButton')}
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {results.length > 0 && (
          <Card className="shadow-none">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5 text-green-500" />
                {t('search.results')}
              </CardTitle>
              <CardDescription className="text-gray-500 dark:text-gray-300">
                Found {results.length} results
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {results.map((result, index) => (
                <Card key={result.id} className="border border-gray-200 dark:border-gray-700">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary">
                          Result {index + 1}
                        </Badge>
                        <Badge variant="outline">
                          {t('search.relevanceScore', { score: result.score.toFixed(4) })}
                        </Badge>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => toggleResultExpansion(index)}
                      >
                        {expandedResults.has(index) ? (
                          <>
                            Collapse <ChevronUp className="w-4 h-4 ml-1" />
                          </>
                        ) : (
                          <>
                            Expand <ChevronDown className="w-4 h-4 ml-1" />
                          </>
                        )}
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div>
                        <h4 className="font-medium text-sm text-gray-600 dark:text-gray-300 mb-2">Content:</h4>
                        <div 
                          className={`text-gray-900 dark:text-gray-100 max-w-none break-words overflow-wrap-anywhere word-break overflow-hidden prose prose-sm dark:prose-invert ${!expandedResults.has(index) ? 'line-clamp-3' : ''}`} 
                          style={{ wordBreak: 'break-word', overflowWrap: 'break-word', hyphens: 'auto', whiteSpace: 'pre-wrap' }}
                        >
                          {result.page_content}
                        </div>
                      </div>
                      
                      {expandedResults.has(index) && (
                        <>
                          {result.metadata && Object.keys(result.metadata).length > 0 && (
                            <div>
                              <h4 className="font-medium text-sm text-gray-600 dark:text-gray-300 mb-2">{t('collections.table.metadata')}:</h4>
                              <pre className="text-xs bg-gray-50 dark:bg-gray-800 dark:text-gray-200 p-3 rounded whitespace-pre-wrap break-words overflow-hidden" style={{ wordBreak: 'break-word', overflowWrap: 'anywhere', maxWidth: '100%' }}>
                                {JSON.stringify(result.metadata, null, 2)}
                              </pre>
                            </div>
                          )}
                          
                          <div>
                            <h4 className="font-medium text-sm text-gray-600 dark:text-gray-300 mb-1">Document ID:</h4>
                            <code className="text-xs bg-gray-100 dark:bg-gray-800 dark:text-gray-200 px-2 py-1 rounded">
                              {result.id}
                            </code>
                          </div>
                        </>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}