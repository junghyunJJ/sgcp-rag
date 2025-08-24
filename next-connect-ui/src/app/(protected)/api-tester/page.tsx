'use client'

import { useState, useEffect } from 'react'
import { Code, Send, Loader2, Check, X, Settings } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import Link from 'next/link'
import { APIEndpoint, APIResponse } from '@/types/api-tester'
import { useTranslation } from '@/hooks/use-translation'


const API_ENDPOINTS = {
  health: [
    {
      id: 'health',
      name: 'Health Check',
      method: 'GET' as const,
      path: '/health',
      description: 'Basic health check endpoint'
    }
  ],
  collections: [
    {
      id: 'get-collections',
      name: 'List Collections',
      method: 'GET' as const,
      path: '/collections',
      description: 'Get all collections'
    },
    {
      id: 'create-collection',
      name: 'Create Collection',
      method: 'POST' as const,
      path: '/collections',
      description: 'Create a new collection',
      body: true
    },
    {
      id: 'get-collection',
      name: 'Get Collection',
      method: 'GET' as const,
      path: '/collections/{collection_id}',
      description: 'Get specific collection by ID',
      params: ['collection_id']
    },
    {
      id: 'update-collection',
      name: 'Update Collection',
      method: 'PATCH' as const,
      path: '/collections/{collection_id}',
      description: 'Update collection metadata',
      params: ['collection_id'],
      body: true
    },
    {
      id: 'delete-collection',
      name: 'Delete Collection',
      method: 'DELETE' as const,
      path: '/collections/{collection_id}',
      description: 'Delete a collection',
      params: ['collection_id']
    }
  ],
  documents: [
    {
      id: 'get-documents',
      name: 'List Documents',
      method: 'GET' as const,
      path: '/collections/{collection_id}/documents',
      description: 'List documents in collection',
      params: ['collection_id']
    },
    {
      id: 'create-documents',
      name: 'Create Documents',
      method: 'POST' as const,
      path: '/collections/{collection_id}/documents',
      description: 'Create documents in collection',
      params: ['collection_id'],
      body: true
    },
    {
      id: 'delete-document',
      name: 'Delete Document',
      method: 'DELETE' as const,
      path: '/collections/{collection_id}/documents/{document_id}',
      description: 'Delete specific document',
      params: ['collection_id', 'document_id']
    },
    {
      id: 'search-documents',
      name: 'Search Documents',
      method: 'POST' as const,
      path: '/collections/{collection_id}/documents/search',
      description: 'Search documents in collection',
      params: ['collection_id'],
      body: true
    }
  ]
}

export default function APITesterPage() {
  const { t } = useTranslation()
  const [selectedGroup, setSelectedGroup] = useState<string>('health')
  const [selectedEndpoint, setSelectedEndpoint] = useState<string>('')
  const [collectionId, setCollectionId] = useState('')
  const [documentId, setDocumentId] = useState('')
  const [requestBody, setRequestBody] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchType, setSearchType] = useState('semantic')
  const [searchLimit, setSearchLimit] = useState(10)
  const [searchFilter, setSearchFilter] = useState('')
  const [newCollectionName, setNewCollectionName] = useState('')
  const [newMetadata, setNewMetadata] = useState('')
  const [collectionName, setCollectionName] = useState('')
  const [metadata, setMetadata] = useState('')
  const [createDocCollectionName, setCreateDocCollectionName] = useState('')
  const [createDocMetadata, setCreateDocMetadata] = useState('')
  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState<APIResponse | null>(null)

  const getCurrentEndpoint = (): APIEndpoint | null => {
    const group = API_ENDPOINTS[selectedGroup as keyof typeof API_ENDPOINTS]
    return group?.find(endpoint => endpoint.id === selectedEndpoint) || null
  }

  useEffect(() => {
    const group = API_ENDPOINTS[selectedGroup as keyof typeof API_ENDPOINTS]
    if (group && group.length > 0) {
      setSelectedEndpoint(group[0].id)
    }
    setResponse(null)
  }, [selectedGroup])

  useEffect(() => {
    setResponse(null)
    setRequestBody('')
    setSearchQuery('')
    setSearchFilter('')
    setNewCollectionName('')
    setNewMetadata('')
    setCollectionName('')
    setMetadata('')
    setCreateDocCollectionName('')
    setCreateDocMetadata('')
  }, [selectedEndpoint])

  const buildRequestUrl = (endpoint: APIEndpoint): string => {
    let url = `/api/${endpoint.path}`
    
    if (endpoint.params?.includes('collection_id') && collectionId) {
      url = url.replace('{collection_id}', collectionId)
    }
    if (endpoint.params?.includes('document_id') && documentId) {
      url = url.replace('{document_id}', documentId)
    }
    
    return url
  }

  const buildRequestBody = (endpoint: APIEndpoint): any => {
    if (!endpoint.body) return null

    switch (endpoint.id) {
      case 'create-collection':
        try {
          const metadataObj = metadata.trim() ? JSON.parse(metadata) : {}
          return {
            name: collectionName || 'New Collection',
            metadata: metadataObj
          }
        } catch {
          return {
            name: collectionName || 'New Collection',
            metadata: {}
          }
        }
      
      case 'update-collection':
        const updateData: any = {}
        
        // Only add name if provided
        if (newCollectionName.trim()) {
          updateData.name = newCollectionName
        }
        
        // Only add metadata if provided
        if (newMetadata.trim()) {
          try {
            updateData.metadata = JSON.parse(newMetadata)
          } catch {
            // Invalid JSON metadata, ignore
          }
        }
        
        return updateData
      
      case 'create-documents':
        try {
          const metadataObj = createDocMetadata.trim() ? JSON.parse(createDocMetadata) : {}
          return {
            collection_name: createDocCollectionName,
            metadata: metadataObj
          }
        } catch {
          return {
            collection_name: createDocCollectionName,
            metadata: {}
          }
        }
      
      case 'search-documents':
        const body: any = {
          query: searchQuery || 'test query',
          limit: searchLimit,
          search_type: searchType
        }
        
        if (searchFilter) {
          try {
            body.filter = JSON.parse(searchFilter)
          } catch {
            // Invalid JSON filter, ignore
          }
        }
        
        return body
      
      default:
        try {
          return requestBody ? JSON.parse(requestBody) : {}
        } catch {
          return {}
        }
    }
  }

  const sendRequest = async () => {
    const endpoint = getCurrentEndpoint()
    if (!endpoint) return

    if (endpoint.params?.includes('collection_id') && !collectionId) {
      toast.error(t('common.collectionIdRequired'))
      return
    }
    if (endpoint.params?.includes('document_id') && !documentId) {
      toast.error(t('apiTester.documentIdRequired'))
      return
    }
    if (endpoint.id === 'search-documents' && !searchQuery) {
      toast.error(t('apiTester.searchQueryRequired'))
      return
    }

    // For create-documents, show message instead of sending request
    if (endpoint.id === 'create-documents') {
      toast.info(t('apiTester.useDocumentUpload'))
      return
    }

    setLoading(true)
    try {
      const url = buildRequestUrl(endpoint)
      const body = buildRequestBody(endpoint)

      const headers: HeadersInit = {}
      
      const options: RequestInit = {
        method: endpoint.method,
        headers: headers
      }

      if (body && (endpoint.method === 'POST' || endpoint.method === 'PATCH')) {
        if (endpoint.id === 'create-documents') {
          // For create-documents, use FormData
          const formData = new FormData()
          formData.append('collection_name', body.collection_name || '')
          formData.append('metadata', JSON.stringify(body.metadata || {}))
          
          options.body = formData
          // Don't set Content-Type header for FormData - let browser set it with boundary
        } else {
          options.body = JSON.stringify(body)
          options.headers = {
            ...headers,
            'Content-Type': 'application/json'
          }
        }
      }

      const response = await fetch(url, options)
      const data = await response.json()

      const apiResponse: APIResponse = {
        success: response.ok,
        status: response.status,
        data: response.ok ? data : undefined,
        error: !response.ok ? (data.message || data.error || 'Request failed') : undefined
      }

      setResponse(apiResponse)

      if (response.ok) {
        toast.success(t('common.success'))
      } else {
        toast.error(`${t('common.error')} (${response.status})`)
      }
    } catch (error) {
      console.error('Request error:', error)
      setResponse({
        success: false,
        status: 0,
        error: error instanceof Error ? error.message : 'Network error'
      })
      toast.error(t('common.error'))
    } finally {
      setLoading(false)
    }
  }

  const currentEndpoint = getCurrentEndpoint()

  return (
    <div className="min-h-screen p-6 bg-background dark:bg-background">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-600 dark:from-gray-100 dark:to-gray-400 bg-clip-text text-transparent flex items-center gap-3">
              <Code className="h-8 w-8 text-blue-500" />
              {t('apiTester.title')}
            </h1>
            <p className="text-gray-600 dark:text-gray-300 mt-1">{t('apiTester.description')}</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="shadow-none">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5 text-blue-500" />
                Request Settings
              </CardTitle>
              <CardDescription className="text-gray-500 dark:text-gray-300">
                Configure API endpoint and parameters
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="group">{t('apiTester.endpoint')} Group</Label>
                  <Select value={selectedGroup} onValueChange={setSelectedGroup}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="health">Health</SelectItem>
                      <SelectItem value="collections">Collections</SelectItem>
                      <SelectItem value="documents">Documents</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="endpoint">{t('apiTester.endpoint')}</Label>
                  <Select value={selectedEndpoint} onValueChange={setSelectedEndpoint}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {API_ENDPOINTS[selectedGroup as keyof typeof API_ENDPOINTS]?.map((endpoint) => (
                        <SelectItem key={endpoint.id} value={endpoint.id}>
                          {endpoint.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {currentEndpoint && (
                <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Badge variant={currentEndpoint.method === 'GET' ? 'secondary' : 
                                  currentEndpoint.method === 'POST' ? 'default' : 
                                  currentEndpoint.method === 'DELETE' ? 'destructive' : 'outline'}>
                      {currentEndpoint.method}
                    </Badge>
                    <code className="text-sm bg-white dark:bg-gray-700 dark:text-gray-200 px-2 py-1 rounded">{currentEndpoint.path}</code>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-300">{currentEndpoint.description}</p>
                </div>
              )}

              {currentEndpoint?.params?.includes('collection_id') && (
                <div className="space-y-2">
                  <Label htmlFor="collectionId">Collection ID</Label>
                  <Input
                    id="collectionId"
                    placeholder={t('apiTester.collectionIdPlaceholder')}
                    value={collectionId}
                    onChange={(e) => setCollectionId(e.target.value)}
                  />
                </div>
              )}

              {currentEndpoint?.params?.includes('document_id') && (
                <div className="space-y-2">
                  <Label htmlFor="documentId">Document ID</Label>
                  <Input
                    id="documentId"
                    placeholder={t('apiTester.documentIdPlaceholder')}
                    value={documentId}
                    onChange={(e) => setDocumentId(e.target.value)}
                  />
                </div>
              )}

              {currentEndpoint?.id === 'search-documents' && (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="searchQuery">{t('common.search')}</Label>
                    <Input
                      id="searchQuery"
                      placeholder={t('search.searchPlaceholder')}
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="searchType">{t('search.searchType')}</Label>
                      <Select value={searchType} onValueChange={setSearchType}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="semantic">Semantic</SelectItem>
                          <SelectItem value="keyword">Keyword</SelectItem>
                          <SelectItem value="hybrid">Hybrid</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="searchLimit">Result Limit</Label>
                      <Select value={searchLimit.toString()} onValueChange={(value) => setSearchLimit(Number(value))}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {[5, 10, 20, 50, 100].map((num) => (
                            <SelectItem key={num} value={num.toString()}>
                              {num}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="searchFilter">Filter (JSON)</Label>
                    <Textarea
                      id="searchFilter"
                      placeholder='{"source": "example.pdf"}'
                      value={searchFilter}
                      onChange={(e) => setSearchFilter(e.target.value)}
                      rows={3}
                      className="font-mono text-sm"
                    />
                  </div>
                </div>
              )}

              {currentEndpoint?.id === 'create-collection' && (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="collectionName">{t('collections.modal.nameLabel')}</Label>
                    <Input
                      id="collectionName"
                      placeholder={t('collections.modal.namePlaceholder')}
                      value={collectionName}
                      onChange={(e) => setCollectionName(e.target.value)}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="metadata">{t('collections.table.metadata')} (JSON)</Label>
                    <Textarea
                      id="metadata"
                      placeholder='{"key": "value"}'
                      value={metadata}
                      onChange={(e) => setMetadata(e.target.value)}
                      rows={4}
                      className="font-mono text-sm"
                    />
                  </div>
                </div>
              )}

              {currentEndpoint?.id === 'create-documents' && (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="createDocCollectionName">Collection Name</Label>
                    <Input
                      id="createDocCollectionName"
                      placeholder={t('collections.modal.namePlaceholder')}
                      value={createDocCollectionName}
                      onChange={(e) => setCreateDocCollectionName(e.target.value)}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="createDocMetadata">{t('collections.table.metadata')} (JSON)</Label>
                    <Textarea
                      id="createDocMetadata"
                      placeholder='{}'
                      value={createDocMetadata}
                      onChange={(e) => setCreateDocMetadata(e.target.value)}
                      rows={4}
                      className="font-mono text-sm"
                    />
                  </div>
                  
                  <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
                    <p className="text-sm text-blue-700 dark:text-blue-300">
                      ℹ️ {t('apiTester.useDocumentUpload')}
                    </p>
                    <Link href="/documents" className="inline-flex items-center gap-1 text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 font-medium mt-2">
                      {t('apiTester.goToDocuments')}
                    </Link>
                  </div>
                </div>
              )}

              {currentEndpoint?.id === 'update-collection' && (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="newCollectionName">New Collection Name (Optional)</Label>
                    <Input
                      id="newCollectionName"
                      placeholder={t('collections.modal.namePlaceholder')}
                      value={newCollectionName}
                      onChange={(e) => setNewCollectionName(e.target.value)}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="newMetadata">New {t('collections.table.metadata')} (JSON, Optional)</Label>
                    <Textarea
                      id="newMetadata"
                      placeholder='{"key": "value"}'
                      value={newMetadata}
                      onChange={(e) => setNewMetadata(e.target.value)}
                      rows={4}
                      className="font-mono text-sm"
                    />
                  </div>
                </div>
              )}

              {currentEndpoint?.body && currentEndpoint.id !== 'search-documents' && currentEndpoint.id !== 'update-collection' && currentEndpoint.id !== 'create-collection' && currentEndpoint.id !== 'create-documents' && (
                <div className="space-y-2">
                  <Label htmlFor="requestBody">{t('apiTester.body')} (JSON)</Label>
                  <Textarea
                    id="requestBody"
                    placeholder={currentEndpoint.id === 'create-collection' ? 
                      '{"name": "My Collection", "metadata": {}}' : 
                      '{}'}
                    value={requestBody}
                    onChange={(e) => setRequestBody(e.target.value)}
                    rows={6}
                    className="font-mono text-sm"
                  />
                </div>
              )}

              <Button 
                onClick={sendRequest} 
                disabled={loading}
                className="w-full bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin mr-2" />
                    {t('common.loading')}
                  </>
                ) : (
                  <>
                    <Send className="w-4 h-4 mr-2" />
                    {t('apiTester.sendRequest')}
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          <Card className="shadow-none">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                {response?.success ? (
                  <Check className="h-5 w-5 text-green-500" />
                ) : response?.success === false ? (
                  <X className="h-5 w-5 text-red-500" />
                ) : (
                  <Code className="h-5 w-5 text-gray-500" />
                )}
                {t('apiTester.response')}
              </CardTitle>
              <CardDescription className="text-gray-500 dark:text-gray-300">
                {response ? (
                  <div className="flex items-center gap-2">
                    <Badge variant={response.success ? 'default' : 'destructive'}>
                      {response.status || 'ERROR'}
                    </Badge>
                    <span>{response.success ? t('common.success') : t('common.error')}</span>
                  </div>
                ) : (
                  t('apiTester.responsePlaceholder')
                )}
              </CardDescription>
            </CardHeader>
            <CardContent className="max-h-96 overflow-y-auto">
              {response ? (
                <div className="space-y-4">
                  {response.success && response.data && (
                    <div>
                      <h4 className="font-medium text-sm text-gray-600 dark:text-gray-300 mb-2">Response Data:</h4>
                      <pre className="text-xs bg-gray-50 dark:bg-gray-800 dark:text-gray-200 p-4 rounded overflow-x-auto whitespace-pre-wrap">
                        {JSON.stringify(response.data, null, 2)}
                      </pre>
                    </div>
                  )}

                  {!response.success && response.error && (
                    <div>
                      <h4 className="font-medium text-sm text-red-600 dark:text-red-400 mb-2">{t('common.error')}:</h4>
                      <pre className="text-xs bg-red-50 dark:bg-red-900/20 p-4 rounded overflow-x-auto text-red-700 dark:text-red-300">
                        {response.error}
                      </pre>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500 dark:text-gray-300">
                  <Code className="h-16 w-16 mx-auto mb-4 text-gray-300 dark:text-gray-500" />
                  <p className="dark:text-gray-300">{t('apiTester.sendRequestPlaceholder')}</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}