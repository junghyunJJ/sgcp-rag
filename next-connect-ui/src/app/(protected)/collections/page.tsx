'use client'

import { useState, useEffect, useCallback } from 'react'
import { RefreshCw, Plus, Trash2, Folder, FileText, Loader2, Database, FolderOpen, Archive, BookOpen, Info, X } from 'lucide-react'
import { CollectionWithStats, Collection } from '@/types/collection'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { CreateCollectionModal } from '@/components/modals/create-collection-modal'
import { useTranslation } from '@/hooks/use-translation'

export default function CollectionsPage() {
  const { t } = useTranslation()
  const [collections, setCollections] = useState<CollectionWithStats[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  
  // Create modal state
  const [showCreateModal, setShowCreateModal] = useState(false)
  
  // Selection states
  const [selectedCollections, setSelectedCollections] = useState<string[]>([])
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [deleting, setDeleting] = useState(false)
  
  // Popover states
  const [openPopovers, setOpenPopovers] = useState<Set<string>>(new Set())

  const fetchCollections = useCallback(async () => {
    try {
      setLoading(true)
      const response = await fetch('/api/collections')
      const res = await response.json()
      if (!res.success) {
        toast.error(t("common.error"), {
          description: t("collections.messages.fetchError")
        })
        setCollections([])
        return
      }

      const collectionsData: CollectionWithStats[] = res.data.map((c: any) => ({
        ...c,
        stats: {
          documents: c.document_count,
          chunks: c.chunk_count,
        }
      }))
      
      setCollections(collectionsData)
    } catch (error) {
      console.error('Failed to fetch collections:', error)
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [t])

  useEffect(() => {
    fetchCollections()
  }, [fetchCollections])

  const handleRefresh = () => {
    setRefreshing(true)
    fetchCollections()
  }

  const togglePopover = (collectionId: string, isOpen: boolean) => {
    setOpenPopovers(prev => {
      const newSet = new Set(prev)
      if (isOpen) {
        newSet.add(collectionId)
      } else {
        newSet.delete(collectionId)
      }
      return newSet
    })
  }


  const handleDeleteSelected = useCallback(async () => {
    if (selectedCollections.length === 0) return

    setDeleting(true)
    let deletedCount = 0
    let failedCount = 0

    for (const uuid of selectedCollections) {
      try {
        const response = await fetch(`/api/collections/${uuid}`, {
          method: 'DELETE',
        })
        
        const result = await response.json()
        
        if (response.ok && result.success) {
          deletedCount++
        } else {
          failedCount++
          console.error(`Failed to delete collection ${uuid}:`, result.message)
        }
      } catch (error) {
        failedCount++
        console.error(`Failed to delete collection ${uuid}:`, error)
      }
    }

    setDeleting(false)
    setShowDeleteConfirm(false)
    setSelectedCollections([])

    if (deletedCount > 0) {
      toast.success(t('collections.messages.deleteSuccess', { count: deletedCount }))
    }
    if (failedCount > 0) {
      toast.error(t('collections.messages.deleteFailed', { count: failedCount }))
    }

    fetchCollections()
  }, [selectedCollections, fetchCollections, t])

  const toggleSelection = (uuid: string) => {
    setSelectedCollections(prev => 
      prev.includes(uuid) 
        ? prev.filter(id => id !== uuid)
        : [...prev, uuid]
    )
  }

  const selectedCollectionNames = selectedCollections.map(uuid => 
    collections.find(c => c.uuid === uuid)?.name || ''
  )

  // Statistics calculation
  const totalDocuments = collections.reduce((sum, collection) => sum + collection.stats.documents, 0)
  const totalChunks = collections.reduce((sum, collection) => sum + collection.stats.chunks, 0)

  // Loading skeleton component
  const LoadingSkeleton = () => (
    <div className="space-y-6">
      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 animate-pulse">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <Skeleton className="h-4 w-4" />
            <Skeleton className="h-4 w-12" />
            <Skeleton className="h-4 w-6" />
          </div>
          <div className="flex items-center gap-2">
            <Skeleton className="h-4 w-4" />
            <Skeleton className="h-4 w-8" />
            <Skeleton className="h-4 w-6" />
          </div>
          <div className="flex items-center gap-2">
            <Skeleton className="h-4 w-4" />
            <Skeleton className="h-4 w-8" />
            <Skeleton className="h-4 w-6" />
          </div>
        </div>
      </div>
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-40" />
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="flex items-center space-x-4">
                <Skeleton className="h-4 w-4" />
                <Skeleton className="h-4 flex-1" />
                <Skeleton className="h-4 w-16" />
                <Skeleton className="h-4 w-16" />
                <Skeleton className="h-4 w-32" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )

  // Empty state component
  const EmptyState = () => (
    <Card className="border-dashed border-2 border-gray-300 dark:border-gray-600 bg-gray-50/50 dark:bg-gray-800/50">
      <CardContent className="flex flex-col items-center justify-center py-12">
        <div className="rounded-full bg-blue-50 dark:bg-blue-900/20 p-6 mb-4">
          <FolderOpen className="h-12 w-12 text-blue-500" />
        </div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">{t('collections.noCollections')}</h3>
        <p className="text-gray-500 dark:text-gray-300 text-center mb-6 max-w-sm">
          {t('collections.noCollectionsDescription')}
        </p>
        <Button 
          onClick={() => setShowCreateModal(true)}
          className="bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600"
        >
          <Plus className="w-4 h-4 mr-2" />
          {t('collections.createFirstCollection')}
        </Button>
      </CardContent>
    </Card>
  )

  return (
    <div className="min-h-screen p-6 bg-background dark:bg-background">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-600 dark:from-gray-100 dark:to-gray-400 bg-clip-text text-transparent flex items-center gap-3">
              <Database className="h-8 w-8 text-blue-500" />
              {t('collections.title')}
            </h1>
            <p className="text-gray-600 dark:text-gray-300 mt-1">{t('collections.description')}</p>
          </div>
          <div className="flex items-center gap-3">
            <Button
              onClick={handleRefresh}
              disabled={refreshing}
              variant="outline"
              size="sm"
              className="flex items-center gap-2"
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
              {t('common.refresh')}
            </Button>
            <Button 
              onClick={() => setShowCreateModal(true)}
              size="sm"
              className="bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600 shadow-lg hover:shadow-xl transition-all duration-200"
            >
              <Plus className="w-4 h-4 mr-2" />
              {t('collections.newCollection')}
            </Button>
          </div>
        </div>

        {/* Loading State */}
        {loading && !refreshing && <LoadingSkeleton />}

        {/* Empty State */}
        {!loading && collections.length === 0 && <EmptyState />}

        {/* Content */}
        {!loading && collections.length > 0 && (
          <>
            {/* Statistics */}
            <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-6">
                  <div className="flex items-center gap-2">
                    <Database className="h-4 w-4 text-blue-500" />
                    <span className="text-sm text-gray-600 dark:text-gray-300">{t('collections.stats.collections')}</span>
                    <span className="font-semibold text-gray-900 dark:text-gray-100">{collections.length}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <FileText className="h-4 w-4 text-green-500" />
                    <span className="text-sm text-gray-600 dark:text-gray-300">{t('collections.stats.documents')}</span>
                    <span className="font-semibold text-gray-900 dark:text-gray-100">{totalDocuments}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Archive className="h-4 w-4 text-purple-500" />
                    <span className="text-sm text-gray-600 dark:text-gray-300">{t('collections.stats.chunks')}</span>
                    <span className="font-semibold text-gray-900 dark:text-gray-100">{totalChunks}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Collections Table Card */}
            <Card className="shadow-none">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex flex-col items-center gap-2">
                    <CardTitle className="flex items-center gap-2">
                      <BookOpen className="h-5 w-5 text-blue-500" />
                      {t('collections.collectionList')}
                    </CardTitle>
                    <CardDescription>
                      {selectedCollections.length > 0 
                        ? t('common.selected', { count: selectedCollections.length }) 
                        : t('common.total', { count: collections.length }) + ' ' + t('collections.stats.collections').toLowerCase()}
                    </CardDescription>
                  </div>
                  
                  {selectedCollections.length > 0 && (
                    <AlertDialog open={showDeleteConfirm} onOpenChange={setShowDeleteConfirm}>
                      <AlertDialogTrigger asChild>
                        <Button variant="destructive" size="sm" className="flex items-center gap-2">
                          <Trash2 className="w-4 h-4" />
                          {t('collections.deleteConfirm.deleteSelected')}
                        </Button>
                      </AlertDialogTrigger>
                      <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>{t('collections.deleteConfirm.title')}</AlertDialogTitle>
                          <AlertDialogDescription>
                            ⚠️ {t('collections.deleteConfirm.description')}
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <div className="my-4">
                          <p className="text-sm text-gray-600 dark:text-gray-300 mb-2">
                            {t('collections.deleteConfirm.collectionsToDelete', { count: selectedCollections.length })}
                          </p>
                          <ul className="text-sm text-gray-700 dark:text-gray-300 mb-3 list-disc pl-5">
                            {selectedCollectionNames.map((name, idx) => (
                              <li key={idx}>{name}</li>
                            ))}
                          </ul>
                          <div className="text-sm text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20 p-2 rounded">
                            ℹ️ {t('collections.deleteConfirm.warningMessage')}
                          </div>
                        </div>
                        <AlertDialogFooter>
                          <AlertDialogCancel disabled={deleting}>{t('common.cancel')}</AlertDialogCancel>
                          <AlertDialogAction
                            onClick={handleDeleteSelected}
                            disabled={deleting}
                            className="bg-red-600 hover:bg-red-700"
                          >
                            {deleting ? (
                              <>
                                <Loader2 className="w-4 h-4 animate-spin mr-2" />
                                {t('collections.deleteConfirm.deleting')}
                              </>
                            ) : (
                              t('collections.deleteConfirm.deleteButton')
                            )}
                          </AlertDialogAction>
                        </AlertDialogFooter>
                      </AlertDialogContent>
                    </AlertDialog>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-200 dark:border-gray-700">
                        <th className="w-8 px-4 py-3 text-left">
                          <input
                            type="checkbox"
                            checked={selectedCollections.length === collections.length && collections.length > 0}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedCollections(collections.map(c => c.uuid))
                              } else {
                                setSelectedCollections([])
                              }
                            }}
                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                          />
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          {t('collections.table.collection')}
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          {t('collections.table.stats')}
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          {t('collections.table.uuid')}
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          {t('collections.table.metadata')}
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
                      {collections.map((collection) => (
                        <tr
                          key={collection.uuid}
                          className={`transition-colors hover:bg-gray-50/50 dark:hover:bg-gray-800/50 ${
                            selectedCollections.includes(collection.uuid) ? 'bg-blue-50/50 dark:bg-blue-900/20' : ''
                          }`}
                        >
                          <td className="px-4 py-4">
                            <input
                              type="checkbox"
                              checked={selectedCollections.includes(collection.uuid)}
                              onChange={() => toggleSelection(collection.uuid)}
                              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                            />
                          </td>
                          <td className="px-4 py-4">
                            <div className="flex items-center space-x-3">
                              <div className="flex-shrink-0">
                                <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-500 flex items-center justify-center">
                                  <Folder className="h-4 w-4 text-white" />
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                <Popover 
                                  open={openPopovers.has(collection.uuid)}
                                  onOpenChange={(isOpen) => togglePopover(collection.uuid, isOpen)}
                                >
                                  <PopoverTrigger asChild>
                                    <button className="text-sm font-medium text-gray-900 dark:text-gray-100 hover:text-blue-600 dark:hover:text-blue-400 transition-colors cursor-pointer flex items-center gap-1">
                                      {collection.name}
                                      <Info className="h-3 w-3 text-gray-400 dark:text-gray-400" />
                                    </button>
                                  </PopoverTrigger>
                                  <PopoverContent className="w-[500px] p-0" align="start">
                                    <div className="p-4">
                                      <div className="flex items-center justify-between mb-4">
                                        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                                          <Folder className="h-5 w-5 text-blue-500" />
                                          {collection.name}
                                        </h3>
                                        <Button
                                          variant="ghost"
                                          size="sm"
                                          onClick={() => togglePopover(collection.uuid, false)}
                                        >
                                          <X className="h-4 w-4" />
                                        </Button>
                                      </div>
                                      
                                      <div className="space-y-4">
                                        {/* 기본 정보 */}
                                        <div>
                                          <h4 className="font-medium text-sm text-gray-600 dark:text-gray-300 mb-2">{t('collections.popover.basicInfo')}</h4>
                                          <div className="space-y-2 text-sm">
                                            <div className="flex justify-between">
                                              <span className="text-gray-500 dark:text-gray-300">UUID:</span>
                                              <code className="text-xs bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded font-mono">
                                                {collection.uuid}
                                              </code>
                                            </div>
                                          </div>
                                        </div>

                                        {/* 통계 정보 */}
                                        <div>
                                          <h4 className="font-medium text-sm text-gray-600 dark:text-gray-300 mb-2">{t('collections.popover.statistics')}</h4>
                                          <div className="grid grid-cols-2 gap-3">
                                            <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg">
                                              <div className="flex items-center gap-2 mb-1">
                                                <FileText className="h-4 w-4 text-blue-500" />
                                                <span className="text-sm font-medium text-blue-700 dark:text-blue-300">{t('collections.stats.documents')}</span>
                                              </div>
                                              <div className="text-lg font-bold text-blue-900 dark:text-blue-100">
                                                {collection.stats.documents}
                                              </div>
                                            </div>
                                            <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded-lg">
                                              <div className="flex items-center gap-2 mb-1">
                                                <Archive className="h-4 w-4 text-purple-500" />
                                                <span className="text-sm font-medium text-purple-700 dark:text-purple-300">{t('collections.stats.chunks')}</span>
                                              </div>
                                              <div className="text-lg font-bold text-purple-900 dark:text-purple-100">
                                                {collection.stats.chunks}
                                              </div>
                                            </div>
                                          </div>
                                        </div>

                                        {/* 메타데이터 */}
                                        {collection.metadata && Object.keys(collection.metadata).length > 0 && (
                                          <div>
                                            <h4 className="font-medium text-sm text-gray-600 dark:text-gray-300 mb-2">{t('collections.table.metadata')}</h4>
                                            <pre className="text-xs bg-gray-50 dark:bg-gray-800 p-3 rounded overflow-x-auto whitespace-pre-wrap">
                                              {JSON.stringify(collection.metadata, null, 2)}
                                            </pre>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  </PopoverContent>
                                </Popover>
                              </div>
                            </div>
                          </td>
                          <td className="px-4 py-4">
                            <div className="flex items-center space-x-2">
                              <Badge variant="secondary" className="text-xs">
                                <FileText className="w-3 h-3 mr-1" />
                                {t('collections.stats.documentsCount', { count: collection.stats.documents })}
                              </Badge>
                              <Badge variant="outline" className="text-xs">
                                {t('collections.stats.chunksCount', { count: collection.stats.chunks })}
                              </Badge>
                            </div>
                          </td>
                          <td className="px-4 py-4">
                            <code className="text-xs text-gray-500 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded font-mono">
                              {collection.uuid.slice(0, 8)}...
                            </code>
                          </td>
                          <td className="px-4 py-4">
                            <div className="text-xs text-gray-500 dark:text-gray-300 max-w-xs truncate">
                              {Object.keys(collection.metadata || {}).length > 0 ? (
                                <code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">
                                  {JSON.stringify(collection.metadata)}
                                </code>
                              ) : (
                                <span className="text-gray-400 dark:text-gray-400 italic">{t('common.none')}</span>
                              )}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>

      {/* Create Collection Modal */}
      <CreateCollectionModal
        open={showCreateModal}
        onOpenChange={setShowCreateModal}
        onSuccess={fetchCollections}
      />
    </div>
  )
}