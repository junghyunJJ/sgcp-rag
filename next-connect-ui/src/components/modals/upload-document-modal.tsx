"use client"

import { useState, useCallback } from "react"
import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import * as z from "zod"
import { Upload, FileText, Settings, Database, X } from "lucide-react"
import { useDropzone } from "react-dropzone"

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { toast } from "sonner"
import { useTranslation } from "@/hooks/use-translation"

interface Collection {
  uuid: string
  name: string
}

interface UploadDocumentModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  collections: Collection[]
  onSuccess?: () => void
}

export function UploadDocumentModal({ 
  open, 
  onOpenChange, 
  collections,
  onSuccess 
}: UploadDocumentModalProps) {
  const { t } = useTranslation()
  const [files, setFiles] = useState<File[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const formSchema = z.object({
    collectionId: z.string().min(1, {
      message: "Please select a collection",
    }),
    chunkSize: z.number().min(100).max(5000),
    chunkOverlap: z.number().min(0).max(1000),
    metadata: z.string().refine((value) => {
      try {
        JSON.parse(value)
        return true
      } catch {
        return false
      }
    }, {
      message: "Please enter valid JSON format",
    }),
  })

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      collectionId: "",
      chunkSize: 1000,
      chunkOverlap: 200,
      metadata: "[]",
    },
  })

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles(prev => [...prev, ...acceptedFiles])
    
    // Auto-generate metadata for new files
    const currentMetadata = JSON.parse(form.getValues("metadata") || "[]")
    const newMetadata = acceptedFiles.map(file => ({
      source: file.name,
      timestamp: new Date().toISOString()
    }))
    
    form.setValue("metadata", JSON.stringify([...currentMetadata, ...newMetadata], null, 2))
  }, [form])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
    },
    multiple: true
  })

  const removeFile = (index: number) => {
    const newFiles = files.filter((_, i) => i !== index)
    setFiles(newFiles)
    
    // Update metadata
    const currentMetadata = JSON.parse(form.getValues("metadata") || "[]")
    const newMetadata = currentMetadata.filter((_: any, i: number) => i !== index)
    form.setValue("metadata", JSON.stringify(newMetadata, null, 2))
  }

  async function onSubmit(values: z.infer<typeof formSchema>) {
    if (files.length === 0) {
      toast.error(t('documents.messages.uploadRequireFiles'))
      return
    }

    setIsLoading(true)

    try {
      const formData = new FormData()
      
      // Add files
      files.forEach(file => {
        formData.append('files', file)
      })
      
      // Add other data
      formData.append('chunk_size', values.chunkSize.toString())
      formData.append('chunk_overlap', values.chunkOverlap.toString())
      formData.append('metadatas_json', values.metadata)

      const response = await fetch(`/api/collections/${values.collectionId}/documents`, {
        method: 'POST',
        body: formData,
      })

      const result = await response.json()

      if (!result.success) {
        throw new Error(result.message || t('documents.messages.uploadError'))
      }

      toast.success(t('documents.messages.uploadSuccess'))
      
      onOpenChange(false)
      form.reset()
      setFiles([])
      onSuccess?.()
    } catch (error: any) {
      console.error("Failed to upload documents:", error)
      toast.error(t('documents.messages.uploadError'), {
        description: error.message || t('documents.messages.uploadError'),
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleCancel = () => {
    onOpenChange(false)
    form.reset()
    setFiles([])
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px] overflow-hidden p-0 max-h-[90vh] overflow-y-auto">
        <DialogHeader className="relative p-6 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20">
          <div className="relative flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-r from-green-500 to-emerald-500 shadow-lg">
              <Upload className="h-6 w-6 text-white" />
            </div>
            <div>
              <DialogTitle className="text-xl bg-gradient-to-r from-green-600 to-emerald-600 dark:from-green-400 dark:to-emerald-400 bg-clip-text text-transparent">
                {t('documents.modal.uploadTitle')}
              </DialogTitle>
              <DialogDescription className="text-muted-foreground dark:text-gray-300">
                {t('documents.modal.uploadDescription')}
              </DialogDescription>
            </div>
            <div className="ml-auto">
              <Database className="h-5 w-5 text-emerald-500 animate-pulse" />
            </div>
          </div>
        </DialogHeader>
        
        <div className="p-6">
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
              <FormField
                control={form.control}
                name="collectionId"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="flex items-center gap-2">
                      <Database className="h-4 w-4 text-blue-500" />
                      {t('search.selectCollection')}
                    </FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder={t('documents.selectCollection')} />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        {collections.map((collection) => (
                          <SelectItem key={collection.uuid} value={collection.uuid}>
                            {collection.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />

              {/* File Upload Area */}
              <div className="space-y-4">
                <label className="text-sm font-medium dark:text-gray-200 flex items-center gap-2">
                  <FileText className="h-4 w-4 text-green-500" />
                  {t('documents.modal.selectFile')}
                </label>
                
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
                    isDragActive 
                      ? 'border-green-500 bg-green-50 dark:bg-green-900/20' 
                      : 'border-gray-300 dark:border-gray-600 hover:border-green-400 dark:hover:border-green-500 hover:bg-green-50/50 dark:hover:bg-green-900/10'
                  }`}
                >
                  <input {...getInputProps()} />
                  <Upload className="mx-auto h-12 w-12 text-gray-400 dark:text-gray-400 mb-4" />
                  {isDragActive ? (
                    <p className="text-green-600 dark:text-green-400">Drop files here...</p>
                  ) : (
                    <>
                      <p className="text-gray-600 dark:text-gray-300 mb-2">Drag and drop or click to upload</p>
                      <p className="text-sm text-gray-500 dark:text-gray-300">{t('documents.modal.supportedFormats')}</p>
                    </>
                  )}
                </div>

                {/* Uploaded Files List */}
                {files.length > 0 && (
                  <div className="space-y-2">
                    <label className="text-sm font-medium dark:text-gray-200">Uploaded files ({files.length})</label>
                    <div className="max-h-32 overflow-y-auto space-y-2">
                      {files.map((file, index) => (
                        <div key={index} className="flex items-center justify-between bg-gray-50 dark:bg-gray-800 p-2 rounded">
                          <div className="flex items-center gap-2">
                            <FileText className="h-4 w-4 text-green-500" />
                            <span className="text-sm dark:text-gray-200">{file.name}</span>
                            <span className="text-xs text-gray-500 dark:text-gray-300">
                              ({(file.size / 1024).toFixed(1)} KB)
                            </span>
                          </div>
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={() => removeFile(index)}
                            className="h-6 w-6 p-0"
                          >
                            <X className="h-3 w-3" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Chunk Settings */}
              <div className="space-y-4">
                <label className="text-sm font-medium dark:text-gray-200 flex items-center gap-2">
                  <Settings className="h-4 w-4 text-purple-500" />
                  Chunk Settings
                </label>
                
                <div className="grid grid-cols-2 gap-4">
                  <FormField
                    control={form.control}
                    name="chunkSize"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Chunk Size</FormLabel>
                        <FormControl>
                          <Input 
                            type="number"
                            min={100}
                            max={5000}
                            step={100}
                            {...field}
                            onChange={(e) => field.onChange(parseInt(e.target.value))}
                          />
                        </FormControl>
                        <FormDescription className="text-gray-500 dark:text-gray-300">Maximum characters per chunk</FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  
                  <FormField
                    control={form.control}
                    name="chunkOverlap"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Chunk Overlap</FormLabel>
                        <FormControl>
                          <Input 
                            type="number"
                            min={0}
                            max={1000}
                            step={50}
                            {...field}
                            onChange={(e) => field.onChange(parseInt(e.target.value))}
                          />
                        </FormControl>
                        <FormDescription className="text-gray-500 dark:text-gray-300">Overlapping characters between chunks</FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
              </div>

              <FormField
                control={form.control}
                name="metadata"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>{t('collections.table.metadata')} (JSON)</FormLabel>
                    <FormControl>
                      <Textarea 
                        {...field}
                        rows={6}
                        className="font-mono text-sm"
                        placeholder='[{"source": "filename.pdf", "timestamp": "2024-01-01T00:00:00.000Z"}]'
                      />
                    </FormControl>
                    <FormDescription className="text-gray-500 dark:text-gray-400">
                      Enter metadata for each file in JSON array format
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <DialogFooter className="pt-6 border-t dark:border-gray-700">
                <div className="flex gap-3 w-full sm:w-auto">
                  <Button 
                    type="button" 
                    variant="outline" 
                    onClick={handleCancel}
                    disabled={isLoading}
                    className="flex-1 sm:flex-initial"
                  >
                    {t('common.cancel')}
                  </Button>
                  <Button 
                    type="submit" 
                    disabled={isLoading || files.length === 0}
                    className="flex-1 sm:flex-initial bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600"
                  >
                    {isLoading ? (
                      <>
                        <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                        {t('documents.modal.uploading')}
                      </>
                    ) : (
                      <>
                        <Upload className="mr-2 h-4 w-4" />
                        {t('documents.uploadDocument')}
                      </>
                    )}
                  </Button>
                </div>
              </DialogFooter>
            </form>
          </Form>
        </div>
      </DialogContent>
    </Dialog>
  )
}