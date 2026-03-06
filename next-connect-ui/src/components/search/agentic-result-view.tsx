'use client'

import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import {
  Search,
  FileCheck,
  Sparkles,
  RefreshCw,
  ShieldCheck,
  ChevronDown,
  ChevronUp,
  FileText,
  Bot,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { AgenticSearchResult } from '@/types/search'
import { useTranslation } from '@/hooks/use-translation'

const STEP_ICON_MAP: Record<string, React.ElementType> = {
  retrieve: Search,
  grade_documents: FileCheck,
  grade_generation: ShieldCheck,
  generate: Sparkles,
  rewrite_query: RefreshCw,
}

function getStepIcon(step: string) {
  const lower = step.toLowerCase()
  const key = Object.keys(STEP_ICON_MAP).find((k) => lower.startsWith(k))
  return key ? STEP_ICON_MAP[key] : FileText
}

const STEP_LABEL_MAP: Record<string, string> = {
  retrieve: 'search.stepRetrieve',
  grade_documents: 'search.stepGrade',
  grade_generation: 'search.stepCheck',
  generate: 'search.stepGenerate',
  rewrite_query: 'search.stepRewrite',
}

function stepLabelKey(step: string): string {
  const lower = step.toLowerCase()
  const match = Object.keys(STEP_LABEL_MAP).find((k) => lower.startsWith(k))
  return match ? STEP_LABEL_MAP[match] : step
}

interface AgenticResultViewProps {
  result: AgenticSearchResult
}

export default function AgenticResultView({ result }: AgenticResultViewProps) {
  const { t } = useTranslation()
  const [docsOpen, setDocsOpen] = useState(true)
  const [stepsOpen, setStepsOpen] = useState(false)
  const [rewritesOpen, setRewritesOpen] = useState(false)

  return (
    <div className="space-y-4">
      {/* AI Answer */}
      <Card className="border-blue-200 dark:border-blue-800">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Bot className="h-5 w-5 text-blue-500" />
            {t('search.aiAnswer')}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {result.generation ? (
            <div className="prose prose-sm dark:prose-invert max-w-none break-words">
              <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
                {result.generation}
              </ReactMarkdown>
            </div>
          ) : (
            <p className="text-muted-foreground italic">{t('search.agenticNoAnswer')}</p>
          )}
          {result.error && (
            <div className="mt-3 p-3 bg-red-50 dark:bg-red-950 rounded-md text-red-700 dark:text-red-300 text-sm">
              {result.error}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Source Documents */}
      {result.relevant_documents.length > 0 && (
        <Collapsible open={docsOpen} onOpenChange={setDocsOpen}>
          <Card>
            <CardHeader className="pb-3">
              <CollapsibleTrigger asChild>
                <Button variant="ghost" className="w-full justify-between p-0 h-auto hover:bg-transparent">
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <FileText className="h-5 w-5 text-green-500" />
                    {t('search.sourceDocuments')}
                    <Badge variant="secondary" className="ml-2">{result.relevant_documents.length}</Badge>
                  </CardTitle>
                  {docsOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </Button>
              </CollapsibleTrigger>
            </CardHeader>
            <CollapsibleContent>
              <CardContent className="space-y-3 pt-0">
                {result.relevant_documents.map((doc, index) => (
                  <div
                    key={index}
                    className="border border-gray-200 dark:border-gray-700 rounded-lg p-4"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <Badge variant="secondary">Doc {index + 1}</Badge>
                      {doc.metadata?.source && (
                        <Badge variant="outline" className="text-xs">
                          {doc.metadata.source}
                        </Badge>
                      )}
                    </div>
                    <p className="text-sm text-gray-900 dark:text-gray-100 whitespace-pre-wrap break-words line-clamp-5">
                      {doc.page_content}
                    </p>
                  </div>
                ))}
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      )}

      {/* Execution Steps */}
      {result.steps.length > 0 && (
        <Collapsible open={stepsOpen} onOpenChange={setStepsOpen}>
          <Card>
            <CardHeader className="pb-3">
              <CollapsibleTrigger asChild>
                <Button variant="ghost" className="w-full justify-between p-0 h-auto hover:bg-transparent">
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <Sparkles className="h-5 w-5 text-purple-500" />
                    {t('search.executionSteps')}
                    <Badge variant="secondary" className="ml-2">{result.steps.length}</Badge>
                  </CardTitle>
                  {stepsOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </Button>
              </CollapsibleTrigger>
            </CardHeader>
            <CollapsibleContent>
              <CardContent className="pt-0">
                <div className="space-y-2">
                  {result.steps.map((step, index) => {
                    const Icon = getStepIcon(step)
                    return (
                      <div key={index} className="flex items-center gap-3 text-sm">
                        <div className="flex items-center justify-center w-6 h-6 rounded-full bg-purple-100 dark:bg-purple-900">
                          <Icon className="h-3.5 w-3.5 text-purple-600 dark:text-purple-300" />
                        </div>
                        <span className="text-gray-700 dark:text-gray-300">{t(stepLabelKey(step))}</span>
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      )}

      {/* Query Rewrites */}
      {result.rewrite_count > 0 && result.query_rewrites.length > 0 && (
        <Collapsible open={rewritesOpen} onOpenChange={setRewritesOpen}>
          <Card>
            <CardHeader className="pb-3">
              <CollapsibleTrigger asChild>
                <Button variant="ghost" className="w-full justify-between p-0 h-auto hover:bg-transparent">
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <RefreshCw className="h-5 w-5 text-orange-500" />
                    {t('search.queryRewrites')}
                    <Badge variant="secondary" className="ml-2">{result.rewrite_count}</Badge>
                  </CardTitle>
                  {rewritesOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </Button>
              </CollapsibleTrigger>
            </CardHeader>
            <CollapsibleContent>
              <CardContent className="pt-0">
                <div className="space-y-2">
                  {result.query_rewrites.map((rewrite, index) => (
                    <div key={index} className="flex items-center gap-2 text-sm">
                      <Badge variant="outline" className="shrink-0">#{index + 1}</Badge>
                      <span className="text-gray-700 dark:text-gray-300">{rewrite}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      )}
    </div>
  )
}
