'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import type { AgenticSearchParams, AgenticSearchResult } from '@/types/search'

export function useAgenticSearch() {
  const [result, setResult] = useState<AgenticSearchResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  // Fix 3: Abort in-flight request on unmount
  useEffect(() => {
    return () => { abortControllerRef.current?.abort() }
  }, [])

  const execute = useCallback(async (collectionId: string, params: AgenticSearchParams) => {
    abortControllerRef.current?.abort()

    const controller = new AbortController()
    abortControllerRef.current = controller

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch(`/api/collections/${collectionId}/agentic-search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
        signal: controller.signal,
      })

      // Fix 4: Check response.ok before parsing JSON
      if (!response.ok) {
        let message = 'Agentic search failed'
        try { const err = await response.json(); message = err.message || message } catch {}
        setError(message)
        return
      }

      const res = await response.json()

      if (res.success) {
        setResult(res.data)
      } else {
        setError(res.message || 'Agentic search failed')
      }
    } catch (err: any) {
      if (err.name === 'AbortError') return
      setError(err.message || 'Agentic search failed')
    } finally {
      if (!controller.signal.aborted) {
        setLoading(false)
      }
    }
  }, [])

  const cancel = useCallback(() => {
    abortControllerRef.current?.abort()
    setLoading(false)
  }, [])

  const reset = useCallback(() => {
    abortControllerRef.current?.abort()
    setResult(null)
    setLoading(false)
    setError(null)
  }, [])

  return { result, loading, error, execute, cancel, reset }
}
