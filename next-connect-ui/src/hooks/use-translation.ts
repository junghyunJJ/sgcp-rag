import { useCallback } from 'react'
import { useLanguage } from '@/providers/language-provider'
import { en } from '@/translations/en'
import { ko } from '@/translations/ko'

type TranslationKeys = typeof en

function getNestedValue(obj: any, path: string): string {
  return path.split('.').reduce((acc, part) => acc && acc[part], obj) || path
}

function interpolate(text: string, values: Record<string, any>): string {
  return text.replace(/\{\{(\w+)\}\}/g, (match, key) => {
    return values[key] !== undefined ? String(values[key]) : match
  })
}

export function useTranslation() {
  const { language } = useLanguage()
  const translations: TranslationKeys = language === 'ko' ? ko : en

  const t = useCallback((key: string, values?: Record<string, any>): string => {
    const translation = getNestedValue(translations, key)
    if (values) {
      return interpolate(translation, values)
    }
    return translation
  }, [translations])

  return { t, language }
}