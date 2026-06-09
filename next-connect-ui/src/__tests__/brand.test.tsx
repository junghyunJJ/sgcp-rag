import { render, screen } from '@testing-library/react'

import { metadata } from '@/app/layout'
import { AppSidebar } from '@/components/layout/app-sidebar'
import { SidebarProvider } from '@/components/ui/sidebar'
import { LanguageProvider } from '@/providers/language-provider'
import { en } from '@/translations/en'
import { ko } from '@/translations/ko'

const DISPLAY_BRAND = 'SGCP-RAG'
const INLINE_BRAND = 'SGCP-RAG'

describe('application brand', () => {
  it('uses SGCP-RAG in app metadata', () => {
    expect(metadata.title).toBe(INLINE_BRAND)
    expect(metadata.description).toBe(INLINE_BRAND)
  })

  it('uses SGCP-RAG in translated landing copy', () => {
    expect(en.main.title).toContain(DISPLAY_BRAND)
    expect(en.main.subtitle).toContain(INLINE_BRAND)
    expect(en.main.about.description).toContain(INLINE_BRAND)

    expect(ko.main.title).toContain(DISPLAY_BRAND)
    expect(ko.main.subtitle).toContain(INLINE_BRAND)
    expect(ko.main.about.description).toContain(INLINE_BRAND)
  })

  it('renders SGCP-RAG in the sidebar header', () => {
    render(
      <LanguageProvider>
        <SidebarProvider>
          <AppSidebar />
        </SidebarProvider>
      </LanguageProvider>
    )

    expect(
      screen.getByText(
        (_content, element) =>
          element?.tagName.toLowerCase() === 'span' && element.textContent === DISPLAY_BRAND
      )
    ).toBeInTheDocument()
    expect(screen.queryByText('LangConnect')).not.toBeInTheDocument()
  })
})
