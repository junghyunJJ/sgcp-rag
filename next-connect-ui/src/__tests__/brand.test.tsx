import { render, screen } from '@testing-library/react'

import { metadata } from '@/app/layout'
import { AppSidebar } from '@/components/layout/app-sidebar'
import { SidebarProvider } from '@/components/ui/sidebar'
import { LanguageProvider } from '@/providers/language-provider'
import { en } from '@/translations/en'
import { ko } from '@/translations/ko'

const BRAND = 'llmwiki'

describe('application brand', () => {
  it('uses llmwiki in app metadata', () => {
    expect(metadata.title).toBe(BRAND)
    expect(metadata.description).toBe(BRAND)
  })

  it('uses llmwiki in translated landing copy', () => {
    expect(en.main.title).toContain(BRAND)
    expect(en.main.subtitle).toContain(BRAND)
    expect(en.main.about.description).toContain(BRAND)

    expect(ko.main.title).toContain(BRAND)
    expect(ko.main.subtitle).toContain(BRAND)
    expect(ko.main.about.description).toContain(BRAND)
  })

  it('renders llmwiki in the sidebar header', () => {
    render(
      <LanguageProvider>
        <SidebarProvider>
          <AppSidebar />
        </SidebarProvider>
      </LanguageProvider>
    )

    expect(screen.getByText(BRAND)).toBeInTheDocument()
    expect(screen.queryByText('LangConnect')).not.toBeInTheDocument()
  })
})
