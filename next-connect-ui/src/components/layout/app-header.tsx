"use client"

import { usePathname } from "next/navigation"
import { SidebarTrigger } from "@/components/ui/sidebar"
import { Separator } from "@/components/ui/separator"
import { Breadcrumb, BreadcrumbItem, BreadcrumbLink, BreadcrumbList, BreadcrumbPage, BreadcrumbSeparator } from "@/components/ui/breadcrumb"
import { Home, Moon, Sun } from "lucide-react"
import { Button } from "../ui/button"
import { useTheme } from "next-themes"
import { useTranslation } from "@/hooks/use-translation"

// Navigation data structure matching sidebar
interface NavigationItem {
  name: string
  href: string
  isHome?: boolean
}

export function AppHeader() {
  const pathname = usePathname()
  const { t } = useTranslation()
  const { theme, setTheme } = useTheme()

  const navigationData: {
    main: NavigationItem[]
  } = {
    main: [
      { name: t("sidebar.main"), href: "/", isHome: true },
      { name: t("sidebar.collections"), href: "/collections" },
      { name: t("sidebar.documents"), href: "/documents" },
      { name: t("sidebar.search"), href: "/search" },
      { name: t("sidebar.apiTester"), href: "/api-tester" },
    ],
  }

  // Generate breadcrumb items based on current path
  const generateBreadcrumb = () => {
    if (pathname === "/") {
      return [{ name: t("sidebar.main"), href: "/", isHome: true }]
    }

    const allItems = [
      ...navigationData.main,
    ]

    const breadcrumbItems = []

    // Find main item
    const mainItem = allItems.find(item => 
      pathname === item.href || pathname.startsWith(item.href + "/")
    )

    if (mainItem) {
      breadcrumbItems.push(mainItem)
    }

    return breadcrumbItems
  }

  const breadcrumbItems = generateBreadcrumb()

  return (
    <header className="flex h-14 shrink-0 items-center gap-2 border-b border-border dark:border-sidebar-border transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-12 justify-between">
      <div className="flex items-center gap-2 px-4 flex-1">
        <SidebarTrigger className="-ml-1" />
        <Separator
          orientation="vertical"
          className="mr-2 data-[orientation=vertical]:h-4 bg-border dark:bg-sidebar-border"
        />
        <Breadcrumb>
          <BreadcrumbList>
            {breadcrumbItems.map((item, index) => (
              <div key={item.href} className="flex items-center">
                {index > 0 && <BreadcrumbSeparator className="text-muted-foreground dark:text-sidebar-foreground/50" />}
                <BreadcrumbItem>
                  {index === breadcrumbItems.length - 1 ? (
                    <BreadcrumbPage className="flex items-center gap-2 text-foreground dark:text-sidebar-foreground">
                      {item.isHome && <Home className="h-4 w-4" />}
                      {item.name}
                    </BreadcrumbPage>
                  ) : (
                    <BreadcrumbLink href={item.href} className="flex items-center gap-2 text-muted-foreground dark:text-sidebar-foreground/70 hover:text-foreground dark:hover:text-sidebar-foreground">
                      {item.isHome && <Home className="h-4 w-4" />}
                      {item.name}
                    </BreadcrumbLink>
                  )}
                </BreadcrumbItem>
              </div>
            ))}
          </BreadcrumbList>
        </Breadcrumb>
      </div>
      <div className="flex items-center gap-2 mr-4">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setTheme(theme === "light" ? "dark" : "light")}
          className="rounded-full hover:bg-accent dark:hover:bg-sidebar-accent"
        >
          <Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
          <Moon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
          <span className="sr-only">{t("common.toggleTheme")}</span>
        </Button>
      </div>
    </header>
  )
}
