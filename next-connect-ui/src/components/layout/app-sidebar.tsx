"use client"

import { usePathname } from "next/navigation"
import {
  FileText,
  Search,
  Home,
  Database,
  Code,
} from "lucide-react"
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { NavMain } from "./nav-main"
import Link from "next/link"
import { useTranslation } from "@/hooks/use-translation"

export function AppSidebar() {
  const pathname = usePathname()
  const { t } = useTranslation()

  const mainItems = [
    {
      name: t("sidebar.main"),
      href: "/",
      icon: Home,
      isActive: pathname === "/",
    },
    {
      name: t("sidebar.collections"),
      href: "/collections",
      icon: Database,
      isActive: pathname.startsWith("/collections"),
    },
    {
      name: t("sidebar.documents"),
      href: "/documents",
      icon: FileText,
      isActive: pathname.startsWith("/documents"),
    },
    {
      name: t("sidebar.search"),
      href: "/search",
      icon: Search,
      isActive: pathname.startsWith("/search"),
    },
    {
      name: t("sidebar.apiTester"),
      href: "/api-tester",
      icon: Code,
      isActive: pathname.startsWith("/api-tester"),
    },
  ]

  return (
    <>
      <Sidebar variant="inset" collapsible="icon" >
        <SidebarHeader>
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton size="lg" asChild>
                <Link href="/">
                  <div className="text-md">🔗</div>
                  <div className="grid flex-1 text-left text-sm leading-tight">
                    <span className="truncate font-medium text-lg">llmwiki</span>
                  </div>
                </Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarHeader>
        <SidebarContent>
          <NavMain title={t("sidebar.mainTitle")} items={mainItems} />
        </SidebarContent>
      </Sidebar>
    </>
  )
}
