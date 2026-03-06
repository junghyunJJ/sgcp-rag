import { NextResponse } from "next/server"
import { serverFetchAPI } from "@/lib/api"

export async function POST(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params

  try {
    const body = await request.json()

    const response = await serverFetchAPI(`/collections/${id}/agentic-search`, {
      method: "POST",
      body: JSON.stringify(body),
    })

    return NextResponse.json({ success: true, data: response }, { status: 200 })
  } catch (error: any) {
    console.error('Agentic search error:', error)
    return NextResponse.json({
      success: false,
      message: error.message || 'Agentic search failed'
    }, { status: 500 })
  }
}
