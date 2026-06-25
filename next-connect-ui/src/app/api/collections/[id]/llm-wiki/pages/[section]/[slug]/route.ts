import { NextResponse } from "next/server"
import { API_URL } from "@/lib/axios"

async function parseBackendResponse(response: Response) {
  try {
    return await response.json()
  } catch {
    return null
  }
}

function errorPayload(data: any, fallback: string) {
  const detail = data?.detail
  return {
    success: false,
    code: detail?.code || data?.code || 'llm_wiki_error',
    message: detail?.message || data?.message || fallback,
  }
}

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string; section: string; slug: string }> }
) {
  const { id, section, slug } = await params
  const response = await fetch(
    `${API_URL}/collections/${id}/llm-wiki/pages/${section}/${slug}`,
    { method: 'GET' }
  )
  const data = await parseBackendResponse(response)

  if (!response.ok) {
    return NextResponse.json(
      errorPayload(data, 'Failed to fetch SNI page'),
      { status: response.status }
    )
  }

  return NextResponse.json({ success: true, data }, { status: 200 })
}
