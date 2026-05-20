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
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params
  const response = await fetch(`${API_URL}/collections/${id}/llm-wiki`, {
    method: 'GET',
  })
  const data = await parseBackendResponse(response)

  if (!response.ok) {
    return NextResponse.json(
      errorPayload(data, 'Failed to fetch LLM Wiki'),
      { status: response.status }
    )
  }

  return NextResponse.json({ success: true, data }, { status: 200 })
}

export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params
  const body = await request.text()
  const response = await fetch(`${API_URL}/collections/${id}/llm-wiki/rebuild`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body || '{}',
  })
  const data = await parseBackendResponse(response)

  if (!response.ok) {
    return NextResponse.json(
      errorPayload(data, 'Failed to rebuild LLM Wiki'),
      { status: response.status }
    )
  }

  return NextResponse.json({ success: true, data }, { status: 200 })
}
