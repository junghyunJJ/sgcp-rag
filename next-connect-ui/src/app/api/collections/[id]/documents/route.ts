import { NextResponse } from "next/server"
import { serverFetchAPI } from "@/lib/api"
import { uploadFormData } from "@/lib/api"

export async function GET(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  const { searchParams } = new URL(request.url)
  const limit = searchParams.get('limit') || '10'
  const offset = searchParams.get('offset') || '0'
    
  try {
    // 백엔드 API 호출 with query parameters
    const response = await serverFetchAPI(`/collections/${id}/documents?limit=${limit}&offset=${offset}`, {
      method: "GET",
    })

    return NextResponse.json({ success: true, data: response }, { status: 200 })
  } catch (error: any) {
    return NextResponse.json({ message: error.message }, { status: 500 })
  }
}

export async function POST(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  
  try {
    // FormData를 받아서 axios로 백엔드에 전달
    const formData = await request.formData()
    
    // Use the Axios function to upload FormData
    const response = await uploadFormData(`/collections/${id}/documents`, formData)

    return NextResponse.json({ success: true, data: response }, { status: 201 })
  } catch (error: any) {
    return NextResponse.json({ 
      success: false, 
      message: error.message || 'Failed to upload documents'
    }, { status: 500 })
  }
}

export async function DELETE(
  request: Request, 
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params
  
  try {
    const body = await request.json()
    
    // Validate the request body
    if (!body.document_ids && !body.file_ids) {
      return NextResponse.json({ 
        success: false, 
        message: 'Either document_ids or file_ids must be provided' 
      }, { status: 400 })
    }
    
    const response = await serverFetchAPI(`/collections/${id}/documents`, {
      method: "DELETE",
      body: JSON.stringify(body),
    })

    // The backend returns the response directly with deleted_count
    return NextResponse.json({ 
      success: true, 
      deleted_count: response.deleted_count || 0,
      ...response 
    }, { status: 200 })
  } catch (error: any) {
    console.error('Failed to delete documents:', error)
    return NextResponse.json({ 
      success: false, 
      message: error.message || 'Failed to delete documents' 
    }, { status: 500 })
  }
}
