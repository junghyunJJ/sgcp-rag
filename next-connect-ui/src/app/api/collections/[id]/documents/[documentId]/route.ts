import { NextResponse } from "next/server"
import { serverFetchAPI } from "@/lib/api"

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string; documentId: string }> }
) {
  const { id, documentId } = await params
  const { searchParams } = new URL(request.url)
  const fileId = searchParams.get('file_id')
  const query = fileId ? `?file_id=${encodeURIComponent(fileId)}` : ''

  try {
    const response = await serverFetchAPI(`/collections/${id}/documents/${documentId}${query}`, {
      method: "GET",
    })

    return NextResponse.json({ success: true, data: response }, { status: 200 })
  } catch (error: any) {
    console.error('Failed to fetch document:', error)
    return NextResponse.json({
      success: false,
      message: error.message || 'Failed to fetch document',
    }, { status: 500 })
  }
}

export async function DELETE(
  request: Request, 
  { params }: { params: Promise<{ id: string; documentId: string }> }
) {
  const { id, documentId } = await params
  const { searchParams } = new URL(request.url)
  const deleteBy = searchParams.get('delete_by') || 'file_id'
  
  try {
    // Call backend API for deletion
    const response = await serverFetchAPI(`/collections/${id}/documents/${documentId}?delete_by=${deleteBy}`, {
      method: "DELETE",
    })

    return NextResponse.json({ success: true, data: response }, { status: 200 })
  } catch (error: any) {
    console.error('Failed to delete document:', error)
    return NextResponse.json({ 
      success: false, 
      message: error.message || 'Failed to delete document' 
    }, { status: 500 })
  }
}
