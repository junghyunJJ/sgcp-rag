import { LLMWikiViewer } from '@/components/wiki/llm-wiki-viewer'

export default async function CollectionWikiPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  return <LLMWikiViewer collectionId={id} />
}
