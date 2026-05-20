import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from uuid import UUID

import pytest

from langconnect.agent.wiki_context import resolve_wiki_context
from langconnect.models.llm_wiki import LLMWikiRebuildRequest, LLMWikiRebuildResponse

pytestmark = pytest.mark.asyncio


class _FakeCollection:
    calls: ClassVar[list[dict[str, int]]] = []
    pages: ClassVar[dict[int, list[dict[str, object]]]] = {}

    def __init__(self, collection_id: str) -> None:
        self.collection_id = collection_id

    async def list(
        self,
        *,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, object]]:
        self.calls.append({"limit": limit, "offset": offset})
        return self.pages.get(offset, [])


class _FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        if "Return one JSON object" in prompt:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "title": "Escaping Source",
                        "summary": "A bounded source summary about beta cells.",
                        "keywords": ["beta", "cells", "source"],
                        "confidence": "high",
                    }
                )
            )
        return SimpleNamespace(
            content=json.dumps(
                {
                    "concepts": [
                        {
                            "title": "../Escaping Concept",
                            "summary": "A concept summary about beta cells.",
                            "keywords": ["beta", "concept"],
                            "source_refs": [
                                {"file_id": "../bad", "chunk_id": "chunk-0"},
                                {"file_id": "../bad", "chunk_id": "chunk-00"},
                            ],
                            "confidence": "medium",
                        }
                    ]
                }
            )
        )


def _chunk(
    chunk_id: str,
    content: str,
    *,
    file_id: str = "../bad",
    source: str = "../../paper.md",
) -> dict[str, object]:
    return {
        "id": chunk_id,
        "content": content,
        "metadata": {"file_id": file_id, "source": source},
        "page_content": content,
    }


async def test_rebuild_pages_collection_and_publishes_generated_artifacts(
    tmp_path,
    monkeypatch,
):
    """Rebuild from paginated chunks and publish Markdown plus runtime pack."""
    from langconnect.services import llm_wiki

    fake_llm = _FakeLLM()
    _FakeCollection.calls = []
    _FakeCollection.pages = {
        0: [_chunk(f"chunk-{index}", "beta cell " * 400) for index in range(100)],
        100: [
            _chunk(f"chunk-{index}", "alpha cell " * 400)
            for index in range(100, 200)
        ],
        200: [_chunk("chunk-200", "terminal chunk")],
    }
    monkeypatch.setattr(llm_wiki, "Collection", _FakeCollection)
    monkeypatch.setattr(llm_wiki, "get_agent_llm", lambda **kwargs: fake_llm)

    result = await llm_wiki.rebuild_llm_wiki(
        "collection-1",
        wiki_root=tmp_path / "llm_wiki",
    )

    assert result.status == "rebuilt"
    assert result.source_page_count == 1
    assert result.concept_page_count == 1
    assert result.page_count == 2
    assert result.chunk_count == 201
    assert _FakeCollection.calls == [
        {"limit": 100, "offset": 0},
        {"limit": 100, "offset": 100},
        {"limit": 100, "offset": 200},
    ]

    collection_dir = tmp_path / "llm_wiki" / "collections" / "collection-1"
    pack_path = tmp_path / "llm_wiki" / "collections" / "collection-1.json"
    assert pack_path == Path(result.pack_path)
    assert (collection_dir / "SCHEMA.md").exists()
    assert (collection_dir / "index.md").exists()
    assert (collection_dir / "log.md").exists()
    manifest = json.loads((collection_dir / "manifest.json").read_text())
    assert manifest["source_count"] == 1
    assert manifest["concept_count"] == 1
    assert manifest["runtime_pack_path"] == str(pack_path)

    source_files = list((collection_dir / "sources").glob("*.md"))
    concept_files = list((collection_dir / "concepts").glob("*.md"))
    assert len(source_files) == 1
    assert len(concept_files) == 1
    assert all(".." not in path.name for path in [*source_files, *concept_files])
    assert source_files[0].resolve().is_relative_to(collection_dir.resolve())
    assert concept_files[0].resolve().is_relative_to(collection_dir.resolve())

    frontmatter = source_files[0].read_text(encoding="utf-8").split("---", 2)[1]
    for field in (
        "title:",
        "type:",
        "summary:",
        "keywords:",
        "source_refs:",
        "generated_at:",
        "updated_at:",
        "confidence:",
    ):
        assert field in frontmatter

    index_text = (collection_dir / "index.md").read_text(encoding="utf-8")
    assert "## Sources" in index_text
    assert "## Concepts" in index_text
    assert index_text.index("## Concepts") < index_text.index("## Sources")
    assert "A bounded source summary" in index_text
    assert "A concept summary" in index_text

    resolved = resolve_wiki_context(
        "collection-1",
        "beta cells",
        wiki_dir=tmp_path / "llm_wiki" / "collections",
    )
    assert resolved.status == "selected"


async def test_rebuild_enforces_prompt_budgets(tmp_path, monkeypatch):
    """Source and concept prompts stay within explicit first-pass budgets."""
    from langconnect.services import llm_wiki

    fake_llm = _FakeLLM()
    _FakeCollection.calls = []
    _FakeCollection.pages = {
        0: [
            _chunk(f"chunk-{index:02d}", str(index) * 5000)
            for index in range(llm_wiki.SOURCE_MAX_CHUNKS + 5)
        ],
        100: [],
    }
    monkeypatch.setattr(llm_wiki, "Collection", _FakeCollection)
    monkeypatch.setattr(llm_wiki, "get_agent_llm", lambda **kwargs: fake_llm)

    await llm_wiki.rebuild_llm_wiki(
        "collection-1",
        wiki_root=tmp_path / "llm_wiki",
    )

    source_prompt = fake_llm.prompts[0]
    concept_prompt = fake_llm.prompts[-1]
    assert source_prompt.count("<chunk ") == llm_wiki.SOURCE_MAX_CHUNKS
    assert len(source_prompt) <= llm_wiki.SOURCE_INPUT_CHAR_LIMIT
    assert len(concept_prompt) <= llm_wiki.CONCEPT_INPUT_CHAR_LIMIT


async def test_rebuild_fails_on_invalid_concept_source_refs(tmp_path, monkeypatch):
    """Malformed LLM concept refs fail cleanly instead of silently defaulting."""
    from langconnect.services import llm_wiki

    class InvalidConceptLLM(_FakeLLM):
        async def ainvoke(self, prompt: str) -> SimpleNamespace:
            if "Return one JSON object" in prompt:
                return await super().ainvoke(prompt)
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "concepts": [
                            {
                                "title": "Invalid Concept",
                                "summary": "Invalid refs.",
                                "keywords": ["invalid"],
                                "source_refs": [
                                    {"file_id": "missing", "chunk_id": "missing"}
                                ],
                                "confidence": "medium",
                            }
                        ]
                    }
                )
            )

    fake_llm = InvalidConceptLLM()
    _FakeCollection.calls = []
    _FakeCollection.pages = {0: [_chunk("chunk-0", "content")], 100: []}
    monkeypatch.setattr(llm_wiki, "Collection", _FakeCollection)
    monkeypatch.setattr(llm_wiki, "get_agent_llm", lambda **kwargs: fake_llm)

    with pytest.raises(llm_wiki.LLMWikiRebuildError, match="valid source_refs"):
        await llm_wiki.rebuild_llm_wiki(
            "collection-1",
            wiki_root=tmp_path / "llm_wiki",
        )

    assert not (tmp_path / "llm_wiki" / "collections" / "collection-1.json").exists()


async def test_rebuild_empty_collection_publishes_empty_pack_without_llm(
    tmp_path,
    monkeypatch,
):
    """Existing empty collections publish an empty pack without LLM credentials."""
    from langconnect.services import llm_wiki

    _FakeCollection.calls = []
    _FakeCollection.pages = {0: []}
    monkeypatch.setattr(llm_wiki, "Collection", _FakeCollection)
    monkeypatch.setattr(
        llm_wiki,
        "get_agent_llm",
        lambda **kwargs: pytest.fail("LLM should not be created for empty rebuild"),
    )

    result = await llm_wiki.rebuild_llm_wiki(
        "collection-1",
        wiki_root=tmp_path / "llm_wiki",
    )

    pack = json.loads(
        (tmp_path / "llm_wiki" / "collections" / "collection-1.json").read_text(
            encoding="utf-8"
        )
    )
    assert result.status == "rebuilt"
    assert result.page_count == 0
    assert result.chunk_count == 0
    assert pack["pages"] == []


async def test_failed_pack_validation_preserves_existing_public_wiki(
    tmp_path,
    monkeypatch,
):
    """Invalid staged pack must not replace the old Markdown or runtime pack."""
    from langconnect.services import llm_wiki

    wiki_root = tmp_path / "llm_wiki"
    collection_dir = wiki_root / "collections" / "collection-1"
    collection_dir.mkdir(parents=True)
    (collection_dir / "log.md").write_text("old log", encoding="utf-8")
    old_pack = {
        "collection_id": "collection-1",
        "pages": [
            {
                "id": "old",
                "title": "Old Page",
                "summary": "old beta",
                "keywords": ["old"],
                "source_refs": [{"file_id": "old-file", "chunk_id": "old-chunk"}],
            }
        ],
    }
    (wiki_root / "collections" / "collection-1.json").write_text(
        json.dumps(old_pack),
        encoding="utf-8",
    )

    fake_llm = _FakeLLM()
    _FakeCollection.calls = []
    _FakeCollection.pages = {0: [_chunk("new-chunk", "new content")], 100: []}
    monkeypatch.setattr(llm_wiki, "Collection", _FakeCollection)
    monkeypatch.setattr(llm_wiki, "get_agent_llm", lambda **kwargs: fake_llm)
    monkeypatch.setattr(llm_wiki, "_validate_runtime_pack", lambda *args: False)

    with pytest.raises(llm_wiki.LLMWikiRebuildError):
        await llm_wiki.rebuild_llm_wiki("collection-1", wiki_root=wiki_root)

    assert (collection_dir / "log.md").read_text(encoding="utf-8") == "old log"
    assert json.loads((wiki_root / "collections" / "collection-1.json").read_text())[
        "pages"
    ][0]["id"] == "old"


async def test_rest_rebuild_endpoint_delegates_llm_overrides(monkeypatch):
    """REST rebuild validates UUID params and forwards LLM override fields."""
    import langconnect.api.llm_wiki as api

    captured: dict[str, object] = {}

    async def fake_rebuild(
        collection_id: str,
        **kwargs: object,
    ) -> LLMWikiRebuildResponse:
        captured.update({"collection_id": collection_id, **kwargs})
        return LLMWikiRebuildResponse(
            collection_id=collection_id,
            source_page_count=1,
            concept_page_count=1,
            page_count=2,
            chunk_count=3,
            pack_path=f"llm_wiki/collections/{collection_id}.json",
        )

    monkeypatch.setattr(api, "rebuild_llm_wiki", fake_rebuild)

    result = await api.llm_wiki_rebuild(
        UUID("00000000-0000-0000-0000-000000000001"),
        LLMWikiRebuildRequest(
            llm_provider="ollama",
            llm_model="qwen",
            llm_temperature=0.1,
        ),
    )

    assert captured == {
        "collection_id": "00000000-0000-0000-0000-000000000001",
        "llm_provider": "ollama",
        "llm_model": "qwen",
        "llm_temperature": 0.1,
    }
    assert result.status == "rebuilt"
