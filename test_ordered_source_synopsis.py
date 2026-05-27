"""Regression tests for ordered LLM Wiki source synopsis extraction."""

from __future__ import annotations

# ruff: noqa: S101, SLF001
import pytest
from langchain_core.documents import Document

from langconnect.services import document_processor, llm_wiki


class _UploadFileStub:
    filename = "agentpaper.md"
    content_type = "text/markdown"

    async def read(self) -> bytes:
        return b"unused by monkeypatched parser"


@pytest.mark.asyncio
async def test_process_document_records_chunk_order_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Store stable chunk order metadata during ingest."""

    def parse(_blob) -> list[Document]:
        return [
            Document(
                page_content=(
                    "First ordered section introduces the paper.\n\n"
                    "Second ordered section expands the overview.\n\n"
                    "Third ordered section closes the synopsis."
                ),
                metadata={"source": "agentpaper.md"},
            )
        ]

    monkeypatch.setattr(document_processor.MIMETYPE_BASED_PARSER, "parse", parse)

    chunks = await document_processor.process_document(
        _UploadFileStub(),
        chunk_size=45,
        chunk_overlap=0,
    )

    assert len(chunks) > 1
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )
    assert {chunk.metadata["chunk_count"] for chunk in chunks} == {len(chunks)}


def _chunk(
    *,
    chunk_id: str,
    content: str,
    chunk_index: int,
) -> llm_wiki._Chunk:
    return llm_wiki._Chunk(
        id=chunk_id,
        content=content,
        metadata={
            "file_id": "agentpaper-file",
            "source": "agentpaper.pdf",
            "chunk_index": chunk_index,
        },
        file_id="agentpaper-file",
        source="agentpaper.pdf",
    )


def test_source_synopsis_uses_ordered_front_chunks_not_uuid_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build synopsis input from source-order front chunks, not UUID order."""
    monkeypatch.delenv(llm_wiki.WIKI_ABSTRACT_SUMMARY_ENV, raising=False)
    monkeypatch.setattr(llm_wiki, "_load_abstract_overrides", dict)

    chunks = [
        _chunk(
            chunk_id="00000000-references-sort-first",
            chunk_index=2,
            content=(
                "# References\n\n"
                "Smith J. Prior citations describe unrelated methods and should "
                "not become the source synopsis. Journal of Examples. 2020."
            ),
        ),
        _chunk(
            chunk_id="ffffffff-front-sort-last",
            chunk_index=0,
            content=(
                "# AgentPaper\n\n"
                "AgentPaper introduces an ordered front-window synopsis for "
                "LLM Wiki source pages. It preserves the opening document context "
                "before later methods and references are considered."
            ),
        ),
        _chunk(
            chunk_id="aaaaaaaa-methods-middle",
            chunk_index=1,
            content=(
                "## Methods\n\n"
                "The methods section contains implementation details that should "
                "not displace the opening synopsis."
            ),
        ),
    ]

    [(file_id, ordered_chunks)] = llm_wiki._group_chunks_by_source(chunks)
    prompt, selected_chunks = llm_wiki._source_prompt(file_id, ordered_chunks)

    assert [chunk.id for chunk in ordered_chunks] == [
        "ffffffff-front-sort-last",
        "aaaaaaaa-methods-middle",
        "00000000-references-sort-first",
    ]
    assert "AgentPaper introduces an ordered front-window synopsis" in prompt
    assert "Prior citations describe unrelated methods" not in prompt
    assert [ref["chunk_id"] for ref in llm_wiki._source_refs(selected_chunks)] == [
        "ffffffff-front-sort-last",
        "aaaaaaaa-methods-middle",
        "00000000-references-sort-first",
    ]


def test_source_synopsis_mode_uses_four_ordered_front_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep abstract-like source synopsis input to the first four chunks."""
    monkeypatch.delenv(llm_wiki.WIKI_ABSTRACT_SUMMARY_ENV, raising=False)
    monkeypatch.setattr(llm_wiki, "_load_abstract_overrides", dict)

    chunks = [
        _chunk(
            chunk_id=f"chunk-{index}",
            chunk_index=index,
            content=(
                f"Chunk {index} contains ordered front-window context for source "
                "synopsis extraction and enough prose to pass lead line filters."
            ),
        )
        for index in range(6)
    ]

    front_window, front_chunks = llm_wiki._source_front_window(chunks)
    _, selected_chunks = llm_wiki._source_prompt("agentpaper-file", chunks)

    assert [chunk.metadata["chunk_index"] for chunk in front_chunks] == [0, 1, 2, 3]
    assert "Chunk 3 contains ordered front-window context" in front_window
    assert "Chunk 4 contains ordered front-window context" not in front_window
    assert [chunk.metadata["chunk_index"] for chunk in selected_chunks] == [0, 1, 2, 3]
