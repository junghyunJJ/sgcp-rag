import json

from langconnect.agent.wiki_context import resolve_wiki_context


def _write_pack(tmp_path, collection_id: str, pages: list[dict[str, object]]) -> None:
    (tmp_path / f"{collection_id}.json").write_text(
        json.dumps({"collection_id": collection_id, "pages": pages}),
        encoding="utf-8",
    )


def test_resolve_wiki_context_selects_top_pages_and_renders_navigation_context(
    tmp_path,
):
    """Select relevant wiki pages and render them as navigation context."""
    _write_pack(
        tmp_path,
        "collection-1",
        [
            {
                "id": "stagent",
                "title": "STAgent",
                "summary": "Interprets pancreatic beta cell maturation.",
                "keywords": ["single-cell", "biological interpretation"],
                "source_refs": [{"file_id": "paper-a", "chunk_id": "chunk-a"}],
            },
            {
                "id": "medea",
                "title": "MEDEA target nomination",
                "summary": "Therapeutic target selection.",
                "keywords": ["target", "cell type"],
                "source_refs": [{"file_id": "paper-b", "chunk_id": "chunk-b"}],
            },
            {
                "id": "unrelated",
                "title": "Unrelated",
                "summary": "No overlap here.",
                "keywords": ["statistics"],
                "source_refs": [{"file_id": "paper-c", "chunk_id": "chunk-c"}],
            },
        ],
    )

    result = resolve_wiki_context(
        "collection-1",
        "How does STAgent interpret beta cell biology?",
        wiki_dir=tmp_path,
    )

    assert result.status == "selected"
    assert [page["id"] for page in result.selected_pages] == ["stagent"]
    assert "non-authoritative navigation memory" in result.context
    assert "STAgent" in result.context
    assert "chunk-a" in result.context


def test_resolve_wiki_context_limits_to_three_pages_and_uses_deterministic_tiebreak(
    tmp_path,
):
    """Limit selection to three pages and keep ties deterministic."""
    _write_pack(
        tmp_path,
        "collection-1",
        [
            {
                "id": "b",
                "title": "Same",
                "summary": "beta",
                "keywords": [],
                "source_refs": [{"file_id": "paper-b", "chunk_id": "chunk-b"}],
            },
            {
                "id": "a",
                "title": "Same",
                "summary": "beta",
                "keywords": [],
                "source_refs": [{"file_id": "paper-a", "chunk_id": "chunk-a"}],
            },
            {
                "id": "c",
                "title": "Alpha",
                "summary": "beta",
                "keywords": [],
                "source_refs": [{"file_id": "paper-c", "chunk_id": "chunk-c"}],
            },
            {
                "id": "d",
                "title": "Zed",
                "summary": "beta",
                "keywords": [],
                "source_refs": [{"file_id": "paper-d", "chunk_id": "chunk-d"}],
            },
        ],
    )

    result = resolve_wiki_context("collection-1", "beta", wiki_dir=tmp_path)

    assert [page["id"] for page in result.selected_pages] == ["c", "a", "b"]


def test_resolve_wiki_context_reports_missing_no_match_and_invalid_inputs(tmp_path):
    """Return finite status values for missing or unusable wiki packs."""
    assert (
        resolve_wiki_context("missing", "anything", wiki_dir=tmp_path).status
        == "missing_pack"
    )

    _write_pack(
        tmp_path,
        "collection-1",
        [
            {
                "id": "alpha",
                "title": "Alpha",
                "summary": "beta",
                "keywords": [],
                "source_refs": [{"file_id": "paper-a", "chunk_id": "chunk-a"}],
            }
        ],
    )
    assert (
        resolve_wiki_context("collection-1", "gamma", wiki_dir=tmp_path).status
        == "no_match"
    )

    (tmp_path / "bad-json.json").write_text("{", encoding="utf-8")
    assert (
        resolve_wiki_context("bad-json", "gamma", wiki_dir=tmp_path).status
        == "invalid_json"
    )

    (tmp_path / "bad-utf8.json").write_bytes(b"\xff")
    assert (
        resolve_wiki_context("bad-utf8", "gamma", wiki_dir=tmp_path).status
        == "invalid_json"
    )

    (tmp_path / "bad-schema.json").write_text(
        json.dumps({"collection_id": "bad-schema", "pages": "not-a-list"}),
        encoding="utf-8",
    )
    assert (
        resolve_wiki_context("bad-schema", "gamma", wiki_dir=tmp_path).status
        == "invalid_schema"
    )

    (tmp_path / "not-file.json").mkdir()
    assert (
        resolve_wiki_context("not-file", "gamma", wiki_dir=tmp_path).status
        == "invalid_schema"
    )

    _write_pack(
        tmp_path,
        "empty-refs",
        [
            {
                "id": "alpha",
                "title": "Alpha",
                "summary": "beta",
                "keywords": [],
                "source_refs": [],
            }
        ],
    )
    assert (
        resolve_wiki_context("empty-refs", "alpha", wiki_dir=tmp_path).status
        == "invalid_schema"
    )


def test_resolve_wiki_context_uses_env_override(tmp_path, monkeypatch):
    """Use LANGCONNECT_WIKI_CONTEXT_DIR when no explicit directory is provided."""
    _write_pack(
        tmp_path,
        "collection-1",
        [
            {
                "id": "alpha",
                "title": "Alpha",
                "summary": "beta",
                "keywords": [],
                "source_refs": [{"file_id": "paper-a", "chunk_id": "chunk-a"}],
            }
        ],
    )
    monkeypatch.setenv("LANGCONNECT_WIKI_CONTEXT_DIR", str(tmp_path))

    result = resolve_wiki_context("collection-1", "alpha")

    assert result.status == "selected"
    assert [page["id"] for page in result.selected_pages] == ["alpha"]


def test_resolve_wiki_context_rejects_unsafe_collection_id(tmp_path):
    """Reject collection ids that could escape the configured wiki directory."""
    escaped = tmp_path.parent / "escaped.json"
    escaped.write_text(
        json.dumps({"collection_id": "escaped", "pages": []}),
        encoding="utf-8",
    )

    result = resolve_wiki_context("../escaped", "anything", wiki_dir=tmp_path)

    assert result.status == "invalid_schema"
    assert result.context == ""
