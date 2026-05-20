import json
from pathlib import Path

import pytest

COLLECTION_ID = "00000000-0000-0000-0000-000000000001"


def _write_generated_wiki(tmp_path: Path) -> Path:
    wiki_root = tmp_path / "llm_wiki"
    collection_dir = wiki_root / "collections" / COLLECTION_ID
    (collection_dir / "sources").mkdir(parents=True)
    (collection_dir / "concepts").mkdir()
    (collection_dir / "index.md").write_text(
        "# Wiki Index\n\n- [Source One](sources/source-one.md)\n",
        encoding="utf-8",
    )
    (collection_dir / "sources" / "source-one.md").write_text(
        "---\ntitle: Source One\n---\n\n# Source One\n\nSource body.",
        encoding="utf-8",
    )
    (collection_dir / "concepts" / "concept-one.md").write_text(
        "---\ntitle: Concept One\n---\n\n# Concept One\n\nConcept body.",
        encoding="utf-8",
    )
    (collection_dir / "SCHEMA.md").write_text("schema", encoding="utf-8")
    (collection_dir / "log.md").write_text("log", encoding="utf-8")
    (collection_dir / "manifest.json").write_text(
        json.dumps(
            {
                "collection_id": COLLECTION_ID,
                "generated_at": "2026-05-19T00:00:00+00:00",
                "sources": [
                    {
                        "type": "source",
                        "title": "Source One",
                        "path": "sources/source-one.md",
                        "slug": "source-one",
                        "id": "source-source-one",
                        "file_id": "file-1",
                        "source": "paper.md",
                        "chunk_count": 2,
                        "reference_count": 2,
                    }
                ],
                "concepts": [
                    {
                        "type": "concept",
                        "title": "Concept One",
                        "path": "concepts/concept-one.md",
                        "slug": "concept-one",
                        "id": "concept-concept-one",
                        "reference_count": 3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return wiki_root


def _artifact_reader(name: str) -> object:
    from langconnect.services import llm_wiki

    assert hasattr(llm_wiki, name), f"{name} should be exposed by llm_wiki"
    return getattr(llm_wiki, name)


def test_read_llm_wiki_index_returns_manifest_navigation(tmp_path):
    """Read the generated index and public navigation lists."""
    wiki_root = _write_generated_wiki(tmp_path)
    read_index = _artifact_reader("read_llm_wiki_index")

    result = read_index(COLLECTION_ID, wiki_root=wiki_root)

    assert result.collection_id == COLLECTION_ID
    assert result.status == "available"
    assert result.generated_at == "2026-05-19T00:00:00+00:00"
    assert result.index_markdown.startswith("# Wiki Index")
    assert [item.slug for item in result.sources] == ["source-one"]
    assert [item.slug for item in result.concepts] == ["concept-one"]


def test_read_llm_wiki_page_uses_manifest_registry_and_strips_frontmatter(tmp_path):
    """Serve only manifest-registered pages and strip YAML frontmatter."""
    wiki_root = _write_generated_wiki(tmp_path)
    read_page = _artifact_reader("read_llm_wiki_page")

    source = read_page(COLLECTION_ID, "sources", "source-one", wiki_root=wiki_root)
    concept = read_page(COLLECTION_ID, "concepts", "concept-one", wiki_root=wiki_root)

    assert source.title == "Source One"
    assert source.path == "sources/source-one.md"
    assert source.markdown.startswith("# Source One")
    assert "title: Source One" not in source.markdown
    assert concept.title == "Concept One"
    assert concept.markdown.startswith("# Concept One")


def test_missing_generated_wiki_raises_stable_not_generated_error(tmp_path):
    """Return a stable missing-wiki error for absent generated artifacts."""
    read_index = _artifact_reader("read_llm_wiki_index")
    artifact_error = _artifact_reader("LLMWikiArtifactError")

    with pytest.raises(artifact_error) as exc:
        read_index(COLLECTION_ID, wiki_root=tmp_path / "llm_wiki")

    assert exc.value.code == "wiki_not_generated"
    assert exc.value.status_code == 404


def test_invalid_manifest_raises_stable_invalid_artifact_error(tmp_path):
    """Return a stable invalid-artifact error for malformed manifests."""
    wiki_root = tmp_path / "llm_wiki"
    collection_dir = wiki_root / "collections" / COLLECTION_ID
    collection_dir.mkdir(parents=True)
    (collection_dir / "index.md").write_text("# Index", encoding="utf-8")
    (collection_dir / "manifest.json").write_text("{", encoding="utf-8")
    read_index = _artifact_reader("read_llm_wiki_index")
    artifact_error = _artifact_reader("LLMWikiArtifactError")

    with pytest.raises(artifact_error) as exc:
        read_index(COLLECTION_ID, wiki_root=wiki_root)

    assert exc.value.code == "invalid_wiki_artifact"
    assert exc.value.status_code == 500


def test_invalid_manifest_page_path_raises_invalid_artifact_error(tmp_path):
    """Reject manifest page paths that do not match the public page registry."""
    wiki_root = _write_generated_wiki(tmp_path)
    manifest_path = wiki_root / "collections" / COLLECTION_ID / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["path"] = "sources/../log.md"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    read_index = _artifact_reader("read_llm_wiki_index")
    artifact_error = _artifact_reader("LLMWikiArtifactError")

    with pytest.raises(artifact_error) as exc:
        read_index(COLLECTION_ID, wiki_root=wiki_root)

    assert exc.value.code == "invalid_wiki_artifact"


@pytest.mark.parametrize(
    ("section", "slug", "code"),
    [
        ("schema", "source-one", "invalid_wiki_page"),
        ("sources", "../log", "invalid_wiki_page"),
        ("sources", "source-one/../../log", "invalid_wiki_page"),
        ("sources", "not-in-manifest", "wiki_page_not_found"),
    ],
)
def test_read_llm_wiki_page_rejects_unsafe_or_unregistered_pages(
    tmp_path,
    section,
    slug,
    code,
):
    """Reject unsafe sections, traversal slugs, and pages outside the manifest."""
    wiki_root = _write_generated_wiki(tmp_path)
    read_page = _artifact_reader("read_llm_wiki_page")
    artifact_error = _artifact_reader("LLMWikiArtifactError")

    with pytest.raises(artifact_error) as exc:
        read_page(COLLECTION_ID, section, slug, wiki_root=wiki_root)

    assert exc.value.code == code
