"""Chunk-size x embedding-model sweep on the REAL 190-paper biomedical corpus.

Unlike compare_embedding_models.py (short LitQA2 gold passages), this chunks the
actual full-text PDFs at several chunk sizes and measures recall against the gold
key-passages with the production token-overlap rule. It exposes the regime where
PubMedBERT's 512-token cap truncates long chunks and a long-context model pulls
ahead. Standalone: no Postgres, no langconnect.config.

Usage:
    # fast pipeline smoke (30 papers, 20 questions) -- validates wiring, NOT a verdict
    uv run python scripts/chunk_model_sweep.py --max-papers 30 --limit 20

    # full sweep
    uv run python scripts/chunk_model_sweep.py --chunk-sizes 1000,3000
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sibling import

from compare_embedding_models import (  # noqa: E402  (path set above)
    TRUST_REMOTE_CODE,
    encode,
    load_cases,
    resolve_prompts,
)

PDF_DIR = REPO_ROOT / "benchmarking/data/lab-bench/LitQA2/open_access_fulltext/pdf"
CACHE_DIR = REPO_ROOT / "reports/cache"
MD_CACHE = CACHE_DIR / "parsed_md"
DEFAULT_OUTPUT = REPO_ROOT / "reports/chunk_model_sweep.json"
DEFAULT_MODELS = ["neuml/pubmedbert-base-embeddings", "Alibaba-NLP/gte-modernbert-base"]
CHUNK_OVERLAP = 200
TOKEN_OVERLAP_THRESHOLD = 0.6


# --- ported from worktrees/feat/scripts/recall_at_k_litqa2.py (production scorer) ---
def _normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


def passage_hit(key_passage: str, chunk_text: str) -> bool:
    key_tokens = set(_normalize(key_passage).split())
    if not key_tokens:
        return False
    chunk_tokens = set(_normalize(chunk_text).split())
    return len(key_tokens & chunk_tokens) / len(key_tokens) >= TOKEN_OVERLAP_THRESHOLD


def parse_pdfs(max_papers: int | None) -> list[str]:
    """Parse PDFs -> cleaned markdown, caching each to disk so it runs once."""
    from langchain_core.documents.base import Blob

    from langconnect.parsers.pymupdf_parser import PyMuPDF4LLMParser

    MD_CACHE.mkdir(parents=True, exist_ok=True)
    parser = PyMuPDF4LLMParser(clean_markdown=True)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if max_papers:
        pdfs = pdfs[:max_papers]
    if not pdfs:
        raise SystemExit(f"No PDFs under {PDF_DIR}")

    texts: list[str] = []
    for i, pdf in enumerate(pdfs, 1):
        cached = MD_CACHE / f"{pdf.stem}.md"
        if cached.exists():
            texts.append(cached.read_text())
            continue
        docs = list(parser.lazy_parse(Blob(data=pdf.read_bytes(), mimetype="application/pdf")))
        md = "\n\n".join(d.page_content for d in docs)
        cached.write_text(md)
        texts.append(md)
        if i % 25 == 0:
            print(f"    parsed {i}/{len(pdfs)} PDFs")
    print(f"  {len(texts)} papers ready (markdown cached in {MD_CACHE})")
    return texts


def chunk_corpus(papers: list[str], chunk_size: int) -> list[str]:
    """Split every paper into chunks; returns the flat haystack of chunk texts."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=CHUNK_OVERLAP
    )
    chunks: list[str] = []
    for paper in papers:
        chunks.extend(splitter.split_text(paper))
    return chunks


def embed_chunks(model, model_id: str, chunks: list[str], chunk_size: int) -> np.ndarray:
    """Embed the haystack with disk caching keyed by (model, chunk_size, n_chunks)."""
    safe = model_id.replace("/", "_")
    npy = CACHE_DIR / f"{safe}__cs{chunk_size}.npy"
    meta = CACHE_DIR / f"{safe}__cs{chunk_size}.json"
    if npy.exists() and meta.exists():
        info = json.loads(meta.read_text())
        if info.get("n_chunks") == len(chunks):
            print(f"    [cache] doc embeddings {npy.name}")
            return np.load(npy)
    _, doc_prompt = resolve_prompts(model_id)
    vecs, secs = encode(model, chunks, doc_prompt)
    np.save(npy, vecs)
    meta.write_text(json.dumps({"n_chunks": len(chunks), "embed_sec": round(secs, 1)}))
    print(f"    embedded {len(chunks)} chunks in {secs:.0f}s ({len(chunks)/secs:.1f}/s)")
    return vecs


def score_cell(model, model_id: str, questions: list[str], gold: list[str],
               chunks: list[str], doc_vecs: np.ndarray, ks: list[int]) -> dict:
    """Recall@k via cosine ranking + token-overlap hit against gold passages."""
    query_prompt, _ = resolve_prompts(model_id)
    q_vecs, _ = encode(model, questions, query_prompt)
    sims = q_vecs @ doc_vecs.T  # normalized -> cosine
    top_k = max(ks)
    # indices of the top-k chunks per question, best first
    top_idx = np.argpartition(-sims, range(top_k), axis=1)[:, :top_k]

    metrics = {"n_chunks": len(chunks), "dim": int(doc_vecs.shape[1])}
    hits_at = {k: 0 for k in ks}
    rr_sum = 0.0
    for qi, key in enumerate(gold):
        ranked = top_idx[qi]
        first_hit = None
        for rank, ci in enumerate(ranked):
            if passage_hit(key, chunks[ci]):
                first_hit = rank
                break
        if first_hit is not None:
            rr_sum += 1.0 / (first_hit + 1)
            for k in ks:
                if first_hit < k:
                    hits_at[k] += 1
    n = len(gold)
    for k in ks:
        metrics[f"recall@{k}"] = round(hits_at[k] / n, 4)
    metrics["mrr"] = round(rr_sum / n, 4)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--chunk-sizes", default="1000,3000")
    ap.add_argument("--ks", default="1,5,10")
    ap.add_argument("--max-papers", type=int, default=None, help="smoke only; deflates distractors")
    ap.add_argument("--limit", type=int, default=None, help="limit questions")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",") if x.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cases = load_cases(REPO_ROOT / "benchmarking/data/lab-bench/LitQA2/litqa2-public.jsonl",
                       args.limit, open_only=False)
    questions = [c["question"] for c in cases]
    gold = [c["passage"] for c in cases]
    print(f"Loaded {len(cases)} questions")

    print("Parsing PDFs...")
    papers = parse_pdfs(args.max_papers)

    # Pre-chunk per size once (shared across models).
    haystacks = {cs: chunk_corpus(papers, cs) for cs in chunk_sizes}
    for cs, ch in haystacks.items():
        print(f"  chunk_size={cs}: {len(ch)} chunks")

    from sentence_transformers import SentenceTransformer

    grid: list[dict] = []
    for model_id in args.models:
        print(f"\n>>> {model_id}")
        model = SentenceTransformer(model_id, device="cpu",
                                    trust_remote_code=model_id in TRUST_REMOTE_CODE)
        for cs in chunk_sizes:
            print(f"  chunk_size={cs}")
            chunks = haystacks[cs]
            doc_vecs = embed_chunks(model, model_id, chunks, cs)
            m = score_cell(model, model_id, questions, gold, chunks, doc_vecs, ks)
            m.update(model=model_id, chunk_size=cs)
            grid.append(m)
            summary = ", ".join(f"recall@{k}={m['recall@' + str(k)]}" for k in ks)
            print(f"    {summary}, mrr={m['mrr']}")
        del model
        gc.collect()

    # grid table: rows = model, cols = chunk_size, value = recall@k for each k
    print("\n=== recall grid (rows=model, cols=chunk_size) ===")
    for k in ks:
        print(f"\nrecall@{k}:")
        header = "model".ljust(40) + "".join(f"cs={cs}".rjust(12) for cs in chunk_sizes)
        print(header)
        for model_id in args.models:
            row = model_id.ljust(40)
            for cs in chunk_sizes:
                cell = next((g for g in grid if g["model"] == model_id and g["chunk_size"] == cs), None)
                row += (f"{cell[f'recall@{k}']:.4f}" if cell else "-").rjust(12)
            print(row)

    args.output.write_text(json.dumps({
        "date": date.today().isoformat(),
        "n_questions": len(cases),
        "n_papers": len(papers),
        "chunk_sizes": chunk_sizes,
        "chunk_overlap": CHUNK_OVERLAP,
        "ks": ks,
        "grid": grid,
    }, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
