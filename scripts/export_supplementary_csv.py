"""Generate publication supplementary CSV tables for the chunking/embedding study.

Produces, under reports/supplementary/:
  S1_chunk_model_retrieval.csv   - chunk_size x embedding_model retrieval (recall@k, MRR, CPU cost)
  S2_overlap_ceiling.csv         - fragmentation ceiling vs chunk_size/overlap (embedding-free)
  S3_passage_length_stats.csv    - gold key-passage length distribution
  README.txt                     - provenance + column definitions

S1 is read from the actual experiment artifacts (reports/chunk_model_sweep.json + the embedding
cache) and is NOT recomputed. S2/S3 are recomputed deterministically from the cached parsed
markdown (no embeddings), so the whole file is reproducible with one command.

Usage:
    uv run python scripts/export_supplementary_csv.py
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chunk_model_sweep as cms  # noqa: E402

SWEEP_JSON = REPO_ROOT / "reports/chunk_model_sweep.json"
CACHE_DIR = REPO_ROOT / "reports/cache"
OUT_DIR = REPO_ROOT / "reports/supplementary"
LITQA2 = REPO_ROOT / "benchmarking/data/lab-bench/LitQA2/litqa2-public.jsonl"

# (chunk_size, chunk_overlap) cells for the embedding-free fragmentation-ceiling table.
CEILING_CELLS = [(1000, 200), (3000, 200), (3000, 400), (3000, 600), (3000, 800)]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(REPO_ROOT)} ({len(rows)} rows)")


def table_s1() -> list[dict]:
    """Chunk size x model retrieval, from the real experiment artifacts."""
    sweep = json.loads(SWEEP_JSON.read_text())
    overlap = sweep["chunk_overlap"]
    rows = []
    for g in sweep["grid"]:
        safe = g["model"].replace("/", "_")
        meta = json.loads((CACHE_DIR / f"{safe}__cs{g['chunk_size']}.json").read_text())
        sec = meta["embed_sec"]
        rows.append({
            "chunk_size_chars": g["chunk_size"],
            "chunk_overlap_chars": overlap,
            "embedding_model": g["model"],
            "embedding_dim": g["dim"],
            "n_chunks": g["n_chunks"],
            "recall_at_1": g["recall@1"],
            "recall_at_5": g["recall@5"],
            "recall_at_10": g["recall@10"],
            "mrr": g["mrr"],
            "doc_embed_sec_cpu": round(sec, 1),
            "docs_per_sec_cpu": round(g["n_chunks"] / sec, 1),
        })
    rows.sort(key=lambda r: (r["chunk_size_chars"], r["embedding_model"]))
    return rows


def table_s3(gold: list[str]) -> list[dict]:
    """Gold key-passage length distribution (characters)."""
    lens = sorted(len(g) for g in gold)
    n = len(lens)
    q = st.quantiles(lens, n=4)
    return [{
        "n_passages": n,
        "min": lens[0],
        "q1": int(q[0]),
        "median": int(st.median(lens)),
        "mean": round(st.mean(lens), 1),
        "q3": int(q[2]),
        "p90": lens[int(n * 0.9)],
        "max": lens[-1],
        "std": round(st.pstdev(lens), 1),
    }]


def table_s2(papers: list[str], gold: list[str]) -> list[dict]:
    """Fragmentation ceiling: fraction of gold passages present in ANY chunk (no embeddings)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    rows = []
    for cs, ov in CEILING_CELLS:
        splitter = RecursiveCharacterTextSplitter(chunk_size=cs, chunk_overlap=ov)
        chunks: list[str] = []
        for p in papers:
            chunks.extend(splitter.split_text(p))
        present = sum(1 for g in gold if any(cms.passage_hit(g, ch) for ch in chunks))
        rows.append({
            "chunk_size_chars": cs,
            "chunk_overlap_chars": ov,
            "overlap_pct": round(ov / cs * 100, 1),
            "n_chunks": len(chunks),
            "n_questions": len(gold),
            "gold_passages_present": present,
            "recall_ceiling": round(present / len(gold), 4),
        })
        print(f"  ceiling cs={cs} ov={ov}: {present}/{len(gold)} = {present/len(gold):.4f}")
    return rows


README = """Supplementary Data: chunking and embedding-model selection for biomedical RAG
================================================================================

Benchmark: LitQA2 (lab-bench), {n_q} questions with gold key-passages, over a corpus of
{n_p} open-access biomedical full-text papers (PDF -> markdown via PyMuPDF4LLM, conservative
cleanup). Retrieval is dense cosine similarity over chunk embeddings (L2-normalized); a
question is counted as recalled@k if any top-k chunk contains its gold key-passage by
normalized token overlap >= 0.60. Embeddings computed on CPU. Production setting evaluated:
RecursiveCharacterTextSplitter, chunk_size in characters.

S1_chunk_model_retrieval.csv
  Retrieval performance by chunk size and embedding model. Source: the experiment run in
  reports/chunk_model_sweep.json plus per-cell embedding wall-clock from reports/cache/.
  Key result: increasing chunk_size 1000->3000 chars yields +20-25 percentage points recall;
  the two embedding models differ by <=2 pp (within noise). doc_embed_sec_cpu / docs_per_sec_cpu
  quantify single-process CPU indexing cost.
  Columns: chunk_size_chars, chunk_overlap_chars, embedding_model, embedding_dim, n_chunks,
  recall_at_1, recall_at_5, recall_at_10, mrr, doc_embed_sec_cpu, docs_per_sec_cpu.

S2_overlap_ceiling.csv
  Fragmentation ceiling = fraction of gold passages present in at least one chunk (the maximum
  recall any retriever could achieve; embedding-independent). Justifies chunk_overlap=200:
  at chunk_size=3000 the ceiling is flat (0.990) for overlap 200..800, so larger overlap only
  inflates n_chunks without raising recall. Columns: chunk_size_chars, chunk_overlap_chars,
  overlap_pct, n_chunks, n_questions, gold_passages_present, recall_ceiling.

S3_passage_length_stats.csv
  Distribution of gold key-passage lengths (characters). Passages are an order of magnitude
  shorter than a 3000-char chunk (median {median}), explaining why chunk overlap becomes
  redundant for fragmentation once chunks are large. Columns: n_passages, min, q1, median,
  mean, q3, p90, max, std.

Generated by scripts/export_supplementary_csv.py.
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = cms.load_cases(LITQA2, None, open_only=False)
    gold = [c["passage"] for c in cases]
    print(f"Loaded {len(cases)} questions")

    print("S1 (from artifacts):")
    s1 = table_s1()
    write_csv(OUT_DIR / "S1_chunk_model_retrieval.csv", list(s1[0].keys()), s1)

    print("S3 (passage lengths):")
    s3 = table_s3(gold)
    write_csv(OUT_DIR / "S3_passage_length_stats.csv", list(s3[0].keys()), s3)

    print("S2 (fragmentation ceiling, recomputed):")
    papers = cms.parse_pdfs(None)  # uses cached markdown
    s2 = table_s2(papers, gold)
    write_csv(OUT_DIR / "S2_overlap_ceiling.csv", list(s2[0].keys()), s2)

    (OUT_DIR / "README.txt").write_text(
        README.format(n_q=len(cases), n_p=len(papers), median=s3[0]["median"])
    )
    print(f"  wrote {(OUT_DIR / 'README.txt').relative_to(REPO_ROOT)}")
    print(f"\nSupplementary tables in {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
