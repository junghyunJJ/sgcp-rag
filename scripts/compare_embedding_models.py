"""Standalone, in-memory bake-off of CPU-friendly embedding models on LitQA2.

This does NOT touch the production pipeline (no Postgres, no langconnect.config).
For each candidate model it embeds the LitQA2 gold passages as the corpus and the
questions as queries, ranks by cosine similarity, and reports recall@k / MRR plus
CPU embedding throughput.

Usage:
    uv run python scripts/compare_embedding_models.py --ks 5,10
    uv run python scripts/compare_embedding_models.py \
        --models neuml/pubmedbert-base-embeddings Alibaba-NLP/gte-modernbert-base \
        --limit 10
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = REPO_ROOT / "benchmarking/data/lab-bench/LitQA2/litqa2-public.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "reports/embedding_model_comparison.json"

DEFAULT_MODELS = [
    "neuml/pubmedbert-base-embeddings",  # baseline to beat
    "Alibaba-NLP/gte-modernbert-base",
    "google/embeddinggemma-300m",
    "BAAI/bge-m3",
]

# Models that ship custom modeling code and need trust_remote_code=True.
TRUST_REMOTE_CODE = ("Alibaba-NLP/gte-modernbert-base",)


# (query_prompt, document_prompt) keyed by a case-insensitive substring of the
# HF model id. Order matters: first match wins, so put more specific keys first.
# Adding a model that needs an instruction (e.g. bge-large-en-v1.5) is one line.
_PROMPT_RULES: list[tuple[str, tuple[str, str]]] = [
    ("embeddinggemma", ("task: search result | query: ", "title: none | text: ")),
    ("bge-large-en", ("Represent this sentence for searching relevant passages: ", "")),
    ("bge-base-en", ("Represent this sentence for searching relevant passages: ", "")),
    ("bge-m3", ("", "")),  # M3 is instruction-free for retrieval
    ("gte-modernbert", ("", "")),
    ("pubmedbert", ("", "")),  # symmetric mean-pooling, no prefix
]


def resolve_prompts(model_id: str) -> tuple[str, str]:
    """Return (query_prompt, document_prompt) prefixes for a model id.

    Matched by case-insensitive substring so org-prefixed ids and minor variants
    still resolve. Unknown ids fall back to no prefix but emit a warning, so a
    typo'd model doesn't quietly look mediocre.
    """
    lid = model_id.lower()
    for key, prompts in _PROMPT_RULES:
        if key in lid:
            return prompts
    print(
        f"    [warn] no prompt rule for {model_id!r}; using no prefix. "
        "Add it to _PROMPT_RULES if this model expects query/doc instructions."
    )
    return "", ""


def load_cases(jsonl_path: Path, limit: int | None, open_only: bool) -> list[dict[str, Any]]:
    """Load LitQA2 records that carry a non-empty gold key-passage."""
    cases: list[dict[str, Any]] = []
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            passage = (rec.get("key-passage") or "").strip()
            question = (rec.get("question") or "").strip()
            if not passage or not question:
                continue
            if open_only and not rec.get("is_opensource", False):
                continue
            cases.append(
                {"id": rec["id"], "question": question, "passage": passage}
            )
            if limit and len(cases) >= limit:
                break
    return cases


def encode(model, texts: list[str], prompt: str) -> tuple[np.ndarray, float]:
    """L2-normalized embeddings + wall-clock seconds. prompt is prepended per text."""
    kwargs: dict[str, Any] = {
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    if prompt:
        kwargs["prompt"] = prompt
    start = time.perf_counter()
    vecs = model.encode(texts, **kwargs)
    return np.asarray(vecs, dtype=np.float32), time.perf_counter() - start


def score_model(model_id: str, cases: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    """Run one model over the corpus and return metrics."""
    from sentence_transformers import SentenceTransformer

    query_prompt, doc_prompt = resolve_prompts(model_id)
    model = SentenceTransformer(
        model_id,
        device="cpu",
        trust_remote_code=model_id in TRUST_REMOTE_CODE,
    )

    questions = [c["question"] for c in cases]
    passages = [c["passage"] for c in cases]

    doc_vecs, doc_secs = encode(model, passages, doc_prompt)
    query_vecs, query_secs = encode(model, questions, query_prompt)

    # Normalized vectors -> cosine similarity is a dot product.
    sims = query_vecs @ doc_vecs.T  # (n_queries, n_docs)
    # Rank doc indices per query, best first.
    ranked = np.argsort(-sims, axis=1)

    n = len(cases)
    gold = np.arange(n)  # case i's needle is doc i
    # rank position (0-based) of the gold doc for each query
    gold_rank = np.argmax(ranked == gold[:, None], axis=1)

    metrics: dict[str, Any] = {
        "model": model_id,
        "dim": int(doc_vecs.shape[1]),
        "n_cases": n,
        "query_prompt": query_prompt,
        "doc_prompt": doc_prompt,
        "doc_embed_sec": round(doc_secs, 2),
        "query_embed_sec": round(query_secs, 2),
        "docs_per_sec": round(n / doc_secs, 1) if doc_secs else None,
        "mrr": round(float(np.mean(1.0 / (gold_rank + 1))), 4),
    }
    for k in ks:
        metrics[f"recall@{k}"] = round(float(np.mean(gold_rank < k)), 4)

    del model, doc_vecs, query_vecs, sims, ranked
    gc.collect()
    return metrics


def print_table(results: list[dict[str, Any]], ks: list[int]) -> None:
    cols = ["model", "dim"] + [f"recall@{k}" for k in ks] + ["mrr", "docs_per_sec"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in results)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--open-only", action="store_true")
    ap.add_argument("--ks", default="5,10", help="comma-separated cutoffs")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    cases = load_cases(args.jsonl, args.limit, args.open_only)
    if not cases:
        raise SystemExit(f"No usable LitQA2 cases found in {args.jsonl}")
    print(f"Loaded {len(cases)} cases (corpus = {len(cases)} gold passages)")

    results = []
    for model_id in args.models:
        print(f"\n>>> {model_id}")
        try:
            results.append(score_model(model_id, cases, ks))
        except Exception as exc:  # keep going so one bad model doesn't sink the run
            print(f"    FAILED: {exc}")
            results.append({"model": model_id, "error": str(exc)})

    ok = [r for r in results if "error" not in r]
    if ok:
        print_table(ok, ks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": date.today().isoformat(),
        "n_cases": len(cases),
        "ks": ks,
        "open_only": args.open_only,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
