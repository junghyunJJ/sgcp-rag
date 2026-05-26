"""Sweep the cosine-similarity threshold for clustering wiki source pages.

Embeds each source page (title + summary + keywords) once, then runs
sentence-transformers community_detection at several thresholds with
min_community_size fixed at 2, reporting cluster count / size distribution /
coverage / silhouette so we can pick a data-driven threshold for dynamic
concept generation. No LLM calls.

    docker cp scripts/concept_cluster_sweep.py langconnect-api:/app/scripts/
    docker exec -w /app -e PYTHONPATH=/app langconnect-api \
        python scripts/concept_cluster_sweep.py --collection-id <id>
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from langconnect.config import get_embeddings

DEFAULT_COLLECTION = "29ee1f13-2b5c-4e2b-8dff-26af9ad00ac7"
WIKI_DIR = "/app/llm_wiki/collections"
MIN_COMMUNITY_SIZE = 2
THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65)


def _page_text(page: dict) -> str:
    keywords = " ".join(str(k) for k in page.get("keywords", []))
    return f"{page['title']} {page['summary']} {keywords}".strip()


def sweep(args: argparse.Namespace) -> int:
    """Embed source pages and report clustering stats across thresholds."""
    import torch
    from sentence_transformers.util import community_detection

    pack = json.loads(
        (Path(WIKI_DIR) / f"{args.collection_id}.json").read_text(encoding="utf-8")
    )
    sources = [p for p in pack["pages"] if p.get("type") == "source"]
    texts = [_page_text(p) for p in sources]
    n = len(texts)
    print(f"source pages: {n} -- embedding...", flush=True)
    vectors = np.asarray(get_embeddings().embed_documents(texts), dtype=np.float32)
    embeddings = torch.from_numpy(vectors)

    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        silhouette_score = None

    print(f"\nmin_community_size={MIN_COMMUNITY_SIZE}")
    print(
        f"{'tau':>5} | {'clusters':>8} | {'clustered':>9} | {'coverage':>8} | "
        f"{'avg_sz':>6} | {'med_sz':>6} | {'max_sz':>6} | {'silhouette':>10}"
    )
    for tau in THRESHOLDS:
        clusters = community_detection(
            embeddings, threshold=tau, min_community_size=MIN_COMMUNITY_SIZE
        )
        sizes = [len(c) for c in clusters]
        clustered = sum(sizes)
        sil = "n/a"
        if silhouette_score is not None and len(clusters) >= 2:
            labels = np.full(n, -1)
            for cid, members in enumerate(clusters):
                for idx in members:
                    labels[idx] = cid
            mask = labels >= 0
            if len(set(labels[mask])) >= 2:
                sil = f"{silhouette_score(vectors[mask], labels[mask], metric='cosine'):.3f}"
        print(
            f"{tau:>5.2f} | {len(clusters):>8} | {clustered:>9} | "
            f"{clustered / n:>7.1%} | "
            f"{statistics.mean(sizes) if sizes else 0:>6.1f} | "
            f"{statistics.median(sizes) if sizes else 0:>6.0f} | "
            f"{max(sizes) if sizes else 0:>6} | {sil:>10}"
        )
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(sweep(parse_args()))
