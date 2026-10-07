"""Searches products in the catalog via Qdrant."""

import time
from dataclasses import dataclass

from qdrant_client.models import ScoredPoint

from ml.config import DEBUG, DEFAULT_TOP_K, QDRANT_COLLECTION
from ml.embedder import encode_text
from ml.indexer import get_qdrant_client


@dataclass
class SearchResult:
    """Search result."""

    item_id: int
    name: str
    group: str
    pp_code: str
    pp_link: str
    score: float


def search(query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
    """
    Searches products by text query.

    Args:
        query: user's text query.
        top_k: how many results to return.

    Returns:
        List of SearchResult, sorted by score descending.
    """
    if DEBUG:
        start = time.perf_counter()

    # 1. Encode the query
    query_vector = encode_text(query)

    # 2. Search in Qdrant
    client = get_qdrant_client()
    hits: list[ScoredPoint] = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    # 3. Build results
    results = []
    for hit in hits:
        payload = hit.payload or {}
        results.append(
            SearchResult(
                item_id=hit.id,
                name=payload.get("name", ""),
                group=payload.get("group", ""),
                pp_code=payload.get("pp_code", ""),
                pp_link=payload.get("pp_link", ""),
                score=hit.score,
            )
        )

    if DEBUG:
        elapsed = (time.perf_counter() - start) * 1000
        print(f"🔍 Search '{query}' took {elapsed:.0f}ms, found {len(results)}")

    return results


def search_and_print(query: str, top_k: int = DEFAULT_TOP_K) -> None:
    """Searches and pretty-prints results (for debugging)."""
    results = search(query, top_k)
    print(f"\n🔍 Query: '{query}'")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        print(f"{i}. [score={r.score:.3f}] {r.name}")
        print(f"   Group: {r.group} | PP code: {r.pp_code}")


if __name__ == "__main__":
    # Test queries
    test_queries = [
        "перфоратор bosch",
        "чем просверлить стену",  # semantic query ("what to drill a wall with")
        "кран на кухню",  # synonym ("tap for kitchen" = faucet)
        "провода для электрики",  # "wires for electrical work"
        "ноутбук для работы",  # "laptop for work"
    ]

    for q in test_queries:
        search_and_print(q, top_k=3)